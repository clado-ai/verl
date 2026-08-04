"""Fused Qwen3.5 attention pre-attention epilogue (QK-RMSNorm + RoPE + gate) in Triton.

Context
-------
In a LoRA fine-tune the base attention projection weights are *frozen*, but the
attention forward still runs every step. For a Qwen3.5-style **gated** GQA block
the per-step pre-attention work after the projection GEMMs is:

  q_proj(x) -> view(t,heads,head_dim*2) -> chunk -> query | gate
  query -> q_norm (per-head RMSNorm, head_dim) -> RoPE
  k_proj(x) -> view(t,kv_heads,head_dim) -> k_norm -> RoPE
  v_proj(x) -> view(t,kv_heads,head_dim)            (passthrough)
  (after attention: attn_out * sigmoid(gate))

HF runs this **eager** (the worker does NOT torch.compile, verified): two fp32
`Qwen3_5RMSNorm` calls plus `apply_rotary_pos_emb` whose `rotate_half` does
`torch.cat([-x2, x1])` on non-contiguous head tensors — a stack of small ops that
round-trip HBM repeatedly.

PARTIAL ROTARY (the real Qwen3.5 config)
----------------------------------------
The real Qwen3.5-4B uses ``partial_rotary_factor=0.25`` -> ``rotary_dim = head_dim//4``
(64 for head_dim 256). HF's ``apply_rotary_pos_emb`` reads ``rotary_dim = cos.shape[-1]``
and rotates ONLY ``q[..., :rotary_dim]`` while passing ``q[..., rotary_dim:]`` through
unrotated; ``rotate_half`` then operates *within* the rotary slice (half-width
``rotary_dim//2`` = 32). The q/k RMSNorm is still applied to the FULL head_dim (256).
The cos/sin tables span ``rotary_dim`` and (non-interleaved RoPE) have
``cos[..., :rotary_dim//2] == cos[..., rotary_dim//2:]``.

This module implements that partial-rotary epilogue directly: RMSNorm over the full
head_dim, RoPE on ``[0, rotary_dim)`` (rotate-half-width ``rotary_dim//2``), and the
normed tail ``[rotary_dim, head_dim)`` copied through unrotated. ``rotary_dim ==
head_dim`` (full rotary) is the natural special case and also works. So the epilogue
now ENGAGES on the real Qwen3.5 partial-rotary config rather than raising.

Empirical result (bench/qkv_result.md, dedicated A40/A6000 bf16):

  * A custom Triton **GEMM** does NOT beat cuBLAS for any q/k/v/o shape or size
    (data-parallel 0.81-0.99x; persistent/split-K/stream-K worse). The stacked
    projection GEMM stays cuBLAS (`F.linear`).
  * Fusing the **entire elementwise epilogue** (q_norm+RoPE, k_norm+RoPE, gate
    extract, v view) into ONE Triton kernel, on top of the cuBLAS GEMM output, is
    a real win vs the worker's eager path at all of {2048,8192,32768} tokens.

So this module ships the **epilogue** kernel (the win), not a GEMM kernel. Callers
keep their cuBLAS stacked projection and call :func:`fused_qkv_epilogue` on the
output. Guarded by :func:`self_test`; a failed self-test means "fall back to eager".
"""

from __future__ import annotations

from chalk.utils import rel_l2

# NOTE: ``torch``/``triton``/``triton.language`` are imported LAZILY (inside the build
# function and the install/self-test paths), NOT at module load — so this module imports
# cleanly on a CPU/no-Triton control plane and only touches Triton once the QKV kernel is
# actually enabled + installed, mirroring triton_embedding/triton_lora/fp8_base.


# Module-level cache for the lazily-built Triton kernel + its launch wrapper. The
# kernel is defined INSIDE ``_build_kernel`` (which imports triton/triton.language
# there), so this module imports cleanly on a CPU/no-Triton control plane and only
# touches Triton once the QKV path is actually enabled + installed.
_FUSED_EPILOGUE = None


def _build_kernel():
    """Import triton/triton.language and define the fused QK-norm+RoPE+gate epilogue
    kernel + its ``fused_qkv_epilogue`` launch wrapper. Returns ``fused_qkv_epilogue``;
    raises on any import/compile problem (the caller keeps the eager path). Cached so the
    ``@triton.jit`` kernel is compiled once. Mirrors how the other opt-in kernel modules
    (``triton_embedding``/``triton_lora``/``fp8_base``) defer their triton import."""
    global _FUSED_EPILOGUE
    if _FUSED_EPILOGUE is not None:
        return _FUSED_EPILOGUE

    import torch
    import triton
    import triton.language as tl

    # -----------------------------------------------------------------------
    # Fused QK-RMSNorm + partial-RoPE + gate-extract epilogue.
    #
    # Layout of the stacked cuBLAS output ``o`` is [tokens, q_raw + k_out + v_out]:
    #   q_raw  = n_q_heads * head_dim * 2   (query | gate, per head: [query | gate])
    #   k_out  = n_kv_heads * head_dim
    #   v_out  = n_kv_heads * head_dim
    # One Triton program handles one (row-block, head). It loads the head's full
    # head_dim, computes the fp32 RMSNorm over the WHOLE head_dim, then:
    #   * dims [0, rotary_dim): RoPE via rotate_half (half-width rotary_dim//2),
    #       out_lo = n_lo*cos - n_hi*sin ;  out_hi = n_hi*cos + n_lo*sin
    #   * dims [rotary_dim, head_dim): copied through (normed, unrotated)
    # cos/sin span ``rotary_dim`` only (== head_dim for full rotary). The gate slice
    # (q heads) is copied through unchanged; v is a passthrough view (no copy).
    # BLOCK_M=32 is fastest (max SM parallelism for this bandwidth-bound op).
    # -----------------------------------------------------------------------
    @triton.jit
    def _qkv_epilogue_kernel(
        o_ptr,
        q_ptr,
        k_ptr,
        g_ptr,
        qn_ptr,
        kn_ptr,
        cos_ptr,
        sin_ptr,
        M,
        stride_om,
        stride_on,
        stride_qm,
        stride_qn,
        stride_km,
        stride_kn,
        stride_gm,
        stride_gn,
        stride_cosm,
        stride_cosd,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROT: tl.constexpr,  # rotary_dim (<= HEAD_DIM); RoPE applied to [0, ROT)
        RHALF: tl.constexpr,  # ROT // 2 (rotate-half width within the rotary slice)
        N_Q: tl.constexpr,
        N_KV: tl.constexpr,
        Q_RAW: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)  # head index in [0, N_Q + N_KV)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)  # full head, for RMSNorm + tail passthrough
        offs_rlo = tl.arange(0, RHALF)  # rotary lo half [0, RHALF)
        offs_rhi = RHALF + tl.arange(0, RHALF)  # rotary hi half [RHALF, ROT)
        m_mask = offs_m[:, None] < M

        # cos/sin span the rotary slice only (width ROT). Non-interleaved RoPE =>
        # cos[..,:RHALF] == cos[..,RHALF:]; load both halves explicitly anyway (robust
        # if that duplication ever changes).
        cos_lo = tl.load(
            cos_ptr + offs_m[:, None] * stride_cosm + offs_rlo[None, :] * stride_cosd,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        cos_hi = tl.load(
            cos_ptr + offs_m[:, None] * stride_cosm + offs_rhi[None, :] * stride_cosd,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        sin_lo = tl.load(
            sin_ptr + offs_m[:, None] * stride_cosm + offs_rlo[None, :] * stride_cosd,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        sin_hi = tl.load(
            sin_ptr + offs_m[:, None] * stride_cosm + offs_rhi[None, :] * stride_cosd,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)

        is_q = pid_n < N_Q
        if is_q:
            base = pid_n * (HEAD_DIM * 2)
            qb = pid_n * HEAD_DIM
            # full-head load -> RMSNorm over the WHOLE head_dim -> store the NORMED head.
            x = tl.load(
                o_ptr + offs_m[:, None] * stride_om + (base + offs_d)[None, :] * stride_on,
                mask=m_mask,
                other=0.0,
            ).to(tl.float32)
            w = tl.load(qn_ptr + offs_d).to(tl.float32)
            var = tl.sum(x * x, axis=1) / HEAD_DIM
            rstd = 1.0 / tl.sqrt(var + EPS)
            n = x * rstd[:, None] * (1.0 + w)[None, :]  # [BLOCK_M, HEAD_DIM] normed
            # Store the normed head; the rotary slice [0, ROT) is overwritten below with
            # the roped values. The tail [ROT, HEAD_DIM) is normed + passed through (no
            # rotation) exactly as HF apply_rotary_pos_emb does — this single power-of-2
            # store covers any tail width (avoids a non-power-of-2 arange on the tail).
            tl.store(
                q_ptr + offs_m[:, None] * stride_qm + (qb + offs_d)[None, :] * stride_qn,
                n.to(q_ptr.dtype.element_ty),
                mask=m_mask,
            )
            # rotary slice [0, ROT): RoPE via rotate_half (lo/hi within ROT) on the NORMED
            # halves. cos/sin span ROT (cos_lo == cos_hi for non-interleaved RoPE).
            wl = tl.load(qn_ptr + offs_rlo).to(tl.float32)
            wh = tl.load(qn_ptr + offs_rhi).to(tl.float32)
            x_rlo = tl.load(
                o_ptr + offs_m[:, None] * stride_om + (base + offs_rlo)[None, :] * stride_on,
                mask=m_mask,
                other=0.0,
            ).to(tl.float32)
            x_rhi = tl.load(
                o_ptr + offs_m[:, None] * stride_om + (base + offs_rhi)[None, :] * stride_on,
                mask=m_mask,
                other=0.0,
            ).to(tl.float32)
            nr_lo = x_rlo * rstd[:, None] * (1.0 + wl)[None, :]
            nr_hi = x_rhi * rstd[:, None] * (1.0 + wh)[None, :]
            out_lo = nr_lo * cos_lo - nr_hi * sin_lo
            out_hi = nr_hi * cos_hi + nr_lo * sin_hi
            tl.store(
                q_ptr + offs_m[:, None] * stride_qm + (qb + offs_rlo)[None, :] * stride_qn,
                out_lo.to(q_ptr.dtype.element_ty),
                mask=m_mask,
            )
            tl.store(
                q_ptr + offs_m[:, None] * stride_qm + (qb + offs_rhi)[None, :] * stride_qn,
                out_hi.to(q_ptr.dtype.element_ty),
                mask=m_mask,
            )
            # gate passthrough: columns [base+HEAD_DIM : base+2*HEAD_DIM]
            g = tl.load(
                o_ptr + offs_m[:, None] * stride_om + (base + HEAD_DIM + offs_d)[None, :] * stride_on,
                mask=m_mask,
                other=0.0,
            )
            tl.store(
                g_ptr + offs_m[:, None] * stride_gm + (qb + offs_d)[None, :] * stride_gn,
                g,
                mask=m_mask,
            )
        else:
            kv_head = pid_n - N_Q
            if kv_head < N_KV:
                base = Q_RAW + kv_head * HEAD_DIM
                kb = kv_head * HEAD_DIM
                x = tl.load(
                    o_ptr + offs_m[:, None] * stride_om + (base + offs_d)[None, :] * stride_on,
                    mask=m_mask,
                    other=0.0,
                ).to(tl.float32)
                w = tl.load(kn_ptr + offs_d).to(tl.float32)
                var = tl.sum(x * x, axis=1) / HEAD_DIM
                rstd = 1.0 / tl.sqrt(var + EPS)
                n = x * rstd[:, None] * (1.0 + w)[None, :]
                tl.store(
                    k_ptr + offs_m[:, None] * stride_km + (kb + offs_d)[None, :] * stride_kn,
                    n.to(k_ptr.dtype.element_ty),
                    mask=m_mask,
                )
                wl = tl.load(kn_ptr + offs_rlo).to(tl.float32)
                wh = tl.load(kn_ptr + offs_rhi).to(tl.float32)
                x_rlo = tl.load(
                    o_ptr + offs_m[:, None] * stride_om + (base + offs_rlo)[None, :] * stride_on,
                    mask=m_mask,
                    other=0.0,
                ).to(tl.float32)
                x_rhi = tl.load(
                    o_ptr + offs_m[:, None] * stride_om + (base + offs_rhi)[None, :] * stride_on,
                    mask=m_mask,
                    other=0.0,
                ).to(tl.float32)
                nr_lo = x_rlo * rstd[:, None] * (1.0 + wl)[None, :]
                nr_hi = x_rhi * rstd[:, None] * (1.0 + wh)[None, :]
                out_lo = nr_lo * cos_lo - nr_hi * sin_lo
                out_hi = nr_hi * cos_hi + nr_lo * sin_hi
                tl.store(
                    k_ptr + offs_m[:, None] * stride_km + (kb + offs_rlo)[None, :] * stride_kn,
                    out_lo.to(k_ptr.dtype.element_ty),
                    mask=m_mask,
                )
                tl.store(
                    k_ptr + offs_m[:, None] * stride_km + (kb + offs_rhi)[None, :] * stride_kn,
                    out_hi.to(k_ptr.dtype.element_ty),
                    mask=m_mask,
                )

    def fused_qkv_epilogue(
        o,
        qn,
        kn,
        cos,
        sin,
        *,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        block_m: int = 32,
    ):
        """Fused QK-RMSNorm + (partial) RoPE + gate-extract over a stacked cuBLAS projection.

        ``o``   : [tokens, q_raw + k_out + v_out] bf16, the output of
                  ``F.linear(x, stacked_qkv_weight)`` where the q block is
                  ``n_q_heads*head_dim*2`` wide (query|gate per head).
        ``qn``/``kn`` : q_norm/k_norm RMSNorm weights, shape [head_dim] (used as 1+w).
        ``cos``/``sin``: [tokens, rotary_dim] rotary tables. ``rotary_dim`` may be
                  ``head_dim`` (full rotary) or smaller (partial rotary, e.g. Qwen3.5's
                  ``rotary_dim = head_dim//4``); the rotary slice is rotated and the
                  tail ``[rotary_dim, head_dim)`` is normed + copied through unrotated,
                  matching HF ``apply_rotary_pos_emb`` (``rotary_dim = cos.shape[-1]``).

        Returns ``(q, k, v, gate)``:
          q    : [tokens, n_q_heads*head_dim]   normed + (partial) roped
          k    : [tokens, n_kv_heads*head_dim]  normed + (partial) roped
          v    : [tokens, n_kv_heads*head_dim]  VIEW into ``o`` (passthrough, no copy)
          gate : [tokens, n_q_heads*head_dim]   raw gate (apply sigmoid post-attention)
        """
        t = o.shape[0]
        q_raw = n_q_heads * head_dim * 2
        q_out = n_q_heads * head_dim
        k_out = n_kv_heads * head_dim
        rot = cos.shape[-1]
        if rot > head_dim or rot % 2 != 0:
            raise ValueError(
                f"fused_qkv_epilogue: rotary_dim (cos width {rot}) must be even and <= head_dim ({head_dim})"
            )
        rhalf = rot // 2
        q = torch.empty((t, q_out), device=o.device, dtype=o.dtype)
        k = torch.empty((t, k_out), device=o.device, dtype=o.dtype)
        g = torch.empty((t, q_out), device=o.device, dtype=o.dtype)
        v = o[:, q_raw + k_out :]  # passthrough view (no copy; matches eager)
        grid = (triton.cdiv(t, block_m), n_q_heads + n_kv_heads)
        with torch.cuda.device(o.device):
            _qkv_epilogue_kernel[grid](
                o,
                q,
                k,
                g,
                qn,
                kn,
                cos,
                sin,
                t,
                o.stride(0),
                o.stride(1),
                q.stride(0),
                q.stride(1),
                k.stride(0),
                k.stride(1),
                g.stride(0),
                g.stride(1),
                cos.stride(0),
                cos.stride(1),
                EPS=eps,
                BLOCK_M=block_m,
                HEAD_DIM=head_dim,
                ROT=rot,
                RHALF=rhalf,
                N_Q=n_q_heads,
                N_KV=n_kv_heads,
                Q_RAW=q_raw,
                num_warps=4,
                num_stages=2,
            )
        return q, k, v, g

    _FUSED_EPILOGUE = fused_qkv_epilogue
    return fused_qkv_epilogue


def fused_qkv_epilogue(o, qn, kn, cos, sin, **kwargs):
    """Public entry point: lazily build (and cache) the Triton epilogue kernel, then
    run it. Importing this module does NOT import triton; the import happens here, on
    first use, behind the enabled/install gate."""
    return _build_kernel()(o, qn, kn, cos, sin, **kwargs)


def build_stacked_qkv_weight(w_q, w_k, w_v):
    """Stack the three frozen projection weights into one [q_raw+k_out+v_out, in]
    matrix for a single cuBLAS ``F.linear`` (read ``x`` once, one launch). The
    GEMM stays cuBLAS — only the epilogue is Triton. ``torch`` is imported lazily so
    this module stays import-safe without triton/torch on a CPU control plane."""
    import torch

    return torch.cat([w_q, w_k, w_v], dim=0).contiguous()


# ===========================================================================================
# TRAINABLE fused QK-RMSNorm + partial-RoPE (fwd+bwd).
#
# The eval-only ``fused_qkv_epilogue`` above operates on a STACKED cuBLAS projection and has NO
# autograd backward, so it engages only on the frozen-base inference path (q/k/v excluded from
# LoRA + no grad). In a REAL LoRA fine-tune the default ``LORA_TARGETS=all-linear`` wraps q/k/v
# as ``lora.Linear`` and grad is enabled, so that epilogue never fires.
#
# This section adds a TRAINABLE fused QK-norm+RoPE that operates on the ALREADY-PROJECTED query /
# key states ``[tokens, n_heads, head_dim]`` (the output of q_proj/k_proj, AFTER any LoRA delta —
# the GEMM+LoRA stays in PyTorch autograd). It replaces the eager ``q_norm(query)`` (an fp32
# Qwen3_5RMSNorm) + ``apply_rotary_pos_emb`` (rotate_half ``cat([-x2,x1])``) stack — two ops with
# an intermediate HBM round-trip — with ONE Triton kernel that has a proper backward, so it
# ENGAGES in training and composes with q/k/v LoRA.
#
# fwd (per head row x[head_dim], weight w[head_dim], partial-rotary tables cos/sin[rot]):
#     var = mean(x^2); rstd = 1/sqrt(var+eps); n_i = x_i * rstd * (1 + w_i)         # RMSNorm
#     rotary [0,rot): y_lo = n_lo*cos_lo - n_hi*sin_lo ;  y_hi = n_hi*cos_hi + n_lo*sin_hi
#     tail   [rot,head_dim): y_i = n_i                                              # normed, unrotated
# bwd (gy -> gn through the orthogonal RoPE transpose, then RMSNorm backward):
#     tail:   gn_i = gy_i
#     rotary: gn_lo = gy_lo*cos_lo + gy_hi*sin_hi ;  gn_hi = -gy_lo*sin_lo + gy_hi*cos_hi
#     gw_i = gn_i * x_i * rstd
#     c = sum_i gn_i * x_i * (1+w_i) ;  gx_j = rstd*(1+w_j)*gn_j - (rstd^3 * x_j / D) * c
# cos/sin are constants (no grad). ``w`` accumulates a grad (q_norm/k_norm weights are trainable
# parameters in HF — small per-head vectors). The whole [tokens, n_heads] grid is one launch per
# direction; one program handles one (row-block, head), same tiling as the eval epilogue.
# ===========================================================================================
_QKNORM_ROPE = None
_OUT_GATE = None


def _build_qknorm_rope():
    """Build the TRAINABLE fused QK-RMSNorm+partial-RoPE fwd+bwd Triton kernels + autograd
    Function. Returns ``qknorm_rope(x, w, cos, sin, eps) -> y`` where ``x``/``y`` are
    ``[tokens, n_heads, head_dim]`` and ``cos``/``sin`` are ``[tokens, rot]`` (rot<=head_dim,
    even). Cached. Raises on any import/compile problem (caller keeps eager)."""
    global _QKNORM_ROPE
    if _QKNORM_ROPE is not None:
        return _QKNORM_ROPE

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _qknorm_rope_fwd_kernel(
        x_ptr,
        w_ptr,
        cos_ptr,
        sin_ptr,
        y_ptr,
        M,
        stride_xm,
        stride_xh,
        stride_xd,
        stride_ym,
        stride_yh,
        stride_yd,
        stride_cosm,
        stride_cosd,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROT: tl.constexpr,
        RHALF: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_rlo = tl.arange(0, RHALF)
        offs_rhi = RHALF + tl.arange(0, RHALF)
        m_mask = offs_m[:, None] < M

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + pid_h * stride_xh + offs_d[None, :] * stride_xd,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + offs_d).to(tl.float32)
        var = tl.sum(x * x, axis=1) / HEAD_DIM
        rstd = 1.0 / tl.sqrt(var + EPS)
        n = x * rstd[:, None] * (1.0 + w)[None, :]
        # store the full normed head (tail passthrough); rotary slice overwritten below
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + pid_h * stride_yh + offs_d[None, :] * stride_yd,
            n.to(y_ptr.dtype.element_ty),
            mask=m_mask,
        )
        cos_lo = tl.load(
            cos_ptr + offs_m[:, None] * stride_cosm + offs_rlo[None, :] * stride_cosd, mask=m_mask, other=0.0
        ).to(tl.float32)
        cos_hi = tl.load(
            cos_ptr + offs_m[:, None] * stride_cosm + offs_rhi[None, :] * stride_cosd, mask=m_mask, other=0.0
        ).to(tl.float32)
        sin_lo = tl.load(
            sin_ptr + offs_m[:, None] * stride_cosm + offs_rlo[None, :] * stride_cosd, mask=m_mask, other=0.0
        ).to(tl.float32)
        sin_hi = tl.load(
            sin_ptr + offs_m[:, None] * stride_cosm + offs_rhi[None, :] * stride_cosd, mask=m_mask, other=0.0
        ).to(tl.float32)
        wl = tl.load(w_ptr + offs_rlo).to(tl.float32)
        wh = tl.load(w_ptr + offs_rhi).to(tl.float32)
        x_lo = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + pid_h * stride_xh + offs_rlo[None, :] * stride_xd,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        x_hi = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + pid_h * stride_xh + offs_rhi[None, :] * stride_xd,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        n_lo = x_lo * rstd[:, None] * (1.0 + wl)[None, :]
        n_hi = x_hi * rstd[:, None] * (1.0 + wh)[None, :]
        out_lo = n_lo * cos_lo - n_hi * sin_lo
        out_hi = n_hi * cos_hi + n_lo * sin_hi
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + pid_h * stride_yh + offs_rlo[None, :] * stride_yd,
            out_lo.to(y_ptr.dtype.element_ty),
            mask=m_mask,
        )
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + pid_h * stride_yh + offs_rhi[None, :] * stride_yd,
            out_hi.to(y_ptr.dtype.element_ty),
            mask=m_mask,
        )

    @triton.jit
    def _qknorm_rope_bwd_kernel(
        x_ptr,
        w_ptr,
        cos_ptr,
        sin_ptr,
        gy_ptr,
        gx_ptr,
        gw_partial_ptr,  # gw_partial: [n_prog_m, n_heads, HEAD_DIM] fp32
        M,
        N_HEADS,
        stride_xm,
        stride_xh,
        stride_xd,
        stride_gym,
        stride_gyh,
        stride_gyd,
        stride_gxm,
        stride_gxh,
        stride_gxd,
        stride_cosm,
        stride_cosd,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROT: tl.constexpr,
        RHALF: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        m_mask = offs_m[:, None] < M

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + pid_h * stride_xh + offs_d[None, :] * stride_xd,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + offs_d).to(tl.float32)
        s = 1.0 + w  # [HEAD_DIM]
        var = tl.sum(x * x, axis=1) / HEAD_DIM
        rstd = 1.0 / tl.sqrt(var + EPS)  # [BLOCK_M]
        gy = tl.load(
            gy_ptr + offs_m[:, None] * stride_gym + pid_h * stride_gyh + offs_d[None, :] * stride_gyd,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        # gn = grad wrt the NORMED head n, computed PER FULL COLUMN d (transpose of the orthogonal
        # RoPE rotation; tail d>=ROT passes gy through). For a column d:
        #   d in [0,RHALF):    partner = d+RHALF (hi); gn = gy[d]*cos[d]    + gy[partner]*sin[partner]
        #   d in [RHALF,ROT):  partner = d-RHALF (lo); gn = gy[d]*cos[d]    - gy[partner]*sin[partner]
        #   d >= ROT:          gn = gy[d]
        # (cos[d]==cos[partner] and sin[d]==sin[partner] for non-interleaved RoPE, so using the
        # per-column cos[d]/sin[partner] is exact and avoids re-deriving the duplicated halves.)
        in_rot = offs_d[None, :] < ROT
        is_lo = offs_d[None, :] < RHALF
        rot_mask = m_mask & in_rot
        cos_d = tl.load(
            cos_ptr + offs_m[:, None] * stride_cosm + offs_d[None, :] * stride_cosd, mask=rot_mask, other=0.0
        ).to(tl.float32)
        partner = tl.where(is_lo, offs_d[None, :] + RHALF, offs_d[None, :] - RHALF)
        gy_partner = tl.load(
            gy_ptr + offs_m[:, None] * stride_gym + pid_h * stride_gyh + partner * stride_gyd,
            mask=rot_mask,
            other=0.0,
        ).to(tl.float32)
        sin_partner = tl.load(
            sin_ptr + offs_m[:, None] * stride_cosm + partner * stride_cosd,
            mask=rot_mask,
            other=0.0,
        ).to(tl.float32)
        # sign of the cross term: +sin for lo columns, -sin for hi columns
        cross = tl.where(is_lo, gy_partner * sin_partner, -gy_partner * sin_partner)
        gn_rot = gy * cos_d + cross
        gn = tl.where(in_rot, gn_rot, gy)

        # RMSNorm backward: gw_i = gn_i * x_i * rstd ; c = sum_i gn_i*x_i*s_i ;
        # gx_j = rstd*s_j*gn_j - (rstd^3 * x_j / D) * c
        gxs = gn * x * s[None, :]  # gn_i * x_i * s_i  [BLOCK_M, HEAD_DIM]
        c = tl.sum(gxs, axis=1)  # [BLOCK_M]
        gx = (
            rstd[:, None] * s[None, :] * gn
            - (rstd[:, None] * rstd[:, None] * rstd[:, None]) * (x / HEAD_DIM) * c[:, None]
        )
        tl.store(
            gx_ptr + offs_m[:, None] * stride_gxm + pid_h * stride_gxh + offs_d[None, :] * stride_gxd,
            gx.to(gx_ptr.dtype.element_ty),
            mask=m_mask,
        )
        # gw_i = gn_i * x_i * rstd, reduced over rows. Accumulate this program's row-block into a
        # per-(prog_m, head) partial (no atomics); the host sums the prog_m axis -> [HEAD_DIM] grad.
        gw = tl.sum(tl.where(m_mask, gn * x * rstd[:, None], 0.0), axis=0)  # [HEAD_DIM]
        gw_off = (pid_m * N_HEADS + pid_h) * HEAD_DIM + offs_d
        tl.store(gw_partial_ptr + gw_off, gw)

    def _launch_fwd(x, w, cos, sin, eps, block_m):
        M, H, D = x.shape
        rot = cos.shape[-1]
        rhalf = rot // 2
        y = torch.empty_like(x)
        grid = (triton.cdiv(M, block_m), H)
        with torch.cuda.device(x.device):
            _qknorm_rope_fwd_kernel[grid](
                x,
                w,
                cos,
                sin,
                y,
                M,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                y.stride(0),
                y.stride(1),
                y.stride(2),
                cos.stride(0),
                cos.stride(1),
                EPS=eps,
                BLOCK_M=block_m,
                HEAD_DIM=D,
                ROT=rot,
                RHALF=rhalf,
                num_warps=4,
                num_stages=2,
            )
        return y

    def _launch_bwd(x, w, cos, sin, gy, eps, block_m):
        M, H, D = x.shape
        rot = cos.shape[-1]
        rhalf = rot // 2
        n_prog_m = triton.cdiv(M, block_m)
        gx = torch.empty_like(x)
        gw_partial = torch.empty((n_prog_m, H, D), device=x.device, dtype=torch.float32)
        grid = (n_prog_m, H)
        with torch.cuda.device(x.device):
            _qknorm_rope_bwd_kernel[grid](
                x,
                w,
                cos,
                sin,
                gy,
                gx,
                gw_partial,
                M,
                H,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                gy.stride(0),
                gy.stride(1),
                gy.stride(2),
                gx.stride(0),
                gx.stride(1),
                gx.stride(2),
                cos.stride(0),
                cos.stride(1),
                EPS=eps,
                BLOCK_M=block_m,
                HEAD_DIM=D,
                ROT=rot,
                RHALF=rhalf,
                num_warps=4,
                num_stages=2,
            )
        # sum the per-head partials over the row-block axis and over heads -> [HEAD_DIM] (q_norm /
        # k_norm weights are shared across heads, exactly like HF's single [head_dim] RMSNorm weight).
        gw = gw_partial.sum(dim=(0, 1)).to(w.dtype)
        return gx, gw

    class _QKNormRopeFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, cos, sin, eps, block_m):
            x = x.contiguous()
            cos = cos.contiguous()
            sin = sin.contiguous()
            y = _launch_fwd(x, w.contiguous(), cos, sin, eps, block_m)
            ctx.save_for_backward(x, w, cos, sin)
            ctx.eps = eps
            ctx.block_m = block_m
            return y

        @staticmethod
        def backward(ctx, gy):
            x, w, cos, sin = ctx.saved_tensors
            gx, gw = _launch_bwd(x, w, cos, sin, gy.contiguous(), ctx.eps, ctx.block_m)
            # grads: x, w, cos(None), sin(None), eps(None), block_m(None)
            return gx, gw, None, None, None, None

    def qknorm_rope(x, w, cos, sin, eps: float = 1e-6, block_m: int = 32):
        """Trainable fused QK-RMSNorm + partial RoPE on ``x`` [tokens, n_heads, head_dim].
        ``w`` is the [head_dim] norm weight (used as 1+w); ``cos``/``sin`` are [tokens, rot]
        (rot<=head_dim, even). Returns y [tokens, n_heads, head_dim] (normed + partial-roped).
        Differentiable wrt ``x`` and ``w`` (cos/sin are constants)."""
        rot = cos.shape[-1]
        if rot > x.shape[-1] or rot % 2 != 0:
            raise ValueError(f"qknorm_rope: rot ({rot}) must be even and <= head_dim ({x.shape[-1]})")
        return _QKNormRopeFn.apply(x, w, cos, sin, eps, block_m)

    _QKNORM_ROPE = qknorm_rope
    return qknorm_rope


def qknorm_rope(x, w, cos, sin, eps: float = 1e-6, block_m: int = 32):
    """Public entry point: lazily build (and cache) the trainable QK-norm+RoPE kernel, then run
    it. Importing this module does NOT import triton; the import happens here, on first use."""
    return _build_qknorm_rope()(x, w, cos, sin, eps=eps, block_m=block_m)


# ===========================================================================================
# PRODUCTION DISPATCH for the fused QK-norm+partial-RoPE op (``load_qkv``).
#
# ``load_qkv`` mirrors ``chalk.ops.rmsnorm.load_rmsnorm`` exactly: it routes through
# ``chalk.ops.arch.load_entry("qkv", ...)`` so a per-arch tuned kernel
# (``chalk/ops/arch/<arch>/qkv.py`` with ``TUNED = True``) is selected on the running GPU when it
# passes the op's live-GPU fwd+bwd self-test vs the fp32 oracle, else the PORTABLE 3D single-tensor
# ``qknorm_rope`` kernel (built above — the exact kernel production has always run) is used. The arch
# tree is a pure, self-validated speedup overlay: it can only accelerate production, never degrade it.
#
# THE ONE CONTRACT: the entry is the shipped PORTABLE contract itself — the per-tensor 3D callable
# ``qknorm_rope(x, w, cos, sin, eps) -> y`` on ``[tokens, heads, head_dim]`` with ``cos``/``sin``
# ``[tokens, rot]`` — so ``load_entry``'s self-test, ``portable=``, each arch ``build()``, AND the
# installed ``_train_forward`` (which calls the entry once per q and once per k) all speak the same
# signature with ZERO layout adaptation. Dispatching an arch kernel is a drop-in kernel swap on the
# existing token-major path; there is no 4D round-trip, so an arch kernel that is faster than the
# portable kernel accelerates production one-for-one, and TUNED=False (or a self-test failure) leaves
# production byte-for-byte on the portable kernel it runs today.
#
# NOTE ON PHASE-2b (sm89/sm90): the two verified qkv kernels were graded by the autoresearch verifier
# on ITS contiguous heads-major ``[b, heads, t, head_dim]`` layout against a 4D-adapter baseline that
# pays a ``transpose(1,2).reshape()`` copy. Production's token-major ``[tokens, heads, head_dim]``
# path never pays that copy, so re-measured on-box in this real dispatched path (arch kernel vs the
# direct portable ``qknorm_rope``) neither kernel is a robust win: sm89 is ~break-even/slightly slower
# and sm90 catastrophically regresses (BLOCK_M=256 backward spills at head_dim=256 -> ~0.2-0.3x). They
# are therefore shipped as research results with ``TUNED = False`` (kept for reference / a future
# token-major-tuned rewrite); load_qkv correctly dispatches the portable kernel until a real win lands.
# ===========================================================================================
def _eager_qknorm_rope(q, k, q_w, k_w, cos, sin, eps):
    """Exact combined fp32 reference matching the qkv ENTRY signature (the manifest oracle
    ``chalk.ops.qkv._eager_qknorm_rope``): ``(q, k, q_w, k_w, cos, sin, eps) -> (q_out, k_out)`` on 4D
    ``[b, heads, t, head_dim]`` q/k with ``cos``/``sin`` ``[b, t, rot]`` (or ``[t, rot]``). Gemma-style
    RMSNorm over the FULL head_dim (``(1 + weight)`` in fp32) then non-interleaved partial RoPE on
    ``[0, rot)`` with a normed, unrotated tail — byte-for-byte the HF ``q_norm``/``k_norm`` +
    ``apply_rotary_pos_emb`` stack the fused kernel replaces."""
    return (
        _eager_qknorm_rope_one(q, q_w, cos, sin, eps),
        _eager_qknorm_rope_one(k, k_w, cos, sin, eps),
    )


def _eager_qknorm_rope_one(a, w, cos, sin, eps):
    """Per-tensor fp32 reference on ``a = [b, heads, t, head_dim]`` (token axis = -2), ``cos``/``sin``
    ``[b, t, rot]`` or ``[t, rot]`` (broadcast over batch+heads)."""
    import torch

    rot = cos.shape[-1]
    rhalf = rot // 2
    t = a.shape[-2]
    af = a.float()
    n = af * torch.rsqrt(af.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())
    n_rot, n_pass = n[..., :rot], n[..., rot:]
    n1, n2 = n_rot[..., :rhalf], n_rot[..., rhalf:]
    rot_half = torch.cat([-n2, n1], dim=-1)
    c = cos.reshape(t, rot).float()[None, None]  # [1, 1, t, rot] -> broadcast over (batch, heads)
    s = sin.reshape(t, rot).float()[None, None]
    return torch.cat([n_rot * c + rot_half * s, n_pass], dim=-1)


def _self_test_entry(fn, tol: float = 2e-2, rotary_dim: int = 64) -> None:
    """Live-GPU fwd+bwd parity of a per-tensor entry ``fn(x, w, cos, sin, eps) -> y`` (the qkv
    dispatch contract) vs the exact eager HF reference (RMSNorm over head_dim then partial RoPE) on
    the real Qwen3.5 partial-rotary config. ``x`` is ``[tokens, heads, head_dim]`` and ``cos``/``sin``
    are ``[tokens, rot]`` — the SAME per-tensor call ``_train_forward`` makes for q and for k. Checks
    y, dx, dw; raises on any mismatch so ``load_entry`` falls back to portable (and ``load_qkv`` keeps
    the eager path). This is exactly the gate ``self_test_trainable`` applies, over a passed entry."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA for qkv self-test")
    dev, eps, hd = "cuda", 1e-6, 256
    rot, rhalf = rotary_dim, rotary_dim // 2
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gen = torch.Generator(device=dev).manual_seed(0)
    for n_heads, t in ((16, 512), (4, 1024)):
        x = torch.randn(t, n_heads, hd, device=dev, dtype=dtype, generator=gen)
        w = (0.1 * torch.randn(hd, device=dev, dtype=torch.float32, generator=gen)).to(dtype)
        pos = torch.arange(t, device=dev).float()
        inv = 1.0 / (1e7 ** (torch.arange(0, rhalf, device=dev).float() / rhalf))
        ang = pos[:, None] * inv[None, :]
        emb = torch.cat([ang, ang], dim=-1)
        cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
        gy = torch.randn(t, n_heads, hd, device=dev, dtype=dtype, generator=gen)

        def rms(a, ww):
            return a * torch.rsqrt(a.float().pow(2).mean(-1, keepdim=True) + eps) * (1.0 + ww.float())

        def rope_partial(a, cos=cos, sin=sin):  # a [t, heads, hd]; cos/sin [t, rot] broadcast over heads
            a_rot, a_pass = a[..., :rot], a[..., rot:]
            a1, a2 = a_rot[..., :rhalf], a_rot[..., rhalf:]
            rot_half = torch.cat([-a2, a1], dim=-1)
            c = cos.float()[:, None]
            s = sin.float()[:, None]
            return torch.cat([a_rot * c + rot_half * s, a_pass], dim=-1)

        xr = x.float().clone().requires_grad_(True)
        wr = w.float().clone().requires_grad_(True)
        ref = rope_partial(rms(xr, wr))
        ref.backward(gy.float())

        xc = x.clone().requires_grad_(True)
        wc = w.clone().requires_grad_(True)
        got = fn(xc, wc, cos, sin, eps)
        torch.cuda.synchronize()
        got.backward(gy)
        torch.cuda.synchronize()

        rel = rel_l2  # fp32 relative-L2 error; shared helper (see chalk.utils.rel_l2)
        for nm, gv, rv in (("y", got, ref), ("dx", xc.grad, xr.grad), ("dw", wc.grad, wr.grad)):
            r = rel(gv, rv)
            if not (r < tol):
                raise RuntimeError(f"qkv self-test failed {nm} heads={n_heads} t={t}: {r:.2e} (tol={tol:.0e})")


def load_qkv():
    """Return the arch-tuned-or-portable per-tensor ``qknorm_rope(x, w, cos, sin, eps) -> y`` entry
    (``x`` = ``[tokens, heads, head_dim]``), gated on its live-GPU fwd+bwd self-test; else ``None``
    (the caller keeps eager). Never raises. Mirrors ``chalk.ops.rmsnorm.load_rmsnorm``: the returned
    entry is self-tested exactly once inside ``load_entry`` (on the arch branch OR the portable one),
    so no second self-test is layered on here."""
    from chalk.ops.arch import load_entry
    from chalk.ops.arch import load_kernel

    def _st(f):
        _self_test_entry(f)

    def _build():
        return load_entry("qkv", _st, portable=_build_qknorm_rope)

    return load_kernel(
        "qkv",
        "fused Triton QK-norm+partial-RoPE (fwd+bwd) enabled",
        "fused Triton QK-norm+partial-RoPE disabled",
        build=_build,
    )


def _build_out_gate():
    """Build the TRAINABLE fused attention OUTPUT-GATE fwd+bwd Triton kernels + autograd
    Function. The Qwen3.5 gated-attention block, AFTER the SDPA core, runs the EAGER
    sigmoid-gate then o_proj::

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # [B,L, H*Dh]
        attn_output = attn_output * torch.sigmoid(gate)                    # <-- this op
        attn_output = self.o_proj(attn_output)

    where ``gate`` is the SECOND chunk of ``q_proj(x)`` (so it carries the LoRA delta and is
    differentiable). This fuses ``y = x * sigmoid(g)`` into ONE Triton launch (fwd) and one
    in-place launch (bwd) — like the SwiGLU activation fuse (the closest analog; the GDN gate
    negative does NOT apply: the GDN gate is tiny [N,<=64] and autograd-dispatch bound, whereas
    this gate is full-width [tokens, H*Dh ~= 2048-5120] and HBM-bound, the SwiGLU-win regime).

    Math::
        y  = x * sigmoid(g)                    (sigmoid in fp32, product cast to dtype)
        dx = gy * sigmoid(g)
        dg = gy * x * sigmoid(g) * (1 - sigmoid(g))

    Returns ``out_gate(x, g) -> y`` (both [..., N], any leading shape). Cached. Raises on any
    import/compile problem (caller keeps the eager gate)."""
    global _OUT_GATE
    if _OUT_GATE is not None:
        return _OUT_GATE

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _out_gate_fwd_kernel(
        x_ptr,
        g_ptr,
        o_ptr,
        row_stride,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # One program per row (grid (M,)), whole-row block next_pow2(N) with a masked tail —
        # the SwiGLU forward launch shape. Elementwise + HBM-bound, so no col-tiling.
        pid = tl.program_id(0).to(tl.int64)
        x_ptr += pid * row_stride
        g_ptr += pid * row_stride
        o_ptr += pid * row_stride
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0)
        g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # sigmoid needs fp32
        sig = tl.sigmoid(g).to(x.dtype)
        tl.store(o_ptr + cols, x * sig, mask=mask)

    @triton.jit
    def _out_gate_bwd_kernel(
        x_ptr,
        g_ptr,
        do_ptr,
        dx_ptr,
        dg_ptr,
        row_stride,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # One pass writing dx/dg to FRESH buffers (NOT in-place). Unlike the SwiGLU activation —
        # where the saved gate/up are MLP projection outputs consumed only by SwiGLU, so the
        # in-place overwrite is safe — here ``x``(=attn_output) and ``g``(=gate) are NOT private to
        # this op: ``gate`` is a chunk-VIEW of the q_proj output whose OTHER chunk (query) is still
        # read by the QK-norm path's backward, and ``attn_output`` is the SDPA output. The forward's
        # ``.contiguous()`` may alias the original storage, so overwriting it corrupts gradients the
        # rest of the attention graph needs (verified E2E: in-place blew up q_proj/q_norm grads).
        pid = tl.program_id(0).to(tl.int64)
        x_ptr += pid * row_stride
        g_ptr += pid * row_stride
        do_ptr += pid * row_stride
        dx_ptr += pid * row_stride
        dg_ptr += pid * row_stride
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        dx = do * sig
        dg = do * x * sig * (1.0 - sig)
        tl.store(dx_ptr + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        tl.store(dg_ptr + cols, dg.to(dg_ptr.dtype.element_ty), mask=mask)

    def _warps(block: int) -> int:
        # forward = 2 loads + 1 store, backward = 3 loads + 2 stores; both are light, so follow
        # SwiGLU's modest curve (wide rows want parallelism to hide HBM latency).
        if block >= 8192:
            return 16
        if block >= 2048:
            return 8
        if block >= 512:
            return 4
        return 2

    class _OutGateFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, g):
            sh = x.shape
            N = sh[-1]
            x2 = x.contiguous().view(-1, N)
            g2 = g.contiguous().view(-1, N)
            M = x2.shape[0]
            out = torch.empty_like(x2)
            block = triton.next_power_of_2(N)
            with torch.cuda.device(x2.device):
                _out_gate_fwd_kernel[(M,)](
                    x2,
                    g2,
                    out,
                    x2.stride(0),
                    N=N,
                    BLOCK_N=block,
                    num_warps=_warps(block),
                )
            ctx.save_for_backward(x2, g2)
            ctx._sh = sh
            return out.view(sh)

        @staticmethod
        def backward(ctx, dout):
            x2, g2 = ctx.saved_tensors
            sh = ctx._sh
            N = sh[-1]
            M = x2.shape[0]
            do2 = dout.contiguous().view(-1, N)
            # FRESH grad buffers (not in-place — see the bwd-kernel comment): x2/g2 may alias tensors
            # the rest of the attention graph still reads.
            dx2 = torch.empty_like(x2)
            dg2 = torch.empty_like(g2)
            block = triton.next_power_of_2(N)
            with torch.cuda.device(x2.device):
                _out_gate_bwd_kernel[(M,)](
                    x2,
                    g2,
                    do2,
                    dx2,
                    dg2,
                    x2.stride(0),
                    N=N,
                    BLOCK_N=block,
                    num_warps=_warps(block),
                )
            return dx2.view(sh), dg2.view(sh)

    def out_gate(x, g):
        return _OutGateFn.apply(x, g)

    _OUT_GATE = out_gate
    return out_gate


def out_gate(x, g):
    """Public entry point: lazily build (and cache) the fused attention output-gate kernel
    (``y = x * sigmoid(g)``, fwd+bwd), then run it. Importing this module does NOT import
    triton; the import happens here, on first use. ``x`` and ``g`` are [..., N] (same shape)."""
    return _build_out_gate()(x, g)


def self_test_out_gate(tol: float = 2e-2) -> bool:
    """Live-GPU fwd+bwd parity of the fused output-gate vs the exact eager reference
    (``x * sigmoid(g)``), at real Qwen3.5 attention-output widths. Checks y, dx, dg in fp32
    against an fp32 oracle (bf16/fp16 toleranced). Returns True iff every rel-err < tol; False on
    any failure (caller keeps the eager gate)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        dev = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        gen = torch.Generator(device=dev).manual_seed(0)
        # Real Qwen3.5 attn-output widths H*head_dim: 0.8B/2B 16*256=4096, 4B 32*256=8192,
        # plus a narrow odd-N case to exercise the masked tail.
        for n in (4096, 8192, 3072, 5000):
            for t in (256, 2048):
                x = torch.randn(t, n, device=dev, dtype=dtype, generator=gen)
                g = torch.randn(t, n, device=dev, dtype=dtype, generator=gen)
                gy = torch.randn(t, n, device=dev, dtype=dtype, generator=gen)

                # fp32 oracle (chalk does sigmoid in fp32) — compare chalk's bf16 grads to fp32
                # truth in fp32 (avoids a bf16-eager-backward spuriously failing a correct kernel).
                xr = x.float().clone().requires_grad_(True)
                gr = g.float().clone().requires_grad_(True)
                ref = xr * torch.sigmoid(gr)
                ref.backward(gy.float())

                xc = x.clone().requires_grad_(True)
                gc = g.clone().requires_grad_(True)
                got = out_gate(xc, gc)
                torch.cuda.synchronize()
                got.backward(gy)
                torch.cuda.synchronize()

                rel = rel_l2  # fp32 relative-L2 error; shared helper (see chalk.utils.rel_l2)

                for nm, gv, rv in (("y", got, ref), ("dx", xc.grad, xr.grad), ("dg", gc.grad, gr.grad)):
                    if not (rel(gv, rv) < tol):
                        print(f"[out-gate] self-test FAIL {nm} n={n} t={t}: {rel(gv, rv):.2e}", flush=True)
                        return False
        return True
    except Exception as e:
        print(f"[out-gate] self-test errored: {type(e).__name__}: {e}", flush=True)
        return False


def self_test_trainable(tol: float = 2e-2, rotary_dim: int = 64) -> bool:
    """Live-GPU fwd+bwd parity of the trainable QK-norm+RoPE vs the exact eager HF reference
    (RMSNorm over head_dim then partial RoPE), at the real Qwen3.5 partial-rotary config. Checks
    y, dx, dw. Returns True iff every rel-err < tol; False on any failure (caller keeps eager)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        dev = "cuda"
        torch.manual_seed(0)
        eps = 1e-6
        hd = 256
        rot = rotary_dim
        rhalf = rot // 2
        for n_heads, t in ((16, 512), (4, 1024)):
            x = torch.randn(t, n_heads, hd, device=dev, dtype=torch.bfloat16)
            w = 0.1 * torch.randn(hd, device=dev, dtype=torch.bfloat16)
            pos = torch.arange(t, device=dev).float()
            inv = 1.0 / (1e7 ** (torch.arange(0, rhalf, device=dev).float() / rhalf))
            ang = pos[:, None] * inv[None, :]
            emb = torch.cat([ang, ang], dim=-1)
            cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
            gy = torch.randn(t, n_heads, hd, device=dev, dtype=torch.bfloat16)

            def rms(a, ww):
                return a * torch.rsqrt(a.float().pow(2).mean(-1, keepdim=True) + eps) * (1.0 + ww.float())

            def rope_partial(a, cos=cos, sin=sin):  # a [t, heads, hd]
                a_rot, a_pass = a[..., :rot], a[..., rot:]
                a1, a2 = a_rot[..., :rhalf], a_rot[..., rhalf:]
                rot_half = torch.cat([-a2, a1], dim=-1)
                c = cos.float()[:, None]
                s = sin.float()[:, None]
                return torch.cat([a_rot * c + rot_half * s, a_pass], dim=-1)

            xr = x.float().clone().requires_grad_(True)
            wr = w.float().clone().requires_grad_(True)
            ref = rope_partial(rms(xr, wr))
            ref.backward(gy.float())

            xc = x.clone().requires_grad_(True)
            wc = w.clone().requires_grad_(True)
            got = qknorm_rope(xc, wc, cos, sin, eps=eps)
            torch.cuda.synchronize()
            got.backward(gy)
            torch.cuda.synchronize()

            rel = rel_l2  # fp32 relative-L2 error; shared helper (see chalk.utils.rel_l2)

            for nm, gv, rv in (("y", got, ref), ("dx", xc.grad, xr.grad), ("dw", wc.grad, wr.grad)):
                if not (rel(gv, rv) < tol):
                    print(f"[qkv-train] self-test FAIL {nm} heads={n_heads} t={t}: {rel(gv, rv):.2e}", flush=True)
                    return False
        return True
    except Exception as e:
        print(f"[qkv-train] self-test errored: {type(e).__name__}: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Self-test guard: True iff the fused epilogue matches an fp32 reference within
# bf16 tolerance — on the REAL Qwen3.5 PARTIAL-rotary config (rotary_dim=64).
# Callers treat False as "fall back to the eager HF path".
# ---------------------------------------------------------------------------
def self_test(tol: float = 1e-2, rotary_dim: int = 64) -> bool:
    try:
        import torch  # lazy: keep the module import-safe without triton/torch

        if not torch.cuda.is_available():
            return False
        dev = "cuda"
        torch.manual_seed(0)
        hidden, n_q, n_kv, hd = 2560, 16, 4, 256
        rot = rotary_dim
        rhalf = rot // 2
        eps = 1e-6
        t = 512
        q_raw, k_out, v_out = n_q * hd * 2, n_kv * hd, n_kv * hd
        x = torch.randn(t, hidden, device=dev, dtype=torch.bfloat16)
        w_q = torch.randn(q_raw, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        w_k = torch.randn(k_out, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        w_v = torch.randn(v_out, hidden, device=dev, dtype=torch.bfloat16) * 0.02
        qn = torch.randn(hd, device=dev, dtype=torch.bfloat16) * 0.1
        kn = torch.randn(hd, device=dev, dtype=torch.bfloat16) * 0.1
        # partial-rotary cos/sin: width rotary_dim, lo half == hi half
        pos = torch.arange(t, device=dev).float()
        inv = 1.0 / (1e7 ** (torch.arange(0, rhalf, device=dev).float() / rhalf))
        ang = pos[:, None] * inv[None, :]  # [t, rhalf]
        emb = torch.cat([ang, ang], dim=-1)  # [t, rot]
        cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
        w_stack = build_stacked_qkv_weight(w_q, w_k, w_v)
        o = torch.nn.functional.linear(x, w_stack)
        q, k, _v, g = fused_qkv_epilogue(o, qn, kn, cos, sin, n_q_heads=n_q, n_kv_heads=n_kv, head_dim=hd, eps=eps)

        # fp32 reference (HF partial-rotary semantics)
        def rms(a, w):
            return a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())

        def rope_partial(a):  # a [t, heads, hd]
            a_rot, a_pass = a[..., :rot], a[..., rot:]
            a1, a2 = a_rot[..., :rhalf], a_rot[..., rhalf:]
            rot_half = torch.cat([-a2, a1], dim=-1)
            c = cos.float()[:, None]
            s = sin.float()[:, None]
            emb_rot = a_rot * c + rot_half * s
            return torch.cat([emb_rot, a_pass], dim=-1)

        qraw = (x.float() @ w_q.float().t()).view(t, n_q, hd * 2)
        query, gate = torch.chunk(qraw, 2, dim=-1)
        kf = (x.float() @ w_k.float().t()).view(t, n_kv, hd)
        rq = rope_partial(rms(query, qn)).reshape(t, n_q * hd)
        rk = rope_partial(rms(kf, kn)).reshape(t, n_kv * hd)
        rg = gate.reshape(t, n_q * hd)
        for got, ref in ((q, rq), (k, rk), (g, rg)):
            rel = (got.float() - ref).abs().max() / (ref.abs().max() + 1e-6)
            if rel.item() > tol:
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Install hook: patch Qwen3_5Attention.forward to run the fused epilogue.
#
# The patched forward reproduces the eager attention pre-attention math but
# replaces the per-head q_norm/k_norm + HF apply_rotary_pos_emb + gate-chunk with
# ONE fused Triton epilogue on a stacked cuBLAS projection. Everything after
# (attention_interface, sigmoid-gate, o_proj) is unchanged.
#
# Install-on-call (the Liger model): calling install_qwen35_attn_epilogue() IS the opt-in — there is
# no env flag — then it is gated by the live-GPU self_test on the real partial-rotary config
# (any failure leaves eager). LoRA-SAFE: the fused path fires only when q/k/v_proj are plain
# bias-free nn.Linear (no PEFT LoRA wrapper); a wrapped/biased projection falls back to the
# original differentiable forward, so the adapter delta math is never bypassed. GRADIENT-SAFE:
# the epilogue has no autograd backward, so the fused path is taken only when grad is disabled
# (inference/eval/no_grad); training builds the eager graph (so upstream/LoRA gradients are
# preserved). This is the same grad/LoRA-safe install-on-call safety model the other
# chalk installers use.
# ---------------------------------------------------------------------------
def install_qwen35_attn_epilogue(model=None) -> bool:
    """Patch ``Qwen3_5Attention.forward`` with the fused Triton QK-norm+RoPE+gate
    epilogue — IFF the live-GPU ``self_test`` passes. Install-on-call: calling this IS
    the opt-in (the Liger model); there is no env flag.
    Never raises (any failure keeps eager). Returns True iff the patch was installed.

    HONEST ACCOUNTING (mirrors the FP8-MLP install): the fused epilogue runs ONLY when
    an attention module's ``q_proj``/``k_proj``/``v_proj`` are plain frozen ``nn.Linear``
    (no bias, no PEFT wrapping). With the worker default ``LORA_TARGETS=all-linear`` PEFT
    wraps q/k/v as ``lora.Linear``, so ``_fused_forward`` would ALWAYS fall back to the
    original eager attention — the kernel would never engage. So when a live ``model`` is
    passed, only install + report enabled if at least one attention module actually has
    plain-frozen q/k/v projections; if none qualify, log an explicit "QKV kernel INACTIVE
    (projections are LoRA-wrapped; exclude q/k/v from LORA_TARGETS to use it)" and return
    ``False`` instead of a misleading "installed". (With no ``model`` there's nothing to
    inspect — the patch is still installed and gated per-call at runtime.)"""
    try:
        import torch  # lazy: keep the module import-safe without triton/torch

        if not torch.cuda.is_available():
            return False
        if not self_test():
            print("[qkv] fused epilogue self-test failed; keeping eager attention", flush=True)
            return False

        import importlib

        import torch.nn as nn

        mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        Attn = getattr(mod, "Qwen3_5Attention", None)
        apply_rope = getattr(mod, "apply_rotary_pos_emb", None)
        if Attn is None or apply_rope is None:
            print("[qkv] no Qwen3_5Attention to patch; keeping eager", flush=True)
            return False
        if getattr(Attn, "_chalk_qkv_patched", False):
            return True

        def _is_plain_linear(p) -> bool:
            # CUDA-ONLY: the fused QKV epilogue stacks the q/k/v weights and runs a Triton
            # kernel, which only executes on CUDA tensors. Under a ``device_map``/offload split
            # an attention block can sit on CPU (or a ``meta`` shell) even in a CUDA process;
            # taking the fused path there would launch Triton on CPU weights and crash instead of
            # falling back. Require the base weight be CUDA-resident so offloaded blocks stay
            # on the original eager attention.
            return type(p) is nn.Linear and getattr(p, "bias", None) is None and p.weight.is_cuda

        # HONEST ACCOUNTING: the fused epilogue only fires on attention modules whose
        # q/k/v projections are plain frozen nn.Linear. Inspect the live model and refuse
        # to report "installed" when NONE qualify (the default LORA_TARGETS=all-linear case,
        # where PEFT wraps q/k/v as lora.Linear), so an A/B run is not misreported as enabled
        # when no attention forward could ever take the fused path. Mirror of the FP8-MLP fix.
        if model is not None:
            attn_mods = [
                m
                for m in model.modules()
                if all(hasattr(m, p) for p in ("q_proj", "k_proj", "v_proj"))
                and hasattr(m, "q_norm")
                and hasattr(m, "k_norm")
            ]
            # No live Qwen3.5-style gated-attention module at all (e.g. a supported
            # NON-Qwen catalog model such as openbmb/MiniCPM5-1B): patching the global
            # Qwen3_5Attention class would be reported as "installed" yet no live module
            # could ever call it. Report INACTIVE and keep eager rather than claim success.
            if not attn_mods:
                print(
                    "[qkv] QKV kernel INACTIVE: this model has no Qwen3.5 gated-attention "
                    "module (q/k/v_proj + q_norm/k_norm), so the fused Qwen3_5Attention epilogue "
                    "can never engage. Keeping eager attention.",
                    flush=True,
                )
                return False
            n_plain = sum(
                1
                for m in attn_mods
                if _is_plain_linear(m.q_proj) and _is_plain_linear(m.k_proj) and _is_plain_linear(m.v_proj)
            )
            if n_plain == 0:
                print(
                    "[qkv] QKV kernel INACTIVE: 0 of "
                    f"{len(attn_mods)} attention modules have plain-frozen q/k/v projections "
                    "(they are LoRA-wrapped under LORA_TARGETS=all-linear), so the fused epilogue "
                    "can never engage — exclude q/k/v (q_proj/k_proj/v_proj) from LORA_TARGETS to "
                    "use the QKV kernel. Keeping eager attention.",
                    flush=True,
                )
                return False

        from transformers.masking_utils import create_causal_mask  # noqa: F401  (parity import)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        _orig_forward = Attn.forward

        def _fused_forward(
            self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs
        ):
            qp, kp, vp = self.q_proj, self.k_proj, self.v_proj
            # Fused path only for the frozen plain-Linear base in inference/eval; a
            # LoRA-wrapped/biased projection or a training forward (grad enabled) uses
            # the exact original differentiable attention. Also require eval mode
            # (``not self.training``): the fused epilogue hard-codes attention
            # ``dropout=0.0``, whereas the eager forward applies ``self.attention_dropout``
            # in training mode — so a no-grad forward taken while the module is still in
            # training mode (e.g. a no_grad eval pass that forgot model.eval()) must NOT
            # take the fused path, or it would silently drop training-mode dropout.
            if not (
                not torch.is_grad_enabled()
                and not self.training
                and hidden_states.is_cuda  # Triton epilogue is CUDA-only; a CPU/offloaded activation falls back
                and _is_plain_linear(qp)
                and _is_plain_linear(kp)
                and _is_plain_linear(vp)
            ):
                return _orig_forward(
                    self,
                    hidden_states,
                    position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            t = int(input_shape.numel())
            x2 = hidden_states.reshape(t, -1)
            n_q = self.config.num_attention_heads
            n_kv = self.config.num_key_value_heads
            hd = self.head_dim
            cos, sin = position_embeddings  # cos/sin: [bs, t, rotary_dim]
            cos2 = cos.reshape(-1, cos.shape[-1]).contiguous()
            sin2 = sin.reshape(-1, sin.shape[-1]).contiguous()

            w_stack = build_stacked_qkv_weight(qp.weight, kp.weight, vp.weight)
            o = torch.nn.functional.linear(x2, w_stack)
            # Liger RMSNorm stores epsilon as ``variance_epsilon``; eager stores ``eps``
            # (same dual-name resolution the embedding installer uses). Read it safely so
            # the fused path doesn't AttributeError on a Liger q_norm and abort eval/rollout.
            qn_eps = float(getattr(self.q_norm, "variance_epsilon", None) or getattr(self.q_norm, "eps", 1e-6))
            q, k, v, gate = fused_qkv_epilogue(
                o,
                self.q_norm.weight,
                self.k_norm.weight,
                cos2,
                sin2,
                n_q_heads=n_q,
                n_kv_heads=n_kv,
                head_dim=hd,
                eps=qn_eps,
            )
            # to [bs, heads, t, hd] for the attention interface
            q = q.view(*input_shape, n_q, hd).transpose(1, 2)
            k = k.view(*input_shape, n_kv, hd).transpose(1, 2)
            v = v.reshape(*input_shape, n_kv, hd).transpose(1, 2)
            gate = gate.reshape(*input_shape, -1)

            if past_key_values is not None:
                k, v = past_key_values.update(k, v, self.layer_idx)

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, mod.eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0,
                scaling=self.scaling,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        Attn.forward = _fused_forward
        Attn._chalk_qkv_patched = True
        print(
            "[qkv] fused Triton QK-norm+partial-RoPE+gate epilogue installed on "
            "Qwen3_5Attention (frozen plain-Linear base, inference/eval path)",
            flush=True,
        )
        return True
    except Exception as e:  # pragma: no cover - defensive: any failure keeps eager
        print(f"[qkv] install failed ({type(e).__name__}: {e}); keeping eager", flush=True)
        return False


# ---------------------------------------------------------------------------
# TRAINABLE install hook: patch Qwen3_5Attention.forward so the q_norm+RoPE and
# k_norm+RoPE steps route through the fused TRAINABLE kernel (fwd+bwd). Unlike the
# eval-only ``install_qwen35_attn_epilogue`` above, this ENGAGES IN TRAINING and works
# with q/k/v in LoRA targets — it operates on the projection OUTPUTS (after any LoRA
# delta, which stays in PyTorch autograd), replacing only the eager q_norm/k_norm +
# apply_rotary_pos_emb elementwise stack. The projections, gate, attention interface and
# o_proj are byte-for-byte the original eager forward.
#
# RoPE is per-(token,head) independent, so applying ``qknorm_rope`` on the
# ``[tokens, heads, head_dim]`` layout (before the ``transpose(1,2)``) is identical to the
# eager ``q_norm(...).transpose(1,2)`` then ``apply_rotary_pos_emb`` on
# ``[bs, heads, seq, head_dim]`` — we just reorder norm+rope-then-transpose. Gated by the
# live-GPU fwd+bwd ``self_test_trainable`` on the real partial-rotary config; any failure
# keeps the exact eager forward (correctness over speed).
# ---------------------------------------------------------------------------
def install_qwen35_qknorm_rope(model=None) -> bool:
    """Patch ``Qwen3_5Attention.forward`` so q_norm+RoPE / k_norm+RoPE run through the fused
    TRAINABLE chalk kernel — IFF the live-GPU ``self_test_trainable`` passes. Install-on-call
    (calling this IS the opt-in; no env flag). Never raises (any failure keeps eager). Returns
    True iff the patch was installed.

    Engages in BOTH training and inference and is independent of LoRA wrapping: it consumes the
    q_proj/k_proj OUTPUTS (after the LoRA delta, which stays differentiable in PyTorch), so it
    fires on the default ``LORA_TARGETS=all-linear`` case where the eval-only QKV epilogue could
    never engage. The q_norm/k_norm + apply_rotary_pos_emb elementwise stack is fused, AND (when its
    own self-test passes) the post-attention OUTPUT gate ``attn_out * sigmoid(gate)`` before o_proj
    is fused into a chalk-native fwd+bwd kernel. The gate split, attention interface and o_proj
    projection stay on the unchanged eager path."""
    try:
        import torch  # lazy: keep the module import-safe without triton/torch

        if not torch.cuda.is_available():
            return False
        # Production dispatch: pick the arch-tuned kernel for the running GPU (else the portable
        # kernel), gated on the combined-q/k fwd+bwd self-test inside load_entry. None => build /
        # self-test failed on every path -> keep the exact eager attention forward.
        qkv_entry = load_qkv()
        if qkv_entry is None:
            print(
                "[qkv-train] fused trainable QK-norm+RoPE unavailable (build/self-test failed); keeping eager",
                flush=True,
            )
            return False

        # Independent self-test for the fused attention OUTPUT-GATE (x * sigmoid(gate), fwd+bwd).
        # If it passes, the patched forward also fuses the post-attention gate; if it fails (or the
        # GPU/triton can't build it) the gate stays eager while QK-norm+RoPE is still fused. The
        # QK-norm fuse already gated the patch ON, so a gate failure must NOT undo that win.
        _use_out_gate = self_test_out_gate()
        if not _use_out_gate:
            print("[qkv-train] fused output-gate self-test declined; keeping the gate eager", flush=True)

        import importlib

        mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        Attn = getattr(mod, "Qwen3_5Attention", None)
        if Attn is None:
            print("[qkv-train] no Qwen3_5Attention to patch; keeping eager", flush=True)
            return False
        if getattr(Attn, "_chalk_qknorm_rope_patched", False):
            return True
        # Don't double-stack with the eval-only epilogue patch (they patch the same forward).
        if getattr(Attn, "_chalk_qkv_patched", False):
            print(
                "[qkv-train] eval-only attn_epilogue already patched this class; skipping the "
                "trainable patch (use one or the other).",
                flush=True,
            )
            return False

        # When a live model is passed, only report installed if it actually has Qwen3.5
        # gated-attention modules (q/k/v_proj + q_norm/k_norm) — otherwise the global class
        # patch could never engage (e.g. a non-Qwen catalog model).
        if model is not None:
            has_attn = any(
                all(hasattr(m, p) for p in ("q_proj", "k_proj", "v_proj"))
                and hasattr(m, "q_norm")
                and hasattr(m, "k_norm")
                for m in model.modules()
            )
            if not has_attn:
                print(
                    "[qkv-train] INACTIVE: model has no Qwen3.5 gated-attention module "
                    "(q/k/v_proj + q_norm/k_norm); keeping eager attention.",
                    flush=True,
                )
                return False

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        apply_rope = getattr(mod, "apply_rotary_pos_emb", None)
        eager_attn = getattr(mod, "eager_attention_forward", None)
        _orig_forward = Attn.forward

        def _train_forward(
            self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs
        ):
            # The fused kernel is CUDA-only. A CPU/offloaded activation (device_map split) or a
            # non-Qwen attn module that somehow reaches this class falls back to the exact eager
            # forward. q_norm/k_norm must be plain Qwen3_5RMSNorm with a [head_dim] weight; if a
            # Liger reassignment changed the norm semantics the eps-read below still matches, but
            # any structural surprise raises and is caught -> eager.
            def _eager():
                # exact eager attention forward — the shared fallback for both the non-CUDA guard
                # and any structural mismatch caught below.
                return _orig_forward(
                    self,
                    hidden_states,
                    position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

            if not hidden_states.is_cuda:
                return _eager()
            try:
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                t = int(input_shape.numel())
                hd = self.head_dim

                # projections (LoRA stays here, differentiable) — identical to eager
                query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, hd * 2), 2, dim=-1)
                gate = gate.reshape(*input_shape, -1)
                key_states = self.k_proj(hidden_states).view(hidden_shape)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings  # [bs, seq, rotary_dim]
                cos2 = cos.reshape(-1, cos.shape[-1]).contiguous()
                sin2 = sin.reshape(-1, sin.shape[-1]).contiguous()

                qn_eps = float(getattr(self.q_norm, "variance_epsilon", None) or getattr(self.q_norm, "eps", 1e-6))
                kn_eps = float(getattr(self.k_norm, "variance_epsilon", None) or getattr(self.k_norm, "eps", 1e-6))

                n_q = query_states.shape[-2]
                n_kv = key_states.shape[-2]
                # Fused q_norm+RoPE / k_norm+RoPE through the arch-tuned-or-portable dispatch entry
                # (``qkv_entry`` from load_qkv), called once per q and once per k on the token-major
                # [tokens, heads, head_dim] layout — the exact per-tensor contract the entry was
                # self-tested on, with no layout round-trip. Portable == the kernel production has
                # always run here, so this is a drop-in kernel swap.
                q_flat = qkv_entry(query_states.reshape(t, n_q, hd), self.q_norm.weight, cos2, sin2, qn_eps)
                k_flat = qkv_entry(key_states.reshape(t, n_kv, hd), self.k_norm.weight, cos2, sin2, kn_eps)
                query_states = q_flat.view(*input_shape, n_q, hd).transpose(1, 2)
                key_states = k_flat.view(*input_shape, n_kv, hd).transpose(1, 2)
            except Exception:
                # any structural mismatch -> exact eager forward
                return _eager()

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attn)
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            # Fused chalk output-gate (x * sigmoid(gate), fwd+bwd) when the self-test passed; else
            # the exact eager gate. ``gate`` carries the LoRA delta (2nd chunk of q_proj) and stays
            # differentiable either way. Any kernel surprise at runtime falls back to eager.
            if _use_out_gate and attn_output.is_cuda and attn_output.shape == gate.shape:
                try:
                    attn_output = out_gate(attn_output, gate)
                except Exception:
                    attn_output = attn_output * torch.sigmoid(gate)
            else:
                attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        _ = apply_rope  # parity reference (the fused kernel replaces it); keep the symbol alive
        Attn.forward = _train_forward
        Attn._chalk_qknorm_rope_patched = True
        Attn._chalk_out_gate_fused = bool(_use_out_gate)
        print(
            "[qkv-train] fused TRAINABLE Triton QK-norm+partial-RoPE installed on "
            f"Qwen3_5Attention (fwd+bwd, engages in training, LoRA-compatible; "
            f"output-gate fused={bool(_use_out_gate)})",
            flush=True,
        )
        return True
    except Exception as e:  # pragma: no cover - defensive: any failure keeps eager
        print(f"[qkv-train] install failed ({type(e).__name__}: {e}); keeping eager", flush=True)
        return False


if __name__ == "__main__":
    print("self_test (partial rotary 64):", self_test(rotary_dim=64))
    print("self_test (full rotary 256):", self_test(rotary_dim=256))
    print("self_test_trainable (partial 64):", self_test_trainable(rotary_dim=64))
    print("self_test_trainable (full 256):", self_test_trainable(rotary_dim=256))
    print("self_test_out_gate:", self_test_out_gate())
