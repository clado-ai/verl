"""Fused Triton LoRA A/B delta matmuls (forward + backward) for Qwen3.5 LoRA SFT/GRPO.

STATUS: KEEP (gated, opt-in). Benchmarked on an A40 (torch 2.10, triton 3.6,
peft 0.19.1) against PEFT's 2-GEMM path. Correct (bf16 grads match autograd to
<1.4e-2) and **faster at production batch sizes**:

    v3 design (v2 forward + fused dA/dB backward), geomean over 36 cases
    (4 proj x r{16,32,64} x tok{2k,8k,32k}):
        fwd = 1.767x (wins 36/36)   bwd = 1.286x (wins 35/36)   fwd+bwd = 1.005x

The forward wins everywhere. Round 2 strengthened the **backward** by fusing the
two M-reduction GEMMs (dA and dB) into a single launch (3->2 launches total),
lifting bwd geomean from 1.13x to 1.29x.

THRESHOLD: see ``install_fused_lora_delta`` for the fwd+bwd guidance, which is
what a user should follow. The A40 numbers above are per-op fwd/bwd geomeans and
still stand; the fwd+bwd threshold they were originally paired with does not. A
later per-op sweep on an A5000 (sm86) and an L40S (sm89) put the crossover at
>=16384 tokens on both, not 8192 -- at 8192 the step measures 0.94-0.97x on sm86
and 0.83-0.88x on sm89, and the best case seen was 1.59x, not 2.05x. That sweep
also found the <=2048 loss is not "a wash": on the portable path it reaches 1.7x
slower than eager. The claim that the small-batch loss is a non-pipelined-
measurement artifact rests on a single stacked-layer run and was not reproduced
in that sweep; do not rely on it.

KEY DESIGN NOTE (why this wins where a naive fusion loses): the obvious single-
launch fusion (one program per output (m,n) tile) RECOMPUTES the full t = x@A^T
tile once per N-tile — at N=9216 that is a ~70x redundant GEMM and it loses
0.5-0.6x. The winning design computes t ONCE per m-block (loop over K, t kept in
SRAM) and then loops over all N-tiles internally to emit the whole output row-
block. The same compute-once principle is applied to the backward dt = dY@B.
A naive (v1) fusion was measured and DISCARDED; see ``bench/lora_result.md``.

ROUND-2 NEGATIVE RESULTS (also in bench/lora_result.md):
  * Fusing the LoRA delta INTO the base GEMM (one kernel for x@W^T +
    scaling*(x@A^T)@B^T) LOSES 0.24-0.87x. A standalone tuned Triton GEMM only
    *matches* cuBLAS on the base weight (~0.96-1.08x); co-mingling the rank-r
    side-accumulation into that kernel breaks the GEMM pipeline (register spill)
    and lands 3-4x slower. cuBLAS owns the big frozen GEMM; keep it separate.
  * A single fully-fused backward kernel (dx + atomic dA/dB accumulation over M)
    LOSES 0.27-0.34x: atomic contention on the small [r,K]/[N,r] grads serializes.
  The win that stuck: fuse only the two independent M-reduction GEMMs (dA, dB)
  into one launch via a split grid (no atomics, no redundant work) -- see
  ``_dAdB_kernel`` below.

WIRING: install-on-call, the Liger model (``apply_liger_kernel_to_qwen3``) — a
consumer calls ``install_fused_lora_delta()``, which runs the per-GPU self-test and, on PASS,
monkeypatches PEFT's vanilla LoRA Linear.forward to route through
``FusedLoRAFunction`` (autograd handles the fused backward). There is no env flag:
calling the installer IS the opt-in, and it falls back to stock PEFT on any failure.
Self-tested exactly like the kernels around it. The fused path wins only where its fused
BACKWARD is actually run at a large per-step token count; under ``no_grad`` (all
generation) or at tiny token counts it only adds launch overhead. That is now ENFORCED,
not just recommended: the patched forward falls back to stock PEFT whenever gradients are
disabled OR the flattened token count is below the ``_FUSE_MIN_TOKENS`` module attribute
(default 4096), so decode/rollout forwards (GRPO/eval) pay no fused-LoRA tax while the
grad-enabled training forwards keep the win.

LoRA math (replicates PEFT exactly): A=[r,d_in], B=[d_out,r], scaling scalar.
    delta = scaling * ((x @ A^T) @ B^T)
Backward, given dY = grad wrt delta:
    dt = (dY @ B) * scaling;  dx = dt @ A;  dA = dt^T @ x;  dB = scaling*(dY^T @ t)
where t = x @ A^T (pre-scale), saved from the forward.
"""

from __future__ import annotations

# Lazy import: triton is only present on the GPU worker. Importing this module on
# the control plane / CPU must not fail.
try:  # pragma: no cover - exercised only on GPU
    import torch
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False

# TOKEN-COUNT CROSSOVER for the fused LoRA-delta path. The fused delta is a Triton
# autograd.Function that fires one kernel launch per LoRA-wrapped projection per layer;
# it WINS at the large per-step token counts training runs at (SFT / GRPO policy-update
# forwards, >=8k tokens: fwd+bwd 1.0-2.05x) but LOSES at the tiny token counts of
# autoregressive DECODE (GRPO/eval rollouts do batch x 1 token), where launch/dispatch
# overhead dominates the small GPU work. Measured on an A100 (Qwen3.5-0.8B LoRA + GSM8K):
# leaving the fused path on during decode made GRPO generation +21% and the whole RL step
# ~18% slower, while the training-forward win was intact. Below this many FLATTENED tokens
# the patched forward falls back to stock PEFT, so decode pays no fused-LoRA tax and
# training keeps the win. 4096 cleanly separates decode (tens of tokens) from training
# forwards (>=8k). Module attribute (not an env read: chalk.ops.* must import clean off the
# GPU worker) — override before install_fused_lora_delta() to tune, e.g.
# ``chalk.ops.lora._FUSE_MIN_TOKENS = 8192``.
_FUSE_MIN_TOKENS = 4096


def _is_sm120_family(cap: tuple[int, int]) -> bool:
    """Return whether a CUDA capability should use the RTX 50-series sm12x config set."""
    return cap[0] == 12


if _HAVE_TRITON:
    # ======================================================================
    # PER-ARCH AUTOTUNE CONFIG SETS (chalk's "winning kernel on every arch")
    # ----------------------------------------------------------------------
    # The fused LoRA-delta Triton GEMMs were originally tuned on Hopper(sm_90)/
    # Ampere(sm_8x): deep ``num_stages`` software pipelining + big N/K tiles that
    # suit the datacenter SM's large shared-memory + TMA/wgmma pipeline. On
    # Blackwell consumer (sm_120, RTX 50-series) that same set LOSES at small
    # token counts (measured RTX 5090: K2560/N4096/R32 @ 4k tok = 0.74x eager;
    # K9216/N2560/R16 @ 4k tok = 0.996x) — the deep-staged big tiles under-occupy
    # the consumer SM at small M, where cuBLAS's tuned small-GEMM path wins.
    # FIX (no eager fallback): give the autotuner an ARCH-APPROPRIATE config set
    # chosen off ``cuda.get_device_capability()`` (the probe ``fp8_base.py`` uses).
    # Blackwell gets smaller-M tiles + shallower stages (2) that keep many CTAs
    # resident at small M; sm_90/sm_8x keep the proven deep set. EVERY config is
    # numerically correct on every arch (same kernel body); the split only steers
    # the autotuner's search so each arch HAS occupancy-good options to pick from.
    # ======================================================================
    def _cap():
        try:
            return torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        except Exception:
            return (0, 0)

    def _uses_sm120_configs() -> bool:
        # The shallow-stage/small-M configs below were motivated by RTX 50-series sm120
        # measurements. Keep sm100 datacenter Blackwell on the Hopper/Ampere-proven
        # base set until it has its own A/B benchmark data.
        return _is_sm120_family(_cap())

    # Save t = x@A^T from the forward (vs recompute in backward) when the contraction K is wide
    # enough that the recompute GEMM's long K-loop costs more than the tiny [M,R] HBM buffer it
    # avoids. Qwen3.5 down_proj has K=9216; gate/up/q/k/v have K=2560. Threshold 4096 catches the
    # wide-K projections only. (Numerically identical; pure speed/memory tradeoff per shape.)
    _SAVE_T_MIN_K = 4096

    # ----------------------------------------------------------------------
    # FORWARD: t = x@A^T computed ONCE per m-block (loop K, t in SRAM), then
    # loop all N-tiles to emit delta = (t@B^T)*scaling (+ base). t is also
    # written out (materialized) so the backward needs no recompute.
    # ----------------------------------------------------------------------
    # Hopper/Ampere base set (deep pipelining, large N tiles) — campaign-tuned.
    _fwd_cfgs_base = [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        # Narrow-N / deep-K / low-warp point (A/B 2026-07: beats all 5 above by ~1.13-1.19x on the
        # LoRA fwd at the production shape M=8192/K=N=4096/rank=32 on A100, order-confirmed). autotune
        # picks it ONLY where it wins, so adding it is zero-regression — it just widens the search.
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=2, num_stages=4),
    ]
    # Blackwell sm_120 set: small-M tiles (32/16) + shallow stages (2) so many
    # CTAs stay resident at small token counts (the regime where the base set
    # lost), plus a couple proven mid tiles. Wider BLOCK_N keeps the N-loop fed.
    _fwd_cfgs_blackwell = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ]
    _fwd_cfgs = _fwd_cfgs_blackwell if _uses_sm120_configs() else _fwd_cfgs_base

    @triton.autotune(configs=_fwd_cfgs, key=["M", "N", "K", "R"])
    @triton.jit
    def _fwd_kernel(
        x_ptr,
        a_ptr,
        b_ptr,
        base_ptr,
        out_ptr,
        t_ptr,
        M,
        N,
        K,
        R,
        scaling,
        sx_m,
        sx_k,
        sa_r,
        sa_k,
        sb_n,
        sb_r,
        so_m,
        so_n,
        st_m,
        st_r,
        HAS_BASE: tl.constexpr,
        WRITE_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < R
        m_mask = offs_m < M
        t_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x_blk = tl.load(
                x_ptr + offs_m[:, None] * sx_m + offs_k[None, :] * sx_k,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            a_blk = tl.load(
                a_ptr + offs_r[:, None] * sa_r + offs_k[None, :] * sa_k,
                mask=r_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            t_acc += tl.dot(x_blk, tl.trans(a_blk))
        t_bf = t_acc.to(out_ptr.dtype.element_ty)
        if WRITE_T:
            tl.store(
                t_ptr + offs_m[:, None] * st_m + offs_r[None, :] * st_r,
                t_bf,
                mask=m_mask[:, None] & r_mask[None, :],
            )
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N
            b_blk = tl.load(
                b_ptr + offs_n[:, None] * sb_n + offs_r[None, :] * sb_r,
                mask=n_mask[:, None] & r_mask[None, :],
                other=0.0,
            )
            d = tl.dot(t_bf, tl.trans(b_blk)) * scaling
            omask = m_mask[:, None] & n_mask[None, :]
            op = out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_n
            if HAS_BASE:
                bb = tl.load(
                    base_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_n,
                    mask=omask,
                    other=0.0,
                )
                d += bb.to(tl.float32)
            tl.store(op, d.to(out_ptr.dtype.element_ty), mask=omask)

    def lora_fwd(x, A, B, scaling, base_out=None, return_t=False):
        """x:[M,K] A:[R,K] B:[N,R] -> delta:[M,N] (+ base_out). If return_t, also
        return t=x@A^T (pre-scale) for a recompute-free backward."""
        M, K = x.shape
        R = A.shape[0]
        N = B.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        t = torch.empty((M, R), device=x.device, dtype=x.dtype) if return_t else out
        BLOCK_R = max(16, triton.next_power_of_2(R))

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        _fwd_kernel[grid](
            x,
            A,
            B,
            base_out if base_out is not None else x,
            out,
            t,
            M,
            N,
            K,
            R,
            scaling,
            x.stride(0),
            x.stride(1),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            out.stride(0),
            out.stride(1),
            t.stride(0),
            t.stride(1),
            HAS_BASE=base_out is not None,
            WRITE_T=return_t,
            BLOCK_R=BLOCK_R,
        )
        return (out, t) if return_t else out

    # ----------------------------------------------------------------------
    # BACKWARD: dt = (dY@B)*scaling computed ONCE per m-block (loop N, dt in
    # SRAM), then loop K internally to emit the full dx row-block + write dt.
    # dA, dB are M-reduction GEMMs; t is supplied from the forward.
    # ----------------------------------------------------------------------
    _dx_cfgs_base = [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    ]
    # Blackwell sm_120: small-M tiles + shallow stages + smaller K (the 256-K
    # stage-3 tile over-uses the consumer SM's shared memory).
    _dx_cfgs_blackwell = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    ]
    _dx_cfgs = _dx_cfgs_blackwell if _uses_sm120_configs() else _dx_cfgs_base

    @triton.autotune(configs=_dx_cfgs, key=["M", "N", "K", "R"])
    @triton.jit
    def _dx_kernel(
        dy_ptr,
        b_ptr,
        a_ptr,
        dx_ptr,
        dt_ptr,
        M,
        N,
        K,
        R,
        scaling,
        sdy_m,
        sdy_n,
        sb_n,
        sb_r,
        sa_r,
        sa_k,
        sdx_m,
        sdx_k,
        sdt_m,
        sdt_r,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < R
        m_mask = offs_m < M
        dt_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N
            dy_blk = tl.load(
                dy_ptr + offs_m[:, None] * sdy_m + offs_n[None, :] * sdy_n,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            b_blk = tl.load(
                b_ptr + offs_n[:, None] * sb_n + offs_r[None, :] * sb_r,
                mask=n_mask[:, None] & r_mask[None, :],
                other=0.0,
            )
            dt_acc += tl.dot(dy_blk, b_blk)
        dt_acc = dt_acc * scaling
        dt_bf = dt_acc.to(dt_ptr.dtype.element_ty)
        tl.store(
            dt_ptr + offs_m[:, None] * sdt_m + offs_r[None, :] * sdt_r,
            dt_bf,
            mask=m_mask[:, None] & r_mask[None, :],
        )
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a_blk = tl.load(
                a_ptr + offs_r[:, None] * sa_r + offs_k[None, :] * sa_k,
                mask=r_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            dx = tl.dot(dt_bf, a_blk)
            tl.store(
                dx_ptr + offs_m[:, None] * sdx_m + offs_k[None, :] * sdx_k,
                dx.to(dx_ptr.dtype.element_ty),
                mask=m_mask[:, None] & k_mask[None, :],
            )

    # dA and dB are independent M-reduction GEMMs (different output shapes). v3
    # fuses them into ONE launch via a split 1D grid: the first cdiv(K,BLOCK_O)
    # programs emit a dA column-tile, the rest emit a dB row-tile. No atomics,
    # no redundant work; just one fewer kernel launch than the v2 3-launch bwd.
    _red_cfgs_base = [
        triton.Config({"BLOCK_O": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_O": 128, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_O": 64, "BLOCK_M": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_O": 128, "BLOCK_M": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_O": 256, "BLOCK_M": 64}, num_warps=8, num_stages=3),
    ]
    # Blackwell sm_120: stages=2 variants of the proven reduction tiles + a small-M
    # variant (the M-reduction GEMMs are skinny [r,K]/[N,r]; deep staging buys little
    # on the consumer pipeline, and small BLOCK_M keeps CTAs resident at small M).
    _red_cfgs_blackwell = [
        triton.Config({"BLOCK_O": 64, "BLOCK_M": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_O": 64, "BLOCK_M": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_O": 128, "BLOCK_M": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_O": 128, "BLOCK_M": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_O": 256, "BLOCK_M": 64}, num_warps=8, num_stages=2),
    ]
    _red_cfgs = _red_cfgs_blackwell if _uses_sm120_configs() else _red_cfgs_base

    @triton.autotune(configs=_red_cfgs, key=["M", "N", "K", "R"])
    @triton.jit
    def _dAdB_kernel(
        dt_ptr,
        x_ptr,
        dy_ptr,
        t_ptr,
        da_ptr,
        db_ptr,
        M,
        N,
        K,
        R,
        scaling,
        sdt_m,
        sdt_r,
        sx_m,
        sx_k,
        sdy_m,
        sdy_n,
        st_m,
        st_r,
        sda_r,
        sda_k,
        sdb_n,
        sdb_r,
        BLOCK_M: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_k_tiles = tl.cdiv(K, BLOCK_O)  # first n_k_tiles programs -> dA, rest -> dB
        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < R
        if pid < n_k_tiles:
            # dA[r,k] = sum_m dt[m,r] * x[m,k]
            offs_k = pid * BLOCK_O + tl.arange(0, BLOCK_O)
            k_mask = offs_k < K
            acc_a = tl.zeros((BLOCK_R, BLOCK_O), dtype=tl.float32)
            for m0 in range(0, M, BLOCK_M):
                offs_m = m0 + tl.arange(0, BLOCK_M)
                mm = offs_m < M
                dt_blk = tl.load(
                    dt_ptr + offs_m[:, None] * sdt_m + offs_r[None, :] * sdt_r,
                    mask=mm[:, None] & r_mask[None, :],
                    other=0.0,
                )
                x_blk = tl.load(
                    x_ptr + offs_m[:, None] * sx_m + offs_k[None, :] * sx_k,
                    mask=mm[:, None] & k_mask[None, :],
                    other=0.0,
                )
                acc_a += tl.dot(tl.trans(dt_blk), x_blk)
            tl.store(
                da_ptr + offs_r[:, None] * sda_r + offs_k[None, :] * sda_k,
                acc_a.to(da_ptr.dtype.element_ty),
                mask=r_mask[:, None] & k_mask[None, :],
            )
        else:
            # dB[n,r] = scaling * sum_m dY[m,n] * t[m,r]
            pid_n = pid - n_k_tiles
            offs_n = pid_n * BLOCK_O + tl.arange(0, BLOCK_O)
            n_mask = offs_n < N
            acc_b = tl.zeros((BLOCK_O, BLOCK_R), dtype=tl.float32)
            for m0 in range(0, M, BLOCK_M):
                offs_m = m0 + tl.arange(0, BLOCK_M)
                mm = offs_m < M
                dy_blk = tl.load(
                    dy_ptr + offs_m[:, None] * sdy_m + offs_n[None, :] * sdy_n,
                    mask=mm[:, None] & n_mask[None, :],
                    other=0.0,
                )
                t_blk = tl.load(
                    t_ptr + offs_m[:, None] * st_m + offs_r[None, :] * st_r,
                    mask=mm[:, None] & r_mask[None, :],
                    other=0.0,
                )
                acc_b += tl.dot(tl.trans(dy_blk), t_blk)
            acc_b = acc_b * scaling
            tl.store(
                db_ptr + offs_n[:, None] * sdb_n + offs_r[None, :] * sdb_r,
                acc_b.to(db_ptr.dtype.element_ty),
                mask=n_mask[:, None] & r_mask[None, :],
            )

    def lora_bwd(dY, x, A, B, scaling, t):
        """Return dx, dA, dB. t is x@A^T (pre-scale) saved from the forward.
        Two launches: _dx_kernel (dx, materializes dt) then _dAdB_kernel (dA+dB)."""
        M, N = dY.shape
        R, K = A.shape
        BLOCK_R = max(16, triton.next_power_of_2(R))
        dx = torch.empty((M, K), device=x.device, dtype=x.dtype)
        dt = torch.empty((M, R), device=x.device, dtype=x.dtype)
        _dx_kernel[lambda m: (triton.cdiv(M, m["BLOCK_M"]),)](
            dY,
            B,
            A,
            dx,
            dt,
            M,
            N,
            K,
            R,
            scaling,
            dY.stride(0),
            dY.stride(1),
            B.stride(0),
            B.stride(1),
            A.stride(0),
            A.stride(1),
            dx.stride(0),
            dx.stride(1),
            dt.stride(0),
            dt.stride(1),
            BLOCK_R=BLOCK_R,
        )
        dA = torch.empty((R, K), device=x.device, dtype=A.dtype)
        dB = torch.empty((N, R), device=x.device, dtype=B.dtype)

        # grid spans dA's K-tiles + dB's N-tiles; the split point (n_k_tiles) is
        # recomputed inside the kernel from K and the autotuned BLOCK_O.
        def _grid(m):
            bo = m["BLOCK_O"]
            return (triton.cdiv(K, bo) + triton.cdiv(N, bo),)

        _dAdB_kernel[_grid](
            dt,
            x,
            dY,
            t,
            dA,
            dB,
            M,
            N,
            K,
            R,
            scaling,
            dt.stride(0),
            dt.stride(1),
            x.stride(0),
            x.stride(1),
            dY.stride(0),
            dY.stride(1),
            t.stride(0),
            t.stride(1),
            dA.stride(0),
            dA.stride(1),
            dB.stride(0),
            dB.stride(1),
            BLOCK_R=BLOCK_R,
        )
        return dx, dA, dB

    class FusedLoRAFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, A, B, scaling, base_out=None):
            # SMALL-M OVERHEAD: at the GRPO regime (M ~2-4k) the fused delta is launch/
            # dispatch-bound — the kernels WIN on raw GPU time (H100 1.08-1.30x at M=4096)
            # but the autograd.Function Python wrapper adds ~0.13 ms vs torch's native
            # C++ autograd. Every Python op skipped here is a direct small-M speedup, so
            # take a no-reshape fast path when x is already 2D (the flattened case): the
            # reshape(-1, H) on a contiguous tensor is a cheap view, but constructing it
            # (and the symmetric out.reshape back) is pure Python overhead we can drop.
            # 3D inputs ([B,S,H]) still reshape; result is bit-identical either way.
            is_2d = x.dim() == 2
            shp = x.shape
            x2 = x if is_2d else x.reshape(-1, shp[-1])
            if base_out is not None:
                base2 = base_out if base_out.dim() == 2 else base_out.reshape(-1, base_out.shape[-1])
            else:
                base2 = None
            K = x2.shape[1]
            # MEMORY vs SPEED on t = x@A^T: by default do NOT materialize/save the [M,R] intermediate
            # — recompute it in backward (a skinny [M,K]@[K,R] GEMM, R<=64) to keep peak memory at
            # eager-PEFT parity. BUT for WIDE-K projections (down_proj: K=9216 on Qwen3.5) that
            # recompute GEMM is NOT cheap — its K-loop is the long one, MEASURED ~0.1 ms at K=9216/
            # M=8192, which alone flipped the fused backward from a win to a loss vs eager on RTX 5090
            # (sm_120). For K >= _SAVE_T_MIN_K the forward already computes t in SRAM, so writing it
            # out costs only a tiny [M,R] bf16 buffer (e.g. M=16384,R=16 -> 0.5 MB) and removes the
            # whole recompute GEMM from the backward — a clean speed win at negligible memory cost.
            # Small-K shapes keep the recompute (cheap GEMM, zero HBM buffer). The result is
            # numerically identical either way (t is the same value, just sourced differently).
            save_t = K >= _SAVE_T_MIN_K
            if save_t:
                out, t = lora_fwd(x2, A, B, scaling, base2, return_t=True)
                ctx.save_for_backward(x2, A, B, t)
            else:
                out = lora_fwd(x2, A, B, scaling, base2, return_t=False)
                ctx.save_for_backward(x2, A, B)
            ctx.save_t = save_t
            ctx.scaling = scaling
            ctx.shp = shp
            ctx.is_2d = is_2d
            ctx.has_base = base_out is not None
            ctx.base_shp = base_out.shape if base_out is not None else None
            return out if is_2d else out.reshape((*shp[:-1], B.shape[0]))

        @staticmethod
        def backward(ctx, grad_out):
            if ctx.save_t:
                x2, A, B, t = ctx.saved_tensors  # wide-K: t saved from the forward (no recompute)
            else:
                x2, A, B = ctx.saved_tensors
                # Recompute t = x@A^T (not saved for small K). fp32-accumulated bf16 GEMM, matches
                # the fused kernel's t within bf16 tolerance (self-test: grads still < 1.5e-2 rel).
                # Recompute UNDER no_grad: t is an internal scratch for lora_bwd, never an output, so
                # it must not build an autograd graph. Without this, a higher-order ``backward(
                # create_graph=True)`` would graph this skinny GEMM (extra memory/overhead) — and the
                # Triton backward doesn't support higher-order grads anyway, so keep the two paths
                # consistent and graph-free.
                with torch.no_grad():
                    t = (x2 @ A.t()).to(x2.dtype)
            # 2D fast path (see forward): grad_out is already [M,N] -> skip the reshape.
            dY = grad_out if ctx.is_2d else grad_out.reshape(-1, grad_out.shape[-1])
            dx, dA, dB = lora_bwd(dY, x2, A, B, ctx.scaling, t)
            dxo = dx if ctx.is_2d else dx.reshape(ctx.shp)
            # base_out is an additive input, so d(base_out) = grad_out — but reshaped to base_out's
            # ORIGINAL shape. The forward supports base_out being 2D while x is 3D (or vice versa), and
            # grad_out carries x's OUTPUT shape, so returning it unreshaped would shape-mismatch the
            # grad against base_out (same element count, different rank).
            grad_base = None
            if ctx.has_base:
                grad_base = grad_out if grad_out.shape == ctx.base_shp else grad_out.reshape(ctx.base_shp)
            return dxo, dA, dB, None, grad_base

    def fused_lora(x, A, B, scaling, base_out=None):
        """Fused LoRA delta with autograd. base_out (frozen base layer output) is
        added inside the kernel so the whole adapter contribution is one launch."""
        return FusedLoRAFunction.apply(x, A, B, scaling, base_out)


# ======================================================================================
# ARCH-DISPATCH ENTRY (delta-only): lora_delta_fn(x, A, B, scaling) -> delta.
# ----------------------------------------------------------------------------------
# This is the contract the per-arch tuned tree implements (``chalk/ops/arch/<arch>/lora.py``
# -> ``build() -> lora_delta_fn``) and the direction the autoresearch verifier grades: the pure
# adapter delta ``scaling * ((x @ A^T) @ B^T)`` with a correct fwd+bwd (grads dx, dA, dB; scaling
# is a scalar so its grad is None). ``load_lora()`` routes through ``chalk.ops.arch.load_entry``
# so the arch kernel is selected when it is a verified win AND passes the live-GPU self-test;
# otherwise it falls back to the PORTABLE fused-Triton kernel (never to torch eager). The install
# path (``install_fused_lora_delta``) obtains the entry via ``load_lora()`` and adds the base
# output OUTSIDE the kernel (exactly as stock PEFT does: ``result + delta``), which is why the
# entry only ever computes the delta and takes a 2D ``[tok, K]`` activation (the caller flattens).
# ======================================================================================


def _eager_lora_delta(x, A, B, scaling):
    """Exact LoRA-delta reference (self-test oracle): ``scaling * ((x @ A^T) @ B^T)``.

    Replicates PEFT's dense LoRA math (A=[r,K], B=[N,r], scalar scaling). Dtype follows the
    inputs, so the self-test feeds fp32 clones to get the trustworthy fp32 truth."""
    return ((x @ A.t()) @ B.t()) * scaling


def _self_test_delta(delta_fn) -> None:
    """Live-GPU fwd+bwd parity of an arch/portable ``lora_delta_fn`` vs the fp32 eager oracle
    (forward, dx, dA, dB). Raises on any mismatch so ``load_entry`` keeps the portable kernel.
    Mirrors the arch verifier's grading direction (fwd+bwd) and its 0.02 rel @ fp32 tolerance."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA for lora self-test")
    dev = "cuda"
    # Exercise the REAL reduced-precision path (bf16 when supported, else fp16), measuring the
    # error in fp32 against an fp32 oracle — the standard reduced-precision-kernel test pattern.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tol = 2e-2
    gen = torch.Generator(device=dev).manual_seed(0)
    # Real Qwen3.5 LoRA shapes (q/k/v/gate/up K=2560, down_proj K=9216) across r in {16,32,64}.
    for K, N, R in ((2560, 4096, 32), (9216, 2560, 16), (2560, 9216, 64)):
        M = 2048
        scaling = 64.0 / (R**0.5)
        x = torch.randn(M, K, device=dev, dtype=dtype, generator=gen) * 0.1
        A = torch.randn(R, K, device=dev, dtype=dtype, generator=gen) * 0.05
        B = torch.randn(N, R, device=dev, dtype=dtype, generator=gen) * 0.05
        g = torch.randn(M, N, device=dev, dtype=dtype, generator=gen)

        # fp32 oracle (chalk's internal math is fp32; a bf16 eager backward is not a valid truth).
        # Disable any ambient training autocast so the reference matmuls stay fp32 — under an active
        # autocast the .float() inputs would be re-cast to bf16 and the gate would compare the kernel
        # to a reduced-precision "oracle".
        xr = x.float().clone().requires_grad_(True)
        Ar = A.float().clone().requires_grad_(True)
        Br = B.float().clone().requires_grad_(True)
        with torch.autocast(device_type="cuda", enabled=False):
            ref = _eager_lora_delta(xr, Ar, Br, scaling)
        ref.backward(g.float())

        xc = x.clone().requires_grad_(True)
        Ac = A.clone().requires_grad_(True)
        Bc = B.clone().requires_grad_(True)
        got = delta_fn(xc, Ac, Bc, scaling)
        torch.cuda.synchronize()
        got.backward(g)
        torch.cuda.synchronize()

        def rel(a, b):
            a32, b32 = a.float(), b.float()
            return (a32 - b32).norm().item() / (b32.norm().item() + 1e-9)

        checks = (
            ("fwd", got, ref),
            ("dx", xc.grad, xr.grad),
            ("dA", Ac.grad, Ar.grad),
            ("dB", Bc.grad, Br.grad),
        )
        for nm, gv, rv in checks:
            r = rel(gv, rv)
            if not (r < tol):
                raise RuntimeError(
                    f"lora self-test failed at K={K} N={N} R={R} dtype={dtype}: {nm}={r:.2e} (tol={tol:.0e})"
                )


def _portable_delta():
    """Portable factory: the shipped fused-Triton LoRA delta as a ``(x, A, B, scaling) -> delta``
    callable (no base fusion). This is the fallback ``load_entry`` uses when there is no verified
    arch kernel — never torch eager."""
    if not _HAVE_TRITON:
        raise RuntimeError("triton unavailable for portable lora kernel")

    def lora_delta_fn(x, A, B, scaling):
        return fused_lora(x, A, B, scaling)

    return lora_delta_fn


def load_lora():
    """Return the LoRA-delta entry (``lora_delta_fn(x, A, B, scaling) -> delta``) for the running
    GPU: the per-arch tuned kernel if it is a verified win and passes the live-GPU self-test, else
    the portable fused-Triton kernel. Never raises — any failure (no torch/triton, no CUDA,
    build/self-test error) returns ``None`` and the caller keeps stock PEFT."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        from chalk.ops.arch import load_entry

        # ``load_entry`` already runs ``_self_test_delta`` on the entry it returns (arch or
        # portable), so do NOT self-test again here — that just doubles startup validation latency.
        fn = load_entry("lora", _self_test_delta, portable=_portable_delta)
        print("[lora] fused LoRA delta (fwd+bwd) enabled (self-test passed)", flush=True)
        return fn
    except Exception as e:  # pragma: no cover - defensive: any failure keeps stock PEFT
        print(f"[lora] disabled (build/self-test failed): {e}", flush=True)
        return None


def _self_test(verbose: bool = False) -> bool:
    """Verify forward + grads match torch autograd (bf16) on a representative case."""
    if not _HAVE_TRITON or not torch.cuda.is_available():
        return False
    dev = "cuda"
    ok = True
    for K, N, R in [(2560, 4096, 32), (9216, 2560, 16), (2560, 9216, 64)]:
        M = 4096
        scaling = 64.0 / (R**0.5)
        x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
        A = (torch.randn(R, K, device=dev, dtype=torch.bfloat16) * 0.05).requires_grad_(True)
        B = (torch.randn(N, R, device=dev, dtype=torch.bfloat16) * 0.05).requires_grad_(True)
        xr, Ar, Br = (v.clone().detach().requires_grad_(True) for v in (x, A, B))
        g = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        ref = (xr @ Ar.t() @ Br.t()) * scaling
        ref.backward(g)
        out = fused_lora(x, A, B, scaling)
        out.backward(g)

        def rel(a, b):
            return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-9)).item()

        e = max(rel(out, ref), rel(x.grad, xr.grad), rel(A.grad, Ar.grad), rel(B.grad, Br.grad))
        if verbose:
            print(f"[triton_lora self-test] K={K} N={N} R={R} rel={e:.2e}")
        ok = ok and (e < 1.5e-2)
    return ok


_PATCHED = False


def install_fused_lora_delta() -> bool:
    """Install-on-call: self-test, then monkeypatch PEFT's vanilla LoRA
    Linear.forward to route the adapter delta through the fused kernel.

    Calling this function IS the opt-in (the Liger model) — there is no env flag.
    Returns True if the patch was installed. Mirrors the self-tested pattern of the
    other fused kernels. Only patches the vanilla (no-variant, no-dropout) path;
    everything else falls back to PEFT untouched.

    Recommended for large per-step token counts. What you get depends on the arch,
    because ``load_lora`` routes through ``load_entry("lora", ...)``: arches with a
    TUNED overlay (sm86/sm90/sm100) run that, the rest fall back to the portable
    fused-Triton kernel. Measured per-op fwd+bwd vs torch eager, K in {2560, 9216},
    N=4096/2560, r=32/16, on the two arches actually benchmarked:

      - sm86 (A5000), running its overlay: 0.94-1.00x at tokens<=8192, 1.17-1.59x
        at tokens>=16384. Never much worse than eager, and the win needs >=16384.
      - sm89 (L40S), running the portable fallback: 0.57-0.88x at tokens<=8192,
        1.17-1.46x at tokens>=16384. The small-batch loss is severe: up to 1.7x
        slower than eager.

    So >=16384 tokens is the regime that pays on both of those. Below it, sm86 is
    roughly a wash while the portable path is a real regression; if you train at short
    sequence x small batch on an arch with no overlay, do not install this.

    Scope: those are two data points, not a law. The sm90 and sm100 overlays are
    different kernels and are unmeasured here, as is portable execution on sm80/sm120
    and on any arch ``load_entry`` does not recognize. Treat >=16384 as the threshold
    established for A5000 and L40S and re-measure before trusting it elsewhere.

    Do not read the small-batch loss as pure dispatch overhead that large batches merely
    amortize -- but this data cannot rule that out either. Regressing portable_ms -
    eager_ms on the token count gives a negative slope with a POSITIVE intercept on all
    four measured (arch, K) sequences (+1.08 and +0.87ms on sm86, +0.35 and +0.21ms on
    sm89), which is the signature of a fixed cost C plus an M-dependent throughput gain:
    C - k*M crosses zero while C itself stays positive. Separating the two would take a
    launch-overhead measurement this sweep does not have. Pick the threshold from the
    arch and the hidden size, not from a flat 8192.
    """
    global _PATCHED
    if _PATCHED:
        return True
    # Clean no-op off the GPU worker (CPU / control plane): no triton or no CUDA means the
    # kernel can't run, so return False quietly — without the scary "self-test FAILED" log,
    # which is reserved for a real GPU where the kernel built but mismatched.
    if not _HAVE_TRITON or not torch.cuda.is_available():
        return False
    # The live self-test (compile/runtime/CUDA OOM) and the monkeypatch install
    # must NEVER abort the run: the kernel is opt-in and promises to fall back to
    # the stock PEFT path on ANY failure. Mirror install_qwen35_rope / install_
    # qwen35_mlp: wrap the whole thing and treat a raise as "keep stock PEFT".
    try:
        from peft.tuners.lora import layer as _lora_layer

        _orig_forward = _lora_layer.Linear.forward

        # Per-device arch-dispatch delta entry cache. Under a multi-GPU device_map the SAME patched
        # ``Linear.forward`` runs on tensors resident on different CUDA devices; a single process-wide
        # entry would be built once for whatever device happened to be current at the first call and
        # be wrong for the others. Key the cache by the input's device and build the entry under that
        # device's context (``load_lora`` selects the per-arch TUNED kernel when it is a verified win
        # AND passes the live-GPU delta self-test, else the portable fused-Triton kernel — never torch
        # eager; None means neither is available on that device).
        _delta_cache: dict = {}

        def _delta(device):
            if device not in _delta_cache:
                with torch.cuda.device(device):
                    _delta_cache[device] = load_lora()
            return _delta_cache[device]

        # Gate the INSTALL on the SAME dispatch the runtime uses: ``load_lora`` builds and delta-
        # self-tests the arch/portable entry on the current CUDA device (M=2048, norm tol) — exactly
        # what the patched forward resolves per device below. A None here means that dispatch is
        # unavailable or failed its self-test, so keep stock PEFT rather than install a forward whose
        # first call would permanently cache None and silently drop to eager (the old install gate on
        # ``_self_test`` used a different shape/tolerance and could pass while this one fails). No live
        # GPU is needed on the CPU / control plane: the CUDA+triton guard above already returned False.
        if load_lora() is None:
            print("[triton_lora] delta self-test FAILED -> not patching PEFT")
            return False

        def _fused_forward(self, x, *args, **kwargs):
            # Only the plain training path (adapters enabled, not merged, no mixed
            # batch, no dropout, no LoRA variant). Anything else -> original PEFT,
            # which knows how to pop variant-only kwargs (e.g. aLoRA's
            # ``alora_offsets``) before delegating to the base layer.
            adapter_names = kwargs.get("adapter_names")
            if self.disable_adapters or self.merged or adapter_names is not None:
                return _orig_forward(self, x, *args, **kwargs)
            # PEFT variants forward layer-only kwargs (e.g. ``alora_offsets``)
            # through these hooks and pop them before calling ``base_layer``; a
            # plain ``nn.Linear`` cannot accept them. Any extra positional/keyword
            # args therefore mean "let stock PEFT handle the variant path" — do NOT
            # forward them blindly into ``self.base_layer`` (that would raise
            # instead of taking the intended no-variant fallback).
            if args or kwargs:
                return _orig_forward(self, x, *args, **kwargs)
            # The fused kernel replicates the *dense* PEFT Linear math only. A
            # bnb-quantized base (peft.tuners.lora.bnb.Linear4bit/Linear8bitLt)
            # defines its own forward and is never reached via this dense patch,
            # but guard per-instance anyway so a non-dense base falls back cleanly.
            if type(self.base_layer) is not torch.nn.Linear:
                return _orig_forward(self, x)
            # Fuse only when the fwd+BACKWARD win will actually be realized. The fused delta's
            # advantage is its fused backward, so under no_grad it can only add its per-projection
            # Triton-launch overhead with no backward to pay it off. That is the entire generation
            # path — GRPO / eval rollout prefill AND autoregressive decode all run under no_grad —
            # which is where the fused path regressed (measured ~18% slower on a GRPO step, all in
            # generation). A pure token-count gate cannot fix this: rollout prefill (batch x prompt
            # len) has the SAME flattened token count as the training forward, so gating on tokens
            # alone would either keep the prefill tax or kill the training win. Gate on grad-enabled
            # instead (this also matches production, where rollouts run in a separate inference
            # engine and chalk only ever sees the grad-enabled training fwd/bwd). Keep a token-count
            # floor too, for the rare tiny-batch training forward where launch overhead still wins.
            # x is [..., in_features]; the leading dims are the token count.
            if not torch.is_grad_enabled() or x.numel() // x.shape[-1] < _FUSE_MIN_TOKENS:
                return _orig_forward(self, x)
            adapters = []
            for ad in self.active_adapters:
                if ad not in self.lora_A or ad in getattr(self, "lora_variant", {}):
                    return _orig_forward(self, x)
                drop = self.lora_dropout[ad]
                if not isinstance(drop, torch.nn.Identity):
                    return _orig_forward(self, x)
                # ``lora_bias=True`` adapters carry a trainable bias on lora_B that
                # the stock forward adds (and accumulates a grad for); the fused
                # kernel takes only the weights, so fall back to preserve it.
                if self.lora_B[ad].bias is not None or self.lora_A[ad].bias is not None:
                    return _orig_forward(self, x)
                A = self.lora_A[ad].weight
                B = self.lora_B[ad].weight
                scaling = self.scaling[ad]
                # CUDA-ONLY: ``fused_lora`` is a Triton kernel and only runs on CUDA tensors that
                # share one device.
                # This patch is installed process-wide, so a dense LoRA layer that later executes
                # on CPU / device_map-offloaded/model-parallel tensors would otherwise reach the
                # Triton kernel and raise instead of using stock PEFT. Fall back to the original
                # differentiable PEFT forward whenever the input or this adapter's A/B weights are
                # not all resident on the same CUDA device.
                if not (x.is_cuda and A.is_cuda and B.is_cuda and A.device == x.device and B.device == x.device):
                    return _orig_forward(self, x)
                adapters.append((A, B, scaling))
            # Resolve the per-device delta entry BEFORE running the base layer: if the dispatch is
            # unavailable (None) we bail to stock PEFT, and the base layer must run exactly once on
            # that path. Running ``self.base_layer(x)`` first and THEN falling to ``_orig_forward``
            # (which recomputes the base) would execute the base layer twice.
            _dfn = _delta(x.device)
            if _dfn is None:  # arch+portable dispatch unavailable at call time -> keep stock PEFT
                return _orig_forward(self, x)
            result = self.base_layer(x)
            rdtype = result.dtype
            # The arch-dispatch entry (``delta_fn``) computes the pure adapter DELTA on a 2D
            # ``[tok, K]`` activation, so flatten x's leading dims once and add the delta OUTSIDE
            # the kernel — exactly as stock PEFT does (``result = result + delta``). This keeps the
            # entry signature identical to what the self-test feeds (and what the verifier graded),
            # so the arch tree's ``[tok, K] -> [tok, N]`` kernels are used verbatim.
            x2 = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
            for A, B, scaling in adapters:
                xi = self._cast_input_dtype(x2, A.dtype)
                delta = _dfn(xi, A, B, scaling)  # [tok, N]
                if delta.shape != result.shape:
                    delta = delta.reshape(result.shape)
                result = result + delta
            return result.to(rdtype)

        _lora_layer.Linear.forward = _fused_forward
        _PATCHED = True
        print(
            "[triton_lora] PEFT (dense) LoRA Linear.forward patched with fused Triton kernel "
            "(QLoRA/bnb Linear4bit/Linear8bitLt keep stock PEFT — unsupported by this kernel)"
        )
        return True
    except Exception as e:  # pragma: no cover - defensive: any failure keeps stock PEFT
        print(f"[triton_lora] patch install failed ({type(e).__name__}: {e}); keeping stock PEFT")
        return False


if __name__ == "__main__":  # pragma: no cover - manual GPU smoke
    print("self-test:", "PASS" if _self_test(verbose=True) else "FAIL")
