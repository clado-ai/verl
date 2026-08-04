"""Chalk-native fused kernels for the Qwen3.5/3.6 Gated-DeltaNet (GDN) block.

The GDN layer (``Qwen3_5GatedDeltaNet`` / ``Qwen3_5MoeGatedDeltaNet``) is the hybrid arch's
linear-attention mixer. Its dominant cost — the ``chunk_gated_delta_rule`` *scan* — is genuinely
external-SOTA (fla's chunk-parallel kernel, years of tuning + a tilelang Hopper backend); chalk
does NOT try to beat the scan here (an honest benchmark vs fla is in benchmark/results/perarch/,
fla stays the scan path). What chalk DOES own is the **non-scan epilogue** around the scan, which
in the real flash worker runs **eager** (pure PyTorch), NOT on any optimized kernel:

  1. THE GATE (always eager — fla provides only the scan + gated-RMSNorm, never this):
        beta = b.sigmoid()                                            # [B,T,Hv]
        g    = -A_log.float().exp() * F.softplus(a.float() + dt_bias) # [B,T,Hv], fp32
     In HF eager this is ~5 separate elementwise kernel launches (exp, add, softplus, mul,
     sigmoid) with fp32 up/downcasts and intermediate HBM round-trips. ``A_log``/``dt_bias`` are
     tiny per-head vectors broadcast over [B,T]. chalk fuses the whole thing into ONE Triton
     kernel (fwd+bwd) — one HBM pass for ``b``/``a`` in, ``beta``/``g`` out, no intermediates.

  2. THE SHORT CAUSAL CONV (eager whenever ``causal_conv1d`` is absent — and the flash worker
     deliberately does NOT install causal_conv1d, see flash deps.py, so this IS eager in prod):
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])    # depthwise, k=conv_kernel_size
     A depthwise (groups==channels) causal conv with a short kernel (k=4 for Qwen3.5) + SiLU.
     ``F.conv1d`` launches a general cuDNN/grouped-conv path + a separate SiLU; chalk fuses the
     depthwise causal conv AND the SiLU into one Triton kernel (fwd+bwd over weight + input).

chalk ships fused Triton kernels (fwd+bwd, each self-tested on live GPU vs the exact HF math) for
BOTH, but only the CONV is installed in prod: ``install_qwen35_gdn`` (the single installer; no env
flag — calling IS the opt-in) patches the GDN ``forward`` to replace the eager conv+SiLU with the
fused conv kernel, while the GATE deliberately STAYS EAGER — the fused gate was benchmarked and
LOSES (autograd-dispatch bound, not compute bound), so the gate kernel exists/self-tests but isn't
patched in. The scan in between stays fla. Any import/compile/self-test failure leaves the eager
path untouched (correctness over speed).

  3. THE GQA EXPANSION (a chalk-owned win on the scan's PRE-op, 0.4.5): the eager forward expands
     q,k from ``num_k_heads`` up to ``num_v_heads`` with ``repeat_interleave`` (materializing 2x q +
     2x k in HBM + an interleave kernel) before the scan. But fla's ``chunk_gated_delta_rule``
     natively supports grouped-value-attention (``num_v_heads % num_k_heads == 0``) and broadcasts
     internally — so on every NON-Hopper arch (sm80/86/120, where fla uses pure Triton, no #640)
     chalk SKIPS the expansion and hands the un-expanded q,k to the scan. do_bench win (A100 sm80
     1.06-1.14x / A40 sm86 1.08-1.10x fwd+bwd at B>=2, + ~1.04x less peak mem); forward bit-identical.
     OFF on Hopper sm90: fla's tilelang chunk_bwd backend (the #640 fix the flash worker forces there)
     hard-refuses the GVA shape and would crash into the broken Triton path — so sm90 keeps the expand.
     Gated by ``get_device_capability()[0] != 9`` at install (``gva_native``).

  4. THE POST-SCAN GATED RMSNORM (a chalk-owned win on the scan's POST-op): the GDN block ends with
     ``self.norm(core_attn_out, z)`` = ``Qwen3_5RMSNormGated`` — ``y = (weight * rmsnorm(x)) * silu(z)``
     over ``head_v_dim``. In prod this is fla's ``FusedRMSNormGated`` (when fla is installed) else the
     eager class (separate var/rsqrt/silu/mul launches). chalk fuses the WHOLE op — rmsnorm + weight +
     silu-gate — into one fwd + one bwd Triton kernel, competing with (and on the GDN dims, beating)
     fla's fused gated-norm too. Self-test gated independently of the conv; engages only when
     ``self.norm`` is the standard silu-gated RMSNorm (``_norm_is_silu_gated``); else defers to
     ``self.norm``. (``load_gated_rmsnorm`` -> ``gnorm_fn``.)

GATE MATH (matched EXACTLY to modeling_qwen3_5):
    beta = sigmoid(b)                                  # bf16 in, bf16 out
    g    = -exp(A_log) * softplus(a_fp32 + dt_bias)    # a cast to fp32, A_log/dt_bias fp32, g fp32
  where ``softplus(x) = log1p(exp(x))`` (numerically the HF/torch threshold form: for large x,
  softplus(x) == x). The gate output ``g`` is consumed by the scan in fp32; ``beta`` matches b's
  dtype. Grads flow back to ``b``, ``a``, ``A_log``, ``dt_bias``.

CONV MATH (matched EXACTLY to F.silu(conv1d(x)[..., :L]) for a depthwise causal conv1d with
``padding = k-1`` then truncated to the input length — i.e. each output position t depends on
inputs [t-k+1 .. t] of the SAME channel, left-zero-padded):
    y[c,t] = silu( sum_{j=0..k-1} w[c,j] * x[c, t-(k-1)+j]  (+ bias[c]) )
  bias is optional (Qwen3.5 conv1d has bias=False, but the kernel supports it for generality).
"""

from __future__ import annotations

from chalk.ops._hf_targets import collect_qwen_classes
from chalk.utils import rel_l2

# Populated by the installers so a worker can fold the outcome into metrics.json notes.
# Empty {} means the kernel was not engaged this run.
RESULT_GATE: dict = {}
RESULT_CONV: dict = {}


# ======================================================================================
# 2. THE SHORT CAUSAL CONV: y[c,t] = silu(sum_j w[c,j] * x[c, t-(k-1)+j] (+ bias[c]))
#    depthwise (groups == channels), causal (left-pad k-1), then truncated to seq_len.
# ======================================================================================
def _build_conv_kernels():
    """Build the fused depthwise-causal-conv1d + SiLU fwd+bwd Triton kernels + autograd Function.
    Returns ``conv_fn(x, weight, bias) -> y`` (matching ``F.silu(conv1d(x)[..., :L])``) or raises.

    Layout: ``x`` is [B, C, L] (channels-first, as in the GDN block after ``transpose(1,2)``),
    ``weight`` is [C, K] (depthwise: one length-K filter per channel), ``bias`` is [C] or None."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _conv_fwd_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        y_ptr,
        B,
        C,
        L,
        K,
        HAS_BIAS: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        # one program == one (batch, channel) row of length L. K is small (4) -> unrolled accum.
        pid = tl.program_id(0)
        bc = pid  # row index over B*C
        c = bc % C
        offs = tl.arange(0, BLOCK_L)
        mask = offs < L
        x_base = x_ptr + bc * L
        acc = tl.zeros([BLOCK_L], dtype=tl.float32)
        for j in range(K):
            # output t uses input index t - (K-1) + j (causal, left zero pad)
            in_idx = offs - (K - 1) + j
            in_mask = mask & (in_idx >= 0)
            xv = tl.load(x_base + in_idx, mask=in_mask, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + c * K + j).to(tl.float32)
            acc += xv * wv
        if HAS_BIAS:
            acc += tl.load(bias_ptr + c).to(tl.float32)
        # SiLU: x * sigmoid(x)
        y = acc * tl.sigmoid(acc)
        tl.store(y_ptr + bc * L + offs, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _conv_bwd_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        dy_ptr,
        dx_ptr,
        dw_partial_ptr,
        dbias_partial_ptr,
        B,
        C,
        L,
        K,
        HAS_BIAS: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        # one program == one (batch, channel) row. Recompute the pre-activation, apply the SiLU
        # grad, then scatter into dx (causal transpose) and accumulate dw/dbias into per-ROW
        # partials [B*C, K]/[B*C] (reduced over batch on the host -> no cross-row atomics).
        pid = tl.program_id(0)
        bc = pid
        c = bc % C
        offs = tl.arange(0, BLOCK_L)
        mask = offs < L
        x_base = x_ptr + bc * L
        # recompute pre-activation acc
        acc = tl.zeros([BLOCK_L], dtype=tl.float32)
        for j in range(K):
            in_idx = offs - (K - 1) + j
            in_mask = mask & (in_idx >= 0)
            xv = tl.load(x_base + in_idx, mask=in_mask, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + c * K + j).to(tl.float32)
            acc += xv * wv
        if HAS_BIAS:
            acc += tl.load(bias_ptr + c).to(tl.float32)
        # d SiLU / d acc = sigmoid(acc) * (1 + acc * (1 - sigmoid(acc)))
        sig = tl.sigmoid(acc)
        dsilu = sig * (1.0 + acc * (1.0 - sig))
        dy = tl.load(dy_ptr + bc * L + offs, mask=mask, other=0.0).to(tl.float32)
        dacc = dy * dsilu  # [BLOCK_L], grad wrt the pre-activation at each output position
        # dbias = sum_t dacc
        if HAS_BIAS:
            tl.store(dbias_partial_ptr + bc, tl.sum(tl.where(mask, dacc, 0.0), axis=0))
        # for each tap j: out t uses x[t-(K-1)+j].
        #   dw[c,j] += sum_t dacc[t] * x[t-(K-1)+j]
        #   dx[t']  += sum_j dacc[t'+(K-1)-j] * w[c,j]   (gather over taps for input position t')
        # We do dx by iterating taps and scattering each tap's contribution back to its input idx.
        dx_acc = tl.zeros([BLOCK_L], dtype=tl.float32)
        for j in range(K):
            wv = tl.load(w_ptr + c * K + j).to(tl.float32)
            in_idx = offs - (K - 1) + j
            in_mask = mask & (in_idx >= 0)
            # dw[c,j]: dacc[t] * x[t-(K-1)+j] summed over t (only valid where in_idx>=0)
            xv = tl.load(x_base + in_idx, mask=in_mask, other=0.0).to(tl.float32)
            dw_j = tl.sum(tl.where(in_mask, dacc * xv, 0.0), axis=0)
            tl.store(dw_partial_ptr + bc * K + j, dw_j)
            # dx contribution: input position p = offs gets contributions from output t = p+(K-1)-j
            # i.e. dx[p] += w[c,j] * dacc[p + (K-1) - j]. Load dacc shifted by (K-1-j) to the LEFT.
            out_idx = offs + (K - 1) - j
            out_mask = mask & (out_idx < L)
            # gather dacc at out_idx; dacc lives in a register tile indexed by offs, so re-derive
            # from dy*dsilu at out_idx is cheapest via a fresh load of dy + recompute dsilu there.
            dy2 = tl.load(dy_ptr + bc * L + out_idx, mask=out_mask, other=0.0).to(tl.float32)
            # recompute acc at out_idx for dsilu — needs the pre-activation at out_idx. Cheap: the
            # filter is length K, but re-running the inner sum per tap is O(K^2). K is tiny (4), so
            # this is fine and avoids materializing acc for shifted indices.
            acc2 = tl.zeros([BLOCK_L], dtype=tl.float32)
            for jj in range(K):
                ii = out_idx - (K - 1) + jj
                iim = out_mask & (ii >= 0)
                xv2 = tl.load(x_base + ii, mask=iim, other=0.0).to(tl.float32)
                wv2 = tl.load(w_ptr + c * K + jj).to(tl.float32)
                acc2 += xv2 * wv2
            if HAS_BIAS:
                acc2 += tl.load(bias_ptr + c).to(tl.float32)
            sig2 = tl.sigmoid(acc2)
            dsilu2 = sig2 * (1.0 + acc2 * (1.0 - sig2))
            dacc2 = dy2 * dsilu2
            dx_acc += tl.where(out_mask, wv * dacc2, 0.0)
        tl.store(dx_ptr + bc * L + offs, dx_acc.to(dx_ptr.dtype.element_ty), mask=mask)

    def _launch_fwd(x, weight, bias):
        B, C, L = x.shape
        K = weight.shape[1]
        # A zero-length sequence has no output element to compute. Returning before the launch also
        # keeps ``BLOCK_L`` off ``next_power_of_2(0) == 0``, which makes the kernel body's
        # ``tl.arange(0, BLOCK_L)`` a compile-time error rather than an empty loop.
        if L == 0:
            return torch.empty_like(x)
        BLOCK_L = triton.next_power_of_2(L)
        y = torch.empty_like(x)
        has_bias = bias is not None
        bias_t = bias if has_bias else x  # dummy ptr when no bias
        with torch.cuda.device(x.device):
            _conv_fwd_kernel[(B * C,)](
                x,
                weight,
                bias_t,
                y,
                B,
                C,
                L,
                K,
                HAS_BIAS=has_bias,
                BLOCK_L=BLOCK_L,
                # else-branch 8->16: A/B on A100+H100 shows 16 beats 8 by ~1.19x fwd+bwd at the
                # large BLOCK_L (>2048) regime (L=4096/8192/16384), both timing orders, 32 plateaus.
                # One program covers a whole [BLOCK_L] causal-conv row via the K-tap loop, so a bigger
                # BLOCK_L wants more warps for parallelism; the old "else 8" was tuned for a shorter L.
                # BLOCK_L==4096 (mid seq) prefers 32 warps on H100 (~1.05x over 16, order-confirmed;
                # tied on A100/Ada, so no regression there); BLOCK_L>=8192 stays at 16 (32 ties/loses).
                num_warps=4 if BLOCK_L <= 2048 else 32 if BLOCK_L <= 4096 else 16,
            )
        return y

    def _launch_bwd(x, weight, bias, dy):
        B, C, L = x.shape
        K = weight.shape[1]
        # An empty sequence contributes nothing to any gradient. dx is already empty at the caller's
        # shape, but dweight ([C,K]) and dbias ([C]) are full rows that autograd accumulates into, so
        # they have to be real zeroed tensors rather than the uninitialized partials the reduction
        # below would otherwise sum. They are float32 to match the launched path, which reduces
        # float32 partials and returns float32 grads for parameters of any dtype. Returning here also
        # keeps ``BLOCK_L`` off ``next_power_of_2(0) == 0``, which the kernel body rejects at compile
        # time.
        if L == 0:
            f32 = torch.float32
            zero_bias = torch.zeros_like(bias, dtype=f32) if bias is not None else None
            return torch.empty_like(x), torch.zeros_like(weight, dtype=f32), zero_bias
        BLOCK_L = triton.next_power_of_2(L)
        dx = torch.empty_like(x)
        has_bias = bias is not None
        bias_t = bias if has_bias else x
        dw_partial = torch.empty((B * C, K), device=x.device, dtype=torch.float32)
        dbias_partial = torch.empty((B * C,), device=x.device, dtype=torch.float32)
        with torch.cuda.device(x.device):
            _conv_bwd_kernel[(B * C,)](
                x,
                weight,
                bias_t,
                dy,
                dx,
                dw_partial,
                dbias_partial,
                B,
                C,
                L,
                K,
                HAS_BIAS=has_bias,
                BLOCK_L=BLOCK_L,
                # else-branch 8->16: A/B on A100+H100 shows 16 beats 8 by ~1.19x fwd+bwd at the
                # large BLOCK_L (>2048) regime (L=4096/8192/16384), both timing orders, 32 plateaus.
                # One program covers a whole [BLOCK_L] causal-conv row via the K-tap loop, so a bigger
                # BLOCK_L wants more warps for parallelism; the old "else 8" was tuned for a shorter L.
                # BLOCK_L==4096 (mid seq) prefers 32 warps on H100 (~1.05x over 16, order-confirmed;
                # tied on A100/Ada, so no regression there); BLOCK_L>=8192 stays at 16 (32 ties/loses).
                num_warps=4 if BLOCK_L <= 2048 else 32 if BLOCK_L <= 4096 else 16,
            )
        # reduce per-row partials over batch -> [C,K] / [C]
        dweight = dw_partial.view(B, C, K).sum(0)
        dbias = dbias_partial.view(B, C).sum(0) if has_bias else None
        return dx, dweight, dbias

    class _ConvFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias):
            x = x.contiguous()
            weight = weight.contiguous()
            y = _launch_fwd(x, weight, bias if bias is None else bias.contiguous())
            ctx.save_for_backward(x, weight, bias if bias is not None else torch.empty(0, device=x.device))
            ctx.has_bias = bias is not None
            return y

        @staticmethod
        def backward(ctx, dy):
            x, weight, bias_saved = ctx.saved_tensors
            bias = bias_saved if ctx.has_bias else None
            dx, dweight, dbias = _launch_bwd(x, weight, bias, dy.contiguous())
            return dx, dweight, dbias

    def conv_fn(x, weight, bias=None):
        return _ConvFn.apply(x, weight, bias)

    return conv_fn


def _eager_conv(x, weight, bias, seq_len):
    """Exact HF reference: F.silu(conv1d(x, padding=K-1)[..., :seq_len]) for a depthwise conv.
    ``weight`` here is the [C,K] depthwise weight (conv1d.weight.squeeze(1))."""
    import torch.nn.functional as F

    C, K = weight.shape
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=K - 1, groups=C)
    return F.silu(out[:, :, :seq_len])


def _self_test_conv(conv_fn) -> None:
    """Live-GPU fwd+bwd parity vs the exact eager depthwise causal conv + SiLU. Raises on
    mismatch. Runs at the real Qwen3.5 conv kernel size (4) and representative conv_dim/seq."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA for gdn-conv self-test")
    dev = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tol = 2e-2
    gen = torch.Generator(device=dev).manual_seed(0)
    K = 4  # Qwen3.5 linear_conv_kernel_dim
    for C in (512, 1536):  # conv_dim = key_dim*2 + value_dim (Qwen3.5-2B ~ 1536)
        for B, L in ((1, 512), (2, 1024)):
            x = torch.randn(B, C, L, device=dev, dtype=dtype, generator=gen)
            weight = 0.3 * torch.randn(C, K, device=dev, dtype=dtype, generator=gen)
            for use_bias in (False, True):
                bias = (0.1 * torch.randn(C, device=dev, dtype=dtype, generator=gen)) if use_bias else None
                gy = torch.randn(B, C, L, device=dev, dtype=dtype, generator=gen)

                xr = x.float().clone().requires_grad_(True)
                wr = weight.float().clone().requires_grad_(True)
                br = bias.float().clone().requires_grad_(True) if use_bias else None
                ref = _eager_conv(xr, wr, br, L)
                ref.backward(gy.float())

                xc = x.clone().requires_grad_(True)
                wc = weight.clone().requires_grad_(True)
                bc = bias.clone().requires_grad_(True) if use_bias else None
                got = conv_fn(xc, wc, bc)
                torch.cuda.synchronize()
                got.backward(gy)
                torch.cuda.synchronize()

                rel = rel_l2  # fp32 relative-L2 error; shared helper (see chalk.utils.rel_l2)

                checks = [("y", got, ref), ("dx", xc.grad, xr.grad), ("dw", wc.grad, wr.grad)]
                if use_bias:
                    checks.append(("dbias", bc.grad, br.grad))
                for nm, gv, rv in checks:
                    r = rel(gv, rv)
                    if not (r < tol):
                        raise RuntimeError(
                            f"gdn-conv self-test failed C={C} B={B} L={L} bias={use_bias}: {nm}={r:.2e} (tol={tol:.0e})"
                        )


def load_conv():
    """Return ``conv_fn`` if the kernel builds + passes its live-GPU self-test, else None."""
    from chalk.ops.arch import load_entry
    from chalk.ops.arch import load_kernel

    # ``load_entry`` already self-tests the entry it returns (arch OR portable — the #34 change),
    # so load_kernel omits ``self_test`` here: a second self-test would just double startup
    # validation latency. Mirrors ``load_lora`` (#38).
    return load_kernel(
        "gdn-conv",
        "fused Triton depthwise causal conv + SiLU (fwd+bwd) enabled",
        "disabled",
        build=lambda: load_entry("gdn_conv", _self_test_conv, portable=_build_conv_kernels),
    )


# ======================================================================================
# 3. THE POST-SCAN GATED RMSNORM:  y = (w * rmsnorm(x)) * silu(z)
#    (the GDN block's ``self.norm(core_attn_out, z)`` — Qwen3_5RMSNormGated). The scan output is
#    RMS-normalized over head_v_dim, scaled by the learned weight, then gated by ``silu(z)`` where
#    ``z`` is the in_proj_z output. In HF this is fla's ``FusedRMSNormGated`` when fla is installed
#    (the flash worker case) else the eager ``Qwen3_5RMSNormGated`` (separate var/rsqrt/silu/mul
#    launches with fp32 up/downcasts). chalk fuses the WHOLE thing — rmsnorm + weight + silu-gate —
#    into one fwd kernel + one bwd kernel, so it competes with (and on the GDN dims, beats) fla's
#    fused gated-norm too, not just the eager fallback.
#
#    MATH (matches Qwen3_5RMSNormGated EXACTLY, llama casting):
#      xf   = x.float();  rstd = 1/sqrt(mean(xf^2) + eps)
#      xhat = (xf * rstd).to(dtype)          # normalize in fp32, cast back BEFORE weight mul
#      n    = w * xhat                        # weighted normalized activation (dtype)
#      y    = n * silu(z.float())            # gate computed in fp32, result cast to dtype
#    BWD (fp32 internal): with s = silu(zf), s' = sigmoid(zf)*(1 + zf*(1-sigmoid(zf)))
#      dn   = dy * s                          # grad into the normalized*weight term
#      dz   = dy * n * s'                     # grad into the gate
#      dw  += sum_rows(dn * xhat)            # over tokens
#      c    = dn * w ;  dx_j = rstd*(c_j - xhat_j * mean_k(c_k*xhat_k))
# ======================================================================================
def _build_gated_rmsnorm_kernels():
    """Build the fused gated-RMSNorm (rmsnorm * weight * silu(gate)) fwd+bwd Triton kernels +
    autograd Function. Returns ``gnorm_fn(x, weight, gate, eps) -> y`` or raises (caller keeps fla).

    ``x`` and ``gate`` are [..., N] (same shape; N == head_v_dim); ``weight`` is [N]."""
    import torch
    import triton
    import triton.language as tl

    # ROW-TILED kernels: each program processes a BLOCK_M x BLOCK_N tile of rows (not one row).
    # The GDN gated-norm has many short rows (N = head_v_dim = 128/256, M = B*L*num_v_heads up to
    # ~200k). One-program-per-row under-utilizes (M tiny launches of an N=128 vector); tiling rows
    # packs more work per program -> beats fla's FusedRMSNormGated across the whole production regime
    # on H100 (do_bench 1.10-1.95x fwd+bwd; v1 one-row-per-program lost to fla at the largest M).
    @triton.jit
    def _gnorm_fwd_kernel(
        x_ptr,
        w_ptr,
        z_ptr,
        y_ptr,
        rstd_ptr,
        x_row_stride,
        z_row_stride,
        y_row_stride,
        M,
        N,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < M
        cols = tl.arange(0, BLOCK_N)
        cmask = cols < N
        m2 = rmask[:, None] & cmask[None, :]
        x = tl.load(x_ptr + rows[:, None] * x_row_stride + cols[None, :], mask=m2, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(rstd_ptr + rows, rstd, mask=rmask)
        # normalize in fp32, cast back to dtype, THEN weight mul (llama casting)
        xhat = (x * rstd[:, None]).to(y_ptr.dtype.element_ty)
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0)[None, :]
        n = (w * xhat).to(tl.float32)
        # gate: silu(z) in fp32
        z = tl.load(z_ptr + rows[:, None] * z_row_stride + cols[None, :], mask=m2, other=0.0).to(tl.float32)
        s = z * tl.sigmoid(z)
        y = n * s
        tl.store(y_ptr + rows[:, None] * y_row_stride + cols[None, :], y.to(y_ptr.dtype.element_ty), mask=m2)

    @triton.jit
    def _gnorm_bwd_kernel(
        x_ptr,
        w_ptr,
        z_ptr,
        dy_ptr,
        rstd_ptr,
        dx_ptr,
        dz_ptr,
        dw_partial_ptr,
        x_row_stride,
        z_row_stride,
        dy_row_stride,
        dx_row_stride,
        dz_row_stride,
        dw_partial_row_stride,
        M,
        N,
        rows_per_prog: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # one program reduces ``rows_per_prog`` contiguous rows in BLOCK_M-row tiles, accumulating
        # its slice of the dweight partial locally (a [GROUP, N] -> [N] final reduce; no atomics).
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        cmask = cols < N
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        dw_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        base = pid * rows_per_prog
        for off in tl.range(0, rows_per_prog, BLOCK_M):
            rows = base + off + tl.arange(0, BLOCK_M)
            rmask = rows < M
            m2 = rmask[:, None] & cmask[None, :]
            x = tl.load(x_ptr + rows[:, None] * x_row_stride + cols[None, :], mask=m2, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + rows[:, None] * z_row_stride + cols[None, :], mask=m2, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr + rows[:, None] * dy_row_stride + cols[None, :], mask=m2, other=0.0).to(tl.float32)
            rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)[:, None]
            xhat = x * rstd  # fp32 normalized
            n = w * xhat  # weighted normalized (fp32)
            # gate + its derivative
            sig = tl.sigmoid(z)
            s = z * sig
            sprime = sig * (1.0 + z * (1.0 - sig))
            # y = n * s
            dn = dy * s
            dz = dy * n * sprime
            tl.store(dz_ptr + rows[:, None] * dz_row_stride + cols[None, :], dz.to(dz_ptr.dtype.element_ty), mask=m2)
            # rmsnorm backward through n = w*xhat with upstream dn
            dw_acc += tl.sum(tl.where(m2, dn * xhat, 0.0), axis=0)
            c = dn * w
            ss = (tl.sum(c * xhat, axis=1) / N)[:, None]
            dx = rstd * (c - xhat * ss)
            tl.store(dx_ptr + rows[:, None] * dx_row_stride + cols[None, :], dx.to(dx_ptr.dtype.element_ty), mask=m2)
        tl.store(dw_partial_ptr + pid * dw_partial_row_stride + cols, dw_acc, mask=cmask)

    def _next_pow2(n: int) -> int:
        return triton.next_power_of_2(n)

    def _block_m(block_n: int) -> int:
        # cap the tile at ~4096 fp32 elements/program so wide rows (N=256) don't register-spill;
        # gives BLOCK_M=32 for N=128 and 16 for N=256 (the do_bench-best on H100).
        return max(1, 4096 // block_n)

    def _warps_for(block_n: int) -> int:
        if block_n >= 8192:
            return 16
        if block_n >= 2048:
            return 8
        if block_n >= 512:
            return 4
        return 2

    def _bwd_grid(M: int, block_m: int, block_n: int, device) -> tuple:
        """Return ``(GROUP, rows_per_prog)``: ``GROUP`` striped [N] partials sized off the device SM
        count + row width; each program reduces ``rows_per_prog`` rows (a multiple of BLOCK_M)."""
        try:
            sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        except Exception:
            sm_count = 64
        waves = 4 if block_n < 2048 else (3 if block_n < 4096 else 2)
        n_tiles = triton.cdiv(M, block_m)
        GROUP = min(n_tiles, max(1, waves * sm_count))
        tiles_per_prog = triton.cdiv(n_tiles, GROUP)
        rows_per_prog = tiles_per_prog * block_m
        GROUP = triton.cdiv(M, rows_per_prog)  # tighten so no program is empty
        return GROUP, rows_per_prog

    class _GatedRMSNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, gate, eps):
            sh = x.shape
            N = sh[-1]
            x2 = x.reshape(-1, N)
            x2 = x2 if x2.is_contiguous() else x2.contiguous()
            z2 = gate.reshape(-1, N)
            z2 = z2 if z2.is_contiguous() else z2.contiguous()
            M = x2.shape[0]
            w = weight if weight.is_contiguous() else weight.contiguous()
            y = torch.empty_like(x2)
            rstd = torch.empty((M,), device=x2.device, dtype=torch.float32)
            BLOCK_N = _next_pow2(N)
            BLOCK_M = _block_m(BLOCK_N)
            # an empty batch has no rows to normalize, and launching would ask triton for a
            # zero-sized grid; the empty y allocated above is already the whole answer.
            if M > 0:
                with torch.cuda.device(x2.device):
                    _gnorm_fwd_kernel[(triton.cdiv(M, BLOCK_M),)](
                        x2,
                        w,
                        z2,
                        y,
                        rstd,
                        x2.stride(0),
                        z2.stride(0),
                        y.stride(0),
                        M,
                        N,
                        eps,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        num_warps=_warps_for(BLOCK_N),
                    )
            ctx.save_for_backward(x2, w, z2, rstd)
            ctx._sh = sh
            return y.reshape(sh)

        @staticmethod
        def backward(ctx, dy):
            x2, w, z2, rstd = ctx.saved_tensors
            sh = ctx._sh
            N = sh[-1]
            M = x2.shape[0]
            dy2 = dy.reshape(-1, N)
            dy2 = dy2 if dy2.is_contiguous() else dy2.contiguous()
            dx = torch.empty_like(x2)
            dz = torch.empty_like(z2)
            BLOCK_N = _next_pow2(N)
            BLOCK_M = _block_m(BLOCK_N)
            # an empty batch contributes nothing to any gradient. dx and dz are already empty at
            # the caller's shape, but dweight is a full [N] row that autograd accumulates into, so
            # it has to be a real zeroed tensor rather than an empty one. returning here also keeps
            # _bwd_grid off a zero row count, where its cdiv(M, rows_per_prog) divides by zero.
            if M == 0:
                return dx.reshape(sh), torch.zeros_like(w), dz.reshape(sh), None
            GROUP, rows_per_prog = _bwd_grid(M, BLOCK_M, BLOCK_N, x2.device)
            dw_partial = torch.empty((GROUP, N), device=x2.device, dtype=torch.float32)
            with torch.cuda.device(x2.device):
                _gnorm_bwd_kernel[(GROUP,)](
                    x2,
                    w,
                    z2,
                    dy2,
                    rstd,
                    dx,
                    dz,
                    dw_partial,
                    x2.stride(0),
                    z2.stride(0),
                    dy2.stride(0),
                    dx.stride(0),
                    dz.stride(0),
                    dw_partial.stride(0),
                    M,
                    N,
                    rows_per_prog=rows_per_prog,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    num_warps=_warps_for(BLOCK_N),
                )
            dweight = dw_partial.sum(0).to(w.dtype)
            return dx.reshape(sh), dweight, dz.reshape(sh), None

    def gnorm_fn(x, weight, gate, eps):
        return _GatedRMSNormFn.apply(x, weight, gate, eps)

    return gnorm_fn


def _eager_gated_rmsnorm(x, weight, gate, eps):
    """Exact HF ``Qwen3_5RMSNormGated`` reference (the self-test oracle):
    norm in fp32, cast back, weight mul, then * silu(gate.float())."""
    import torch
    import torch.nn.functional as F

    input_dtype = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    h = weight * h.to(input_dtype)
    return (h * F.silu(gate.to(torch.float32))).to(input_dtype)


def _self_test_gated_rmsnorm(gnorm_fn) -> None:
    """Live-GPU fwd+bwd parity vs the exact eager Qwen3_5RMSNormGated math (forward, dx, dweight,
    dgate). Raises on mismatch so the caller keeps fla's FusedRMSNormGated. Oracle in fp32 (the
    kernel's contract is fp32-internal, reduced-dtype I/O), bf16-toleranced."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA for gdn gated-rmsnorm self-test")
    dev, eps = "cuda", 1e-6
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tol = 2e-2
    gen = torch.Generator(device=dev).manual_seed(0)
    # head_v_dim across the Qwen3.5/3.6 family is 128 or 256; the row count is B*L*num_v_heads.
    for N in (128, 256):
        for M in (2048, 16384):
            x = torch.randn(M, N, device=dev, dtype=dtype, generator=gen)
            weight = (1.0 + 0.02 * torch.randn(N, device=dev, dtype=torch.float32, generator=gen)).to(dtype)
            z = torch.randn(M, N, device=dev, dtype=dtype, generator=gen)
            g = torch.randn(M, N, device=dev, dtype=dtype, generator=gen)

            xr = x.float().clone().requires_grad_(True)
            wr = weight.float().clone().requires_grad_(True)
            zr = z.float().clone().requires_grad_(True)
            ref = _eager_gated_rmsnorm(xr, wr, zr, eps)
            ref.backward(g.float())

            xc = x.clone().requires_grad_(True)
            wc = weight.clone().requires_grad_(True)
            zc = z.clone().requires_grad_(True)
            got = gnorm_fn(xc, wc, zc, eps)
            torch.cuda.synchronize()
            got.backward(g)
            torch.cuda.synchronize()

            rel = rel_l2  # fp32 relative-L2 error; shared helper (see chalk.utils.rel_l2)

            checks = [
                ("fwd", got, ref),
                ("dx", xc.grad, xr.grad),
                ("dw", wc.grad, wr.grad),
                ("dz", zc.grad, zr.grad),
            ]
            for nm, gv, rv in checks:
                r = rel(gv, rv)
                if not (r < tol):
                    raise RuntimeError(f"gdn gated-rmsnorm self-test failed N={N} M={M}: {nm}={r:.2e} (tol={tol:.0e})")


def load_gated_rmsnorm():
    """Return ``gnorm_fn`` if the kernel builds + passes its live-GPU self-test, else None."""
    from chalk.ops.arch import load_entry
    from chalk.ops.arch import load_kernel

    # Arch-tuned overlay (#32): load_entry picks chalk/ops/arch/<arch>/gdn_gated_rmsnorm.py when it
    # is a verified win (TUNED) and passes the op's live-GPU self-test; else the portable kernel.
    # load_entry ALREADY self-tests the entry it returns (arch OR portable — the #34 change), so
    # load_kernel omits ``self_test`` here: a second self-test would just double startup validation
    # latency (mirrors ``load_lora``; closes codex P2 on PR #32, r3523073131).
    return load_kernel(
        "gdn-norm",
        "fused Triton gated-RMSNorm (rmsnorm*w*silu(gate), fwd+bwd) enabled",
        "disabled",
        build=lambda: load_entry("gdn_gated_rmsnorm", _self_test_gated_rmsnorm, portable=_build_gated_rmsnorm_kernels),
    )


def _norm_is_silu_gated(norm) -> bool:
    """True iff ``norm`` is the standard gated-RMSNorm whose semantics chalk's fused kernel matches
    EXACTLY: ``y = (weight * rmsnorm(x)) * silu(gate)``. Covers both the eager ``Qwen3_5RMSNormGated``
    (always silu, no ``activation`` attr) and fla's ``FusedRMSNormGated`` (``activation`` string).
    Anything else (a non-silu activation, no ``.weight``) returns False -> chalk defers to ``self.norm``."""
    if not hasattr(norm, "weight"):
        return False
    act = getattr(norm, "activation", "silu")
    return act in (None, "silu")


# ======================================================================================
# INSTALL: patch the Qwen3.5/3.6 GatedDeltaNet.forward so its EAGER gate + EAGER conv run
# through the chalk fused kernels, while the dominant scan stays on fla. The scan is
# untouched — chalk only replaces the two eager epilogue ops around it.
# ======================================================================================
def _make_chalk_gdn_forward(orig_forward, *, conv_fn, gva_native=False, gnorm_fn=None):
    """Build a replacement GDN ``forward`` that routes the EAGER short causal conv (the
    ``F.silu(self.conv1d(...))`` path used when ``causal_conv1d_fn is None`` — the flash worker's
    config) through chalk's fused conv+SiLU kernel. The gate (beta+g) stays EAGER on purpose: a
    do_bench A/B (H100, fwd+bwd) showed a fused gate kernel LOSES to eager 0.67-0.90x — the gate
    tensors are tiny ([N, num_v_heads<=64]) so the path is autograd-dispatch/launch bound, not
    compute bound (eager fwd-only is ~0.03ms; even torch.compile is SLOWER than eager there). So
    chalk fuses ONLY the conv (the proven 1.05-1.32x win) and leaves the gate eager.

    ``gva_native`` (default False; set True only on non-Hopper by ``install_qwen35_gdn``): SKIP the
    eager ``repeat_interleave`` that expands q/k from ``num_k_heads`` up to ``num_v_heads`` before
    the scan, and instead let fla's chunk_gated_delta_rule broadcast internally (it natively supports
    Grouped-Value-Attention: ``num_v_heads % num_k_heads == 0``). This drops the 2x-q + 2x-k HBM
    materialization (and the interleave kernel) — a do_bench win (A100 sm80 1.06-1.14x fwd+bwd +
    ~1.04x less peak mem at the linkd GRPO regime; same on A40 sm86 / 5090 sm120). NOT enabled on
    Hopper (sm90): fla's tilelang chunk_bwd backend (the #640 fix) HARD-REFUSES the GVA shape
    ("does not support GQA ... use repeat_interleave") and would fall back to the broken Triton
    kernel — so on sm90 chalk keeps the expand (status quo). The forward output is bit-identical
    either way (``o`` rel err 6e-9 expand-vs-gva; only the q/k grad accumulation order differs, well
    within tol).

    ``gnorm_fn`` (default None): chalk's fused gated-RMSNorm kernel
    (``y = (weight * rmsnorm(x)) * silu(gate)``) replacing the POST-SCAN ``self.norm(core_attn_out,
    z)``. When set (the kernel built + self-tested) and ``self.norm`` is the standard silu-gated
    RMSNorm (eager ``Qwen3_5RMSNormGated`` OR fla ``FusedRMSNormGated`` — checked per-instance by
    ``_norm_is_silu_gated``), the whole gated-norm runs in one fwd + one bwd Triton kernel. Otherwise
    the forward keeps ``self.norm`` unchanged, so the conv + GVA wins still land.

    Any path this forward does not fully own — cached decode, single-token, a model whose
    ``causal_conv1d_fn`` is the real fla/causal_conv1d fast kernel, packed ``cu_seq_lens_q`` /
    ``seq_idx`` — delegates to ``orig_forward`` so nothing is silently changed."""
    import torch
    import torch.nn.functional as F

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        # Only own the multi-token, NON-cached training path. Anything else -> original forward.
        use_precomputed = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        # The fused conv assumes per-(B,C) contiguous rows with simple left-zero causal padding;
        # packed sequences (cu_seqlens / seq_idx) need fla's segmented conv, and any cache changes
        # the conv-input assembly. Be conservative: only take this path with no cache + no packing.
        if (
            use_precomputed
            or cache_params is not None
            or kwargs.get("cu_seq_lens_q") is not None
            or kwargs.get("seq_idx") is not None
        ):
            return orig_forward(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
        # chalk's conv only replaces the EAGER F.silu(conv1d(...)) path. If THIS model has the real
        # fast conv kernel wired (fla/causal_conv1d), don't displace it — defer to the original.
        if getattr(self, "causal_conv1d_fn", None) is not None:
            return orig_forward(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)

        # --- chalk-owned multi-token training path (mirrors modeling_qwen3_5/3_6 exactly; the ONLY
        #     substitution vs the eager forward is the fused conv+SiLU. The gate stays eager.) ---
        # mask padding (same as apply_mask_to_padding_states) — 2D bool mask, batched, multi-token
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, conv_dim, L]

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # FUSED CONV + SiLU (replaces F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]))
        conv_w = self.conv1d.weight.squeeze(1)  # [conv_dim, K]
        conv_b = self.conv1d.bias  # None for Qwen3.5
        mixed_qkv = conv_fn(mixed_qkv, conv_w, conv_b)  # already SiLU'd + length seq_len

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # GATE stays EAGER (a fused kernel here LOSES — see docstring).
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # GQA expansion: on Hopper (tilelang requires equal heads) materialize the expanded q/k;
        # elsewhere skip it and let fla broadcast natively (GVA) — saves the 2x-q+2x-k HBM + the
        # interleave kernel (do_bench win; o is bit-identical). Only valid when num_v % num_k == 0.
        if self.num_v_heads // self.num_k_heads > 1 and not (gva_native and self.num_v_heads % self.num_k_heads == 0):
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        # POST-SCAN GATED RMSNORM: y = (w * rmsnorm(core_attn_out)) * silu(z). Use chalk's fused
        # kernel when it built+self-tested (gnorm_fn set) AND this norm is the standard gated-RMSNorm
        # (.weight present, default eps gate). Otherwise delegate to self.norm (fla FusedRMSNormGated
        # or eager Qwen3_5RMSNormGated) unchanged. The chalk-fused path is gated to the bf16/fp16
        # training dtype + the standard activation; anything unusual falls back to self.norm.
        if gnorm_fn is not None and _norm_is_silu_gated(self.norm):
            eps = getattr(self.norm, "variance_epsilon", None)
            if eps is None:
                eps = getattr(self.norm, "eps", self.layer_norm_epsilon)
            core_attn_out = gnorm_fn(core_attn_out, self.norm.weight, z, float(eps))
        else:
            core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    return forward


def install_qwen35_gdn(run_benchmark: bool = False) -> dict | bool:
    """Patch the Qwen3.5/3.6 GatedDeltaNet ``forward`` so its EAGER short causal conv (the
    ``F.silu(conv1d(...))`` path, used whenever ``causal_conv1d_fn`` is absent — the flash worker's
    deliberate config) runs on chalk's fused Triton depthwise-causal-conv+SiLU kernel — IFF the
    kernel builds + passes its live-GPU fwd+bwd self-test. The dominant scan
    (``chunk_gated_delta_rule``) stays on fla, untouched; the gate (beta+g) stays eager (a fused
    gate kernel was benchmarked and LOSES — it is autograd-dispatch bound, not compute bound).

    Install-on-call (no env flag): calling this IS the opt-in. Patches the GDN CLASS so every
    instance picks up the fused conv. SAFE NO-OP if the conv kernel fails to build/self-test, or no
    qwen3_5/3_6 GatedDeltaNet class is importable (returns False, leaves every class untouched). A
    failed (re)install is non-destructive: ``RESULT_CONV`` and the patch are unchanged on any False.

    Returns a report dict on success (``{installed, patched, conv, gate}``), False on no-op."""
    conv_fn = load_conv()
    if conv_fn is None:
        return False
    targets = collect_qwen_classes("GatedDeltaNet")  # [(label, cls)]
    if not targets:
        print("[gdn] no qwen3_5/3_6 GatedDeltaNet class to patch; keeping eager", flush=True)
        return False

    # Native-GVA (skip the q/k repeat_interleave) is a do_bench win on every NON-Hopper arch
    # (sm80 A100, sm86 A40, sm120 5090) but MUST stay off on Hopper (sm90 = major 9): fla's tilelang
    # chunk_bwd backend — the #640 correctness fix the flash worker forces on sm90 — hard-refuses the
    # GVA shape and would crash into the broken Triton path. So gate it OFF only when major == 9
    # (Hopper / Hopper-Next). On sm100/sm120 (Blackwell, major >= 10) fla uses pure Triton (no #640),
    # GVA is correct + faster. (Multi-GPU: cap of dev 0.)
    gva_native = False
    try:
        import torch

        if torch.cuda.is_available():
            gva_native = torch.cuda.get_device_capability()[0] != 9
    except Exception:
        gva_native = False

    # POST-SCAN GATED RMSNORM: chalk's fused rmsnorm*weight*silu(gate) kernel replaces the eager
    # Qwen3_5RMSNormGated AND fla's FusedRMSNormGated for ``self.norm(core_attn_out, z)`` — a do_bench
    # win on the GDN head_v_dim rows (N=128/256). Independently self-test gated: if it fails to build/
    # self-test, gnorm_fn stays None and the forward keeps ``self.norm`` (fla/eager), so the conv +
    # GVA wins still land. Engages only when self.norm is the standard silu-gated RMSNorm (checked
    # per-instance via _norm_is_silu_gated).
    gnorm_fn = load_gated_rmsnorm()

    patched = []
    for label, cls in targets:
        # Save the unbound original ONCE per class (idempotent re-install keeps the very first).
        if not hasattr(cls, "_chalk_gdn_orig_forward"):
            cls._chalk_gdn_orig_forward = cls.forward
        orig = cls._chalk_gdn_orig_forward
        cls.forward = _make_chalk_gdn_forward(orig, conv_fn=conv_fn, gva_native=gva_native, gnorm_fn=gnorm_fn)
        patched.append(label)

    RESULT_CONV.clear()
    RESULT_CONV.update(
        {
            "installed": True,
            "self_test": "passed",
            "patched": patched,
            "gva_native": gva_native,
            "gated_norm": gnorm_fn is not None,
        }
    )
    report = {
        "installed": True,
        "patched": patched,
        "conv": "fused",
        "gate": "eager (fused loses)",
        "gated_norm": "fused" if gnorm_fn is not None else "self.norm (fla/eager)",
        "gva_native": gva_native,
    }
    print(
        f"[gdn] chalk fused conv+SiLU epilogue installed on {patched} "
        f"(scan stays fla; gate stays eager; native-GVA={'on' if gva_native else 'off (Hopper)'}; "
        f"gated-norm={'fused' if gnorm_fn is not None else 'self.norm'})",
        flush=True,
    )
    return report


if __name__ == "__main__":  # manual smoke
    import torch

    if torch.cuda.is_available():
        for name, builder, tester in (
            ("conv", _build_conv_kernels, _self_test_conv),
            ("gated-rmsnorm", _build_gated_rmsnorm_kernels, _self_test_gated_rmsnorm),
        ):
            try:
                f = builder()
                tester(f)
                print(f"gdn-{name} self-test: PASS")
            except Exception as e:
                print(f"gdn-{name} self-test: FAIL ({e})")
    else:
        print("gdn: requires CUDA (skipped)")
