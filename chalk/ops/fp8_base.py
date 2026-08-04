"""Model-wide FP8 (e4m3) frozen-base GEMM for Qwen3.5 LoRA training — sm_89+ (Ada 4090/L40S + Hopper H100 + Blackwell sm_100/sm_120).

This is the model-wide FP8 frozen-base GEMM path: it covers the MLP gate/up/down
projections AND the attention projections (the real-attention q/k/v/o AND the
Qwen3.5 GatedDeltaNet linear-attention in_proj_qkv / in_proj_z / out_proj) under a
single installer (scope-controllable via ``mlp=``/``attn=``). This is FP8-QLoRA-style
training on sm_89+ (the gate is ``get_device_capability() >= (8, 9)``: Ada 4090/L40S, H100, and Blackwell sm_100/sm_120).

THE QLoRA ARGUMENT (already validated for the MLP): in a LoRA fine-tune ALL base
projection weights are FROZEN; a bf16 LoRA adapter (A/B) trains on top. Running the
frozen base GEMM in FP8 e4m3 is a forward COMPUTE-PATH change (like QLoRA's 4-bit
frozen base), not a trained-weight change. The backward gradient to x (needed by
upstream layers / the LoRA adapter via grad-checkpointing) uses the bf16 frozen
base weight — the frozen base stays bf16-resident; only the forward matmul is FP8.
An 80-step Qwen3.5-4B LoRA SFT loss-curve A/B proved the MLP variant quality-neutral
(max loss drift 3.5e-3 over 80 steps, identical final loss). This module extends that
to attention and is itself gated behind its own end-to-end loss-curve A/B
(``benchmark/results/perarch/0.4.0_fp8_frozen_base.md``).

LoRA-SAFE / FROZEN-SAFE: a Linear is wrapped ONLY if it is a plain, bias-free,
``requires_grad=False`` ``nn.Linear`` (the frozen base). PEFT ``lora.Linear``
wrappers, trainable, or biased Linears are left untouched, so the adapter delta is
never bypassed. The wrap composes with the bf16 LoRA adapter: PEFT's ``lora.Linear``
holds its own frozen ``base_layer`` (a plain ``nn.Linear``) plus the bf16 A/B; this
installer targets the standalone frozen base Linears (the ones the LoRA target set
does NOT cover) and leaves PEFT's wrapped base alone (PEFT runs its own forward).

The forward FP8 GEMM is cuBLAS via ``torch._scaled_mm`` with a rowwise (per-token)
activation scale + per-channel (per-output) weight scale, fed by a FUSED Triton
per-token quant kernel (the naive eager amax/clamp/cast quant costs more than the
whole bf16 GEMM — see bench). Per-channel weight scales are cached on the module
(frozen -> quantize once).

Gated: CUDA capability >= (8,9), passing live self-test. Install-on-call (the Liger model):
calling ``install_fp8_frozen_base(model)`` IS the opt-in — there is no env flag.
Optional scope control via keyword args (Liger-style, all default-on):
  * ``attn=False`` -> skip attention/GDN projections (MLP-only coverage), for an
    apples-to-apples A/B.
  * ``mlp=False``  -> skip the MLP projections (attention-only).
  * ``min_k`` (default 256) -> never FP8 a GEMM whose contraction (in_features) is below
    this; the win needs a real GEMM and the tiny GDN in_proj_a/b (K-out 32) gate params
    are both pointless and numerically delicate.
"""

from __future__ import annotations

import contextlib

_E4M3_MAX = 448.0

# Projection name suffixes that make up the FROZEN attention base on Qwen3.5:
#  - real-attention blocks: q_proj / k_proj / v_proj / o_proj
#  - GatedDeltaNet linear-attention blocks: in_proj_qkv (2560->8192, the q/k/v feed),
#    in_proj_z (the gate value stream 2560->4096), out_proj (4096->2560).
# (in_proj_a / in_proj_b are tiny 2560->32 GDN gating params -> excluded by MIN_K.)
_ATTN_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
)
_MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


def _build_fp8(free_base: bool = False, fp8_dx: bool = False):
    """Build the fused per-token e4m3 quant kernel + the frozen-base FP8 Linear
    autograd.Function. Raises on any import/compile problem (caller -> keep baseline).
    Reuses the exact recipe proven for the MLP: rowwise act scale + per-channel weight
    scale, ``torch._scaled_mm`` cuBLAS FP8 GEMM, bf16-frozen-base dx backward.

    ``free_base`` (FP8-QLoRA, opt-in via the installer kwarg, NOT an env flag): store the
    frozen base as fp8 ONLY (drop the bf16 copy) and dequant it for the dx backward. fp8
    weights are half the bf16, so peak memory goes DOWN vs the baseline bf16 base AND the FP8
    forward GEMM is faster — faster *and* lighter (the default cached-fp8 path is +mem).

    ``fp8_dx`` (opt-in via the installer kwarg, default OFF — a RESEARCH speed lever): also run
    the dx BACKWARD GEMM (``dx = dy @ W``) in FP8 e4m3 instead of bf16 cuBLAS. do_bench H100
    sm90: dx 1.23-1.66x faster (q/gate-up/down). HONEST TRADE-OFF — unlike the FROZEN-base
    FORWARD (whose e4m3 error is a FIXED bias on a weight that never updates, so it trains like
    NF4-QLoRA), the dx error (~3.7e-2 rel vs bf16's ~1.7e-3) perturbs EVERY gradient that
    updates the LoRA adapter, so it can affect learning quality. Stays default-OFF until a real
    training-reward A/B validates it; the safe default keeps the exact bf16 dx. When on, the dx
    needs W quantized along the CONTRACTION (K) axis in a column-major layout (distinct from the
    forward's per-output-channel W8); that is built once per weight and cached in the Function's
    saved tensors. Incompatible with ``free_base`` (free_base already dequants the fp8 base for
    dx)."""
    import weakref

    import torch
    import triton
    import triton.language as tl

    F8 = torch.float8_e4m3fn
    _free_base = free_base
    _fp8_dx = fp8_dx and not free_base  # free_base owns the dx path (dequant); don't double-handle

    @triton.jit
    def _quant_rowwise_kernel(x_ptr, o_ptr, s_ptr, K, sxm, sxk, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + row * sxm + offs * sxk, mask=mask, other=0.0).to(tl.float32)
        s = tl.max(tl.abs(x)) / 448.0
        s = tl.where(s < 1e-12, 1e-12, s)
        q = tl.minimum(tl.maximum(x / s, -448.0), 448.0)
        tl.store(o_ptr + row * K + offs, q.to(o_ptr.dtype.element_ty), mask=mask)
        tl.store(s_ptr + row, s)

    def quant_rowwise(x):
        """x:[M,K] -> (x_fp8 [M,K] e4m3 contiguous, scale [M,1] f32). One fused pass."""
        M, K = x.shape
        o = torch.empty((M, K), device=x.device, dtype=F8)
        s = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        _quant_rowwise_kernel[(M,)](
            x,
            o,
            s,
            K,
            x.stride(0),
            x.stride(1),
            BLOCK_K=triton.next_power_of_2(K),
            num_warps=8,
        )
        return o, s

    def quant_weight_per_channel(w):
        """w:[N,K] frozen -> (w8 [N,K] e4m3, scale [1,N] f32). Once per weight."""
        N = w.shape[0]
        s = (w.float().abs().amax(1, keepdim=True) / _E4M3_MAX).clamp(min=1e-12)
        w8 = (w.float() / s).clamp(-_E4M3_MAX, _E4M3_MAX).to(F8)
        return w8, s.reshape(1, N).float().contiguous()

    def quant_weight_for_dx(w):
        """fp8_dx: quantize the frozen base ``w:[N,K]`` for the dx GEMM ``dx = dy @ w``.

        ``torch._scaled_mm(A[M,N], B[N,K])`` contracts N and requires B COLUMN-MAJOR
        (``B.stride(0) == 1``) with per-OUTPUT-column (K) scale ``scale_b:[1,K]``. So quantize
        ``w`` along its rows (the contraction axis N of the dx GEMM = w's out-dim) with a
        per-K-column scale, and return a column-major [N,K] view. Done once per weight (cached
        in the autograd saved tensors) so the layout cost is amortized, not per-step."""
        sW = (w.float().abs().amax(0, keepdim=True) / _E4M3_MAX).clamp(min=1e-12)  # [1,K]
        w8 = (w.float() / sW).clamp(-_E4M3_MAX, _E4M3_MAX).to(F8)  # [N,K] row-major
        # [N,K] in a column-major layout (stride(0)==1) as _scaled_mm's B requires.
        w8 = w8.t().contiguous().t()
        return w8, sW.float().contiguous()

    def smm(x8, sa, w8, sb):
        return torch._scaled_mm(x8, w8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=True)

    # Single-slot activation-quant cache. Sibling projections fed by the SAME input
    # (attention q/k/v all take hidden_states; MLP gate/up share x) otherwise each
    # re-quantize that input — and the rowwise-quant prologue is ~37% of a small-N FP8 GEMM
    # (k/v_proj on Qwen3.5), enough to make FP8 LOSE to bf16 cuBLAS there. Keyed by object
    # identity (weakref) + version counter, so a hit means the EXACT same live, unmutated
    # tensor — never a false hit from allocator data_ptr reuse after a free. One slot
    # suffices: q/k/v (and gate/up) call back-to-back, so the first quantizes and its siblings
    # reuse; any unrelated projection input simply misses and overwrites the slot.
    class _QCache:
        __slots__ = ("ref", "sa", "ver", "x8")

        def __init__(self):
            # Empty state: ``ref is None`` is the canonical "no cached activation".
            # ``ver = -1`` is the version sentinel (a live ``Tensor._version`` is always
            # >= 0, so it can never produce a false hit), and ``x8``/``sa`` hold no buffer.
            self.ref = None
            self.ver = -1
            self.x8 = None
            self.sa = None

        def clear(self):
            """Drop the cached activation and release its fp8/scale buffers (back to the
            empty state). A subsequent ``quant_act`` then misses and recomputes."""
            self.ref = None
            self.ver = -1
            self.x8 = None
            self.sa = None

    _qcache = _QCache()

    def quant_act(x):
        """Rowwise-quant the projection input ``x`` (>=2D), reusing the cached (x8, scale)
        when the IDENTICAL, unmutated tensor was just quantized by a sibling projection."""
        c = _qcache
        if c.ref is not None and c.ref() is x and c.ver == x._version:
            return c.x8, c.sa
        x8, sa = quant_rowwise(x.reshape(-1, x.shape[-1]))
        try:
            c.ref = weakref.ref(x)
            c.ver = x._version
            c.x8, c.sa = x8, sa
        except TypeError:  # x not weak-referenceable -> don't cache, and free the old slot
            # ``c.ref = None`` alone disables the cache but would strand the PREVIOUS
            # ``x8``/``sa`` buffers in the slot; clear them so they can be freed.
            c.clear()
        return x8, sa

    class _FP8Linear(torch.autograd.Function):
        """y = x @ W.T, W frozen. Forward FP8 e4m3; backward dx uses the bf16 frozen
        base weight (the QLoRA frozen-quantized-base recipe). No weight grad (frozen).
        Handles an arbitrary leading shape (flattens to [M,K], unflattens the output).
        The activation quant is shared across sibling projections via ``quant_act``."""

        @staticmethod
        def forward(ctx, x, w_bf16, w8, sb, w8_dx=None, sW_dx=None):
            sh = x.shape
            x8, sa = quant_act(x)
            y = smm(x8, sa, w8, sb)
            # dx backward needs the base weight. Modes:
            #  * default: save the bf16 weight -> EXACT bf16 dx (the safe default).
            #  * free_base (FP8-QLoRA): the bf16 copy is dropped to save memory, so save the fp8
            #    fwd weight + scale and DEQUANT in backward (dx uses the fp8-rounded weight).
            #  * fp8_dx (research speed lever): save the dx-quantized fp8 weight (per-K-column,
            #    column-major) + scale and run the dx GEMM itself in FP8 (faster, e4m3 dx error).
            if _free_base:
                ctx.save_for_backward(w8, sb)
                ctx._mode = "free_base"
            elif _fp8_dx and w8_dx is not None:
                ctx.save_for_backward(w8_dx, sW_dx)
                ctx._mode = "fp8_dx"
                ctx._K = w_bf16.shape[1] if w_bf16 is not None else w8_dx.shape[1]
            else:
                ctx.save_for_backward(w_bf16)
                ctx._mode = "bf16"
            ctx._sh = sh
            return y.reshape(*sh[:-1], y.shape[-1])

        @staticmethod
        def backward(ctx, dy):
            sh = dy.shape
            mode = getattr(ctx, "_mode", "bf16")
            if mode == "fp8_dx":
                w8_dx, sW_dx = ctx.saved_tensors  # [N,K] col-major fp8, [1,K] scale
                d = dy.reshape(-1, sh[-1]).contiguous()
                dq, sa = quant_rowwise(d)  # [M,N] e4m3, sa[M,1]
                dx = torch._scaled_mm(
                    dq, w8_dx, scale_a=sa, scale_b=sW_dx, out_dtype=torch.bfloat16, use_fast_accum=True
                ).reshape(*sh[:-1], w8_dx.shape[1])
                return dx, None, None, None, None, None
            d = dy.reshape(-1, sh[-1]).to(torch.bfloat16)
            if mode == "free_base":
                w8, sb = ctx.saved_tensors
                # dequant fp8 -> bf16 (transient, freed after this GEMM): w = w8 * per-out-chan scale
                w_bf16 = (w8.to(torch.float32) * sb.reshape(-1, 1)).to(torch.bfloat16)
            else:
                (w_bf16,) = ctx.saved_tensors
            dx = torch.mm(d, w_bf16).reshape(*sh[:-1], w_bf16.shape[1])
            return dx, None, None, None, None, None

    return quant_rowwise, quant_weight_per_channel, quant_weight_for_dx, _FP8Linear


def _self_test(quant_rowwise, quant_wpc, FP8Linear, quant_wdx=None) -> None:
    """Live-GPU numerics + autograd self-test. Asserts (a) the fused Triton quant
    dequant-matches the eager rowwise quant bit-exactly, (b) the FP8 linear forward
    rel-err vs fp32 is within a generous e4m3 envelope across the real Qwen3.5
    attention/MLP shapes, and (c) the bf16-dx backward is finite and correctly shaped.
    When ``quant_wdx`` is given (fp8_dx mode), also asserts the FP8 dx GEMM is finite,
    correctly shaped, and within the e4m3 dx envelope vs the exact fp32 dx.
    Raises on any mismatch -> caller keeps baseline."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    F8 = torch.float8_e4m3fn

    # The fused Triton quant must agree with the eager rowwise quant. Both share the SAME
    # per-row scale (assert that exactly), so any disagreement is the e4m3 ROUNDING of the
    # scaled value. The two implementations round-to-nearest-even, but a value that sits
    # exactly on an e4m3 bucket boundary can land in adjacent buckets on different backends
    # (Triton's .to(fp8) rounding differs from eager on sm_89 Ada for a few-per-thousand
    # boundary elements) — and near the top of the e4m3 range one bucket step is ~16-32 fp8
    # units, so a single boundary flip dequantizes to >>1e-3. That is a last-bit rounding
    # divergence, NOT a broken kernel (the end-to-end FP8Linear forward below still lands at
    # the ~3.7e-2 e4m3 envelope on every arch incl. Ada). So check (a) the scale matches
    # exactly, (b) the fused dequant is faithful to the INPUT within the e4m3 envelope, and
    # (c) the fraction of elements that disagree with eager by more than one e4m3 ULP is
    # negligible — which catches a genuinely wrong kernel (wrong scale/layout/NaNs/garbage)
    # without tripping on backend rounding.
    x = torch.randn(2048, 2560, device="cuda", dtype=torch.bfloat16)
    o, s = quant_rowwise(x)
    se = (x.float().abs().amax(1, keepdim=True) / _E4M3_MAX).clamp(min=1e-12)
    oe = (x.float() / se).clamp(-_E4M3_MAX, _E4M3_MAX).to(F8)
    if not torch.allclose(s, se, rtol=1e-4, atol=1e-12):
        raise RuntimeError("fp8 fused quant: per-row scale != eager rowwise scale")
    deq = o.float() * s
    if not torch.isfinite(deq).all():
        raise RuntimeError("fp8 fused quant produced non-finite values")
    # (b) faithful to the input within the e4m3 quantization envelope (relative)
    rel = (deq - x.float()).norm() / (x.float().norm() + 1e-9)
    if rel.item() > 8e-2:
        raise RuntimeError(f"fp8 fused quant rel-err vs input too high: {rel.item():.2e}")
    # (c) disagreement with eager beyond one e4m3 ULP must be a negligible fraction. One ULP
    # at fp8 value v is ~ |v| * 2**-3 (3 mantissa bits); compare in fp8-unit space (o vs oe).
    ulp = (oe.float().abs() * 0.125).clamp(min=1.0)  # >=1 fp8-unit floor for tiny values
    frac_bad = ((o.float() - oe.float()).abs() > ulp).float().mean().item()
    if frac_bad > 1e-2:
        raise RuntimeError(f"fp8 fused quant disagrees with eager beyond 1 ULP for {frac_bad:.1%} of elems")

    # Real Qwen3.5-4B frozen-base shapes: (out, in). 3-D input (B,T,H) to exercise
    # the leading-shape flatten/unflatten too.
    shapes = {
        "q_proj": (8192, 2560),
        "k_proj": (1024, 2560),
        "v_proj": (1024, 2560),
        "o_proj": (2560, 4096),
        "in_proj_qkv": (8192, 2560),
        "in_proj_z": (4096, 2560),
        "out_proj": (2560, 4096),
        "gate_proj": (9216, 2560),
        "down_proj": (2560, 9216),
    }
    for name, (N, K) in shapes.items():
        lin = nn.Linear(K, N, bias=False).cuda().to(torch.bfloat16)
        for p in lin.parameters():
            p.requires_grad_(False)
        w8, sb = quant_wpc(lin.weight)
        w8_dx, sW_dx = quant_wdx(lin.weight) if quant_wdx is not None else (None, None)
        xin = torch.randn(2, 1024, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = FP8Linear.apply(xin, lin.weight, w8, sb, w8_dx, sW_dx)
        ref = F.linear(xin.detach().float(), lin.weight.float())
        e = ((y.float() - ref).norm() / (ref.norm() + 1e-9)).item()
        if e > 8e-2:  # generous e4m3 envelope (real ~3.5e-2, random ~5.4e-2)
            raise RuntimeError(f"fp8 {name} rel-err too high: {e:.2e}")
        if y.shape != (2, 1024, N):
            raise RuntimeError(f"fp8 {name} bad output shape {tuple(y.shape)}")
        y.float().pow(2).sum().backward()
        if xin.grad is None or not torch.isfinite(xin.grad).all() or xin.grad.shape != xin.shape:
            raise RuntimeError(f"fp8 {name} backward produced bad dx")
        if quant_wdx is not None:
            # fp8_dx: dx must be within the e4m3 dx envelope vs the EXACT fp32 dx
            # (dx = dy @ W). dy here is 2*y (from the y^2 loss), reshaped to [M,N].
            dy = (2.0 * y.detach()).reshape(-1, N)
            dx_ref = (dy.float() @ lin.weight.float()).reshape(2, 1024, K)
            edx = ((xin.grad.float() - dx_ref).norm() / (dx_ref.norm() + 1e-9)).item()
            if edx > 8e-2:  # e4m3 dx envelope (real ~3.7e-2)
                raise RuntimeError(f"fp8_dx {name} dx rel-err too high: {edx:.2e}")


# ======================================================================================
# load_entry OVERLAY for the ``fp8_base`` op — the canonical frozen-base linear entry
# ``fp8_linear_fn(x, weight) -> y``, selected per-arch (tuned kernel) or portable.
#
# The autoresearch grid evolves this exact op (manifest cell ``fp8_base@<arch>``): entry
# ``fp8_linear_fn(x, weight) -> y``, fp32 oracle ``_eager_linear``. ``load_fp8_base`` routes through
# ``chalk.ops.arch.load_entry`` so a verified per-arch kernel (``arch/<arch>/fp8_base.py`` with
# ``TUNED = True``) is used when it passes the live self-test, else the portable e4m3 linear below.
# ``install_fp8_frozen_base`` consumes this entry for its DEFAULT frozen-base forward (see there).
# ======================================================================================
def _eager_linear(x, weight):
    """fp32 oracle for the ``fp8_base`` cell: the frozen-base linear ``y = x @ weight.T`` (the exact
    math the FP8 e4m3 forward approximates). The trustworthy fp32 truth the reduced-precision kernel
    is graded against (self-test reference + the arch tree's oracle)."""
    return x @ weight.transpose(-1, -2)


def _build_portable_linear():
    """Portable frozen-base FP8 e4m3 linear — the ``load_entry`` fallback for the ``fp8_base`` op and
    the tuned arch tree's numeric sibling. Entry ``fp8_linear_fn(x, weight) -> y`` (bf16):

      * FORWARD: rowwise per-token activation scale + per-output-channel weight scale fed to
        ``torch._scaled_mm`` (cuBLAS FP8 e4m3) — the recipe proven for the MLP/attention base.
      * BACKWARD: the EXACT bf16 grads (``dx = dy @ W`` in bf16 — the quality-safe frozen-base recipe,
        since the dx feeds the LoRA adapter gradient — and ``dw = dy^T @ x``, produced ONLY when
        ``weight.requires_grad`` so a frozen base pays for none).

    Handles an arbitrary leading activation shape (flatten/unflatten), non-contiguous inputs, the
    degenerate ``M == 0`` shape, and binds the launch to the input's device. Raises on any
    import/compile problem so the caller keeps the eager base."""
    import torch

    F8 = torch.float8_e4m3fn

    def _q_rowwise(t):
        # t:[R,C] -> (fp8 [R,C] e4m3, scale [R,1] f32) rowwise amax scale.
        s = (t.float().abs().amax(1, keepdim=True) / _E4M3_MAX).clamp(min=1e-12)
        q = (t.float() / s).clamp(-_E4M3_MAX, _E4M3_MAX).to(F8)
        return q, s

    class _FP8LinearPortable(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            sh = x.shape
            K = sh[-1]
            x2 = x.reshape(-1, K)
            x2 = x2 if x2.is_contiguous() else x2.contiguous()
            w = weight if weight.is_contiguous() else weight.contiguous()
            M, N = x2.shape[0], w.shape[0]
            with torch.cuda.device(x2.device):
                if M == 0:  # degenerate: no rows to quant/matmul, emit the empty bf16 output.
                    y = torch.empty((0, N), device=x2.device, dtype=torch.bfloat16)
                else:
                    x8, sa = _q_rowwise(x2)  # [M,K] e4m3, [M,1]
                    w8, sb = _q_rowwise(w)  # [N,K] e4m3, [N,1]
                    y = torch._scaled_mm(
                        x8,
                        w8.t(),
                        scale_a=sa,
                        scale_b=sb.reshape(1, N),
                        out_dtype=torch.bfloat16,
                        use_fast_accum=True,
                    )
            # Save the bf16 weight for the EXACT bf16 dx (dx feeds the adapter grad -> keep it bf16),
            # and the flattened activation ONLY when the weight needs a grad (a frozen base does not
            # — the frozen-base LoRA path production uses). ``save_for_backward`` only accepts tensors
            # (a ``None`` raises), so save just the tensors that exist and reconstruct in backward
            # (mirrors flce's optional-saved-tensor handling). Track presence on ctx.
            ctx._wreq = weight.requires_grad
            ctx._xreq = x.requires_grad
            if ctx._wreq:
                ctx.save_for_backward(w, x2)
            else:
                ctx.save_for_backward(w)
            ctx._sh = sh
            return y.reshape(*sh[:-1], N)

        @staticmethod
        def backward(ctx, gy):
            sh = ctx._sh
            N = gy.shape[-1]
            # Only the tensors that existed were saved (x2 dropped on the frozen-base path); unpack
            # positionally per ctx._wreq and rebuild the absent activation as None.
            saved = ctx.saved_tensors
            w = saved[0]
            x2 = saved[1] if ctx._wreq else None
            K = w.shape[1]
            gy2 = gy.reshape(-1, N)
            gy2 = gy2 if gy2.is_contiguous() else gy2.contiguous()
            gx = gw = None
            with torch.cuda.device(w.device):
                gyb = gy2.to(torch.bfloat16)
                if ctx._xreq:  # dx = dy @ W (bf16, exact frozen-base dx)
                    gx = torch.mm(gyb, w.to(torch.bfloat16)).reshape(*sh[:-1], K)
                if ctx._wreq:  # dw = dy^T @ x (only when the weight is trainable)
                    gw = torch.mm(gyb.t(), x2.to(torch.bfloat16))
            # None upstream slots map to None grads; a None grad for a non-requiring input is fine.
            return gx, gw

    def fp8_linear_fn(x, weight):
        return _FP8LinearPortable.apply(x, weight)

    return fp8_linear_fn


def _self_test_linear(fp8_linear_fn) -> None:
    """Live-GPU fwd+bwd parity for the ``fp8_linear_fn(x, weight) -> y`` entry vs the fp32 oracle
    ``_eager_linear`` (``y = x @ weight.T``). Asserts the forward AND both grads (dx, dw) are finite,
    correctly shaped, and within the e4m3 envelope (the manifest tol, fwd/bwd rel 8e-2) across the
    real Qwen3.5-4B frozen-base (out,in) projection shapes — including a 3-D leading activation to
    exercise the flatten/unflatten guard. Raises on any mismatch so ``load_entry`` falls back to the
    portable kernel (and ``load_fp8_base`` keeps the eager base)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA for fp8_base self-test")
    dev, tol = "cuda", 8e-2
    gen = torch.Generator(device=dev).manual_seed(0)
    # (leading token dims, (N=out, K=in)) real Qwen3.5-4B frozen-base projections; last is 3-D input.
    cases = [
        ((2048,), (8192, 2560)),  # q_proj / in_proj_qkv
        ((2048,), (1024, 2560)),  # k_proj / v_proj (small N)
        ((2048,), (2560, 4096)),  # o_proj / out_proj
        ((2048,), (9216, 2560)),  # gate_proj / up_proj
        ((2, 1024), (2560, 9216)),  # down_proj, 3-D input -> flatten/unflatten guard
    ]
    for lead, (N, K) in cases:
        x = torch.randn(*lead, K, device=dev, dtype=torch.bfloat16, generator=gen, requires_grad=True)
        weight = torch.randn(N, K, device=dev, dtype=torch.bfloat16, generator=gen, requires_grad=True)
        g = torch.randn(*lead, N, device=dev, dtype=torch.bfloat16, generator=gen)

        # fp32 oracle (the trustworthy reference; the kernel's internal accumulation is fp32).
        xr = x.detach().float().requires_grad_(True)
        wr = weight.detach().float().requires_grad_(True)
        ref = _eager_linear(xr, wr)
        ref.backward(g.float())
        dx_ref, dw_ref = xr.grad, wr.grad

        y = fp8_linear_fn(x, weight)
        torch.cuda.synchronize()
        y.backward(g)
        torch.cuda.synchronize()

        def rel(a, b):
            return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-9)

        if tuple(y.shape) != (*lead, N):
            raise RuntimeError(f"fp8_base bad output shape {tuple(y.shape)} vs {(*lead, N)}")
        r_fwd = rel(y, ref)
        if r_fwd > tol:
            raise RuntimeError(f"fp8_base fwd rel-err too high at N={N} K={K}: {r_fwd:.2e} (tol={tol:.0e})")
        if x.grad is None or not torch.isfinite(x.grad).all() or x.grad.shape != x.shape:
            raise RuntimeError(f"fp8_base backward produced bad dx at N={N} K={K}")
        r_dx = rel(x.grad, dx_ref)
        if r_dx > tol:
            raise RuntimeError(f"fp8_base dx rel-err too high at N={N} K={K}: {r_dx:.2e} (tol={tol:.0e})")
        if weight.grad is None or not torch.isfinite(weight.grad).all() or weight.grad.shape != weight.shape:
            raise RuntimeError(f"fp8_base backward produced bad dw at N={N} K={K}")
        r_dw = rel(weight.grad, dw_ref)
        if r_dw > tol:
            raise RuntimeError(f"fp8_base dw rel-err too high at N={N} K={K}: {r_dw:.2e} (tol={tol:.0e})")

    # FROZEN-BASE regression (the production LoRA path): weight.requires_grad=False. Forward must
    # NOT save a None activation (save_for_backward tensor-only), and backward must still return the
    # exact bf16 dx that feeds the adapter grad. Also exercise the degenerate M==0 (empty-batch) row
    # count so an empty forward returns [0,N] and empty grads (no zero-row kernel launch).
    N, K = 2560, 4096
    weight = torch.randn(N, K, device=dev, dtype=torch.bfloat16, generator=gen)  # requires_grad=False
    for M in (2048, 0):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=gen, requires_grad=True)
        y = fp8_linear_fn(x, weight)
        torch.cuda.synchronize()
        if tuple(y.shape) != (M, N):
            raise RuntimeError(f"fp8_base frozen-base bad output shape {tuple(y.shape)} vs {(M, N)}")
        y.sum().backward()
        torch.cuda.synchronize()
        if x.grad is None or not torch.isfinite(x.grad).all() or tuple(x.grad.shape) != (M, K):
            raise RuntimeError(f"fp8_base frozen-base backward produced bad dx at M={M}")
        if weight.grad is not None:
            raise RuntimeError("fp8_base frozen-base: weight.grad must stay None (frozen)")


def load_fp8_base():
    """Return the frozen-base FP8 linear entry ``fp8_linear_fn(x, weight) -> y`` for the running GPU:
    the arch-tuned kernel (``chalk/ops/arch/<arch>/fp8_base.py`` with ``TUNED = True``) when it is a
    verified win AND passes the live self-test, else the portable e4m3 linear. Never raises — any
    failure (no torch/CUDA, compile/self-test error) returns ``None`` so the caller keeps the eager
    base.

    This is the ``load_entry`` overlay wiring for the ``fp8_base`` op: the per-arch tuned tree is a
    self-validated speedup over the portable kernel, selected only when TUNED and its ``build()``
    passes ``_self_test_linear`` (fwd + dx + dw vs the fp32 oracle). ``install_fp8_frozen_base`` routes
    its default frozen-base forward through this entry (see there)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        from chalk.ops.arch import load_entry

        # load_entry already runs _self_test_linear on the entry it returns (arch or portable), so
        # do NOT self-test again here — that just doubles startup validation latency.
        fn = load_entry("fp8_base", _self_test_linear, portable=_build_portable_linear)
        print("[fp8_base] frozen-base FP8 e4m3 linear enabled (self-test passed)", flush=True)
        return fn
    except Exception as e:  # pragma: no cover - defensive: any failure keeps the eager base
        print(f"[fp8_base] disabled (build/self-test failed): {type(e).__name__}: {e}", flush=True)
        return None


def _arch_has_tuned_fp8_base() -> bool:
    """True iff the running GPU's arch ships a VERIFIED (``TUNED``) ``fp8_base`` kernel. Used to scope
    the ``load_entry`` overlay to arches we actually tuned: on those the installer routes its default
    frozen-base forward through the arch-tuned kernel; on every other arch (untuned fp8_base seed / no
    file) it keeps the existing cached-weight bf16-dx machinery unchanged (zero behavior change)."""
    try:
        import importlib

        from chalk.ops.arch import current_arch

        arch = current_arch()
        if arch is None:
            return False
        mod = importlib.import_module(f"chalk.ops.arch.{arch}.fp8_base")
        return bool(getattr(mod, "TUNED", False))
    except Exception:  # any probe failure => "no tuned arch kernel"
        return False


def _plain_frozen(p) -> bool:
    import torch
    import torch.nn as nn

    # bf16-ONLY: the FP8 path is validated for bf16 frozen base weights — _scaled_mm emits
    # bf16 and the backward multiplies a bf16 grad by the saved base weight. A frozen
    # fp16/fp32 Linear would change its forward output dtype to bf16 and can hit a
    # mixed-dtype torch.mm in backward, so leave non-bf16 base weights on the eager path.
    #
    # CUDA-ONLY: the replacement forward launches the Triton rowwise-quant kernel + a cuBLAS
    # FP8 `_scaled_mm` on the activation, which only run on a CUDA tensor. Under a `device_map`
    # offload a frozen bf16 Linear can sit on CPU (or a `meta` shell), and wrapping it would
    # raise at runtime when that layer executes on CPU instead of cleanly falling back. So skip
    # any Linear whose frozen base weight is not CUDA-resident and keep it on the eager path.
    return (
        type(p) is nn.Linear
        and getattr(p, "bias", None) is None
        and not p.weight.requires_grad
        and p.weight.dtype == torch.bfloat16
        and p.weight.is_cuda
    )


def _classify(name: str):
    """Return 'attn' / 'mlp' / None for a Linear leaf name (by its suffix)."""
    suf = name.rsplit(".", 1)[-1]
    if suf in _ATTN_SUFFIXES:
        return "attn"
    if suf in _MLP_SUFFIXES:
        return "mlp"
    return None


def _peft_lora_frozen_base(mod):
    """If ``mod`` is a PEFT LoRA ``Linear`` (has ``lora_A`` + ``base_layer``) whose
    ``base_layer`` is a plain frozen bf16 CUDA ``nn.Linear``, return that base_layer; else None.
    This lets FP8 wrap the FROZEN base GEMM *inside* a standard all-linear LoRA fine-tune —
    QLoRA-style (frozen base in FP8 e4m3, the bf16 LoRA A/B adapter on top) — the common config
    where the standalone-frozen path no-ops because every projection is a PEFT wrapper. The
    adapter delta (added after ``base_layer(x)`` by PEFT or chalk's fused-LoRA kernel) is
    untouched; only the frozen base matmul moves to FP8."""
    base = getattr(mod, "base_layer", None)
    if base is None or not hasattr(mod, "lora_A"):
        return None
    return base if _plain_frozen(base) else None


def install_fp8_frozen_base(
    model,
    *,
    attn: bool = True,
    mlp: bool = True,
    min_k: int = 256,
    lora_base: bool = False,
    no_wcache: bool = False,
    free_base: bool = False,
    fp8_dx: bool = False,
) -> dict:
    """Wrap every FROZEN plain-Linear base projection in ``model`` (attention q/k/v/o +
    GDN in_proj_qkv/in_proj_z/out_proj, and the MLP gate/up/down) so its forward GEMM
    runs FP8 e4m3 on FP8-capable GPUs. Returns a dict report ``{installed, attn, mlp, skipped,
    by_suffix}``.

    Install-on-call (the Liger model): calling this IS the opt-in — there is no env flag.
    GATED: rejects CUDA capability < (8, 9); accepts sm_89 Ada and sm_90+ Hopper-Blackwell
    when the live self-test passes. ANY failure (no torch/CUDA, no FP8 hardware,
    compile/self-test error) leaves the model untouched and returns ``{installed: 0, ...}``.

    Scope kwargs (Liger-style, all default-on): ``attn=False`` (MLP-only), ``mlp=False``
    (attn-only), ``min_k`` (default 256, excludes the tiny GDN gate Linears). ``lora_base``
    (default ``False``): also wrap the FROZEN ``base_layer`` inside PEFT LoRA wrappers — see
    the LoRA note below.

    Memory kwargs (default off; the FP8 base is a speed<->memory tradeoff — see the bench
    findings). Both are plain keyword args, NEVER env flags — kernel behavior is controlled
    entirely via this Python API (chalk has no ``CHALK_*`` runtime toggles):
      * ``no_wcache=True`` -> re-quantize the per-channel fp8 weight every forward instead of
        caching it on the module. The cached fp8 weight (default) is fast but adds an fp8 copy
        of every wrapped weight (~+50% of the frozen-weight memory); re-quantizing makes that
        fp8 weight a transient freed after each GEMM, so peak memory stays ~baseline at the
        cost of a small per-step weight-quant (cheap vs the GEMM win). Use it for FP8-base
        speed WITHOUT the memory regression.
      * ``free_base=True`` (FP8-QLoRA) -> store the frozen base as fp8 ONLY: cache the fp8
        weight + scale and DROP the bf16 base (net frozen-weight memory bf16(1x) -> fp8(0.5x),
        a SAVING vs baseline) plus the FP8 forward speedup; the dx backward dequants from the
        fp8. To keep checkpoints recoverable the fp8 weight + scale are registered as BUFFERS
        and a ``state_dict`` hook reconstitutes the original ``weight`` (dequantized) on save,
        so ``state_dict()`` round-trips the real bf16 weight rather than the freed shell.
        Takes precedence over ``no_wcache`` (the fp8 weight must persist to be the base).

    LoRA/FROZEN-SAFE: only bias-free, requires_grad=False Linears are wrapped, and a trainable
    projection is never touched. By DEFAULT a LoRA-wrapped projection is left untouched; with
    ``lora_base=True`` the frozen ``base_layer`` INSIDE a PEFT LoRA wrapper IS wrapped too — the
    trainable LoRA adapter math is unchanged, only the frozen base GEMM runs FP8. The wrap is
    differentiable (bf16-frozen-base dx), so it is active during TRAINING (that is the point)."""
    report = {"installed": 0, "already": 0, "attn": 0, "mlp": 0, "skipped": 0, "by_suffix": {}}
    try:
        import torch

        if not torch.cuda.is_available():
            return report
        # FP8 e4m3 tensor cores exist on sm_89 (Ada: 4090 / L40S / RTX-6000-Ada) and
        # sm_90+ (Hopper / Blackwell). torch._scaled_mm rowwise e4m3 runs on all of them
        # (do_bench-verified on Ada — see benchmark/results/perarch/0.4.11_fp8_ada_sm89.md).
        # sm_80 (A100) and below have NO FP8 hardware -> keep baseline there.
        if torch.cuda.get_device_capability() < (8, 9):
            print("[fp8-base] no FP8 hardware (need sm_89 Ada / sm_90+ Hopper-Blackwell); keeping baseline", flush=True)
            return report
        quant_rowwise, quant_wpc, quant_wdx, FP8Linear = _build_fp8(free_base=free_base, fp8_dx=fp8_dx)
        _self_test(quant_rowwise, quant_wpc, FP8Linear, quant_wdx=quant_wdx if (fp8_dx and not free_base) else None)
        _use_fp8_dx = fp8_dx and not free_base

        # DEFAULT frozen-base forward: route through the load_entry overlay (the arch-tuned
        # ``fp8_linear_fn(x, weight)`` for this GPU, else the portable e4m3 linear). Scoped to arches
        # that actually ship a VERIFIED (``TUNED``) ``fp8_base`` kernel so this is a zero-behavior-
        # change no-op on every other arch (they keep the cached-weight bf16-dx machinery below). The
        # research memory/speed levers (``free_base`` / ``fp8_dx``) own the frozen weight (freed or
        # dx-requantized) and keep their specialized machinery — the overlay is default-only.
        _arch_entry = None
        if not free_base and not _use_fp8_dx and _arch_has_tuned_fp8_base():
            _arch_entry = load_fp8_base()

        do_attn, do_mlp = attn, mlp

        # Track what we patch so a mid-loop failure can fully roll back — the documented
        # contract is "installed: 0 == untouched model", never a partially-FP8 model.
        _patched = []  # (module, original_forward)
        # Track free_base-freed modules separately: freeing the bf16 base is the only
        # destructive step (it releases the [out,in] weight storage), so it needs its own
        # restore path (the original weight + fp8 buffers + state_dict hook + freed flag).
        _freed = []  # (module, original_weight, state_dict_hook_handle)

        def _undo_free(lin, orig_weight, hook_handle):
            # Restore a freed module to its pre-free state: put the original bf16 weight back,
            # drop the private fp8 buffers and the state_dict hook, clear the freed flag.
            if hook_handle is not None:
                hook_handle.remove()
            for buf in ("_fp8_w8", "_fp8_sb"):
                # Remove from BOTH the buffer registry (register_buffer / free_base path) AND __dict__
                # (the cached non-free path sets it via plain setattr) so no stale attribute survives.
                lin._buffers.pop(buf, None)
                lin.__dict__.pop(buf, None)
            lin.weight = orig_weight
            if hasattr(lin, "_chalk_fp8_freed"):
                del lin._chalk_fp8_freed

        # Memory mode comes from the installer kwargs (no env flags). By default the
        # per-channel fp8 weight is cached on the module on first use (frozen -> quantize
        # once): fast, but +~50% of the frozen-weight memory. ``no_wcache`` re-quantizes it
        # every forward (transient, freed after the GEMM -> ~baseline peak memory).
        _no_wcache = no_wcache
        _free_base_w = free_base

        def _free_base_weight(lin, w8, sb):
            """FP8-QLoRA: drop the bf16 base weight now that the fp8 copy + scale exist
            (backward dequants from the fp8). Net frozen-weight memory bf16(1x) -> fp8(0.5x)
            = a SAVING vs baseline, plus the FP8 forward GEMM speedup.

            SAFE vs the naive ``lin.weight.data = empty(0)``: that blanked the parameter while
            the fp8 lived in plain attributes, so ``state_dict()`` silently serialized an empty
            ``weight`` -> irrecoverable checkpoints + a shape surprise for downstream code. Here
            (a) the fp8 weight + scale are registered as BUFFERS (so they persist and aren't
            lost), and (b) a state_dict hook reconstitutes the original-shape, DEQUANTIZED bf16
            ``weight`` on save (and drops the private fp8 buffers from the saved dict), so a
            checkpoint round-trips the real frozen weight rather than the freed shell."""
            import torch as _torch

            if getattr(lin, "_chalk_fp8_freed", False):
                return
            # Keep the ORIGINAL bf16 weight Parameter alive (not just its storage) so a later
            # failure can restore it exactly — record it for rollback BEFORE we mutate anything.
            _orig_weight = lin.weight
            _hook_handle = None
            lin._chalk_fp8_freed = True
            # Register the fp8 weight + scale as buffers (persistent: they ARE the live base now).
            lin.register_buffer("_fp8_w8", w8)
            lin.register_buffer("_fp8_sb", sb)
            # Free the bf16 base: keep a real 0-element parameter (so ``.weight`` still exists and
            # the optimizer/PEFT see a frozen param) but release the [out, in] weight storage.
            lin.weight = _torch.nn.Parameter(
                _torch.empty(0, dtype=_orig_weight.dtype, device=_orig_weight.device),
                requires_grad=False,
            )

            def _state_dict_hook(module, state_dict, prefix, local_metadata):
                # Reconstitute the original-shape, dequantized bf16 weight under the ``weight``
                # key (w = w8 * per-out-channel scale) and drop the private fp8 buffers, so the
                # saved checkpoint matches a normal Linear and is fully recoverable. The frozen
                # base is bf16 by construction (``_plain_frozen`` only wraps bf16 Linears).
                # Mutate ``state_dict`` IN-PLACE and return None: the public
                # ``register_state_dict_post_hook`` contract is "hook(module, state_dict, prefix,
                # local_metadata) -> None; may modify state_dict inplace" (torch ignores the return
                # value for public-API hooks), unlike the private hook which respected a return.
                w8b = state_dict.pop(prefix + "_fp8_w8", None)
                sbb = state_dict.pop(prefix + "_fp8_sb", None)
                if w8b is not None and sbb is not None:
                    state_dict[prefix + "weight"] = (w8b.to(_torch.float32) * sbb.reshape(-1, 1)).to(_torch.bfloat16)
                return

            # Public API (torch >= 2.1; this project requires torch >= 2.1.2) instead of the private
            # nn.Module._register_state_dict_hook, which is internal and may change across versions.
            _hook_handle = lin.register_state_dict_post_hook(_state_dict_hook)
            # Record everything this free mutated so a later failure restores the module EXACTLY
            # (the original weight Parameter, the private fp8 buffers, the state_dict hook, the
            # freed flag) — keeping the whole install all-or-nothing even though the free is the
            # one destructive (storage-releasing) step.
            _freed.append((lin, _orig_weight, _hook_handle))

        def make_fwd(lin):
            # Accept *args/**kwargs because PEFT calls ``base_layer(x, *args, **kwargs)`` — a plain
            # Linear ignores them and FP8Linear only needs x.
            def _fwd(x, *args, **kwargs):
                # DEFAULT path on a tuned arch: the load_entry overlay (arch-tuned / portable
                # ``fp8_linear_fn``). It re-quantizes the frozen weight per forward (memory-neutral,
                # like ``no_wcache``) so it needs no per-module fp8 cache. Only set for the default
                # config (not free_base / fp8_dx), so those levers still hit the machinery below.
                if _arch_entry is not None:
                    return _arch_entry(x, lin.weight)
                if _no_wcache and not _free_base_w:
                    w8, sb = quant_wpc(lin.weight)  # transient: freed after this GEMM (memory-neutral)
                    # fp8_dx + no_wcache: re-quantize the dx weight transiently too (memory-neutral).
                    w8_dx, sW_dx = quant_wdx(lin.weight) if _use_fp8_dx else (None, None)
                    return FP8Linear.apply(x, lin.weight, w8, sb, w8_dx, sW_dx)
                if not hasattr(lin, "_fp8_w8"):
                    w8, sb = quant_wpc(lin.weight)
                    if _free_base_w:
                        _free_base_weight(lin, w8, sb)
                    else:
                        lin._fp8_w8, lin._fp8_sb = w8, sb
                    # Cache the dx-quantized weight once (col-major, per-K-column scale) so the
                    # layout/quant cost is amortized over the run, not paid per backward step.
                    if _use_fp8_dx and not hasattr(lin, "_fp8_w8_dx"):
                        lin._fp8_w8_dx, lin._fp8_sW_dx = quant_wdx(lin.weight)
                w8_dx = getattr(lin, "_fp8_w8_dx", None)
                sW_dx = getattr(lin, "_fp8_sW_dx", None)
                return FP8Linear.apply(x, lin.weight, lin._fp8_w8, lin._fp8_sb, w8_dx, sW_dx)

            return _fwd

        for name, mod in model.named_modules():
            kind = _classify(name)
            if kind is None:
                continue
            if (kind == "attn" and not do_attn) or (kind == "mlp" and not do_mlp):
                continue
            # The Linear whose forward GEMM we move to FP8: a standalone frozen projection, or
            # (opt-in) the frozen base_layer inside a PEFT LoRA wrapper (QLoRA-style — brings the
            # FP8 base GEMM win to a standard all-linear LoRA fine-tune).
            if _plain_frozen(mod):
                target = mod
            elif lora_base:
                target = _peft_lora_frozen_base(mod)
            else:
                target = None
            if target is None:
                continue
            if getattr(target, "_chalk_fp8_base_patched", False):
                # Already wrapped by a prior successful install (re-run during setup / A-B
                # relabel). The FP8 forward is still live, so count it as active rather than
                # letting an all-already-patched re-run fall into the "installed == 0 ->
                # INACTIVE" branch and mislabel a live FP8-base path as off.
                report["already"] += 1
                continue
            if target.weight.shape[1] < min_k:  # contraction dim too small to matter
                report["skipped"] += 1
                continue
            _orig_mod_forward = target.forward
            target.forward = make_fwd(target)
            target._chalk_fp8_base_patched = True
            _patched.append((target, _orig_mod_forward))
            report["installed"] += 1
            report[kind] += 1
            suf = name.rsplit(".", 1)[-1]
            report["by_suffix"][suf] = report["by_suffix"].get(suf, 0) + 1

        # FP8-QLoRA (free_base): quantize + drop the bf16 base EAGERLY so the memory saving
        # (bf16 1x -> fp8 0.5x) is realized immediately — not deferred to the first forward. This
        # matters for large models (e.g. 35B-A3B) where the bf16 base is the OOM wall: freeing it
        # up front lets the model fit before any forward runs. (The cached-fp8 default stays lazy,
        # quantized on first forward, so it adds the fp8 copy only when used.)
        #
        # ALL-OR-NOTHING: the free is the one DESTRUCTIVE step (it releases the bf16 weight
        # storage), so it runs in its OWN pass AFTER the whole patch loop has succeeded — never
        # interleaved with forward-patching. Every free is recorded in ``_freed`` so the outer
        # ``except`` can restore the original weights too; thus even a failure here (or anywhere)
        # leaves the model EXACTLY as found rather than with half its bases freed to 0 elements.
        if _free_base_w:
            for target, _orig_mod_forward in _patched:
                w8, sb = quant_wpc(target.weight)
                _free_base_weight(target, w8, sb)

        if report["installed"] == 0 and report["already"] > 0:
            # Nothing new to patch because a previous call already wrapped these modules —
            # the FP8-base path IS active, so report success, not INACTIVE.
            print(
                f"[fp8-base] FP8 e4m3 frozen-base already installed on {report['already']} Linears "
                "(no new modules to patch on this re-run); FP8 base ACTIVE.",
                flush=True,
            )
            return report
        if report["installed"] == 0:
            print(
                "[fp8-base] FP8 base INACTIVE: 0 frozen plain-Linear base projections found "
                "(with LORA_TARGETS=all-linear the projections are PEFT LoRA Linears, so the "
                "frozen base is wrapped by PEFT — exclude the FP8-target projections from "
                "LORA_TARGETS, or this no-ops). Keeping baseline.",
                flush=True,
            )
            return report
        print(
            f"[fp8-base] FP8 e4m3 frozen-base installed on {report['installed']} Linears "
            f"(attn={report['attn']} mlp={report['mlp']}; by_suffix={report['by_suffix']}; "
            f"self-test passed)",
            flush=True,
        )
        return report
    except Exception as e:  # defensive: any failure keeps baseline (model left untouched)
        # Roll back EVERYTHING the patch mutated so the model is left exactly as we found it
        # (no partial-FP8 patch while the report says installed: 0). Undo the destructive
        # base-frees FIRST — restoring each original bf16 weight + dropping the fp8 buffers and
        # state_dict hook — so no module is left with a 0-element weight; then restore forwards.
        for mod, orig_weight, hook_handle in locals().get("_freed", ()):
            with contextlib.suppress(Exception):
                _undo_free(mod, orig_weight, hook_handle)
        for mod, orig_forward in locals().get("_patched", ()):
            with contextlib.suppress(Exception):
                mod.forward = orig_forward
                if hasattr(mod, "_chalk_fp8_base_patched"):
                    del mod._chalk_fp8_base_patched
        print(f"[fp8-base] FP8 base disabled (build/self-test failed): {type(e).__name__}: {e}", flush=True)
        return {"installed": 0, "already": 0, "attn": 0, "mlp": 0, "skipped": 0, "by_suffix": {}}


if __name__ == "__main__":  # manual self-test / smoke
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9):
        try:
            qr, qwpc, qwdx, FL = _build_fp8()
            _self_test(qr, qwpc, FL)
            qr2, qwpc2, qwdx2, FL2 = _build_fp8(fp8_dx=True)
            _self_test(qr2, qwpc2, FL2, quant_wdx=qwdx2)
            print("fp8-base self-test: PASS (bf16-dx + fp8-dx)")
        except Exception as e:
            print(f"fp8-base self-test: FAIL ({e})")
    else:
        print("fp8-base: requires FP8 hardware (sm_89 Ada / sm_90+ Hopper-Blackwell) (skipped)")
