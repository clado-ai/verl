"""fp8_base@sm89 — chalk autoresearch tuned kernel (one file per layer, per arch).

Cell: fp8_base@sm89
Entry: fp8_linear_fn(x, weight) -> y   (direction: fwd+bwd)
Oracle: chalk.ops.fp8_base._eager_linear   tol fwd_rel=0.08/bwd_rel=0.08 @ fp32
Measured 1.1305x (fwd+bwd) on 2026-07-04 against a retired anchor rule, on a backward this file no
longer ships. See ANCHOR and BACKWARD below before quoting the number.

STATUS: VERIFIED WIN, ADOPTED BY PRODUCTION DISPATCH. ``TUNED = True`` so ``chalk.ops.arch.load_entry``
selects this file on sm89 once ``build()`` passes the op's live-GPU self-test (``fp8_base._self_test_linear``);
any numeric/autograd mismatch falls back to the portable e4m3 linear. Verified on a real sm89 GPU by the
chalk autoresearch verifier (correctness + generalization + timing + roofline + anti-cheat) on 2026-07-04:
1.1305x, roofline_fraction=0.257, all gates green, no cheat flags.

ANCHOR: the 2026-07-04 scorer divided by ``min`` over the APPLICABLE baselines. Liger was never one of
them for this op: ``autoresearch/hive/eval/baselines.py`` registers only rmsnorm, rmsnorm_llama,
swiglu, rope, and flce in ``_LIGER_BUILDERS``, and ``fp8_base`` appears in none of the 17 historical
revisions of that file, so ``liger_callable("fp8_base")`` has always returned ``None`` and that arm
was never timed. CURRENT was not the portable e4m3 linear either -- ``autoresearch/hive/eval/eval.sh``
materialized it as ``git show ${HIVE_BASE_REF:-origin/main}:<this file>``, the PREVIOUS revision of
this same overlay. So the min ran over eager and that prior revision. #86 (a055885, 2026-07-28)
retired the rule for CURRENT-only, explicitly "never min-of-baselines". The run recorded only the
ratio, not the per-baseline milliseconds, so which arm was the min cannot be recovered and 1.1305x
cannot be restated against today's anchor.

BACKWARD: 1.1305x is a fwd+bwd number measured on a backward this file no longer contains. It was
graded at 8b00849, whose backward quantized the cotangent (``_quant_rows(gy, wcol=sw)``) and ran an
FP8 dx GEMM through ``_launch``. Later the same day 9baca37 replaced that with a bf16 ``torch.mm``
against the unquantized weight -- the quality-safe default documented below. So the *backward* half
of the ratio measures code that is gone, independently of which anchor it divided by.

The forward's compute path is untouched: for ``M > 0`` it is the same rowwise quant plus tiled e4m3
GEMM that was graded. 9baca37 added an ``M == 0`` guard (no zero-row launch) and changed which
tensors are stashed for backward (the unquantized ``weight`` instead of the quantized ``wf, sw``),
and 75b30bb applied ruff rewrites to the helpers (``range(0, n)`` -> ``range(n)``, lambda -> def);
none of these alter the forward math.

A rerun will not re-derive it either: ``_CHALK_RESOLVERS["fp8_base"]`` resolves CURRENT through
``_chalk_fp8_base`` -> ``load_fp8_base()``, which is arch-aware and returns THIS file on sm89, so the
anchor becomes the candidate and the cell times this kernel against itself (~1.0x). Re-measuring
against the portable e4m3 linear means pinning the anchor to it explicitly.

The verified FORWARD compute below (the FP8 e4m3 tiled GEMM + fused rowwise quant) is kept byte-for-byte as
graded. The BACKWARD computes grad_x in bf16 from the frozen base weight — the quality-safe DEFAULT this op
documents: FP8-dx is an opt-in research lever owned by the ``fp8_base`` installer's separate machinery, not
this default entry (its e4m3 dx error, ~3.7e-2 rel, would otherwise perturb every LoRA-adapter grad on the
documented ``fp8_dx=False`` default). ``build()`` layers the PRODUCTION guards required by dispatch: an
arbitrary leading activation shape (flatten/unflatten to the 2-D [M,K] the cell entry expects), the
degenerate ``M == 0`` empty-batch shape (returns an empty [0,N]/[B,0,N] with no zero-row kernel launch),
non-contiguous inputs (the rowwise quant reads via strides), and device binding around the kernel launches.
"""

import torch
import triton
import triton.language as tl

TUNED = True
SPEEDUP = 1.1305
# doubly unrestatable: the 2026-07-04 min-of-baselines anchor is gone (see ANCHOR above) and the
# backward half of this fwd+bwd figure measures code 9baca37 replaced (see BACKWARD above). None
# keeps dispatch from printing it as "verified".
SPEEDUP_ANCHOR = None

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


# ----------------------------- fp8 GEMM ------------------------------------
def _configs():
    return [
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 64, "GROUP": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "GROUP": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64, "BN": 64, "BK": 128, "GROUP": 8}, num_warps=4, num_stages=4),
    ]


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _gemm(
    A,
    B,
    C,
    ascale,
    bscale,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    HAS_BSC: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(num_pid_m - first_pid_m, GROUP)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_k = tl.arange(0, BK)
    a_ptrs = A + (offs_m[:, None] * sam + offs_k[None, :] * sak)
    b_ptrs = B + (offs_k[:, None] * sbk + offs_n[None, :] * sbn)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(tl.cdiv(K, BK)):
        kmask = offs_k < K - k * BK
        a = tl.load(a_ptrs, mask=mask_m[:, None] & kmask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=kmask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk

    asc = tl.load(ascale + offs_m, mask=mask_m, other=0.0)
    acc = acc * asc[:, None]
    if HAS_BSC:
        bsc = tl.load(bscale + offs_n, mask=mask_n, other=0.0)
        acc = acc * bsc[None, :]

    c_ptrs = C + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def _launch(A, B, ascale, bscale, M, N, K, sam, sak, sbk, sbn, out_dtype):
    C = torch.empty((M, N), device=A.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)

    _gemm[grid](A, B, C, ascale, bscale, M, N, K, sam, sak, sbk, sbn, C.stride(0), C.stride(1), bscale is not None)
    return C


# --------------------------- fused fp8 quant --------------------------------
@triton.jit
def _qrow(X, Q, S, R, C, sx0, sx1, DOSCALE: tl.constexpr, W, BC: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BC)
    acc = tl.zeros([BC], dtype=tl.float32)
    nblk = tl.cdiv(C, BC)
    for k in range(nblk):
        cols = k * BC + offs
        m = cols < C
        x = tl.load(X + row * sx0 + cols * sx1, mask=m, other=0.0).to(tl.float32)
        if DOSCALE:
            w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
            x = x * w
        acc = tl.maximum(acc, tl.abs(x))
    amax = tl.max(acc)
    amax = tl.maximum(amax, 1e-8)
    inv = 448.0 / amax
    tl.store(S + row, amax * (1.0 / 448.0))
    for k in range(nblk):
        cols = k * BC + offs
        m = cols < C
        x = tl.load(X + row * sx0 + cols * sx1, mask=m, other=0.0).to(tl.float32)
        if DOSCALE:
            w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
            x = x * w
        q = (x * inv).to(Q.dtype.element_ty)
        tl.store(Q + row * C + cols, q, mask=m)


def _quant_rows(t, wcol=None):
    R, C = t.shape
    q = torch.empty((R, C), device=t.device, dtype=FP8)
    s = torch.empty((R,), device=t.device, dtype=torch.float32)
    BC = 256 if C >= 256 else triton.next_power_of_2(C)
    _qrow[(R,)](t, q, s, R, C, t.stride(0), t.stride(1), wcol is not None, wcol if wcol is not None else t, BC=BC)
    return q, s


# ------------------------------ autograd ------------------------------------
class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        M, K = x.shape
        N = weight.shape[0]
        # Device binding: the rowwise-quant + FP8 GEMM Triton launches must run on x's device even
        # when the ambient current device differs (multi-GPU) — matches the portable kernel's guard.
        with torch.cuda.device(x.device):
            if M == 0:  # degenerate empty batch: no rows to quant/matmul, emit the empty bf16 output
                y = torch.empty((0, N), device=x.device, dtype=torch.bfloat16)
            else:
                xf, sx = _quant_rows(x)  # [M,K], [M]
                wf, sw = _quant_rows(weight)  # [N,K], [N]
                # y[m,n] = sum_k x[m,k]*w[n,k]; B logical [K,N]=w^T -> sbk=1, sbn=K
                y = _launch(xf, wf, sx, sw, M, N, K, K, 1, 1, K, torch.bfloat16)
        ctx.wreq = weight.requires_grad
        ctx.xreq = x.requires_grad
        # dx is computed in bf16 from the frozen base weight (the quality-safe DEFAULT: FP8-dx is an
        # opt-in research lever owned by the ``fp8_base`` installer's separate machinery, NOT this
        # default entry — its e4m3 dx error would otherwise perturb every LoRA-adapter grad). Save the
        # bf16 ``weight`` for that dx, and the activation ONLY when the weight is trainable (dw).
        if ctx.wreq:
            ctx.save_for_backward(weight, x)
        else:
            ctx.save_for_backward(weight)
        return y

    @staticmethod
    def backward(ctx, gy):
        weight = ctx.saved_tensors[0]
        gy = gy.contiguous()
        gx = gw = None
        with torch.cuda.device(weight.device):
            gyb = gy.to(torch.bfloat16)
            if ctx.xreq:
                # grad_x[m,k] = sum_n gy[m,n] * weight[n,k] = gy @ weight, in bf16 (the exact
                # frozen-base dx — feeds the LoRA-adapter grad, so keep it off the FP8 path).
                # M==0 flows through mm as an empty [0,K] with no kernel launch.
                gx = torch.mm(gyb, weight.to(torch.bfloat16))
            if ctx.wreq:
                x = ctx.saved_tensors[1]
                # grad_w[n,k] = sum_m gy[m,n] * x[m,k] = gy^T @ x. bf16 (fp8 here blows the bwd_rel=0.08
                # tol on the M-length accumulation); cuBLAS reads gy.t() as a strided view (no copy).
                gw = torch.mm(gy.t(), x if x.dtype == gy.dtype else x.to(gy.dtype))
        return gx, gw


def build():
    """Return the tuned entry ``fp8_linear_fn(x, weight) -> y`` with production guards.

    The verified cell entry is 2-D ([M,K] activation, [N,K] frozen weight). Production feeds an
    arbitrary leading activation shape (e.g. [B,T,H]) from the wrapped Linear, so wrap the verified
    2-D ``_FP8Linear`` in a differentiable flatten/unflatten. The flatten is autograd-tracked so the
    grad flows back to the original-shape activation; the frozen weight passes through unchanged.
    ``M == 0`` (empty batch) is guarded in ``_FP8Linear.forward`` (empty [0,N] output, empty grads —
    no zero-row kernel launch); non-contiguous inputs read via strides in the rowwise quant. The
    forward GEMM math is byte-for-byte the graded kernel; the dx backward is the bf16 default."""

    def fp8_linear_fn(x, weight):
        if x.dim() == 2:
            return _FP8Linear.apply(x, weight)
        sh = x.shape
        y = _FP8Linear.apply(x.reshape(-1, sh[-1]), weight)
        return y.reshape(*sh[:-1], y.shape[-1])

    return fp8_linear_fn
