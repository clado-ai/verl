"""gdn_gated_rmsnorm@sm100 — chalk arch-tuned kernel (one file per layer, per arch).

Cell: gdn_gated_rmsnorm@sm100
Entry: gated_rmsnorm_fn(x, weight, z, eps) -> y   (direction: fwd+bwd)
Oracle: chalk.ops.gdn._eager_gated_rmsnorm   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32

STATUS: ADOPTED (TUNED). Verified by the chalk autoresearch verifier (correctness + generalization
+ timing + roofline + anti-cheat) on a real sm100 (B200) GPU on 2026-07-04: 1.977x (fwd+bwd), all
gates green, no cheat flags. Selected by ``chalk.ops.arch.load_entry`` when the running GPU is sm100
and this ``build()`` passes ``chalk.ops.gdn._self_test_gated_rmsnorm``; otherwise dispatch falls back
to the op's portable kernel (never eager).

ANCHOR: 1.977x divides by a bar that no longer exists, so do not read it as a ratio against the
shipped chalk kernel. The 2026-07-04 scorer took ``min`` over the applicable baselines. Two of the
three arms are known: Liger never applied (``gdn_gated_rmsnorm`` is absent from ``_LIGER_BUILDERS``
in all 17 historical revisions of ``autoresearch/hive/eval/baselines.py``, so that arm was never
timed), and CURRENT was not the portable kernel -- ``autoresearch/hive/eval/eval.sh`` materialized it
as ``git show ${HIVE_BASE_REF:-origin/main}:<this file>``, the PREVIOUS revision of this same overlay.
So the min ran over eager and that prior revision. #86 (a055885, 2026-07-28) retired the rule: the
anchor is now CURRENT alone, explicitly "never min-of-baselines". The run recorded only the ratio,
not the per-baseline milliseconds, so which arm was the min cannot be recovered and this number
cannot be restated against today's anchor. Treat it as a 2026-07-04 measurement of a retired rule.

The kernel code below is byte-identical to the code that earned 1.977x (verified by AST comparison
against cffc4ca, docstrings excluded), so the number does describe THIS kernel -- just not against
today's anchor. A rerun will not re-derive it either: ``_CHALK_RESOLVERS["gdn_gated_rmsnorm"]``
resolves CURRENT through ``chalk.ops.gdn.load_gated_rmsnorm``, which is arch-aware and returns THIS
file on sm100, so the anchor becomes the candidate and the cell times the kernel against itself
(~1.0x). Re-measuring against the portable kernel means pinning the anchor to
``_build_gated_rmsnorm_kernels`` explicitly.

Production contract (vs the op's portable ``_build_gated_rmsnorm_kernels``): entry signature
``gated_rmsnorm_fn(x, weight, z, eps)`` matched to the op self-test; kernel launches bound to the
input device via ``torch.cuda.device``; contiguous/reshape handling; degenerate M==0 handled without
a launch; correct fwd + bwd (dx, dz, dw) with ``None`` returned for the ``eps`` grad and for a missing
upstream grad. Backward uses a grid-strided persistent-program layout with a ``needs_input_grad``-gated
dweight partial reduction. build() returns the entry callable.
"""

import torch
import triton
import triton.language as tl


def _warps_for(block):
    w = block // 256
    if w < 1:
        w = 1
    if w > 16:
        w = 16
    return w


@triton.jit
def _fwd_kernel(X, Z, W, Y, Rstd, stride_m, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    base = row * stride_m + cols
    x = tl.load(X + base, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    tl.store(Rstd + row, rstd)
    s = z * tl.sigmoid(z)
    y = x * rstd * w * s
    tl.store(Y + base, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel(
    X,
    Z,
    W,
    Rstd,
    DY,
    DX,
    DZ,
    DWp,
    tokens,
    stride_m,
    N,
    N_PROG: tl.constexpr,
    COMPUTE_DW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    invN = 1.0 / N
    row = pid
    while row < tokens:
        base = row * stride_m + cols
        x = tl.load(X + base, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(Z + base, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + base, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)
        sig = tl.sigmoid(z)
        s = z * sig
        sprime = sig * (1.0 + z * (1.0 - sig))
        nr = x * rstd
        o = w * nr
        c = dy * s * w
        c = tl.where(mask, c, 0.0)
        dot = tl.sum(c * x, axis=0)
        dx = rstd * c - (rstd * rstd) * nr * (dot * invN)
        dz = dy * o * sprime
        tl.store(DX + base, dx.to(DX.dtype.element_ty), mask=mask)
        tl.store(DZ + base, dz.to(DZ.dtype.element_ty), mask=mask)
        if COMPUTE_DW:
            dw_acc += dy * s * nr
        row += N_PROG
    if COMPUTE_DW:
        tl.store(DWp + pid * N + cols, dw_acc, mask=mask)


class _GatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, weight, eps):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().reshape(-1, N)
        z2 = z.contiguous().reshape(-1, N)
        w = weight.contiguous()
        M = x2.shape[0]
        y = torch.empty_like(x2)
        rstd = torch.empty((M,), dtype=torch.float32, device=x2.device)
        BLOCK = triton.next_power_of_2(N)
        nw = _warps_for(BLOCK)
        if M > 0:
            with torch.cuda.device(x2.device):
                _fwd_kernel[(M,)](x2, z2, w, y, rstd, x2.stride(0), N, eps, BLOCK=BLOCK, num_warps=nw)
        ctx.save_for_backward(x2, z2, w, rstd)
        ctx.orig_shape = orig_shape
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_y):
        x2, z2, w, rstd = ctx.saved_tensors
        if grad_y is None:
            return (None, None, None, None)
        N = x2.shape[-1]
        M = x2.shape[0]
        dy2 = grad_y.contiguous().reshape(-1, N)
        dx = torch.empty_like(x2)
        dz = torch.empty_like(z2)
        BLOCK = triton.next_power_of_2(N)
        nw = _warps_for(BLOCK)
        need_w = ctx.needs_input_grad[2]

        props = torch.cuda.get_device_properties(x2.device)
        n_prog = min(M, props.multi_processor_count * 8)
        if n_prog < 1:
            n_prog = 1

        if need_w:
            dw_part = torch.empty((n_prog, N), dtype=torch.float32, device=x2.device)
        else:
            dw_part = torch.empty((1, 1), dtype=torch.float32, device=x2.device)

        if M > 0:
            with torch.cuda.device(x2.device):
                _bwd_kernel[(n_prog,)](
                    x2,
                    z2,
                    w,
                    rstd,
                    dy2,
                    dx,
                    dz,
                    dw_part,
                    M,
                    x2.stride(0),
                    N,
                    N_PROG=n_prog,
                    COMPUTE_DW=need_w,
                    BLOCK=BLOCK,
                    num_warps=nw,
                )

        dw = dw_part.sum(0).to(w.dtype) if (need_w and M > 0) else (torch.zeros_like(w) if need_w else None)
        dx = dx.reshape(ctx.orig_shape)
        dz = dz.reshape(ctx.orig_shape)
        return dx, dz, dw, None


TUNED = True
SPEEDUP = 1.977
# the 2026-07-04 min-of-baselines anchor this divided by no longer exists (see ANCHOR above), so the
# figure cannot be restated against today's bar. None keeps dispatch from printing it as "verified".
SPEEDUP_ANCHOR = None


def build():
    def gated_rmsnorm_fn(x, weight, z, eps):
        return _GatedRMSNorm.apply(x, z, weight, eps)

    return gated_rmsnorm_fn
