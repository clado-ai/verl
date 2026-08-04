"""gdn_gated_rmsnorm@sm89 — chalk arch-tuned kernel (one file per layer, per arch).

Cell: gdn_gated_rmsnorm@sm89
Entry: gated_rmsnorm_fn(x, weight, z, eps) -> y   (direction: fwd+bwd)
Oracle: chalk.ops.gdn._eager_gated_rmsnorm   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32

STATUS: ADOPTED (TUNED). Verified by the chalk autoresearch verifier (correctness + generalization
+ timing + roofline + anti-cheat) on a real sm89 GPU on 2026-07-04: 2.4165x (fwd+bwd), all gates
green, no cheat flags. Selected by ``chalk.ops.arch.load_entry`` when the running GPU is sm89 and
this ``build()`` passes ``chalk.ops.gdn._self_test_gated_rmsnorm``; otherwise dispatch falls back to
the op's portable kernel (never eager).

ANCHOR: 2.4165x divides by a bar that no longer exists, so do not read it as a ratio against the
shipped chalk kernel. The 2026-07-04 scorer took ``min`` over the applicable baselines. Two of the
three arms are known: Liger never applied (``gdn_gated_rmsnorm`` is absent from ``_LIGER_BUILDERS``
in all 17 historical revisions of ``autoresearch/hive/eval/baselines.py``, so that arm was never
timed), and CURRENT was not the portable kernel -- ``autoresearch/hive/eval/eval.sh`` materialized it
as ``git show ${HIVE_BASE_REF:-origin/main}:<this file>``, the PREVIOUS revision of this same overlay.
So the min ran over eager and that prior revision. #86 (a055885, 2026-07-28) retired the rule: the
anchor is now CURRENT alone, explicitly "never min-of-baselines". The run recorded only the ratio,
not the per-baseline milliseconds, so which arm was the min cannot be recovered and this number
cannot be restated against today's anchor. Treat it as a 2026-07-04 measurement of a retired rule.

The kernel code below is byte-identical to the code that earned 2.4165x (verified by AST comparison
against cffc4ca, docstrings excluded), so the number does describe THIS kernel -- just not against
today's anchor. A rerun will not re-derive it either: ``_CHALK_RESOLVERS["gdn_gated_rmsnorm"]``
resolves CURRENT through ``chalk.ops.gdn.load_gated_rmsnorm``, which is arch-aware and returns THIS
file on sm89, so the anchor becomes the candidate and the cell times the kernel against itself
(~1.0x). Re-measuring against the portable kernel means pinning the anchor to
``_build_gated_rmsnorm_kernels`` explicitly.

Production contract (vs the op's portable ``_build_gated_rmsnorm_kernels``): entry signature
``gated_rmsnorm_fn(x, weight, z, eps)`` matched to the op self-test; kernel launches bound to the
input device via ``torch.cuda.device``; contiguous/reshape handling; degenerate M==0 handled without
a launch; correct fwd + bwd (dx, dz, dw) with ``None`` returned for the ``eps`` grad and for a missing
upstream grad. This arch uses a 2-D row-tiled backward (BLOCK_M rows/program). build() returns entry.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(X, Z, W, Y, Rstd, stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * stride + cols
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    tl.store(Rstd + row, r)
    xn = x * r
    sig = tl.sigmoid(z)
    silu = z * sig
    y = xn * w * silu
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel(X, Z, W, DY, Rstd, DX, DZ, DW, stride, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < M
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    off = rows[:, None] * stride + cols[None, :]
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + off, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Rstd + rows, mask=rmask, other=0.0)[:, None]
    w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
    sig = tl.sigmoid(z)
    silu = z * sig
    xn = x * r
    g = xn * w
    dsilu = sig * (1.0 + z * (1.0 - sig))
    dz = dy * g * dsilu
    tl.store(DZ + off, dz.to(DZ.dtype.element_ty), mask=mask)
    sw = dy * silu * w
    mean_cxn = tl.sum(sw * xn, axis=1)[:, None] / N
    dx = r * (sw - xn * mean_cxn)
    tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=mask)
    dw_partial = tl.sum(dy * silu * xn, axis=0)
    tl.atomic_add(DW + cols, dw_partial, mask=cmask)


class _GatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, weight, eps):
        x = x.contiguous()
        z = z.contiguous()
        weight = weight.contiguous()
        shp = x.shape
        N = shp[-1]
        x2 = x.reshape(-1, N)
        z2 = z.reshape(-1, N)
        M = x2.shape[0]
        y = torch.empty_like(x2)
        rstd = torch.empty((M,), device=x.device, dtype=torch.float32)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = min(max(BLOCK_N // 256, 1), 16)
        if M > 0:
            with torch.cuda.device(x2.device):
                _fwd_kernel[(M,)](x2, z2, weight, y, rstd, x2.stride(0), N, eps, BLOCK_N=BLOCK_N, num_warps=num_warps)
        ctx.save_for_backward(x2, z2, weight, rstd)
        ctx.eps = eps
        ctx.shp = shp
        return y.reshape(shp)

    @staticmethod
    def backward(ctx, dy):
        x2, z2, weight, rstd = ctx.saved_tensors
        if dy is None:
            return (None, None, None, None)
        N = x2.shape[1]
        M = x2.shape[0]
        dy = dy.contiguous().reshape(-1, N)
        dx = torch.empty_like(x2)
        dz = torch.empty_like(x2)
        dw = torch.zeros((N,), device=x2.device, dtype=torch.float32)
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_M = 32
        num_warps = 4
        if M > 0:
            grid = (triton.cdiv(M, BLOCK_M),)
            with torch.cuda.device(x2.device):
                _bwd_kernel[grid](
                    x2,
                    z2,
                    weight,
                    dy,
                    rstd,
                    dx,
                    dz,
                    dw,
                    x2.stride(0),
                    M,
                    N,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    num_warps=num_warps,
                )
        return (dx.reshape(ctx.shp), dz.reshape(ctx.shp), dw.to(weight.dtype), None)


TUNED = True
SPEEDUP = 2.4165
# the 2026-07-04 min-of-baselines anchor this divided by no longer exists (see ANCHOR above), so the
# figure cannot be restated against today's bar. None keeps dispatch from printing it as "verified".
SPEEDUP_ANCHOR = None


def build():
    def gated_rmsnorm_fn(x, weight, z, eps):
        return _GatedRMSNorm.apply(x, z, weight, eps)

    return gated_rmsnorm_fn
