"""rmsnorm@sm100 — chalk autoresearch kernel (one file per layer, per arch).

Cell: rmsnorm@sm100
Entry: rmsnorm_fn(x, weight, eps) -> y   (direction: fwd+bwd)
Oracle: chalk.ops.rmsnorm._eager_rmsnorm   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the shipped portable chalk kernel   target speedup: up to 1.2x

STATUS: RESEARCH RESULT (not yet adopted by production dispatch on this branch — no TUNED, so load_entry falls back to the portable kernel; adoption = wiring the op's load_*() to load_entry + TUNED=True + aligning build()'s entry signature with the op's production _self_test, all of which ships in the kernel PRs). Verified on a real sm100 GPU: 1.471x vs eager (fwd+bwd), all gates green, no cheat flags. That
figure was measured 2026-07-03 against the eager anchor of the time; #86 moved the scored anchor to
the shipped portable chalk kernel, so 1.471x is NOT a margin over the bar named above and cannot be
adopted as one without a re-run.
Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm100 GPU. build() returns the entry callable.
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
def _rms_fwd(X, W, Y, Rstd, stride_m, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    base = row * stride_m + cols
    x = tl.load(X + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    tl.store(Rstd + row, rstd)
    y = x * rstd * w
    tl.store(Y + base, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rms_bwd(
    X, W, Rstd, DY, DX, DWp, tokens, stride_m, N, N_PROG: tl.constexpr, COMPUTE_DW: tl.constexpr, BLOCK: tl.constexpr
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
        dy = tl.load(DY + base, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)
        xhat = x * rstd
        wdy = w * dy
        wdy = tl.where(mask, wdy, 0.0)
        s = tl.sum(xhat * wdy, axis=0) * invN
        dx = (wdy - xhat * s) * rstd
        tl.store(DX + base, dx.to(DX.dtype.element_ty), mask=mask)
        if COMPUTE_DW:
            dw_acc += dy * xhat
        row += N_PROG
    if COMPUTE_DW:
        tl.store(DWp + pid * N + cols, dw_acc, mask=mask)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().reshape(-1, N)
        w = weight.contiguous()
        M = x2.shape[0]
        y = torch.empty_like(x2)
        rstd = torch.empty((M,), dtype=torch.float32, device=x2.device)
        BLOCK = triton.next_power_of_2(N)
        nw = _warps_for(BLOCK)
        if M > 0:
            _rms_fwd[(M,)](x2, w, y, rstd, x2.stride(0), N, eps, BLOCK=BLOCK, num_warps=nw)
        ctx.save_for_backward(x2, w, rstd)
        ctx.orig_shape = orig_shape
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_y):
        x2, w, rstd = ctx.saved_tensors
        N = x2.shape[-1]
        M = x2.shape[0]
        dy2 = grad_y.contiguous().reshape(-1, N)
        dx = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        nw = _warps_for(BLOCK)
        need_w = ctx.needs_input_grad[1]

        props = torch.cuda.get_device_properties(x2.device)
        n_prog = min(M, props.multi_processor_count * 8)
        if n_prog < 1:
            n_prog = 1

        if need_w:
            dw_part = torch.empty((n_prog, N), dtype=torch.float32, device=x2.device)
        else:
            dw_part = torch.empty((1, 1), dtype=torch.float32, device=x2.device)

        if M > 0:
            _rms_bwd[(n_prog,)](
                x2,
                w,
                rstd,
                dy2,
                dx,
                dw_part,
                M,
                x2.stride(0),
                N,
                N_PROG=n_prog,
                COMPUTE_DW=need_w,
                BLOCK=BLOCK,
                num_warps=nw,
            )

        # at M == 0 the kernel launch above is skipped, so dw_part still holds
        # uninitialized torch.empty memory. weight has real elements at every M,
        # and autograd accumulates into dw, so it must be zeroed rather than
        # summed from that garbage.
        if need_w:
            dw = dw_part.sum(0).to(w.dtype) if M > 0 else torch.zeros_like(w)
        else:
            dw = None
        dx = dx.reshape(ctx.orig_shape)
        return dx, dw, None


def build():
    def rmsnorm_fn(x, weight, eps):
        return _RMSNorm.apply(x, weight, eps)

    return rmsnorm_fn
