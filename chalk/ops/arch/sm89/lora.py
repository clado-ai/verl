"""lora@sm89 — chalk autoresearch kernel (one file per layer, per arch).

Cell: lora@sm89
Entry: lora_delta_fn(x, lora_a, lora_b, scaling) -> delta   (direction: fwd+bwd)
Oracle: chalk.ops.lora._eager_lora_delta   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the shipped portable chalk kernel   target speedup: up to 2.0x

STATUS: RESEARCH RESULT (not yet adopted by production dispatch on this branch — no TUNED, so load_entry falls back to the portable kernel; adoption = wiring the op's load_*() to load_entry + TUNED=True + aligning build()'s entry signature with the op's production _self_test, all of which ships in the kernel PRs). Verified on a real sm89 GPU: 1.156x vs the PORTABLE chalk kernel (fwd+bwd), roofline_fraction=0.170, all gates green, no cheat flags.
Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm89 GPU. build() returns the entry callable.
"""

import torch


class _LoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, scaling):
        s = float(scaling)
        h = torch.mm(x, a.t())  # (M, R)
        delta = torch.mm(h, b.t())  # (M, N)
        delta.mul_(s)
        ctx.save_for_backward(x, a, b, h)
        ctx.scaling = s
        return delta

    @staticmethod
    def backward(ctx, g):
        x, a, b, h = ctx.saved_tensors
        s = ctx.scaling
        g = g.contiguous()
        # dh = s * (g @ B)   -> fold scale here (small M x R) instead of full M x N
        dh = torch.mm(g, b)
        dh.mul_(s)
        dx = torch.mm(dh, a)  # (M, K)
        dA = torch.mm(dh.t(), x)  # (R, K)
        dB = torch.mm(g.t(), h)  # (N, R)
        dB.mul_(s)
        return dx, dA, dB, None


def build():
    def lora_delta_fn(x, lora_a, lora_b, scaling):
        return _LoRA.apply(x, lora_a, lora_b, scaling)

    return lora_delta_fn
