"""lora@sm86 — chalk autoresearch kernel (one file per layer, per arch).

Cell: lora@sm86
Entry: lora_delta_fn(x, lora_a, lora_b, scaling) -> delta   (direction: fwd+bwd)
Oracle: chalk.ops.lora._eager_lora_delta   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the arch-tuned kernel this file already ships   target speedup: up to 2.0x

STATUS: ADOPTED — verified on a real sm86 GPU: 1.612x vs the PORTABLE chalk kernel (fwd+bwd),
roofline_fraction=0.307, all gates green, no cheat flags. The baseline is portable, not eager:
``lora`` is registered in ``_CHALK_RESOLVERS`` (autoresearch/hive/eval/baselines.py), so the
verifier anchors it on the shipped kernel, and portable is also what ``load_entry`` falls back to
when this file is absent — so vs-portable is the delta a user actually gets. Re-measured directly
on an sm86 A5000 at tokens=4096, K=N=4096, r=32: 1.6957x vs portable against 1.0172x vs eager,
confirming which baseline the 1.612 figure came from. That anchor moved after this was measured:
since #99 ``chalk_current_callable`` resolves ``lora`` arch-aware, so on sm86 it now returns THIS
file and a new candidate is scored against it, which is why the header names a different bar than
the 1.612. The figure stays as measured — it is the delta a user got by adopting this — but it is
no longer the target. Selected by production dispatch
(``TUNED = True``); ``chalk.ops.lora.load_lora`` routes through ``load_entry("lora", ...)`` and
guards it with the live-GPU delta self-test, so this file only runs when it matches the fp32
oracle. ``build()`` returns the entry callable ``lora_delta_fn(x, A, B, scaling) -> delta``.

The win folds ``scaling`` into the small [r,K] A factor (never scaling the big [tok,N] output in
either direction) and keeps t=x@(sA)^T saved from the forward so the backward has no recompute.
Math is a custom autograd.Function (three cuBLAS GEMMs fwd, four bwd); ``scaling`` is a scalar so
its grad is None. The production caller (install_fused_lora_delta) always feeds a 2D [tok,K]
activation (it flattens leading dims) and adds the base output OUTSIDE this kernel.
"""

import torch

TUNED = True
SPEEDUP = 1.612
SPEEDUP_ANCHOR = "the portable chalk kernel"


def build():
    class LoRA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, A, B, scaling):
            s = float(scaling)
            As = A * s  # [r,K] small; folds scaling
            hs = torch.mm(x, As.transpose(0, 1))  # [M,r] = s*(x@A^T)
            delta = torch.mm(hs, B.transpose(0, 1))  # [M,N]
            ctx.save_for_backward(x, As, B, hs)
            ctx.s = s
            return delta

        @staticmethod
        def backward(ctx, g):
            # None upstream grad (delta unused) -> no grads flow.
            if g is None:
                return None, None, None, None
            x, As, B, hs = ctx.saved_tensors
            s = ctx.s
            dh = torch.mm(g, B)  # [M,r] = g@B (unscaled)
            dx = torch.mm(dh, As)  # [M,K] = s*(g@B)@A
            dA = torch.mm(dh.transpose(0, 1), x)  # [r,K] = (g@B)^T@x
            dA.mul_(s)  # small
            dB = torch.mm(g.transpose(0, 1), hs)  # [N,r] = s*(g^T@h)
            return dx, dA, dB, None  # scaling is a scalar -> None

    def lora_delta_fn(x, lora_a, lora_b, scaling):
        return LoRA.apply(x, lora_a, lora_b, scaling)

    return lora_delta_fn
