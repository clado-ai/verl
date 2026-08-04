"""lora@sm100 — chalk autoresearch kernel (one file per layer, per arch).

Cell: lora@sm100
Entry: lora_delta_fn(x, lora_a, lora_b, scaling) -> delta   (direction: fwd+bwd)
Oracle: chalk.ops.lora._eager_lora_delta   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the arch-tuned kernel this file already ships   target speedup: up to 2.0x

STATUS: ADOPTED — verified on a real sm100 GPU: 1.2559x vs the PORTABLE chalk kernel (fwd+bwd),
all gates green, no cheat flags. The baseline is portable, not eager: when this file was measured,
``lora`` resolved through ``_chalk_lora`` in autoresearch/hive/eval/baselines.py, which returned the
module-level ``fused_lora`` portable entry, and it is still portable that ``load_entry`` falls back
to when this file is absent — so vs-portable is the delta a user actually gets. (Since #99 the same
op resolves via ``_ChalkAnchor(... "load_lora")``, which is arch-aware and would anchor a NEW sm100
candidate on this overlay; the 1.2559 figure predates that and is not a vs-eager number.)
Selected by production dispatch (``TUNED = True``); ``chalk.ops.lora.load_lora``
routes through ``load_entry("lora", ...)`` guarded by the live-GPU delta self-test, so this file
only runs when it matches the fp32 oracle. ``build()`` returns ``lora_delta_fn(x, A, B, scaling)``.

APPROACH: F.linear(u, W) == u @ W^T fused into ONE autograd node (native C++ backward), so the
whole op is 3 graph nodes vs eager's mm+t+mm+t+mul (5) — fewer nodes and fewer forward dispatches,
which is what matters in this launch/engine-bound regime. Being plain differentiable torch ops,
autograd handles non-contiguous inputs, degenerate (tok==0) shapes, correct dx/dA/dB, and None
upstream grads natively — the caller feeds a 2D [tok,K] activation and adds the base OUTSIDE this
kernel.
"""

import torch.nn.functional as F

TUNED = True
SPEEDUP = 1.2559
SPEEDUP_ANCHOR = "the portable chalk kernel"


def build():
    def lora_delta_fn(x, lora_a, lora_b, scaling):
        # delta = scaling * ((x @ A^T) @ B^T)
        return F.linear(F.linear(x, lora_a), lora_b) * scaling

    return lora_delta_fn
