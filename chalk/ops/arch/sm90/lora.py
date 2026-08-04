"""lora@sm90 — chalk autoresearch kernel (one file per layer, per arch).

Cell: lora@sm90
Entry: lora_delta_fn(x, lora_a, lora_b, scaling) -> delta   (direction: fwd+bwd)
Oracle: chalk.ops.lora._eager_lora_delta   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the arch-tuned kernel this file already ships   target speedup: up to 2.0x

STATUS: ADOPTED — verified on a real sm90 GPU: 1.1314x vs the PORTABLE chalk kernel (fwd+bwd),
all gates green, no cheat flags. The baseline is portable, not eager: when this file was measured,
``lora`` resolved through ``_chalk_lora`` in autoresearch/hive/eval/baselines.py, which returned the
module-level ``fused_lora`` portable entry, and it is still portable that ``load_entry`` falls back
to when this file is absent — so vs-portable is the delta a user actually gets. (Since #99 the same
op resolves via ``_ChalkAnchor(... "load_lora")``, which is arch-aware and would anchor a NEW sm90
candidate on this overlay; the 1.1314 figure predates that and is not a vs-eager number.)
Selected by production dispatch (``TUNED = True``); ``chalk.ops.lora.load_lora``
routes through ``load_entry("lora", ...)`` guarded by the live-GPU delta self-test, so this file
only runs when it matches the fp32 oracle. ``build()`` returns ``lora_delta_fn(x, A, B, scaling)``.

APPROACH: lean all-cuBLAS composition (no fused Triton kernel, no custom autograd.Function). Two
skinny GEMMs via F.linear so autograd builds the minimal C++ node graph (fewer dispatches than
eager's mm/transpose/mul chain), and ``scaling`` is folded ONCE into the small token-independent
[r,K] A tensor so neither the forward nor the backward ever scales the big [tok,N] output. Native
bf16 tensor-core matmuls (matches the eager baseline dtype); the fp32 accumulation keeps both fwd
and bwd well inside the 0.02 rel tolerance. Being plain differentiable torch ops, autograd handles
non-contiguous inputs, degenerate (tok==0) shapes, correct dx/dA/dB, and None upstream grads
natively — the caller feeds a 2D [tok,K] activation and adds the base output OUTSIDE this kernel.
"""

import torch  # noqa: F401
import torch.nn.functional as F

TUNED = True
SPEEDUP = 1.1314
SPEEDUP_ANCHOR = "the portable chalk kernel"


def build():
    def lora_delta_fn(x, lora_a, lora_b, scaling):
        # Fold scaling into the small, token-independent [r, K] A factor: one cheap elementwise
        # pass instead of scaling the [tok, N] output. Both projections are plain F.linear so
        # cuBLAS owns the two skinny GEMMs and autograd's backward is the minimal node chain.
        a_s = lora_a * scaling
        h = F.linear(x, a_s)  # [tok, r]  = x @ (scaling*A)^T
        return F.linear(h, lora_b)  # [tok, N]  = h @ B^T

    return lora_delta_fn
