"""rmsnorm_llama@sm89 — chalk autoresearch kernel (one file per layer, per arch).

Cell: rmsnorm_llama@sm89
Entry: rmsnorm_fn(x, weight, eps) -> y   (direction: fwd+bwd)
Oracle: chalk.ops.rmsnorm._eager_rmsnorm (gemma=False)   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the shipped portable chalk kernel   target speedup: up to 1.2x

STATUS: SEED — the portable plain-weight (Llama / MiniCPM) RMSNorm kernel
(``chalk.ops.rmsnorm._build_kernels(gemma=False)``): normalize in fp32, cast BACK to the input
dtype FIRST, then multiply by the PLAIN weight (no ``1+`` offset). The per-arch tuned ``rmsnorm``
tree is GEMMA-only (every arch/sm89/rmsnorm.py hardcodes ``(1+w)`` and is graded against the
Gemma oracle), so the Llama-convention cell seeds from the portable plain-weight builder here
rather than the (1+w) tuned kernels. Evolve a tuned plain-weight kernel and set ``TUNED=True`` +
``SPEEDUP=<x>`` only after the chalk autoresearch verifier confirms a real win on a real sm89 GPU.
build() returns the entry callable ``rmsnorm_fn(x, weight, eps) -> y`` (fwd+bwd, differentiable
wrt x and weight).
"""

from chalk.ops.rmsnorm import _build_kernels


def build():
    # gemma=False -> Llama / MiniCPM plain-weight, cast-before convention (matches the
    # rmsnorm_llama oracle chalk.ops.rmsnorm._eager_rmsnorm(gemma=False)).
    return _build_kernels(gemma=False)
