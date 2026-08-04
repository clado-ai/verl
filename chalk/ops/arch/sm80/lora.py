"""lora@sm80 — chalk autoresearch kernel (one file per layer, per arch).

Cell: lora@sm80
Entry: lora_delta_fn(x, lora_a, lora_b, scaling) -> delta   (direction: fwd+bwd)
Oracle: chalk.ops.lora._eager_lora_delta   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the shipped portable chalk kernel   target speedup: up to 2.0x

STATUS: SEED (eager reference, ~1.0x) — autoresearch subagent optimizes this in place.
Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm80 GPU. build() returns the entry callable.
"""

import torch  # noqa: F401


def build():
    def lora_delta_fn(x, lora_a, lora_b, scaling):
        d = (x.float() @ lora_a.float().t()) @ lora_b.float().t()
        return (d * float(scaling)).to(x.dtype)

    return lora_delta_fn
