"""embedding@sm80 — chalk autoresearch kernel (one file per layer, per arch).

Cell: embedding@sm80
Entry: embedding_fn(input_ids, weight) -> y   (direction: fwd)
Oracle: chalk.ops.embedding._eager_embedding   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: eager   target speedup: up to 1.15x

STATUS: SEED (eager reference, ~1.0x) — autoresearch subagent optimizes this in place.
Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm80 GPU. build() returns the entry callable.
"""

import torch  # noqa: F401


def build():
    def embedding_fn(input_ids, weight):
        return weight[input_ids]

    return embedding_fn
