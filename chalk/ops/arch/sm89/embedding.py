"""embedding@sm89 — chalk autoresearch kernel (one file per layer, per arch).

Cell: embedding@sm89
Entry: embedding_fn(input_ids, weight) -> y   (direction: fwd)
Oracle: chalk.ops.embedding._eager_embedding   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: eager   target speedup: up to 1.15x

STATUS: NO WIN: autoresearch found no kernel beating eager fwd+bwd on sm89 (eager optimal here). Last: Correctness/generalization/roofline pass (no cheat flags) but timing unwinnable: eager w[ids] baseline is already optimal (~11us), and the verifier's weight[0,0] sensitivity probe forces a mandatory c
Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm89 GPU. build() returns the entry callable.
"""

import torch  # noqa: F401


def build():
    def embedding_fn(input_ids, weight):
        return weight[input_ids]

    return embedding_fn
