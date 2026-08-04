"""embedding@sm90 — chalk autoresearch kernel (one file per layer, per arch).

Cell: embedding@sm90
Entry: embedding_fn(input_ids, weight) -> y   (direction: fwd)
Oracle: chalk.ops.embedding._eager_embedding   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: eager   target speedup: up to 1.15x

STATUS: NO WIN: autoresearch found no kernel beating eager fwd+bwd on sm90 (eager optimal here). Last: Correctness/generalization/roofline/sandbox all green (no cheat flags), but timing gate needs >1.0x vs torch weight[input_ids] and best achievable was 0.555x; launch-overhead-bound gather — torch's fu
Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm90 GPU. build() returns the entry callable.
"""

import torch  # noqa: F401


def build():
    def embedding_fn(input_ids, weight):
        return weight[input_ids]

    return embedding_fn
