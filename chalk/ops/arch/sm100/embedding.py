"""embedding@sm100 — chalk autoresearch kernel (one file per layer, per arch).

Cell: embedding@sm100
Entry: embedding_fn(input_ids, weight) -> y   (direction: fwd)
Oracle: chalk.ops.embedding._eager_embedding   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: eager   target speedup: up to 1.15x

STATUS: NO WIN: autoresearch found no kernel beating eager fwd+bwd on sm100 (eager optimal here). Last: All gates green except timing: torch's weight[input_ids] gather is bandwidth-optimal baseline; triton ~2-3x slower (0.33x), torch primitives only tie and the sole >1.0x route (index_select out= buffer
Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm100 GPU. build() returns the entry callable.
"""

import torch  # noqa: F401


def build():
    def embedding_fn(input_ids, weight):
        return weight[input_ids]

    return embedding_fn
