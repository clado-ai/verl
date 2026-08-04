"""embedding@sm86 — chalk autoresearch kernel (one file per layer, per arch).

Cell: embedding@sm86
Entry: embedding_fn(input_ids, weight) -> y   (direction: fwd)
Oracle: chalk.ops.embedding._eager_embedding   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: eager   target speedup: up to 1.15x

STATUS: NO WIN: autoresearch found no kernel beating eager fwd+bwd on sm86 (eager optimal here). Last: Sandbox/correctness/generalization/roofline all green, cheat_flags empty; timing never crossed 1.0x (memory-bound gather; best median 0.977x iter5, torch indexing at gather roofline ~0.52). Two key fi
Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm86 GPU. build() returns the entry callable.
"""

import torch  # noqa: F401


def build():
    def embedding_fn(input_ids, weight):
        return weight[input_ids]

    return embedding_fn
