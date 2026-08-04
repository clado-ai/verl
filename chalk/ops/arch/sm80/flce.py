"""flce@sm80 — chalk autoresearch kernel (one file per layer, per arch).

Cell: flce@sm80
Entry: flce_fn(hidden, lm_head_weight, labels, ignore_index=-100, reduction='mean',
       label_smoothing=0.0) -> loss   (direction: fwd+bwd)
Oracle: chalk.ops.flce._eager_flce   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the shipped portable chalk kernel   target speedup: up to 6.0x

STATUS: SEED (eager reference, ~1.0x) — autoresearch subagent optimizes this in place. A seed is the
starting point handed to the optimizer, not a verifier result: nothing here has been measured
against either anchor.

REACHABLE since #175: ``chalk.ops.flce.load_flce`` now builds through ``load_entry``, the only
loader that consults ``chalk/ops/arch/``. Before that it called ``load_kernel(build=_build_kernels)``
directly, so this file could not dispatch on any arch no matter how it scored. Adoption still needs
TWO more things, and the second is the one that bites: a ``TUNED = True`` line, AND an entry whose
signature matches the one declared above. ``build()`` below exposes only ``(hidden, lm_head_weight,
labels)``; production calls the entry with ``ignore_index``/``reduction``/``label_smoothing`` as
KEYWORDS (``chalk/ops/flce.py`` ``_causal_lm_loss``), and ``_self_test`` passes ``label_smoothing``.
Flipping TUNED without widening the signature raises TypeError at argument binding, ``load_entry``
swallows it, and the overlay ships inert while claiming a verified win -- the #112/PR #99 defect
shape. ``test/test_correctness_entry_kwargs.py`` grades exactly this.

build() returns the entry callable.
"""

import torch  # noqa: F401


def build():
    import torch.nn.functional as F

    def flce_fn(hidden, lm_head_weight, labels):
        hf = hidden.float()
        wf = lm_head_weight.float()
        n = hf.shape[0]
        vocab = wf.shape[0]
        # Chunk the rows so the materialized fp32 logits tile stays bounded (~4M elements),
        # derived from the RUNTIME vocab — never a benchmark-shape literal. (chalk's static
        # anti-cheat gate flags an in-scope dim hardcoded as a constant, e.g. a literal 2048.)
        rows = min(n, max(1, (1 << 22) // max(vocab, 1)))
        total = hf.new_zeros(())
        for s in range(0, n, rows):
            logits = hf[s : s + rows] @ wf.t()
            total = total + F.cross_entropy(logits, labels[s : s + rows], reduction="sum")
        return total / max(n, 1)

    return flce_fn
