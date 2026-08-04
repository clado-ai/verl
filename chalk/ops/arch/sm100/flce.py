"""flce@sm100 — chalk autoresearch kernel (one file per layer, per arch).

Cell: flce@sm100
Entry: flce_fn(hidden, lm_head_weight, labels, ignore_index=-100, reduction='mean',
       label_smoothing=0.0) -> loss   (direction: fwd+bwd)
Oracle: chalk.ops.flce._eager_flce   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the shipped portable chalk kernel   target speedup: up to 6.0x

STATUS: NO WIN AGAINST THE 2026-07-03 EAGER ANCHOR. Not a claim that eager is optimal, and not a
verdict against the bar named above: #86 moved the scored anchor from eager to the shipped portable
chalk kernel, and this cell has not been re-run since, so nothing here has been measured against the
portable kernel it now has to beat.

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

Last run: numerically correct (fwd_rel ~1e-6, grad_hidden ~2.5e-3, grad_weight ~2.0e-3 vs the fp32
oracle), stopped by the verifier's input-sensitivity cheat-probe, which perturbs lm_head_weight[0,0]
-- a single vocab item -- and requires the loss to move. See #18: that probe false-flags honest
kernels, so this is a probe verdict, not evidence the kernel is wrong.

Measured by the chalk autoresearch verifier on a real sm100 GPU. build() returns the entry callable.
"""

import torch  # noqa: F401


def build():
    import torch.nn.functional as F

    def flce_fn(hidden, lm_head_weight, labels, chunk=2048):
        hf = hidden.float()
        wf = lm_head_weight.float()
        n = hf.shape[0]
        total = hf.new_zeros(())
        for s in range(0, n, chunk):
            logits = hf[s : s + chunk] @ wf.t()
            total = total + F.cross_entropy(logits, labels[s : s + chunk], reduction="sum")
        return total / max(n, 1)

    return flce_fn
