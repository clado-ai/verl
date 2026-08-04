"""Per-architecture tuned-kernel tree — one folder per SM arch, one file per layer.

Each ``chalk/ops/arch/<arch>/<layer>.py`` defines ``build() -> entry``: the fused, differentiable
compute callable for that ``(layer, GPU-architecture)`` cell (e.g. ``swiglu_fn(gate, up) -> y``).
These files are BOTH the shipped per-arch kernels AND the live targets of the autoresearch grid —
the harness under ``autoresearch/`` evolves these exact files behind the reward-hacking-proof
verifier, and a verified win is committed here.

Production dispatch is :func:`load_entry`. It is a pure, self-validated *speedup overlay* on top of
each op's portable kernel:

* an arch file is only ever selected when it is flagged a verified win (``TUNED = True``) AND its
  ``build()`` passes the op's live-GPU numeric self-test vs the fp32 oracle;
* a seed, a missing file, a broken build, or a numerically-wrong kernel falls back to the op's
  portable kernel — so the arch tree can never *degrade* production, only accelerate it.

Import-time pure (no torch/triton): ``import chalk`` never pulls torch through this tree; the arch
modules import torch/triton lazily inside ``build()``, and are themselves imported only lazily by
:func:`load_entry` from inside an op's ``load_*`` (which already runs on a live GPU).
"""

from __future__ import annotations

import importlib

from collections.abc import Callable

# torch.cuda.get_device_capability() tuple -> arch string. Mirrors
# ``autoresearch.contract.SM_TO_CAPABILITY`` (the mapping chalk's installers arch-gate on); kept
# here too so production dispatch has zero dependency on the (dev-only) autoresearch package.
_CAP_TO_ARCH: dict[tuple[int, int], str] = {
    (8, 0): "sm80",
    (8, 6): "sm86",
    (8, 9): "sm89",
    (9, 0): "sm90",
    (10, 0): "sm100",
    (12, 0): "sm120",
}


def current_arch() -> str | None:
    """Arch string for the running CUDA device, or ``None`` (no CUDA / unrecognized capability)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return _CAP_TO_ARCH.get(tuple(torch.cuda.get_device_capability()))
    except Exception:  # any probe failure => "no arch override"
        return None


def _speedup_claim(mod: object) -> str:
    """Render an overlay's ``SPEEDUP`` for the dispatch line, at the strength its evidence supports.

    ``SPEEDUP`` is documentation: it never gates selection (``TUNED`` + ``build()`` + the op's
    self-test do that), so this print is the only thing repo-wide that reads it, and therefore the
    only place a stale figure can reach a user.

    A number is only worth the anchor it divided by. Some shipped overlays were graded on 2026-07-04
    under the retired ``min``-of-baselines rule against a bar that no longer exists (see #86 /
    a055885), and their own docstrings say so: the figure describes that kernel, but cannot be
    restated against today's anchor. Printing those as "verified" asserts more than the file claims.

    So the strong word is OPT-IN: an overlay earns "verified" by setting ``SPEEDUP_ANCHOR`` to the
    baseline the figure divides by, and anything else prints "unrestated". ``None`` and an omitted
    field render identically on purpose: both mean no anchor is on record, and the reasons differ
    per overlay (a retired bar, a baseline unrecoverable from the file's own notes, a field simply
    not filled in), so the line states the absence and not a cause it cannot know. Absence is the
    weak reading, never the strong one -- a new overlay that forgets the field under-claims
    (harmless) instead of inheriting a guarantee nobody checked.
    """
    speedup = getattr(mod, "SPEEDUP", None)
    if not speedup:
        return ""
    anchor = getattr(mod, "SPEEDUP_ANCHOR", None)
    if anchor is None:
        # Says only what the field records: no anchor. WHY there is none differs per overlay -- sm89
        # gdn_gated_rmsnorm was graded against a bar that has since been retired, sm80/swiglu's
        # baseline is unrecoverable from its own notes, and a new overlay may simply have omitted the
        # field -- so naming any one cause here would assert provenance for the other two.
        return f", {speedup}x unrestated (no anchor recorded, so the figure cannot be restated)"
    return f", verified {speedup}x vs {anchor}"


def load_entry(
    op: str,
    self_test: Callable[[Callable], None],
    *,
    portable: Callable[[], Callable],
) -> Callable:
    """Return the arch-tuned entry for ``op`` on the running GPU, else the portable kernel.

    The arch file ``chalk.ops.arch.<arch>.<op>`` is selected ONLY if (a) it exists, (b) it is
    flagged a verified win (module attribute ``TUNED`` truthy — seeds are not), and (c) its
    ``build()`` result passes ``self_test`` (the op's own live-GPU parity check vs its fp32
    oracle, which raises on any numeric/autograd mismatch). Any failure of (a)/(b)/(c) — or no
    recognized arch — falls back to ``portable()``. Never raises for an arch-kernel reason.

    The returned entry is ALWAYS self-tested exactly once before it is returned: on the arch
    branch above, and — when falling back — on the portable kernel too. A portable self-test
    failure PROPAGATES (it is not an arch-kernel reason) so the op's ``load_*`` catches it and
    keeps stock PEFT; the returned entry is thus guaranteed validated for the caller to patch in.

    ``self_test`` is the op's existing ``_self_test`` (takes the entry, raises on mismatch);
    ``portable`` is a zero-arg factory returning the op's shipped portable kernel.
    """
    arch = current_arch()
    if arch is not None:
        try:
            mod = importlib.import_module(f"{__name__}.{arch}.{op}")
            if not getattr(mod, "TUNED", False):
                raise LookupError("seed (no verified arch kernel)")
            build = getattr(mod, "build", None)
            if build is None:
                raise AttributeError(f"arch/{arch}/{op}.py defines no build()")
            entry = build()
            if entry is None:
                raise LookupError("build() returned None")
            self_test(entry)  # raises on numeric/autograd mismatch vs the op's oracle
            print(f"[{op}] arch-tuned kernel selected ({arch}{_speedup_claim(mod)})", flush=True)
            return entry
        except Exception as e:  # fall back to portable on ANY arch-kernel issue
            print(f"[{op}] no arch-tuned kernel for {arch} ({type(e).__name__}: {e}); using portable", flush=True)
    # Portable fallback: self-test it too, so a portable build/numeric failure surfaces HERE
    # (and the caller keeps stock PEFT) instead of at the first training forward after PEFT is
    # patched. The arch branch already self-tested the entry it returned above, so this is the
    # ONLY self-test on the portable path — the returned entry is validated exactly once on
    # either branch, with no double-test on the arch branch. A failure propagates (never caught
    # here) so the op's ``load_*`` treats it as "keep stock PEFT" — never torch eager.
    entry = portable()
    self_test(entry)  # raises on numeric/autograd mismatch vs the op's fp32 oracle
    return entry


def load_kernel(
    tag: str,
    enabled_desc: str,
    disabled_desc: str,
    *,
    build: Callable[[], Callable],
    self_test: Callable[[Callable], None] | None = None,
) -> Callable | None:
    """Build an op's fused kernel and gate it on its live-GPU self-test — the shared loader body.

    Returns the entry callable when ``build()`` succeeds (and ``self_test`` passes, when given);
    otherwise ``None`` (the caller keeps the eager/portable path). Never raises — no CUDA, a
    build/compile error, or a self-test mismatch all fall to ``None``. On success prints
    ``[{tag}] {enabled_desc} (self-test passed)``; on any failure prints
    ``[{tag}] {disabled_desc} (build/self-test failed): {e}``.

    ``self_test`` is OPTIONAL: pass it when ``build`` returns an un-validated kernel (e.g. a direct
    portable build). OMIT it when ``build`` already self-tests what it returns — e.g. it wraps
    :func:`load_entry`, which self-tests both the arch and the portable entry it returns, so a
    second self-test here would only double startup validation latency.

    torch is imported lazily inside, so ``import chalk`` stays torch-free through this module.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        fn = build()
        if self_test is not None:
            self_test(fn)
        print(f"[{tag}] {enabled_desc} (self-test passed)", flush=True)
        return fn
    except Exception as e:  # pragma: no cover - defensive: any failure keeps the eager/portable baseline
        print(f"[{tag}] {disabled_desc} (build/self-test failed): {e}", flush=True)
        return None
