"""Shared helpers for the chalk transformers installers (aggregator-side glue)."""

from __future__ import annotations

import traceback


def run_installer(report: dict, name: str, installer, *, tag: str) -> None:
    """Run one chalk installer, recording its result under ``name`` in ``report``.

    A failure of ONE installer must never abort the rest — so any exception is caught, logged
    (prefixed with ``[{tag}]``), and stored as an ``{"error": ...}`` marker in the report instead
    of propagating. Each installer is already self-test + arch gated and no-ops safely off-GPU;
    this guard covers the residual case where one raises anyway.
    """
    try:
        report[name] = installer()
    except Exception as exc:  # deliberately broad: one kernel must not abort the others
        print(
            f"[{tag}] installer {name!r} raised; skipping it (other kernels still run):\n" + traceback.format_exc(),
            flush=True,
        )
        report[name] = {"error": repr(exc)}
