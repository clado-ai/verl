"""Shared discovery of the Qwen3.5/3.6 HuggingFace modeling classes chalk patches.

The installers monkeypatch a fused kernel onto a specific layer class per model family (the dense
+ MoE variants of Qwen3.5 and Qwen3.6). Each family ships its own modeling module with the class
named ``<Prefix><Suffix>`` (e.g. ``Qwen3_5MoeMLP``). This helper collects every importable such
class for a given layer suffix, so each installer states only its suffix instead of re-listing the
four modules. Import-light: ``importlib`` only, no torch/transformers pulled at module load.
"""

from __future__ import annotations

import importlib

# (modeling module name, class-name prefix) for the four Qwen3.5/3.6 families chalk targets. The
# class chalk patches is ``<prefix><suffix>`` inside ``transformers.models.<mod>.modeling_<mod>``.
_QWEN_MODULES = (
    ("qwen3_5", "Qwen3_5"),
    ("qwen3_5_moe", "Qwen3_5Moe"),
    ("qwen3_6", "Qwen3_6"),
    ("qwen3_6_moe", "Qwen3_6Moe"),
)


def collect_qwen_classes(suffix: str):
    """Return ``[(label, cls)]`` for every importable Qwen3.5/3.6 ``<prefix>{suffix}`` class.

    ``label`` is ``f"{mod_name}.{prefix}{suffix}"``. A module that does not import (that family is
    absent from the installed transformers) or a class that is missing / has no ``forward`` is
    skipped, so the result is empty when none of the four families expose the layer. Iteration
    order matches ``_QWEN_MODULES``.
    """
    out = []
    for mod_name, prefix in _QWEN_MODULES:
        try:
            mod = importlib.import_module(f"transformers.models.{mod_name}.modeling_{mod_name}")
        except Exception:
            continue
        cls_name = f"{prefix}{suffix}"
        cls = getattr(mod, cls_name, None)
        if cls is not None and hasattr(cls, "forward"):
            out.append((f"{mod_name}.{cls_name}", cls))
    return out
