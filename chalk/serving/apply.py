"""Forward-only serving composition of chalk's kernels.

``apply_chalk_serving_kernels(model)`` installs the inference-relevant chalk kernels and nothing else.
It reuses the exact same tuned ops + class-patching installers as training
(``chalk.transformers.apply_chalk_kernel_to_*``) — the forward math is identical — but fixes a
serving policy: the training-only kernels are OFF (fused-linear-cross-entropy = the loss, the
trainable LoRA-delta, the trainable attention epilogue), and the *eval* attention epilogue is ON
(at inference q/k/v come from the frozen base weights, not a LoRA delta).

This is the one place a serving-specific divergence would live if the ``serving_bench`` A/B ever
justifies it (e.g. decode-shape ``M=1`` autotuned configs, or forward-only kernel variants that skip
save-for-backward). Until then it is a thin, explicit policy over the shared kernels.
"""

from __future__ import annotations

from collections.abc import Mapping

# Qwen3.5/3.6 forward kernels for serving. Keys mirror ``apply_chalk_kernel_to_qwen35``'s signature.
SERVING_QWEN_KERNELS: dict[str, bool] = {
    "rope": True,
    "rmsnorm": True,
    "swiglu": True,
    "fused_embedding": True,
    "gdn": True,
    "attn_epilogue": True,  # eval variant: base q/k/v (no LoRA) — the right one for serving
    "trainable_attn_epilogue": False,
    "fused_linear_cross_entropy": False,  # training loss — generation uses the plain LM head
    "fused_lora_delta": False,  # base-model forward has no trainable adapter delta
}

# The Llama family (incl. MiniCPM, which loads as LlamaForCausalLM) only has rmsnorm + swiglu kernels.
SERVING_LLAMA_KERNELS: dict[str, bool] = {"rmsnorm": True, "swiglu": True}


def _is_llama_family(model_or_name) -> bool:
    """True for MiniCPM / Llama-family models — both served through chalk's llama-mode rmsnorm+swiglu.
    Matches the base-model id (``openbmb/MiniCPM5-1B``, ``meta-llama/*``) or the HF class name
    (``LlamaForCausalLM``, ``MiniCPM*``)."""
    name = (model_or_name if isinstance(model_or_name, str) else type(model_or_name).__name__).lower()
    return "minicpm" in name or "llama" in name


def serving_kernel_flags(base_model: str, *, fp8_base: bool = False) -> dict[str, bool]:
    """The chalk kwargs this model runs at inference. ``fp8_base`` enables the FP8-frozen-base kernel
    (for the FP8-checkpoint path); off for the bf16 base A/B."""
    if _is_llama_family(base_model):
        return dict(SERVING_LLAMA_KERNELS)
    flags = dict(SERVING_QWEN_KERNELS)
    flags["fp8_frozen_base"] = fp8_base
    return flags


def apply_chalk_serving_kernels(model, *, base_model: str | None = None, fp8_base: bool = False, **overrides) -> dict:
    """Install chalk's forward kernels on a live HF model for serving. Returns chalk's install report.

    ``base_model`` selects the family (Llama/MiniCPM vs Qwen3.5/3.6); if omitted it is inferred from
    the model class name. ``overrides`` let a caller flip an individual kernel (e.g. ``gdn=False``).
    Does not swallow a hard ``apply_*`` error — a signature/import failure must surface (never silently
    run eager while reporting chalk installed).
    """
    key = base_model if base_model is not None else model
    if _is_llama_family(key):
        from chalk.transformers import apply_chalk_kernel_to_llama

        flags = {**SERVING_LLAMA_KERNELS, **{k: v for k, v in overrides.items() if k in SERVING_LLAMA_KERNELS}}
        return apply_chalk_kernel_to_llama(model, **flags)

    from chalk.transformers import apply_chalk_kernel_to_qwen35

    flags = serving_kernel_flags(base_model or type(model).__name__, fp8_base=fp8_base)
    flags.update(overrides)
    return apply_chalk_kernel_to_qwen35(model, **flags)


def engaged_serving_kernels(report: Mapping[str, object] | None) -> list[str]:
    """Kernels that truly installed (truthy, non-error), excluding the deprecated ``liger`` key."""
    return sorted(
        k
        for k, v in (report or {}).items()
        if k != "liger" and v not in (False, None) and not (isinstance(v, dict) and "error" in v)
    )
