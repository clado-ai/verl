"""One-call entry point for chalk's Qwen3.5/3.6 kernels — chalk STANDALONE (Liger fully replaced).

chalk installs its OWN fused kernels for every layer that matters:
  * ``rms_norm`` + ``swiglu`` (the SiLU-gated MLP activation) + ``fused_linear_cross_entropy``
    (the LM head + cross-entropy, chunked so ``[N, V]`` logits never materialize) — the three
    layers Liger Kernel used to own — PLUS
  * RoPE, the trainable QK-norm + partial-RoPE attention epilogue, fused LoRA-delta, fused
    embedding, the Gated-DeltaNet short-conv + SiLU, and the FP8 frozen base — layers Liger has
    no kernel for at all.

So the model runs entirely on chalk kernels with ZERO Liger involvement: chalk's per-layer kernels
match-or-beat Liger's and cover strictly
more of the model. This is the "fully replace Liger" path — and it is now the only path. chalk has
no ``liger_kernel`` dependency, import, or fallback.

Each installer is self-test + arch gated and no-ops safely off-GPU / on a shape or arch mismatch, so
applying chalk is always safe: a kernel that can't engage cleanly leaves the exact eager module
untouched (the report dict records what actually installed).

The legacy ``liger=`` argument (which used to select a Liger-composition path) is DEPRECATED and
ignored — chalk no longer composes with or depends on Liger. It is still accepted so existing callers
do not break; passing a truthy value emits a ``DeprecationWarning`` and still runs chalk standalone.
"""

from __future__ import annotations

import warnings

from chalk.transformers._common import run_installer


def _install_rope():
    """Install the fused RoPE kernel (self-test + arch gated; a clean no-op returning False
    off-GPU)."""
    from chalk.ops.rope import install_qwen35_rope

    return install_qwen35_rope()


def apply_chalk_kernel_to_qwen35(
    model,
    *,
    liger: bool | str = False,
    rope: bool = True,
    rmsnorm: bool = True,
    swiglu: bool = True,
    fused_linear_cross_entropy: bool = True,
    fused_lora_delta: bool = True,
    attn_epilogue: bool = False,
    trainable_attn_epilogue: bool = True,
    fused_embedding: bool = True,
    gdn: bool = True,
    moe: bool = True,
    fp8_frozen_base: bool = False,
    fp8_lora_base: bool = False,
    fp8_no_wcache: bool = False,
    fp8_free_base: bool = False,
    fp8_dx: bool = False,
) -> dict:
    """Apply chalk's Qwen3.5/3.6 kernels — standalone, in one call (Liger fully replaced).

    Mirrors the old ``apply_liger_kernel_to_qwen3_5(...)`` surface: enablement is the call itself
    (no env flag) and each kernel is a boolean keyword. Returns a report dict keyed by kernel name
    with each installer's result (``True``/``False`` / count / report / ``None`` when a kernel isn't
    present in this build, or ``{"error": ...}`` if an installer raised).

    Args:
        liger: DEPRECATED and ignored. chalk is standalone and has fully replaced Liger — there is no
            composition path and no ``liger_kernel`` dependency. Accepted only for back-compat: a
            truthy value emits a ``DeprecationWarning`` and still runs chalk-only. Remove the argument.
        rmsnorm/swiglu/fused_linear_cross_entropy: enable the corresponding fused chalk kernel — the
            three layers Liger used to own, now owned by chalk. ``rmsnorm`` patches
            ``Qwen3_5RMSNorm.forward`` on the class so the fused Triton RMSNorm stands in for the eager
            one. ``swiglu`` swaps the elementwise ``silu(gate)*up`` MLP activation for the fused Triton
            kernel (the gate/up/down projections stay as-is, so it stays differentiable and composes
            with LoRA on them). ``fused_linear_cross_entropy`` binds the causal-LM ``forward`` so the
            LM head + CE run through the chunked kernel (no ``[N, V]`` logits materialization) during
            training. Each is self-test + arch gated; a failure keeps the exact eager module.
        rope/fused_lora_delta/attn_epilogue/fused_embedding/gdn/fp8_frozen_base: enable the
            corresponding chalk kernel (Liger has no kernel for these — chalk always owns them when
            enabled). Most default ON. ``gdn`` fuses the Gated-DeltaNet block's EAGER short causal conv
            + SiLU into one chalk Triton kernel (the path the flash worker actually runs, since
            causal_conv1d is deliberately not installed) — ~1.14-1.15x on the whole GDN block (H100),
            1.05-1.32x on the conv itself. The dominant scan (``chunk_gated_delta_rule``) stays on fla
            and the gate stays eager (a fused gate kernel LOSES — autograd-dispatch bound). Self-test
            gated + conservative path-guards (only the multi-token, non-cached training path). ``moe``
            (default ON) replaces the Qwen3.5/3.6-MoE sparse block's 128-expert Python-loop expert
            compute with chalk's loop-free batched grouped-GEMM kernel (one big ``bmm`` per projection
            over the expert axis instead of 128 tiny ones — ~50x on an H100 at the Qwen3.5-MoE shape).
            Self-test gated; only owns the standard CUDA stacked-3D-expert + ``norm_topk_prob`` shape
            (any other block delegates to the stock forward), so it never eager-regresses.
            ``trainable_attn_epilogue`` (default ON) fuses the attention block's eager q_norm/k_norm +
            ``apply_rotary_pos_emb`` (QK-RMSNorm + partial RoPE) into ONE chalk Triton kernel with a
            real backward (fwd+bwd) — it consumes the q_proj/k_proj OUTPUTS (after any LoRA delta,
            which stays differentiable), so it ENGAGES IN TRAINING with the default
            ``LORA_TARGETS=all-linear`` (where the eval-only ``attn_epilogue`` could never fire). On
            H100 it is ~1.9-8.2x faster fwd+bwd than the eager norm+RoPE stack it replaces (the win
            grows with sequence length). Self-test (fwd+bwd, both partial- and full-rotary) gated; any
            failure keeps the exact eager attention. Skipped when the eval-only ``attn_epilogue`` is
            requested (they patch the same forward). ``attn_epilogue`` (eval-only; needs q/k/v excluded
            from ``LORA_TARGETS``) and ``fp8_frozen_base`` (sm_89+: Ada / Hopper / Blackwell) default
            off. When enabled, ``fp8_frozen_base`` FP8s the frozen base WHEREVER it lives — both
            standalone frozen ``Linear``s AND the frozen ``base_layer`` INSIDE PEFT LoRA wrappers (the
            production all-linear LoRA case, where every projection is PEFT-wrapped and a
            standalone-only scan would no-op). The bf16 LoRA adapter delta is untouched; only the
            frozen base GEMM moves to FP8. Verified quality-neutral (loss-curve A/B) and GRPO-safe (the
            frozen FP8 base is deterministic, so trainer and rollout see identical weights).
        fp8_lora_base: explicit synonym for the PEFT-base wrap (sm_89+: Ada / Hopper / Blackwell,
            default off). Now that ``fp8_frozen_base`` already covers the PEFT LoRA base by default,
            this just forces the FP8-base path on; kept for back-compat. Off by default.
        fp8_no_wcache/fp8_free_base: FP8-base memory modes (default off), forwarded to
            ``install_fp8_frozen_base`` as ``no_wcache`` / ``free_base``. The FP8 base is a
            speed<->memory tradeoff: ``fp8_no_wcache`` re-quantizes the fp8 weight every forward (no
            persistent fp8 copy -> ~baseline peak memory); ``fp8_free_base`` keeps the fp8 weight and
            DROPS the bf16 base (FP8-QLoRA: net memory SAVING, checkpoints stay recoverable via a
            state_dict hook). These are plain kwargs, never env flags.
        fp8_dx: (default off, a RESEARCH speed lever forwarded to ``install_fp8_frozen_base``) also run
            the dx BACKWARD GEMM in FP8 (do_bench H100: dx 1.23-1.66x faster). HONEST trade-off: unlike
            the FROZEN forward (whose e4m3 error is a fixed bias on a non-updating weight), the dx error
            perturbs every gradient that updates the LoRA adapter, so it can affect learning quality and
            stays OFF until a training-reward A/B validates it. Ignored when ``fp8_free_base`` is on
            (free_base already dequants the fp8 base for dx).

    Never raises on a kernel failure (each installer keeps the eager path).
    """
    if liger:
        warnings.warn(
            "apply_chalk_kernel_to_qwen35(liger=...) is deprecated and ignored: chalk now runs "
            "standalone and has fully replaced Liger (it owns rms_norm / swiglu / fused-linear-CE "
            "plus RoPE / LoRA-delta / embedding / GDN / FP8). Remove the liger= argument.",
            DeprecationWarning,
            stacklevel=2,
        )
    report: dict = {}

    def _run(name: str, installer) -> None:
        run_installer(report, name, installer, tag="chalk.apply")

    report["liger"] = False  # chalk standalone — Liger fully replaced, never applied

    # chalk owns every enabled layer. Each call is independently self-test + arch gated and no-ops
    # safely off-GPU, and is additionally wrapped by ``_run`` so one installer's failure can't abort
    # the remaining installers.
    if rope:
        _run("rope", _install_rope)
    if rmsnorm:
        from chalk.ops.rmsnorm import install_qwen35_rmsnorm

        _run("rmsnorm", install_qwen35_rmsnorm)
    if swiglu:
        from chalk.ops.swiglu import install_qwen35_swiglu

        _run("swiglu", install_qwen35_swiglu)
    if fused_linear_cross_entropy:
        from chalk.ops.flce import install_qwen35_flce

        _run("fused_linear_cross_entropy", lambda: install_qwen35_flce(model))
    if fused_lora_delta:
        from chalk.ops.lora import install_fused_lora_delta

        _run("fused_lora_delta", install_fused_lora_delta)
    if attn_epilogue:
        from chalk.ops.qkv import install_qwen35_attn_epilogue

        _run("attn_epilogue", lambda: install_qwen35_attn_epilogue(model))
    if trainable_attn_epilogue and not attn_epilogue:
        from chalk.ops.qkv import install_qwen35_qknorm_rope

        _run("trainable_attn_epilogue", lambda: install_qwen35_qknorm_rope(model))
    if fused_embedding:
        from chalk.ops.embedding import install_qwen35_fused_embedding

        _run("fused_embedding", lambda: install_qwen35_fused_embedding(model))
    if gdn:
        from chalk.ops.gdn import install_qwen35_gdn

        _run("gdn", install_qwen35_gdn)
    if moe:
        from chalk.transformers.moe import install_qwen35_moe

        _run("moe", install_qwen35_moe)
    if fp8_frozen_base or fp8_lora_base:
        from chalk.ops.fp8_base import install_fp8_frozen_base

        _run(
            "fp8_frozen_base",
            lambda: install_fp8_frozen_base(
                model, lora_base=True, no_wcache=fp8_no_wcache, free_base=fp8_free_base, fp8_dx=fp8_dx
            ),
        )

    return report
