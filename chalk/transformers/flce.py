"""Install chalk's chunked Fused-Linear-Cross-Entropy Triton kernel (LM-head + CE, fwd+bwd).

This installer patches the Qwen3.5/3.6 causal-LM loss path on the CLASS (or a single instance when
given ``model``): it binds the ``*ForCausalLM.forward`` so the LM head + CE run through the chunked
kernel (no ``[N, V]`` logits materialization) during training. Self-test + arch gated, with an
exact-eager fallback on any failure. chalk is standalone — Liger is fully replaced and there is no
composition path (see ``chalk.transformers.apply``).
"""

from chalk.ops.flce import install_qwen35_flce

__all__ = ["install_qwen35_flce"]
