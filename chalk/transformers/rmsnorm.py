"""Install chalk's fused Qwen3.5/3.6 RMSNorm Triton kernel (fwd+bwd).

This installer patches ``Qwen3_5RMSNorm.forward`` on the CLASS, so chalk's fused kernel runs in
place of the eager RMSNorm on every instance. Self-test + arch gated, with an exact-eager fallback
on any failure. chalk is standalone — Liger is fully replaced and there is no composition path
(see ``chalk.transformers.apply``).
"""

from chalk.ops.rmsnorm import install_qwen35_rmsnorm

__all__ = ["install_qwen35_rmsnorm"]
