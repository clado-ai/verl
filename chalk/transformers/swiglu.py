"""Install chalk's fused Qwen3.5/3.6 SwiGLU activation Triton kernel (fwd+bwd).

This installer patches ``Qwen3_5MLP.forward`` on the CLASS, so chalk's fused ``silu(gate)*up``
activation runs in place of the eager ``act_fn(gate)*up`` (the gate/up/down projections stay
as-is). Self-test + arch gated, with an exact-eager fallback on any failure. chalk is standalone —
Liger is fully replaced and there is no composition path (see ``chalk.transformers.apply``).
"""

from chalk.ops.swiglu import install_qwen35_swiglu

__all__ = ["install_qwen35_swiglu"]
