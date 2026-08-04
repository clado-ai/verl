"""Install the fused Qwen3.5 QK-norm + (partial) RoPE attention kernels.

- ``install_qwen35_attn_epilogue`` — the EVAL-ONLY fused epilogue (QK-norm + RoPE + gate over a
  stacked cuBLAS projection; no backward, fires only with q/k/v excluded from LoRA + grad disabled).
- ``install_qwen35_qknorm_rope`` — the TRAINABLE fused QK-norm + partial-RoPE (fwd+bwd) that engages
  in real training and composes with q/k/v in LoRA targets (the production path).
"""

from chalk.ops.qkv import install_qwen35_attn_epilogue
from chalk.ops.qkv import install_qwen35_qknorm_rope

__all__ = ["install_qwen35_attn_epilogue", "install_qwen35_qknorm_rope"]
