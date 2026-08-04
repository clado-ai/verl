"""Install the fused Qwen3.5 embedding-gather + layer-0 RMSNorm kernel."""

from chalk.ops.embedding import install_qwen35_fused_embedding

__all__ = ["install_qwen35_fused_embedding"]
