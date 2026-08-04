"""Install the model-wide FP8 (e4m3) frozen-base GEMM path (sm_89+ Ada / Hopper / Blackwell)."""

from chalk.ops.fp8_base import install_fp8_frozen_base

__all__ = ["install_fp8_frozen_base"]
