"""
Chalk model integrations — installers that monkeypatch chalk kernels into HuggingFace models.

Mirrors Liger Kernel's ``liger_kernel.transformers`` layer: each ``install_*`` entry point is
install-on-call — calling it IS the opt-in (the Liger model, ``apply_liger_kernel_to_qwen3``),
with no env flag — then arch-gated, runs a numeric self-test on install, and silently falls back
to the eager path on any import / compile / self-test failure. All installers but one
patch only frozen ``nn.Linear`` layers (never trainable / PEFT-wrapped layers); the exception is
``install_fused_lora_delta``, which intentionally accelerates the *trainable* PEFT LoRA path by
patching PEFT's dense ``lora.Linear.forward`` — gated by a self-test that checks the fused forward
and the ``x`` / ``A`` / ``B`` gradients against autograd, with per-call fallback to stock PEFT for
any unsupported adapter (merged / disabled, mixed-batch, variants, dropout, ``lora_bias``,
quantized / non-dense base, non-CUDA). Per-kernel scope is controlled by keyword args (e.g.
``install_fp8_frozen_base(model, attn=False)``), again mirroring Liger.

``apply_chalk_kernel_to_qwen35(model, ...)`` is the one-call entry point that installs every chalk
kernel in one shot. chalk is standalone — it has fully replaced Liger, with no ``liger_kernel``
dependency, import, or composition path; the legacy ``liger=`` argument is deprecated and ignored
(see ``chalk.transformers.apply``).

All kernel modules import torch/triton lazily, so importing this package (and listing the
installers) is safe on a CPU-only box; calling an installer without a GPU is a no-op that returns
its 'did nothing' value (``False`` / ``0`` / an empty report).
"""

from chalk.transformers.apply import apply_chalk_kernel_to_qwen35
from chalk.transformers.embedding import install_qwen35_fused_embedding
from chalk.transformers.flce import install_qwen35_flce
from chalk.transformers.fp8_base import install_fp8_frozen_base
from chalk.transformers.gdn import install_qwen35_gdn
from chalk.transformers.lora import install_fused_lora_delta
from chalk.transformers.minicpm import apply_chalk_kernel_to_llama
from chalk.transformers.minicpm import apply_chalk_kernel_to_minicpm
from chalk.transformers.moe import install_qwen35_moe
from chalk.transformers.qkv import install_qwen35_attn_epilogue
from chalk.transformers.qkv import install_qwen35_qknorm_rope
from chalk.transformers.rmsnorm import install_qwen35_rmsnorm
from chalk.transformers.rope import install_qwen35_rope
from chalk.transformers.swiglu import install_qwen35_swiglu

__all__ = [
    "apply_chalk_kernel_to_llama",
    "apply_chalk_kernel_to_minicpm",
    "apply_chalk_kernel_to_qwen35",
    "install_fp8_frozen_base",
    "install_fused_lora_delta",
    "install_qwen35_attn_epilogue",
    "install_qwen35_flce",
    "install_qwen35_fused_embedding",
    "install_qwen35_gdn",
    "install_qwen35_moe",
    "install_qwen35_qknorm_rope",
    "install_qwen35_rmsnorm",
    "install_qwen35_rope",
    "install_qwen35_swiglu",
]
