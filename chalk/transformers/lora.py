"""Install the fused Triton LoRA-delta kernel into PEFT ``lora.Linear`` layers."""

from chalk.ops.lora import install_fused_lora_delta

__all__ = ["install_fused_lora_delta"]
