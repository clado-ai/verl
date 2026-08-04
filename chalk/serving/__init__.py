"""chalk.serving — the serving (inference) kernel surface.

Separate from ``chalk.transformers`` (the training surface) but built on the SAME ``chalk.ops``
kernels: the forward math of RMSNorm / SwiGLU / RoPE / GDN-conv / embedding is identical whether or
not a backward pass follows, so there is one tuned kernel per op and this package only changes the
*composition* — it installs the forward-relevant kernels and deliberately leaves out the training-
only ones (fused-linear-cross-entropy = the loss, the trainable LoRA-delta, the trainable attention
epilogue, GRPO). See ``policy.py`` for the per-model selection and the rationale.

Why a separate folder at all (rather than just calling ``apply_chalk_kernel_to_qwen35`` with flags):
this is the home for anything that genuinely diverges for serving — the forward-only composition
policy (``apply.py``) AND the decode-shape kernels (``decode.py``). The ``serving_bench`` A/B showed
decode divergence is real: training tunes for long sequences, but at decode (M≈8, forward-only) the
training kernels are actually slower than eager, so ``decode.py`` ships kernels tuned for that regime
(``apply_chalk_decode_kernels``), found by the serve autoresearch and E2E-validated.

Import is cheap and CPU-safe — the ops load lazily, exactly like ``chalk.transformers``.
"""

from chalk.serving.apply import apply_chalk_serving_kernels
from chalk.serving.apply import engaged_serving_kernels
from chalk.serving.apply import serving_kernel_flags
from chalk.serving.decode import apply_chalk_decode_kernels
from chalk.serving.vllm import install_vllm_serving_kernels
from chalk.serving.vllm import is_gemma_family

__all__ = [
    "apply_chalk_decode_kernels",
    "apply_chalk_serving_kernels",
    "engaged_serving_kernels",
    "install_vllm_serving_kernels",
    "is_gemma_family",
    "serving_kernel_flags",
]
