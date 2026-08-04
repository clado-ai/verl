"""
Chalk operators — raw Triton/CUDA kernels and their ``torch.autograd.Function`` wrappers.

Mirrors Liger Kernel's layout: ``chalk.ops`` holds the low-level kernel implementations
(``@triton.jit`` functions, autograd Functions, FP8 GEMM helpers), while ``chalk.transformers``
holds the model-level installers that monkeypatch these kernels into HuggingFace modules.

This namespace starts empty by design — kernels are landed one at a time, each with its own
benchmark evidence.
"""

__all__: list[str] = []
