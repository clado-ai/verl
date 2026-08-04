"""Small device/runtime helpers, mirroring Liger Kernel's ``utils`` surface.

torch is imported lazily so that importing :mod:`chalk` stays light on machines
without a GPU stack (the kernels themselves require torch + triton at call time).
"""


def is_npu_available() -> bool:
    """Detect Ascend NPU availability."""
    try:
        import torch
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def infer_device() -> str:
    """Return the current device name based on available devices.

    Falls back to ``"cpu"`` when torch is not installed.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():  # Works for both Nvidia and AMD
        return "cuda"
    if is_npu_available():
        return "npu"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def rel_l2(a, b) -> float:
    """Relative L2 error ``||a - b|| / (||b|| + eps)`` between two tensors, measured in fp32.

    Both sides are cast up to fp32 before the norm so a reduced-dtype (bf16/fp16) output is
    compared against its reference WITHOUT re-introducing the storage dtype's rounding into the
    metric. Shared by the ops' live-GPU self-tests to grade a fused kernel against its eager
    oracle. Tensor-method only (no module-level torch import), so importing ``chalk.utils`` stays
    light on a machine without a GPU stack.
    """
    a32, b32 = a.float(), b.float()
    return (a32 - b32).norm().item() / (b32.norm().item() + 1e-9)
