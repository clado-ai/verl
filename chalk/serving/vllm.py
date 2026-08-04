"""vLLM serving surface for chalk's forward kernels.

``chalk.serving.apply`` / ``chalk.serving.decode`` target HF ``transformers`` modules. This is the
counterpart for **vLLM** engines: it patches chalk's forward kernels into vLLM's OWN layer classes
(``GemmaRMSNorm`` / ``RMSNorm`` / ``SiluAndMul``) so a live vLLM serving engine runs them.

``vllm`` is an optional, lazily-imported dependency (chalk does not depend on it); everything imports
inside the functions, so importing this module is cheap and CPU-safe. It is a no-op (empty install
list) off-GPU or when the classes cannot be patched.

Placement + convention rules the caller must respect (learned the hard way integrating this into a
production vLLM serving stack — see the tests for the failure each rule prevents):

1. **Process.** vLLM V1 runs the model in a separate ``EngineCore`` worker process. Call this INSIDE
   that worker (e.g. via a ``worker_cls`` override), not in the API/front-end process — the latter
   has no GPU and never touches the running model.

2. **Timing.** vLLM's ``CustomOp.__init__`` binds ``self._forward_method`` at construction, so patch
   BEFORE ``load_model`` builds the layers, or the patch is ignored. That is also when VRAM is free
   for the Triton self-test.

3. **Right class.** Qwen3.5/3.6 import ``GemmaRMSNorm`` (the ``1 + w`` convention) as their norm, NOT
   plain ``RMSNorm``. Patch ``GemmaRMSNorm`` for gemma-convention families, ``RMSNorm`` for llama.

4. **Eager only.** Under CUDA-graph compilation vLLM's inductor fusion traces the layers and fuses
   silu/norm+quant — better than standalone Triton and incompatible with the tracer. Install only
   when the engine runs ``enforce_eager``.

Every op self-tests against a plain-torch reference and reverts to stock vLLM on any numeric
mismatch, so only faithful patches take effect.
"""

from __future__ import annotations

from collections.abc import Mapping

_ATOL = 2e-2
_RTOL = 2e-2


def is_gemma_family(base_model: str | None) -> bool:
    """Qwen3.5/3.6 norms use the (1+weight) gemma convention; MiniCPM/llama use plain weight."""
    return "minicpm" not in (base_model or "").lower()


def _ref_rmsnorm(x, w, eps: float, gemma: bool):
    import torch

    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    scale = (w.float() + 1.0) if gemma else w.float()
    return (xn * scale).to(x.dtype)


def _patch_rmsnorm(gemma: bool) -> tuple[str | None, str]:
    import torch

    from chalk.ops.rmsnorm import load_rmsnorm

    fn = load_rmsnorm(gemma=gemma)
    if fn is None:
        return None, f"load_rmsnorm(gemma={gemma}) returned None (no kernel for this arch)"

    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm

    target = GemmaRMSNorm if gemma else RMSNorm
    label = "GemmaRMSNorm" if gemma else "RMSNorm"
    # Fall back to the ORIGINAL forward_native, never forward_cuda: GemmaRMSNorm.forward_cuda delegates
    # to self.forward_native, which we patch, so a forward_cuda fallback recurses infinitely.
    orig_native = target.forward_native
    try:
        w = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(8, 2560, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6
        if not torch.allclose(_ref_rmsnorm(x, w, eps, gemma), fn(x, w, eps), atol=_ATOL, rtol=_RTOL):
            maxd = (_ref_rmsnorm(x, w, eps, gemma).float() - fn(x, w, eps).float()).abs().max().item()
            return None, f"{label} self-test mismatch (max|Δ|={maxd:.4g} > atol {_ATOL})"
    except Exception as e:  # report, never raise
        return None, f"{label} self-test raised {type(e).__name__}: {e}"

    def patched(self, x, residual=None):
        if residual is not None:
            return orig_native(self, x, residual)
        try:
            return fn(x, self.weight.data, self.variance_epsilon)
        except Exception:
            return orig_native(self, x, residual)

    target.forward_native = patched
    target.forward_cuda = patched
    return f"rmsnorm[{label}]", "ok"


def _ref_swiglu(gate, up):
    import torch.nn.functional as f

    return f.silu(gate.float()).mul(up.float()).to(gate.dtype)


def _patch_swiglu() -> tuple[str | None, str]:
    import torch

    from chalk.ops.swiglu import load_swiglu

    fn = load_swiglu()
    if fn is None:
        return None, "load_swiglu() returned None (no kernel for this arch)"

    from vllm.model_executor.layers.activation import SiluAndMul

    orig_native = SiluAndMul.forward_native

    def patched(*args):
        # SiluAndMul is weightless: the runtime binds (self, x) but vLLM's inductor fusion matcher
        # calls SiluAndMul.forward_native(x) UNBOUND. The input is always the last positional arg.
        x = args[-1]
        try:
            d = x.shape[-1] // 2
            return fn(x[..., :d].contiguous(), x[..., d:].contiguous())
        except Exception:
            return orig_native(*args)

    try:
        d = 4096
        gate = torch.randn(8, d, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(8, d, device="cuda", dtype=torch.bfloat16)
        if not torch.allclose(_ref_swiglu(gate, up), fn(gate, up), atol=_ATOL, rtol=_RTOL):
            maxd = (_ref_swiglu(gate, up).float() - fn(gate, up).float()).abs().max().item()
            return None, f"self-test mismatch (max|Δ|={maxd:.4g} > atol {_ATOL})"
    except Exception as e:  # report, never raise
        return None, f"self-test raised {type(e).__name__}: {e}"

    SiluAndMul.forward_native = patched
    SiluAndMul.forward_cuda = patched
    return "swiglu", "ok"


def install_vllm_serving_kernels(base_model: str | None = None, *, gemma: bool | None = None) -> Mapping[str, object]:
    """Patch chalk's forward kernels into vLLM's layer classes. Safe no-op on any failure.

    Call inside the EngineCore worker, before ``load_model``, in ``enforce_eager`` mode (see module
    docstring). ``gemma`` selects the RMSNorm convention; when ``None`` it is inferred from
    ``base_model``. Returns ``{"installed": [...], "skipped": [...]}`` (or ``{"error": ...}`` if the
    chalk/vLLM import failed — the engine keeps running on stock vLLM regardless).
    """
    if gemma is None:
        gemma = is_gemma_family(base_model)
    try:
        results = [_patch_rmsnorm(gemma), _patch_swiglu()]
        installed = [name for name, _ in results if name]
        skipped = [reason for name, reason in results if not name]
        return {"installed": installed, "skipped": skipped}
    except Exception as e:  # a broken hook must never take serving down
        return {"error": str(e), "installed": []}
