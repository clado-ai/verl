"""One-call entry point for chalk's kernels on the MiniCPM / Llama family (plain-weight RMSNorm + SwiGLU).

Covers both the custom-remote-code MiniCPM (``openbmb/MiniCPM-2B-*``, classes ``MiniCPMRMSNorm`` /
``MiniCPMMLP``) AND models that transformers serves through the native **Llama** classes
(``LlamaRMSNorm`` / ``LlamaMLP``) — which is what newer checkpoints resolve to: e.g.
``openbmb/MiniCPM5-1B`` loads as ``LlamaForCausalLM`` (``model_type: "llama"``), so its norm/MLP are
``LlamaRMSNorm`` / ``LlamaMLP``, not the old MiniCPM classes. Both families use the identical LLAMA
plain-weight RMSNorm + ``silu(gate)*up`` SwiGLU, so chalk's kernels apply unchanged; only the class
name to patch differs. ``apply_chalk_kernel_to_llama`` is an alias of this entry point.

These classes have no single fixed import path (remote code, or just built at load), so this installer
patches by walking ``model.modules()`` AFTER the model is built and swapping the kernel on the CLASS
of each matched instance (matched by ``type(m).__name__`` ∈ the accepted sets below), so every
current and future instance of that class picks up the fused kernel.

Two chalk kernels apply:

* ``rmsnorm`` — ``{MiniCPM,Llama}RMSNorm.forward`` is ``rms_layernorm(h, weight, eps)`` =
  ``h=(h*rsqrt(var+eps)).to(dtype); return h*weight``: normalize in fp32, cast BACK to the input
  dtype, THEN multiply by the PLAIN weight (NO ``(1+weight)`` Gemma offset). This is the LLAMA
  convention, so chalk's fused RMSNorm is installed in ``gemma=False`` (plain-weight / cast-before)
  mode. muP scaling (``scale_depth`` / ``dim_model_base`` / ``scale_emb``) is applied on the
  residuals/embeddings/logits OUTSIDE the norm, so it does not touch this kernel.
* ``swiglu`` — ``{MiniCPM,Llama}MLP`` is the standard ``down_proj(silu(gate_proj(x)) * up_proj(x))``,
  the SAME activation as Qwen3.5, so chalk's convention-agnostic fused SwiGLU (``chalk.ops.swiglu``)
  applies unchanged.

Every install is live-GPU self-test gated at the MODEL's actual dims and is a SAFE no-op on any
mismatch (kernel build/self-test failure, no matching class, non-CUDA): a layer that can't be
verified stays EAGER. The installer NEVER raises — a failure keeps eager for that layer and is
recorded in the report dict (``{"installed": True, ...}`` engaged / ``False`` verified-off /
``None`` no such module in this model / ``{"error": ...}`` unexpected). Mirrors
``apply_chalk_kernel_to_qwen35``.
"""

from __future__ import annotations

from chalk.transformers._common import run_installer

# Norm / MLP classes this installer patches, by ``type(m).__name__``. The MiniCPM remote-code classes
# AND the native transformers Llama classes (what MiniCPM5 and llama-family checkpoints resolve to) —
# identical plain-weight RMSNorm + silu-gated MLP, so one installer covers both.
_RMSNORM_CLASS_NAMES = ("MiniCPMRMSNorm", "LlamaRMSNorm")
_MLP_CLASS_NAMES = ("MiniCPMMLP", "LlamaMLP")

# Populated by apply_chalk_kernel_to_minicpm so a worker can fold the outcome into metrics. Empty
# {} means nothing was engaged this run.
RESULT: dict = {}


def _restore_forward(classes, flag_attr, orig_attr):
    """Undo any prior chalk patch on these classes (used when a later (re)apply fails self-test):
    a trust_remote_code class patch lives on the CLASS, so a failed reapply must restore the saved
    original ``forward`` rather than leave a stale/possibly-wrong kernel installed."""
    for cls in classes:
        if getattr(cls, flag_attr, False) and hasattr(cls, orig_attr):
            cls.forward = getattr(cls, orig_attr)
            setattr(cls, flag_attr, False)


def _make_rmsnorm_forward(fn):
    """Class-level ``MiniCPMRMSNorm.forward`` that routes CUDA inputs through chalk's fused
    llama-mode RMSNorm and falls back to the EXACT eager ``rms_layernorm`` (plain-weight,
    cast-before) for any non-CUDA / mis-shaped call (the class patch is global)."""

    def forward(self, hidden_states):
        import torch

        # HF stores eps as ``variance_epsilon``; use an explicit ``is None`` check (a valid eps of
        # 0.0 is falsy, so an ``or`` fallback would be wrong).
        eps = getattr(self, "variance_epsilon", None)
        if eps is None:
            eps = getattr(self, "eps", 1e-6)
        eps = float(eps)
        # Only take the fused path when the weight dtype matches the activation dtype: under AMP a
        # fp32 norm weight over bf16 activations makes eager's ``h*weight`` promote the output to
        # fp32, which the kernel (storing into ``empty_like(hidden_states)``) would not match.
        if (
            hidden_states.is_cuda
            and self.weight.is_cuda
            and hidden_states.shape[-1] == self.weight.numel()
            and self.weight.dtype == hidden_states.dtype
        ):
            return fn(hidden_states, self.weight, eps)
        # Eager LLAMA fallback == MiniCPM's rms_layernorm exactly: normalize in fp32, cast BACK to
        # the input dtype, THEN multiply by the plain weight.
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = (h * torch.rsqrt(var + eps)).to(input_dtype)
        return h * self.weight

    return forward


def _make_swiglu_forward(fn):
    """Class-level ``MiniCPMMLP.forward`` that swaps the elementwise ``silu(gate)*up`` activation
    for chalk's fused SwiGLU while leaving the gate/up/down projections exactly as the model has
    them (so it stays differentiable and composes with LoRA). Non-CUDA activations fall back to the
    exact eager ``act_fn(gate) * up``."""

    def forward(self, x):

        g = self.gate_proj(x)
        u = self.up_proj(x)
        # chalk's SwiGLU hard-codes silu(gate)*up. Match SiLU by class name so it also covers
        # HF's ACT2FN['silu'] wrapper (SiLUActivation), not only torch.nn.SiLU.
        _is_silu = type(self.act_fn).__name__ in ("SiLU", "SiLUActivation")
        if g.is_cuda and u.is_cuda and _is_silu:
            act = fn(g, u)
        else:
            act = self.act_fn(g) * u
        return self.down_proj(act)

    return forward


def _install_minicpm_rmsnorm(model):
    """Install chalk's fused RMSNorm (llama / plain-weight mode) on every ``MiniCPMRMSNorm`` class
    in ``model`` — IFF the live-GPU self-test at the model's ACTUAL norm dims passes. Returns a
    report dict on success, ``False`` if the kernel can't be verified (stay eager), ``None`` if the
    model has no ``MiniCPMRMSNorm`` (nothing to do)."""
    from chalk.ops.rmsnorm import _self_test
    from chalk.ops.rmsnorm import load_rmsnorm

    norms = [m for m in model.modules() if type(m).__name__ in _RMSNORM_CLASS_NAMES]
    if not norms:
        return None
    classes = {type(m) for m in norms}
    # Build + self-test the llama (plain-weight, cast-before) kernel. load_rmsnorm(gemma=False)
    # already gates at default hidden sizes; below we ALSO gate at the model's real dims.
    fn = load_rmsnorm(gemma=False)
    if fn is None:
        _restore_forward(classes, "_chalk_minicpm_rmsnorm", "_chalk_minicpm_orig_rmsnorm")
        return False
    # Every distinct weight width the model actually norms over: hidden (2304 for MiniCPM-2B) plus
    # any small head_dim norms (q/k norms) if a variant has them.
    dims = sorted({int(m.weight.numel()) for m in norms if getattr(m, "weight", None) is not None}) or [2304]
    try:
        _self_test(fn, gemma=False, hiddens=tuple(dims))
    except Exception as e:  # a self-test miss = keep eager (safe), not an error
        # restore any prior patch on these classes so a failed reapply doesn't leave a stale kernel
        _restore_forward(classes, "_chalk_minicpm_rmsnorm", "_chalk_minicpm_orig_rmsnorm")
        print(f"[minicpm.rmsnorm] self-test at model dims {dims} failed; keeping eager: {e}", flush=True)
        return False
    fwd = _make_rmsnorm_forward(fn)
    patched = []
    for cls in classes:
        # Save the pristine forward once so a later failed reapply can restore it; always (re)assign
        # so a repeat apply refreshes to the current kernel closure.
        if not getattr(cls, "_chalk_minicpm_rmsnorm", False) and not hasattr(cls, "_chalk_minicpm_orig_rmsnorm"):
            cls._chalk_minicpm_orig_rmsnorm = cls.forward
        cls.forward = fwd
        cls._chalk_minicpm_rmsnorm = True
        patched.append(cls.__name__)
    print(f"[minicpm.rmsnorm] fused llama-mode RMSNorm installed on {sorted(set(patched))} dims={dims}", flush=True)
    return {
        "installed": True,
        "self_test": "passed",
        "convention": "llama",
        "dims": dims,
        "patched": sorted(set(patched)),
        "count": len(norms),
    }


def _install_minicpm_swiglu(model):
    """Install chalk's fused SwiGLU activation on every ``MiniCPMMLP`` class in ``model`` — IFF the
    live-GPU self-test passes. Returns a report dict on success, ``False`` if the kernel can't be
    verified (stay eager), ``None`` if the model has no ``MiniCPMMLP`` (nothing to do)."""
    from chalk.ops.swiglu import _self_test
    from chalk.ops.swiglu import load_swiglu

    mlps = [m for m in model.modules() if type(m).__name__ in _MLP_CLASS_NAMES]
    if not mlps:
        return None
    classes = {type(m) for m in mlps}

    # pretraining_tp > 1 makes MiniCPMMLP.forward use sliced gate/up/down projections (tensor-parallel
    # checkpoint numerics); our dense fusion would change those, so keep the original forward. The
    # branch is controlled by each MLP module's OWN self.config, so check those (not just the
    # top-level model.config, which can be missing/stale in wrapped/adapter models).
    def _tp(m):
        return int(getattr(getattr(m, "config", None), "pretraining_tp", 1) or 1)

    if int(getattr(getattr(model, "config", None), "pretraining_tp", 1) or 1) > 1 or any(_tp(m) > 1 for m in mlps):
        _restore_forward(classes, "_chalk_minicpm_swiglu", "_chalk_minicpm_orig_swiglu")
        print("[minicpm.swiglu] pretraining_tp>1 (tensor-parallel MLP); keeping eager", flush=True)
        return False
    if not all(hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj") for m in mlps):
        _restore_forward(classes, "_chalk_minicpm_swiglu", "_chalk_minicpm_orig_swiglu")
        print("[minicpm.swiglu] MLP lacks gate_proj/up_proj/down_proj (alias variant); keeping eager", flush=True)
        return False
    fn = load_swiglu()
    if fn is None:
        _restore_forward(classes, "_chalk_minicpm_swiglu", "_chalk_minicpm_orig_swiglu")
        return False
    # Extra gate at the model's real intermediate size (the SwiGLU activation is elementwise, so
    # correctness is size-independent — this just exercises the exact width). Best-effort: skip if
    # the config isn't introspectable (load_swiglu already self-tested at the default widths).
    inters = []
    try:
        isz = int(getattr(model.config, "intermediate_size", 0))
        if isz > 0:
            inters = [isz]
    except Exception:
        inters = []
    if inters:
        try:
            _self_test(fn, inters=tuple(inters))
        except Exception as e:  # a self-test miss = keep eager (safe
            _restore_forward(classes, "_chalk_minicpm_swiglu", "_chalk_minicpm_orig_swiglu")
            print(f"[minicpm.swiglu] self-test at model inter {inters} failed; keeping eager: {e}", flush=True)
            return False
    fwd = _make_swiglu_forward(fn)
    patched = []
    for cls in classes:
        # Save the pristine forward once so a later failed reapply can restore it; always (re)assign.
        if not getattr(cls, "_chalk_minicpm_swiglu", False) and not hasattr(cls, "_chalk_minicpm_orig_swiglu"):
            cls._chalk_minicpm_orig_swiglu = cls.forward
        cls.forward = fwd
        cls._chalk_minicpm_swiglu = True
        patched.append(cls.__name__)
    print(f"[minicpm.swiglu] fused SwiGLU installed on {sorted(set(patched))} inter={inters}", flush=True)
    return {
        "installed": True,
        "self_test": "passed",
        "inter": inters,
        "patched": sorted(set(patched)),
        "count": len(mlps),
    }


def apply_chalk_kernel_to_minicpm(model, *, rmsnorm: bool = True, swiglu: bool = True) -> dict:
    """Apply chalk's fused kernels to a built MiniCPM / Llama-family model, in one call.

    Covers the custom-remote-code MiniCPM (``openbmb/MiniCPM-2B-*``) AND models that transformers
    serves through native **Llama** classes (``LlamaRMSNorm`` / ``LlamaMLP``) — e.g.
    ``openbmb/MiniCPM5-1B`` loads as ``LlamaForCausalLM``. Both share the identical LLAMA plain-weight
    RMSNorm + silu-gated MLP, so the same chalk kernels apply; only the class name differs.
    ``apply_chalk_kernel_to_llama`` is an alias.

    Mirrors ``apply_chalk_kernel_to_qwen35``: enablement is the call itself (no env flag), each
    kernel is a boolean keyword, and every installer is live-GPU self-test gated and a SAFE no-op on
    any mismatch (a layer that can't be verified stays EAGER). Because the classes may be loaded via
    ``trust_remote_code`` (no fixed import path), the installers walk ``model.modules()`` and patch
    the class of each matched instance by NAME — so call this AFTER the model is built.

    Args:
        rmsnorm: install chalk's fused Triton RMSNorm on ``{MiniCPM,Llama}RMSNorm`` in LLAMA /
            plain-weight mode (``h.to(dtype) * weight``, no ``(1+weight)`` offset), gated by a
            self-test at the model's real norm dims (hidden + any head_dim norms).
        swiglu: swap the ``silu(gate)*up`` activation in ``{MiniCPM,Llama}MLP`` for chalk's fused
            Triton SwiGLU (gate/up/down projections stay as-is, so it composes with LoRA), self-test
            gated.

    Returns a report dict keyed by kernel name. Each value is ``{"installed": True, ...}`` when the
    kernel engaged, ``False`` when it was verified-off (stay eager), ``None`` when the model has no
    matching module, or ``{"error": ...}`` if an installer raised. NEVER raises on a kernel failure.
    """
    report: dict = {}

    def _run(name: str, installer) -> None:
        run_installer(report, name, installer, tag="chalk.minicpm")

    if rmsnorm:
        _run("rmsnorm", lambda: _install_minicpm_rmsnorm(model))
    if swiglu:
        _run("swiglu", lambda: _install_minicpm_swiglu(model))

    RESULT.clear()
    RESULT.update(report)
    return report


# Semantic alias: this entry point also patches the native Llama classes (``LlamaRMSNorm`` /
# ``LlamaMLP``), so llama-family callers can use the honestly-named function.
apply_chalk_kernel_to_llama = apply_chalk_kernel_to_minicpm
