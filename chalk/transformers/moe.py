"""Install chalk's fast batched grouped-GEMM MoE kernel into Qwen3.5/3.6-MoE.

The Qwen3.5-MoE sparse block (``Qwen3_5MoeSparseMoeBlock``) routes each token to its ``top_k``
experts and runs a per-expert SwiGLU MLP over the routed tokens — the experts hold their weights as
STACKED 3D parameters (``self.experts.gate_up_proj`` [E, 2*inter, hidden], ``self.experts.down_proj``
[E, hidden, inter], matching ``chalk.ops.moe``'s tensor contract) — then scatters the router-weighted
expert outputs back. The stock forward runs the expert compute as a 128-iteration Python loop of tiny
GEMMs; chalk replaces the WHOLE expert compute with the loop-free batched-``bmm`` grouped-GEMM kernel
(``chalk.ops.moe.load_moe`` -> ``grouped_gemm_fn``): one big batched GEMM per projection over the
expert axis instead of 128 small ones (~50x on an H100 at the Qwen3.5-MoE shape).

``install_qwen35_moe`` patches the ``SparseMoeBlock`` CLASS ``forward`` (like chalk's swiglu/gdn
installers patch their layer class) so every current and future instance routes through the chalk
kernel — IFF the kernel builds + passes its live-GPU fwd+bwd self-test (``load_moe``). Install-on-call
(no env flag): calling it IS the opt-in.

SAFE — NEVER a chalk->eager regression on a real GPU shape:
  * kernel build / self-test failure (``load_moe`` -> None) or no importable Qwen3.5/3.6 MoE block
    -> return False, leave every class untouched;
  * per-instance, the patched forward only takes the chalk path for the shape it OWNS (CUDA hidden
    states + stacked-3D expert params + integer ``top_k`` + ``norm_topk_prob`` renorm, which is the
    Qwen3.5-MoE default); anything else delegates to the stock forward unchanged. A block whose
    experts aren't the stacked-3D-param variant is never touched.

A failed (re)install is non-destructive: ``RESULT`` and the class patch are unchanged on any False.
"""

from __future__ import annotations

from chalk.ops._hf_targets import collect_qwen_classes
from chalk.ops.moe import load_moe

__all__ = ["install_qwen35_moe"]

# Populated on a successful install so a worker can fold the outcome into metrics.json.
RESULT: dict = {}


def _make_forward(orig_forward, fn):
    """Build the patched ``SparseMoeBlock.forward``. Routes the block's expert compute through
    chalk's batched grouped-GEMM kernel ``fn`` for the shape chalk owns; delegates everything else
    to ``orig_forward`` unchanged. Kept differentiable (``fn`` is autograd-native), so it composes
    with training and returns the standard ``(hidden_states, router_logits)`` Qwen3-MoE contract."""
    import torch

    def forward(self, hidden_states, *args, **kwargs):
        experts = getattr(self, "experts", None)
        gate = getattr(self, "gate", None)
        gup = getattr(experts, "gate_up_proj", None)
        dwn = getattr(experts, "down_proj", None)
        # top_k: the block stores it as ``self.top_k`` (== config.num_experts_per_tok); fall back to
        # the experts module or the config so we track whichever attribute this build exposes.
        top_k = getattr(self, "top_k", None)
        if top_k is None:
            top_k = getattr(experts, "top_k", None)
        if top_k is None:
            top_k = getattr(getattr(self, "config", None), "num_experts_per_tok", None)
        norm = getattr(self, "norm_topk_prob", True)  # Qwen3.5-MoE default; the kernel always renorms

        # A block with a SHARED-EXPERT path (a Qwen2-MoE-style ``shared_expert`` + ``shared_expert_gate``,
        # or a ``shared_experts`` module) adds a shared MLP output to the routed-expert output. Our
        # kernel computes ONLY the routed-expert output, so owning such a block would silently DROP the
        # shared contribution and diverge from HF in both forward and grads. Qwen3.5-MoE is routed-only
        # (no shared expert — see MODEL_DIMS), so this never fires on the real target; it just makes any
        # shared-expert variant delegate to the stock forward (correct, unaccelerated) instead of wrong.
        no_shared = (
            getattr(self, "shared_expert", None) is None
            and getattr(self, "shared_experts", None) is None
            and getattr(self, "shared_expert_gate", None) is None
        )

        # NEVER-EAGER-REGRESSION guard: only own the standard CUDA Qwen3.5-MoE expert shape with
        # stacked-3D expert params + integer top_k + norm_topk_prob renorm + NO shared expert. Any
        # deviation (extra forward args, CPU offload, per-expert-MLP variant, norm_topk_prob=False,
        # a shared-expert path) -> stock forward.
        owns = (
            not args
            and not kwargs
            and gate is not None
            and torch.is_tensor(hidden_states)
            and hidden_states.is_cuda
            and torch.is_tensor(gup)
            and gup.dim() == 3
            and torch.is_tensor(dwn)
            and dwn.dim() == 3
            and isinstance(top_k, int)
            and bool(norm)
            and no_shared
        )
        if not owns:
            return orig_forward(self, hidden_states, *args, **kwargs)

        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, orig_shape[-1])  # [T, hidden]; carries the block's autograd graph
        router_logits = gate(x)  # [T, E]
        y = fn(x, gup, dwn, router_logits, top_k)  # loop-free batched grouped-GEMM
        y = y.reshape(orig_shape)
        return y, router_logits

    return forward


def install_qwen35_moe() -> bool:
    """Patch ``Qwen3_5MoeSparseMoeBlock.forward`` (and the qwen3_6_moe equivalent) so the expert
    compute runs through chalk's loop-free batched grouped-GEMM kernel — IFF the kernel builds +
    passes its live-GPU fwd+bwd self-test.

    Install-on-call (no env flag). Patches the CLASS forward (not individual instances), so every
    current and future block picks up the fast kernel, mirroring chalk's swiglu/gdn installers.
    Returns True iff the kernel was installed, False on any safe no-op (kernel disabled / no
    importable Qwen3.5/3.6 MoE block). A failed (re)install leaves the class + ``RESULT`` untouched.
    """
    fn = load_moe()
    if fn is None:
        return False

    # Only the MoE families expose a SparseMoeBlock (the dense qwen3_5/qwen3_6 modules are skipped by
    # collect_qwen_classes). Collect BEFORE mutating, so a "nothing to patch" outcome disturbs nothing.
    targets = collect_qwen_classes("SparseMoeBlock")  # [(label, cls)]
    if not targets:
        print("[moe] no qwen3_5/3_6 MoE SparseMoeBlock class to patch; keeping stock", flush=True)
        return False

    patched = []
    for label, cls in targets:
        # Save the unbound original ONCE per class (idempotent re-install keeps the very first).
        if not hasattr(cls, "_chalk_moe_orig_forward"):
            cls._chalk_moe_orig_forward = cls.forward
        orig = cls._chalk_moe_orig_forward
        cls.forward = _make_forward(orig, fn)
        patched.append(label)

    RESULT.clear()
    RESULT.update({"installed": True, "self_test": "passed", "patched": patched})
    print(f"[moe] chalk batched grouped-GEMM MoE installed on {patched} (self-test passed)", flush=True)
    return True
