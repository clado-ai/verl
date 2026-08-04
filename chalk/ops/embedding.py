"""Fused gather + layer-0 RMSNorm Triton kernel for the Qwen3.5 LoRA worker.

WHAT THIS FUSES
---------------
The very first thing a Qwen3.5 forward does is::

    inputs_embeds = embed_tokens(input_ids)          # [tokens, hidden] gather
    # ... decoder layer 0:
    residual      = inputs_embeds
    hidden        = input_layernorm(inputs_embeds)    # layer-0 RMSNorm

i.e. a memory-bound embedding gather immediately followed by the layer-0 RMSNorm.
The baseline (production) path materializes ``inputs_embeds`` to HBM and reads it
back for the RMSNorm; this kernel gathers the row, RMS-normalizes it in registers,
and writes the normalized result in ONE launch, so the layer-0 norm never round-
trips the embedding through HBM a second time.

  baseline  = F.embedding(ids, table)  THEN  (Liger/eager) RMSNorm
  fused      = gather row table[id] -> fp32 var -> x*rstd -> *(1+w) -> cast -> store

RMSNorm SEMANTICS (must match the real model EXACTLY)
-----------------------------------------------------
Qwen3.5's ``Qwen3_5RMSNorm.forward`` is::

    output = x.float() * rsqrt(x.float().pow(2).mean(-1) + eps)
    output = output * (1.0 + self.weight.float())        # fp32 multiply
    return output.type_as(x)                              # cast to bf16 at the end

so the weight carries an implicit ``+1.0`` OFFSET and the weight multiply happens in
fp32 BEFORE the final cast. This is Liger ``casting_mode="gemma"`` + ``offset=1.0``
(which is exactly how ``apply_liger_kernel_to_qwen3_5`` patches every ``input_layernorm``
instance — verified on the pod). NOTE: the earlier ``bench/triton_embedding_rmsnorm.py``
prototype used ``casting_mode="llama"`` (cast-then-multiply, plain ``weight``); that is
WRONG for Qwen3.5. The kernel here uses the gemma+offset semantics and is self-tested
against the eager ``Qwen3_5RMSNorm`` math.

WHY A FORWARD-ONLY KERNEL IS CORRECT
------------------------------------
Under LoRA the recipe targets ``all-linear`` with no ``modules_to_save`` /
``trainable_token_indices`` and there is no full-fine-tune path, so ``embed_tokens``
AND ``input_layernorm.weight`` are both FROZEN: the embed -> layer-0-RMSNorm subgraph
builds NO backward graph. A forward-only fused kernel is therefore production-correct.
To stay safe even if a future config trained either, the patched RMSNorm takes the
fused path ONLY when grad is disabled OR the inputs are non-trainable (no backward
needed); otherwise it falls back to the original (Liger or eager) differentiable path.

HONEST IMPACT (see bench/embedding_result.md)
---------------------------------------------
Per-op the fused kernel is ~1.8x vs the Liger two-op path on EVERY arch (it is purely
memory-bound, so the win is architecture-independent). But it runs ONCE per forward
(layer 0 only) and is ~0.04% of step time. It is a CORRECT, UNIVERSAL, TINY opt-in
win. There is no larger embedding-adjacent fusion available on this model: weight tying
(``tie_word_embeddings=true``) routes the big embedding-table cost through the
lm_head/cross-entropy GEMM, which Liger's fused-linear-CE already owns.

GATING / SAFETY
---------------
Install-on-call (the Liger model): calling ``install_qwen35_fused_embedding()`` IS the opt-in —
there is no env flag — then it is gated by a live-GPU numeric self-test (ANY
import/compile/self-test failure leaves the Liger/eager path untouched). The installer
patches ONLY layer-0's ``input_layernorm`` + arranges the gather, no-ops safely if the
model shape, the embedding module, or the Liger RMSNorm offset/casting does not match
what this kernel implements. Import-safe on a CPU control plane (triton/torch imported
lazily).
"""

from __future__ import annotations

import contextlib


def _build_kernel():
    """Import torch/triton and define the fused gather+RMSNorm kernel. Returns
    ``fused_gather_rmsnorm`` or raises on any import/compile problem (the caller treats a
    raise as "keep the Liger/eager path")."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_gather_rmsnorm_kernel(
        ids_ptr,  # [n_tokens] int (flattened)
        table_ptr,  # [vocab, hidden] embedding table (== lm_head when tied)
        w_ptr,  # [hidden] RMSNorm weight (effective weight is 1+w)
        out_ptr,  # [n_tokens, hidden] normalized output
        hidden,
        eps,
        OFFSET: tl.constexpr,  # 1.0 for Qwen3.5 (weight is zeros-init, effective 1+w)
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        tok = tl.load(ids_ptr + row)
        h_off = tl.arange(0, BLOCK_H)
        mask = h_off < hidden
        x = tl.load(table_ptr + tok * hidden + h_off, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, axis=0) / hidden
        rstd = 1.0 / tl.sqrt(var + eps)
        # gemma casting_mode: the (offset+weight) multiply stays in fp32, cast LAST.
        normed = xf * rstd
        w = tl.load(w_ptr + h_off, mask=mask, other=0.0).to(tl.float32) + OFFSET
        y = (normed * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * hidden + h_off, y, mask=mask)

    def fused_gather_rmsnorm(ids, table, weight, eps=1e-6, offset=1.0):
        """Gather ``table[ids]`` and apply Qwen3.5 RMSNorm (gemma casting, ``1+weight``)
        in one kernel.

        ids: [n_tokens] int (any shape, flattened); table: [vocab, hidden]; weight:
        [hidden]. Returns [n_tokens, hidden] in ``table.dtype``. Inputs are made
        contiguous defensively (production buffers already are)."""
        assert table.is_cuda
        assert ids.is_cuda
        assert table.ndim == 2
        if not ids.is_contiguous():
            ids = ids.contiguous()
        ids = ids.view(-1)
        if not table.is_contiguous():
            table = table.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        n_tokens, hidden = ids.numel(), table.shape[1]
        out = torch.empty((n_tokens, hidden), device=table.device, dtype=table.dtype)
        BLOCK_H = triton.next_power_of_2(hidden)
        _fused_gather_rmsnorm_kernel[(n_tokens,)](
            ids, table, weight, out, hidden, eps, OFFSET=float(offset), BLOCK_H=BLOCK_H, num_warps=8
        )
        return out

    return fused_gather_rmsnorm


def _self_test(fused_gather_rmsnorm) -> None:
    """Live-GPU numeric self-test vs the EXACT eager ``Qwen3_5RMSNorm`` math
    (fp32 var, fp32 ``*(1+w)``, cast last). Raises on mismatch so the caller keeps the
    Liger/eager path."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    dev, vocab, hidden, eps = "cuda", 4096, 2560, 1e-6
    table = torch.randn(vocab, hidden, device=dev, dtype=torch.bfloat16)
    # weight is zeros-init in Qwen3.5; use a small perturbation (effective 1+w).
    weight = 0.02 * torch.randn(hidden, device=dev, dtype=torch.bfloat16)
    for n in (256, 2048):
        ids = torch.randint(0, vocab, (n,), device=dev)
        emb = F.embedding(ids, table)
        xf = emb.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        ref = (normed * (1.0 + weight.float())).to(emb.dtype).float()
        got = fused_gather_rmsnorm(ids, table, weight, eps=eps, offset=1.0).float()
        rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        if not (rel < 2e-2):
            raise RuntimeError(f"triton_embedding self-test failed at n={n}: rel={rel:.2e}")


def load_fused_embedding():
    """Return ``fused_gather_rmsnorm`` if the kernel builds and passes its live-GPU
    self-test; otherwise return ``None`` (keep the Liger/eager path). Never raises —
    any failure (no torch/triton, no CUDA, compile/self-test error) -> ``None``.

    Deliberately NOT routed through ``load_entry``, unlike the other fused ops: this kernel is
    gather+layer-0-RMSNorm, ``fn(ids, table, weight, eps=, offset=)``, while the ``embedding``
    contract op an overlay author writes to is ``embedding_fn(input_ids, weight)``. A
    contract-shaped overlay is rejected by ``_self_test`` above with "too many positional
    arguments", so wiring it would open an adoption path no author can satisfy: the same
    signature mismatch that made the LoRA overlay unshippable. ``baselines`` records the same
    fact by listing embedding as deliberately unregistered."""
    from chalk.ops.arch import load_kernel

    return load_kernel(
        "embed",
        "fused Triton gather+RMSNorm kernel enabled",
        "fused Triton embedding kernel disabled",
        build=_build_kernel,
        self_test=_self_test,
    )


def _resolve_text_model(model):
    """Return the inner Qwen3.5 text model (the one that owns ``embed_tokens`` + the
    decoder ``layers``), or None if the structure does not match. Handles the
    ForConditionalGeneration (``model.model.language_model``), ForCausalLM
    (``model.model``), and bare-TextModel layouts, and unwraps a PEFT ``PeftModel``
    (``trainer.model`` is the PEFT wrapper during training) via ``get_base_model()``."""
    roots = [model]
    # PEFT wrapper: the real HF model is under get_base_model() (the bare LoraModel
    # walk does NOT expose embed_tokens/layers at the expected paths).
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        with contextlib.suppress(Exception):
            roots.append(getter())
    for root in roots:
        for path in (
            ("model", "language_model"),
            ("model",),
            (),
        ):
            obj = root
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and hasattr(obj, "embed_tokens") and hasattr(obj, "layers"):
                return obj
    return None


def install_qwen35_fused_embedding(model) -> bool:
    """Wire the fused gather + layer-0-RMSNorm kernel onto a loaded Qwen3.5 model — IFF
    the live-GPU self-test passes. Install-on-call: calling this IS the opt-in (the Liger
    model); there is no env flag.

    Patches ONLY layer-0's ``input_layernorm`` (the single RMSNorm fed by the embedding):
    a stashing ``embed_tokens.forward`` records the ``input_ids`` of the current step, and
    the patched ``input_layernorm`` re-runs the gather fused with the RMSNorm so the layer-0
    norm avoids the embedding HBM round-trip. The raw embedding (needed for the residual)
    is still produced by the normal ``embed_tokens`` gather, so the residual path is exact.

    SAFE NO-OP CONDITIONS (any -> return False, leave the model untouched):
      * kernel disabled / build / self-test failure;
      * model structure (text model / embed_tokens / layers / layer-0 input_layernorm)
        does not match;
      * the layer-0 RMSNorm offset/casting does not match what the kernel implements
        (Liger present but NOT patched with offset=1.0/gemma, OR an unexpected weight
        shape) — verified by a per-instance numeric check against the module's OWN
        forward before swapping it in.

    CORRECTNESS GATE: the fused path is taken only when no backward graph needs to flow
    through the layer-0 norm (grad disabled, or both the embedding table and the norm
    weight are frozen) — the production LoRA recipe freezes both. Otherwise it falls back
    to the original ``input_layernorm`` forward. Never raises: any failure keeps the
    original path. Returns True iff the patch was installed."""
    fn = load_fused_embedding()
    if fn is None:
        return False
    try:
        import torch

        tm = _resolve_text_model(model)
        if tm is None:
            print("[embed] no matching Qwen3.5 text model (embed_tokens/layers); keeping baseline", flush=True)
            return False
        embed = tm.embed_tokens
        layers = tm.layers
        if not hasattr(embed, "weight") or len(layers) == 0:
            print("[embed] embedding/layers shape mismatch; keeping baseline", flush=True)
            return False
        layer0 = layers[0]
        ln = getattr(layer0, "input_layernorm", None)
        if ln is None or not hasattr(ln, "weight"):
            print("[embed] no layer-0 input_layernorm; keeping baseline", flush=True)
            return False
        table = embed.weight
        if table.ndim != 2 or ln.weight.numel() != table.shape[1]:
            print("[embed] embed/norm dim mismatch; keeping baseline", flush=True)
            return False
        if getattr(ln, "_chalk_embed_patched", False):
            return True

        # Resolve eps + offset. Qwen3.5 norms carry offset=1.0; Liger sets ``offset`` /
        # ``variance_epsilon`` on the instance, eager has ``eps``. If Liger patched it with
        # a NON-1.0 offset or a non-gemma casting mode, we cannot match it -> bail.
        eps = float(getattr(ln, "variance_epsilon", None) or getattr(ln, "eps", 1e-6))
        offset = float(getattr(ln, "offset", 1.0))
        casting = getattr(ln, "casting_mode", "gemma")
        if casting not in ("gemma", None) or offset != 1.0:
            print(
                f"[embed] layer-0 norm casting/offset unsupported (casting={casting}, offset={offset}); keeping baseline",
                flush=True,
            )
            return False

        _orig_ln_forward = ln.forward

        # Per-instance numeric check: the fused gather+RMSNorm must match the layer-0
        # norm's OWN forward (Liger or eager) applied to the real embedding, on the real
        # table/weight. This catches any semantic drift (offset/casting/eps) that the
        # attribute check above missed, on the ACTUAL module — before we swap it in.
        try:
            with torch.no_grad():
                n = 64
                dev = table.device
                ids_chk = torch.randint(0, table.shape[0], (n,), device=dev)
                emb_chk = torch.nn.functional.embedding(ids_chk, table)
                ref = _orig_ln_forward(emb_chk).float()
                got = fn(ids_chk, table, ln.weight, eps=eps, offset=offset).float()
                # Sync before reading the result. This per-instance check runs from
                # ``install_qwen35_fused_embedding`` during ``apply_chalk_kernel_to_qwen35`` with the
                # full Qwen3.5 model resident; under the default async stream the fused Triton kernel
                # can still be in flight when ``.item()`` reads ``got``, yielding a garbage rel
                # (~1e9) that spuriously DISABLES a correct kernel. Force completion so the gate
                # measures the real output.
                if dev.type == "cuda":
                    # Sync the device the tensors are ON (``dev`` from ``table.device``), not the
                    # current device: no-arg ``synchronize()`` only flushes the current device, so
                    # on a multi-GPU run where ``table`` lives on a non-current GPU the fused
                    # kernel could still be in flight when ``.item()`` reads ``got`` — spuriously
                    # failing the numeric gate and disabling a correct kernel.
                    torch.cuda.synchronize(dev)
                rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            if not (rel < 2e-2):
                print(f"[embed] per-instance norm check failed (rel={rel:.2e}); keeping baseline", flush=True)
                return False
        except Exception as e:
            print(f"[embed] per-instance norm check errored ({type(e).__name__}: {e}); keeping baseline", flush=True)
            return False

        # Stash the current step's input_ids on the text model so the patched layer-0 norm
        # can re-gather fused. embed_tokens still produces the raw embedding (residual).
        _orig_embed_forward = embed.forward

        def _embed_forward(input_ids, *args, **kwargs):
            try:
                tm._chalk_last_ids = input_ids
            except Exception:
                tm._chalk_last_ids = None
            return _orig_embed_forward(input_ids, *args, **kwargs)

        embed_table = table  # capture (tied weight, frozen)

        def _ln_forward(hidden_states, *args, **kwargs):
            # SINGLE-USE STASH: consume the recorded ids unconditionally at entry (read into
            # a local, immediately clear it on the model). The stash is only valid for the
            # ONE layer-0 norm call that directly follows THIS step's embed_tokens gather; by
            # clearing it before we branch, a forward that supplies inputs_embeds directly
            # (so embed_tokens never ran and the stash is stale from a prior step) can never
            # re-gather table[stale_ids] in place of the real hidden_states, even if its
            # token count coincidentally matches.
            ids = getattr(tm, "_chalk_last_ids", None)
            tm._chalk_last_ids = None
            # Take the fused path only when: we recorded this step's ids; the incoming
            # hidden_states are the raw embedding (same shape as a fresh gather) AND no
            # backward needs to flow through this norm (grad disabled, or both the table
            # and the norm weight are frozen). Anything else -> original differentiable
            # forward (exact, never drops gradients).
            try:
                if (
                    ids is not None
                    and isinstance(ids, torch.Tensor)
                    and ids.is_cuda
                    and hidden_states.ndim >= 2
                    and hidden_states.shape[-1] == embed_table.shape[1]
                    and hidden_states.numel() // hidden_states.shape[-1] == ids.numel()
                    and (not torch.is_grad_enabled() or (not embed_table.requires_grad and not ln.weight.requires_grad))
                ):
                    out = fn(ids, embed_table, ln.weight, eps=eps, offset=offset)
                    return out.reshape(hidden_states.shape)
            except Exception:
                pass  # any runtime hiccup -> exact original path below
            return _orig_ln_forward(hidden_states, *args, **kwargs)

        embed.forward = _embed_forward
        ln.forward = _ln_forward
        ln._chalk_embed_patched = True
        print(
            "[embed] fused Triton gather+RMSNorm installed on layer-0 input_layernorm "
            "(forward-only, frozen-embed path)",
            flush=True,
        )
        return True
    except Exception as e:  # pragma: no cover - defensive
        print(f"[embed] install failed ({type(e).__name__}: {e}); keeping baseline", flush=True)
        return False


if __name__ == "__main__":  # manual self-test / smoke
    fn = load_fused_embedding()
    print("fused gather+rmsnorm loaded:", fn is not None)
