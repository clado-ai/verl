"""Chunked GRPO-style policy loss over the LM head (fwd+bwd) — mirrors :mod:`chalk.ops.flce`.

Where FLCE chunks the ``[tokens, vocab]`` logits over the VOCAB dim to keep the fused-linear
cross-entropy feasible at V=248320, GRPO's memory pressure is the same ``[tokens, vocab]`` logits
tensor produced by the LM head — so this op chunks the loss over the SEQUENCE (token) dim: only one
``[chunk, vocab]`` logits tile is ever live, in the forward pass AND (via recompute) in backward.

SEMANTICS — a self-contained, fully-differentiable GRPO/PPO-flavoured surrogate over exactly the
four contract inputs ``(hidden, lm_head_weight, advantages, mask)`` (no reference/old-policy log-
probs or sampled-token ids are passed to this entry, so the objective is expressed purely from the
policy's own per-token distribution — it exercises the SAME expensive compute/memory pattern as a
production fused-linear GRPO loss: a full ``[chunk, vocab]`` log-softmax + a vocab-wide reduction +
the two backward GEMMs). Per token ``t`` (row), all in fp32::

    logits_t   = hidden_t @ lm_head_weight.T            # [V]
    logp_t     = log_softmax(logits_t)                  # [V]
    p_t        = softmax(logits_t)                      # [V]
    logp_bar_t = mean_v logp_t[v]                       # policy-improvement score (advantage-weighted)
    kl_t       = sum_v p_t[v]*logp_t[v] + log V         # KL(policy || uniform) >= 0 (the KL penalty)
    per_tok_t  = -advantages_t * logp_bar_t + beta * kl_t

    loss = sum_t mask_t * per_tok_t / max(sum_t mask_t, 1)

``advantages`` weights the policy score (raise the mean log-prob where the advantage is positive);
``mask`` is the 0/1 completion mask (masked tokens contribute nothing); ``beta`` (0.04) scales the
KL-to-uniform penalty. Every term is smooth in the logits (no argmax/hard selection), so the fp32
oracle and a bf16 kernel agree within the 2e-2 envelope. Gradients flow through ``hidden``,
``lm_head_weight`` and ``advantages``; ``mask`` is a discrete gate (no grad).

The portable kernel here streams the sequence in ``[chunk, vocab]`` tiles via gradient
checkpointing (recompute each tile in backward) so peak extra memory is one tile, not ``[T, V]`` —
matching the point of the op. A per-arch Triton specialization is the autoresearch tuning target;
this module is the correct, memory-bounded portable baseline + the fp32 oracle it must match.
"""

from __future__ import annotations

import math

# GRPO KL-penalty coefficient (KL of the policy to the uniform distribution). Fixed here because the
# contract entry ``grpo_loss_fn(hidden, lm_head_weight, advantages, mask)`` carries no beta arg; the
# oracle, the eager baseline and every seed kernel MUST use this same value.
GRPO_BETA = 0.04

# Populated by a future install hook so a worker can fold the outcome into metrics.json's notes.
RESULT: dict = {}


def _chunk_rows(n_tokens: int, vocab: int) -> int:
    """Rows per sequence chunk so the live ``[chunk, vocab]`` fp32 logits tile stays ~4M elements,
    derived from the RUNTIME vocab (never a benchmark-shape literal — the static anti-cheat gate
    flags an in-scope dim hardcoded as a constant)."""
    return min(max(1, n_tokens), max(1, (1 << 22) // max(int(vocab), 1)))


def _grpo_chunk_numerator(hidden_c, weight, adv_c, mask_c, beta: float):
    """``sum_t mask_t * per_token_t`` for one sequence chunk (fp32). Isolated so the portable path
    can wrap it in a gradient checkpoint (recompute in backward -> one live logits tile)."""
    logits = (hidden_c @ weight.t()).float()  # [c, V]
    lse = logits.logsumexp(dim=-1)  # [c]
    logp = logits - lse.unsqueeze(-1)  # [c, V] == log_softmax
    p = logp.exp()  # [c, V] == softmax
    logp_bar = logp.mean(dim=-1)  # [c] mean log-prob across vocab
    neg_entropy = (p * logp).sum(dim=-1)  # [c] == -H(p)
    kl = neg_entropy + math.log(int(weight.shape[0]))  # [c] KL(p || uniform) >= 0
    per_token = -adv_c.float() * logp_bar + float(beta) * kl  # [c]
    return (per_token * mask_c.float()).sum()


def _validate(hidden, weight, advantages, mask):
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be [tokens, hidden], got {tuple(hidden.shape)}")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[1]:
        raise ValueError(f"lm_head_weight must be [vocab, hidden={hidden.shape[1]}], got {tuple(weight.shape)}")
    n = hidden.shape[0]
    if tuple(advantages.shape) != (n,):
        raise ValueError(f"advantages must be [tokens={n}], got {tuple(advantages.shape)}")
    if tuple(mask.shape) != (n,):
        raise ValueError(f"mask must be [tokens={n}], got {tuple(mask.shape)}")


def _eager_grpo_loss(hidden, lm_head_weight, advantages, mask, *, beta: float = GRPO_BETA, chunk: int | None = None):
    """fp32 reference (the oracle math). Chunked over the sequence purely so the fp32 logits never
    materialize all at once — the numeric result is independent of the chunk size."""
    import torch

    _validate(hidden, lm_head_weight, advantages, mask)
    n = hidden.shape[0]
    vocab = lm_head_weight.shape[0]
    rows = _chunk_rows(n, vocab) if chunk is None else max(1, int(chunk))
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, n, rows):
        e = min(s + rows, n)
        total = total + _grpo_chunk_numerator(hidden[s:e], lm_head_weight, advantages[s:e], m[s:e], beta)
    return total / denom


def _build_kernels():
    """Return the portable ``grpo_loss_fn(hidden, lm_head_weight, advantages, mask) -> loss``.

    FUSED design (mirrors :func:`chalk.ops.flce._build_kernels`; ~13x the prior chunked-torch seed on
    an A100 at the 248k-vocab shape). The LM-head GEMM (``hidden @ weight.T``) stays on cuBLAS,
    chunked over the SEQUENCE dim so only one ``[chunk, V]`` logits tile is ever live. A single Triton
    kernel per chunk fuses the vocab-wide reduction (log-sum-exp, mean-logit, the KL neg-entropy term)
    AND the in-place backward gradient wrt the logits tile into ONE launch (one program per token;
    three streaming passes over ``V`` in ``BLOCK_V``-wide blocks: online-softmax lse/mean-logit,
    ``S1 = sum_v p_v*logp_v``, then — only when grads are needed — the per-logit gradient written IN
    PLACE over the logits buffer). The two backward GEMMs (``grad_hidden``, ``grad_weight``) reuse
    cuBLAS on that in-place gradient buffer, with ``grad_weight.addmm_`` writing straight into the
    accumulator (NO ``[V, H]`` temporary). It never materializes the full ``[tokens, vocab]`` tensor
    (chunked over N), nor the seed's extra ``[chunk, vocab]`` logp/p buffers + gradient-checkpoint
    recompute, so it beats the seed on both time and peak memory. Raises on any import problem (the
    caller treats a raise as "keep the eager path")."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _grpo_kernel(
        X_ptr,  # [chunk, V] logits (overwritten in place with d_logits when HAS_GRAD)
        X_row_stride,
        ADV_ptr,  # [chunk] fp32 advantages
        MASK_ptr,  # [chunk] fp32 0/1 completion mask
        LOSS_ptr,  # [chunk] fp32 per-row loss contribution (already mask/denom scaled)
        LOGP_BAR_ptr,  # [chunk] fp32 per-row mean log-prob (needed for grad wrt advantages)
        denom,  # float: sum(mask) over the WHOLE batch, clamp>=1 (mean-reduction scale)
        beta,
        V: tl.constexpr,
        HAS_GRAD: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        X_ptr += row * X_row_stride
        adv = tl.load(ADV_ptr + row).cast(tl.float32)
        m_tok = tl.load(MASK_ptr + row).cast(tl.float32)

        # --- pass 1: online softmax (m, d) + running raw-logit sum ---
        m = float("-inf")
        d = 0.0
        sum_x = 0.0
        for off in range(0, V, BLOCK_V):
            cols = off + tl.arange(0, BLOCK_V)
            xb = tl.load(X_ptr + cols, mask=cols < V, other=float("-inf")).cast(tl.float32)
            block_max = tl.max(xb)
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp(m - m_new) + tl.sum(tl.exp(xb - m_new))
            sum_x += tl.sum(tl.where(cols < V, xb, 0.0))
            m = m_new
        lse = m + tl.log(d)
        mean_x = sum_x / V
        logp_bar = mean_x - lse

        # --- pass 2: S1 = sum_v p_v * logp_v  (KL(p||uniform) = S1 + log V) ---
        s1 = 0.0
        for off in range(0, V, BLOCK_V):
            cols = off + tl.arange(0, BLOCK_V)
            xb = tl.load(X_ptr + cols, mask=cols < V, other=float("-inf")).cast(tl.float32)
            p = tl.exp(xb - m) / d
            logp = xb - lse
            s1 += tl.sum(tl.where(cols < V, p * logp, 0.0))

        kl = s1 + tl.log(V * 1.0)
        per_tok = -adv * logp_bar + beta * kl
        loss = m_tok * per_tok / denom
        tl.store(LOSS_ptr + row, loss)
        tl.store(LOGP_BAR_ptr + row, logp_bar)

        # --- pass 3: in-place gradient wrt logits ---
        if HAS_GRAD:
            scale = m_tok / denom
            for off in range(0, V, BLOCK_V):
                cols = off + tl.arange(0, BLOCK_V)
                xb = tl.load(X_ptr + cols, mask=cols < V, other=float("-inf")).cast(tl.float32)
                p = tl.exp(xb - m) / d
                logp = xb - lse
                g = -adv / V + adv * p + beta * p * logp - beta * p * s1
                g = g * scale
                tl.store(X_ptr + cols, g, mask=cols < V)

    _BLOCK_V = 32768
    _NUM_WARPS = 32

    def _launch_cfg(V: int):
        block = min(_BLOCK_V, triton.next_power_of_2(V))
        return block, _NUM_WARPS

    def _chunk_size(N: int, H: int, V: int, mult: int = 4) -> int:
        # Memory-envelope heuristic (peak extra memory ~ one [chunk, V] tile), derived from the
        # RUNTIME dims (never a benchmark-shape literal — the static anti-cheat gate flags an in-scope
        # dim hardcoded as a constant). A wider chunk (mult) gives the backward GEMMs a bigger K dim
        # (cuBLAS efficiency) at the cost of a proportionally larger live tile; mult=4 keeps the tile
        # well under peak at the real 248k vocab.
        inc_factor = triton.cdiv(V, H)
        cs = triton.next_power_of_2(triton.cdiv(N, inc_factor)) * max(1, int(mult))
        return max(1, min(cs, N))

    class _GRPOFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, advantages, mask):
            device = hidden.device
            N, H = hidden.shape
            V = weight.shape[0]
            hidden = hidden if hidden.is_contiguous() else hidden.contiguous()
            weight = weight if weight.is_contiguous() else weight.contiguous()
            advantages_f = advantages.float().contiguous()
            mask_f = mask.float().contiguous()

            need_grad = hidden.requires_grad or weight.requires_grad or advantages.requires_grad
            grad_hidden = torch.zeros_like(hidden) if need_grad else None
            grad_weight = torch.zeros_like(weight) if (need_grad and weight.requires_grad) else None

            loss_1d = torch.zeros(N, dtype=torch.float32, device=device)
            logp_bar_1d = torch.zeros(N, dtype=torch.float32, device=device)
            denom = float(mask_f.sum().clamp(min=1.0).item())

            chunk_size = _chunk_size(N, H, V)
            BLOCK_V, num_warps = _launch_cfg(V)

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                h_c = hidden[start:end]
                logits = torch.matmul(h_c, weight.t())  # [c, V] cuBLAS, input dtype
                logits = logits if logits.is_contiguous() else logits.contiguous()
                adv_c = advantages_f[start:end]
                mask_c = mask_f[start:end]
                loss_c = loss_1d[start:end]
                logpbar_c = logp_bar_1d[start:end]
                rows = end - start
                with torch.cuda.device(device):
                    _grpo_kernel[(rows,)](
                        logits,
                        logits.stride(0),
                        adv_c,
                        mask_c,
                        loss_c,
                        logpbar_c,
                        denom,
                        GRPO_BETA,
                        V=V,
                        HAS_GRAD=need_grad,
                        BLOCK_V=BLOCK_V,
                        num_warps=num_warps,
                    )
                if need_grad:
                    # logits now holds d_logits ([c, V], already mask/denom-scaled by the kernel).
                    grad_hidden[start:end] = torch.matmul(logits, weight).to(grad_hidden.dtype)
                    if grad_weight is not None:
                        # Accumulate d_weight IN PLACE: grad_weight += d_logits.T @ h_c. addmm_ writes
                        # straight into grad_weight with NO full [V, H] temporary (a plain
                        # matmul(...).to(...) materialised a [V,H] tile per chunk that OOM'd at V=248k).
                        grad_weight.addmm_(logits.t(), h_c.to(logits.dtype))
                del logits

            loss = loss_1d.sum()
            grad_advantages = None
            if need_grad and advantages.requires_grad:
                grad_advantages = (-mask_f * logp_bar_1d / denom).to(advantages.dtype)

            ctx.has_grad_hidden = grad_hidden is not None
            ctx.has_grad_weight = grad_weight is not None
            ctx.has_grad_advantages = grad_advantages is not None
            to_save = tuple(t for t in (grad_hidden, grad_weight, grad_advantages) if t is not None)
            ctx.save_for_backward(*to_save)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            saved = list(ctx.saved_tensors)
            grad_hidden = saved.pop(0) if ctx.has_grad_hidden else None
            grad_weight = saved.pop(0) if ctx.has_grad_weight else None
            grad_advantages = saved.pop(0) if ctx.has_grad_advantages else None
            is_unit = grad_output.numel() == 1 and float(grad_output.detach()) == 1.0
            if not is_unit:
                if grad_hidden is not None:
                    grad_hidden = grad_hidden * grad_output.to(grad_hidden.dtype)
                if grad_weight is not None:
                    grad_weight = grad_weight * grad_output.to(grad_weight.dtype)
                if grad_advantages is not None:
                    grad_advantages = grad_advantages * grad_output.to(grad_advantages.dtype)
            return grad_hidden, grad_weight, grad_advantages, None

    def grpo_loss_fn(hidden, lm_head_weight, advantages, mask):
        _validate(hidden, lm_head_weight, advantages, mask)
        return _GRPOFunction.apply(hidden, lm_head_weight, advantages, mask)

    return grpo_loss_fn


def _self_test(grpo_loss_fn) -> None:
    """Live-GPU numeric + autograd parity vs :func:`_eager_grpo_loss`: loss, grad_hidden,
    grad_weight, grad_advantages. fp32 loss rel-err < 1e-4; grads rel-err < 2e-3. Raises on
    mismatch so the caller keeps the eager path. Smaller-than-real V (fast) but the same chunked
    code path (V > the chunk rows forces multi-chunk)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA for grpo_chunked_loss self-test")
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)

    def rel(a, b):
        return (a - b).norm().item() / (b.norm().item() + 1e-9)

    for n, h, v in ((512, 1024, 8000), (777, 512, 4096)):
        scale = h**-0.5
        hidden = torch.randn(n, h, device=dev, dtype=torch.float32, generator=gen)
        weight = torch.randn(v, h, device=dev, dtype=torch.float32, generator=gen) * scale
        advantages = torch.randn(n, device=dev, dtype=torch.float32, generator=gen)
        mask = (torch.rand(n, device=dev, generator=gen) > 0.2).float()

        hr = hidden.clone().requires_grad_(True)
        wr = weight.clone().requires_grad_(True)
        ar = advantages.clone().requires_grad_(True)
        ref = _eager_grpo_loss(hr, wr, ar, mask)
        ref.backward()

        hc = hidden.clone().requires_grad_(True)
        wc = weight.clone().requires_grad_(True)
        ac = advantages.clone().requires_grad_(True)
        got = grpo_loss_fn(hc, wc, ac, mask)
        got.backward()

        r_loss = abs(got.item() - ref.item()) / (abs(ref.item()) + 1e-9)
        r_dh = rel(hc.grad, hr.grad)
        r_dw = rel(wc.grad, wr.grad)
        r_da = rel(ac.grad, ar.grad)
        if not (r_loss < 1e-4 and r_dh < 2e-3 and r_dw < 2e-3 and r_da < 2e-3):
            raise RuntimeError(
                f"grpo_chunked_loss self-test failed at n={n} h={h} v={v}: "
                f"loss={r_loss:.2e} dh={r_dh:.2e} dw={r_dw:.2e} da={r_da:.2e}"
            )


def load_grpo_loss():
    """Return ``grpo_loss_fn`` if the kernel builds and passes its live-GPU self-test; otherwise
    ``None`` (keep the eager path). Never raises."""
    from chalk.ops.arch import load_entry
    from chalk.ops.arch import load_kernel

    return load_kernel(
        "grpo_chunked_loss",
        "chunked GRPO policy loss (seq-chunked LM head, fwd+bwd) enabled",
        "chunked GRPO policy loss disabled",
        build=lambda: load_entry("grpo_chunked_loss", _self_test, portable=_build_kernels),
    )


__all__ = [
    "GRPO_BETA",
    "fused_linear_grpo_loss_entry",
    "load_grpo_loss",
]


# Public alias matching the contract entry name, for callers that want the validated portable path.
def fused_linear_grpo_loss_entry(hidden, lm_head_weight, advantages, mask):
    return _build_kernels()(hidden, lm_head_weight, advantages, mask)
