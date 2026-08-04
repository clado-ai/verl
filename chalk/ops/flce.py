"""Custom chunked Fused-Linear-Cross-Entropy kernel (LM-head + CE, fwd+bwd) for Qwen3.5/3.6 —
matches-or-beats Liger's ``fused_linear_cross_entropy`` on speed AND peak memory.

This is the hardest, highest-value layer: the LM head projects the hidden states to a vocab of
**V = 248320** for Qwen3.5/3.6 (a multimodal vocab — NOT ~152k), so a naive
``logits = hidden @ lm_head_W.T`` materializes an ``[N_tokens, V]`` tensor that is multiple GB and
dominates training memory. Liger's ``LigerFusedLinearCrossEntropyFunction`` avoids it by CHUNKING
over the token dim and never materializing the full logits; this module is chalk's own version so
chalk can stand ALONE (drop the Liger dependency for this layer) at speed/memory/quality parity.

ALGORITHM (matched to Liger ``fused_linear_cross_entropy_forward``):
  Split the ``N`` tokens into chunks of ``chunk_size`` rows (``chunk_size`` chosen so the live
  logits tile ``[chunk, V]`` is ~ ``[N, H]`` scale, i.e. ``inc_factor = ceil(V/H)`` and
  ``chunk_size = next_pow2(ceil(N / inc_factor))``). Per chunk:
    1. ``logits_c = hidden_c @ W.T``  (cuBLAS ``torch.matmul``, input precision) -> ``[chunk, V]``
       (ONLY ONE chunk live at a time -> peak extra mem ~ ``[chunk, V]`` not ``[N, V]``).
    2. A Triton kernel (``_ce_kernel``, ONE program per row) computes, in fp32:
         * online-softmax ``m`` (row max) + ``d`` (sum exp) over V in ``BLOCK_V`` blocks,
         * ``lse = m + log d``; per-row loss ``= lse - x_y`` (+ label-smoothing term),
         * the gradient wrt logits IN PLACE into the same buffer:
             ``d_logit_i = (softmax_i - eps)``,  ``d_logit_y -= (1 - label_smoothing)``,
           all divided by ``n_non_ignore`` for ``reduction="mean"`` (``eps = label_smoothing/V``).
       Reuses the logits buffer for the gradient -> no second ``[chunk, V]`` allocation.
    3. ``grad_hidden_c = d_logits @ W``;  ``grad_W += d_logits.T @ hidden_c``  (fp32 accumulated
       across chunks).  Free the chunk.
  Loss ``= sum(per-row losses)`` (the kernel already divided each by ``n_non_ignore`` for "mean").

SEMANTICS: matches ``F.cross_entropy(F.linear(hidden, weight), labels, ignore_index=...,
reduction="mean"|"sum", label_smoothing=...)``. ``ignore_index`` rows contribute zero loss and
zero gradient. The math (loss, ``grad_hidden``, ``grad_weight``) is verified vs a naive
``F.linear`` + ``F.cross_entropy`` reference (fp32 rel-err < 1e-2 on loss; bf16 envelope on grads).

WHY IT CAN BEAT LIGER (the levers): same chunked-GEMM structure (cuBLAS is unbeatable on the dense
``[chunk,V]`` matmul), so the win is in (a) the CE Triton kernel's launch config — Liger hardwires
``num_warps=32`` and ``BLOCK_SIZE=min(32768, next_pow2(V))``; chalk tunes both to the A40 — and (b)
the chunk size (memory/occupancy tradeoff). At parity of GEMM cost the kernels are close; chalk
targets matching-or-beating wall time at <= Liger's peak memory. Be HONEST in the bench about which.

Install-on-call (the Liger model): ``install_qwen35_flce(model)`` swaps the loss path of the live
``Qwen3_5ForCausalLM`` (binds a ``forward`` that runs the model body, then routes the LM-head + CE
through this kernel instead of materializing logits) — mirroring how Liger patches
``Qwen3_5ForCausalLM.forward`` with its ``lce_forward``. Self-test gated; any build/self-test
failure leaves the model's eager/Liger forward untouched.
"""

from __future__ import annotations

# Populated by install_qwen35_flce so a worker can fold the outcome into metrics.json's notes.
# Empty {} means the kernel was not engaged this run.
RESULT: dict = {}

_DEFAULT_CE_CFG = {"BLOCK_V": 32768, "num_warps": 32, "chunk_mult": 16}

_TUNING_ARCHES = (
    {
        "name": "Blackwell",
        "capabilities": ("sm_100", "sm_120"),
        "chunk_mult": 16,
        "note": "B200/sm_100 A/B (2026-07): 16 confirmed as the memory/speed knee (self_test passes; "
        "chunk_mult=16 keeps flce's 2-chunk memory optimization). mult=32 is 1.02-1.12x faster (1.12x "
        "@tok=2048 -> ~1.02x @16384) but for V=248320 that drives chunk==N -> a single chunk == full "
        "[N,V] logit materialization (~2x the [chunk,V] transient), so 16 is retained -- see the "
        "SUPERSEDED block in _build_kernels. sm_120 is a non-target: triton miscompiles on consumer "
        "Blackwell, so flce self_test -> eager there regardless.",
    },
    {
        "name": "Hopper",
        "capabilities": ("sm_90",),
        "chunk_mult": 16,
        "note": "H100 sm_90 A/B (2026-07): chunk_mult=16 beats 8 at EVERY training tok (1.23x@2048 -> "
        "1.04x@16384, both runs, no reversals; correct rel~1e-7). The prior 'mult=8 wins' note only "
        "benchmarked chunk_mult<=6, never 16.",
    },
    {
        "name": "Ampere",
        "capabilities": ("sm_80", "sm_86"),
        "chunk_mult": 16,
        "note": "A100 sm_80 A/B (2026-07): chunk_mult=16 beats 8 at every tok 1024-16384 (1.25x@1024 -> "
        "tie@16384, monotonic; correct rel<2.3e-3). The old 'small-token decision shape' note that "
        "justified 8 was contradicted head-on: the win is LARGEST at small tokens.",
    },
)


def flce_tuning_metadata() -> dict:
    """CPU-safe description of the measured FLCE launch policy.

    This is intentionally metadata only: changing FLCE's actual runtime knobs still requires GPU
    A/B through scripts/bench_all.py. Keeping the measured arch policy structured makes benchmark JSON
    and CI assertions agree on why FLCE currently uses one config across Blackwell/Hopper/Ampere.
    """
    return {
        "ce": dict(_DEFAULT_CE_CFG),
        "accumulator": {"dtype": "weight"},
        "requires_per_arch_split": False,
        "architectures": [
            {
                "name": arch["name"],
                "capabilities": list(arch["capabilities"]),
                "chunk_mult": arch["chunk_mult"],
                "note": arch["note"],
            }
            for arch in _TUNING_ARCHES
        ],
        "benchmark": {
            "command": "python scripts/bench_all.py --kernels flce_sweep --baselines eager,liger --json out.json",
            "rows": "results[].{vocab,tok,chunk_mult,chalk_ms,chalk_peak_gb}",
        },
    }


def _build_kernels():
    """Import torch/triton and define the per-chunk CE Triton kernel + the chunked
    ``autograd.Function``. Returns ``flce_fn`` with signature
    ``(hidden, weight, labels, ignore_index=-100, reduction="mean", label_smoothing=0.0) -> loss``
    or raises on any import/compile problem (the caller treats a raise as "keep eager/Liger")."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _ce_kernel(
        X_ptr,  # [chunk, V] logits (overwritten in place with d_logits when HAS_GRAD)
        X_row_stride,
        Y_ptr,  # [chunk] int64 labels
        loss_ptr,  # [chunk] fp32 per-row loss
        n_non_ignore,  # float: number of non-ignored rows in the WHOLE batch (mean scale)
        ignore_index,
        V: tl.constexpr,  # vocab size. constexpr: the ``range(0, V, BLOCK_V)`` loops below are
        # Python ``range()`` over the bound, so V MUST be compile-time (Triton can't unroll a
        # runtime ``range()``). Vocab is fixed per model, so Triton JIT-specializes once per V.
        label_smoothing: tl.constexpr,
        reduction: tl.constexpr,  # "mean" | "sum"
        HAS_GRAD: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        # One program == one row (token). Online softmax over V in BLOCK_V-wide blocks, then a
        # second pass writes the in-place gradient. Mirrors liger_cross_entropy_kernel for the
        # hard-label + (optional) label-smoothing, mean/sum case (no ce_weight/softcap/z-loss).
        row = tl.program_id(0).to(tl.int64)
        y = tl.load(Y_ptr + row)
        X_ptr += row * X_row_stride

        # ``ignore_index`` is handled by PREDICATION, not a Python ``if`` on the runtime label
        # ``y`` (Triton can't branch on a runtime ``tl.load`` value — that ``if`` either fails to
        # JIT or is miscompiled, which would keep FLCE from ever enabling). Every ignored row runs
        # the SAME straight-line code as a normal row, then has its loss + gradient masked to 0:
        #   * ``is_ignored`` is the per-row predicate (one program == one row, so it's a scalar);
        #   * ``y_safe`` clamps the label to 0 for ignored rows so ``tl.load(X_ptr + y)`` and the
        #     ``cols == y`` true-class term stay IN BOUNDS (an ignored label is ``ignore_index``,
        #     e.g. -100, which would otherwise index out of the [0, V) logit row). Column 0 is an
        #     arbitrary safe target — the whole ignored row's grad/loss is zeroed below anyway.
        # This reproduces ``F.cross_entropy(..., ignore_index=...)``: an ignored row contributes
        # EXACTLY 0 loss and 0 gradient, and (see the reduction guard) never feeds NaN/inf into the
        # softmax. Non-ignored rows are bit-for-bit unchanged (``is_ignored`` is False for them).
        is_ignored = y == ignore_index
        y_safe = tl.where(is_ignored, 0, y)

        # --- pass 1: online softmax (max m, sum d) + label-smoothing accumulator ---
        m = float("-inf")
        d = 0.0
        eps = label_smoothing / V
        scaled_x_sum = 0.0
        ori_x_y = tl.load(X_ptr + y_safe).cast(tl.float32)  # logit of the true class (for the loss)
        for off in range(0, V, BLOCK_V):
            cols = off + tl.arange(0, BLOCK_V)
            xb = tl.load(X_ptr + cols, mask=cols < V, other=float("-inf")).cast(tl.float32)
            # Guard the reduction for ignored rows: replace their logits with a finite 0.0 so the
            # row reduces over zeros (``m=0``, ``d`` finite & >0, ``lse=log d``) instead of feeding
            # ``-inf`` into ``m``/``exp`` (which would poison the running max into a NaN). Only the
            # ignored row's OWN m/d/lse are touched (per-row scalars), and its loss/grad are zeroed
            # later, so this finite filler is discarded. Non-ignored rows keep the ``other=-inf``
            # masking of out-of-range cols (so those lanes stay excluded from the reduction).
            xb = tl.where(is_ignored, 0.0, xb)
            block_max = tl.max(xb)
            if label_smoothing > 0:
                # sum over valid cols of (-eps * x_i); masked lanes are -inf so guard with where.
                scaled_x_sum += tl.sum(tl.where(cols < V, -eps * xb, 0.0))
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp(m - m_new) + tl.sum(tl.exp(xb - m_new))
            m = m_new
        lse = m + tl.log(d)

        # --- pass 2: in-place gradient wrt logits (only when we need grads) ---
        if HAS_GRAD:
            for off in range(0, V, BLOCK_V):
                cols = off + tl.arange(0, BLOCK_V)
                xb = tl.load(X_ptr + cols, mask=cols < V, other=float("-inf")).cast(tl.float32)
                xb = tl.where(is_ignored, 0.0, xb)  # same finite guard so ``sm`` stays finite
                sm = tl.exp(xb - m) / d  # softmax_i
                gb = sm - eps  # smoothing: dx_i = softmax_i - eps
                gb = tl.where(cols != y_safe, gb, gb - (1.0 - label_smoothing))  # dx_y extra term
                if reduction == "mean":
                    gb = gb / n_non_ignore
                # PREDICATE the gradient: an ignored row writes exactly 0.0 into every logit column
                # (zero gradient), matching ``ignore_index``. Non-ignored rows write ``gb``.
                gb = tl.where(is_ignored, 0.0, gb)
                tl.store(X_ptr + cols, gb, mask=cols < V)

        # --- loss: lse - x_y, plus the label-smoothing correction, then mean scale ---
        loss = lse - ori_x_y
        if label_smoothing > 0:
            smooth = scaled_x_sum + label_smoothing * lse
            loss = loss * (1.0 - label_smoothing) + smooth
        if reduction == "mean":
            loss = loss / n_non_ignore
        # PREDICATE the loss write: an ignored row stores exactly 0.0 (it must contribute 0 to the
        # summed loss), matching ``ignore_index``. Non-ignored rows store their computed ``loss``.
        loss = tl.where(is_ignored, 0.0, loss)
        tl.store(loss_ptr + row, loss)

    # ------- launch-config policy (the speed lever vs Liger) -------
    # Liger hardwires BLOCK_SIZE = min(32768, next_pow2(V)) and num_warps=32. For Qwen3.5
    # V=248320, next_pow2=262144 so Liger's BLOCK is 32768 (V looped in 8 blocks) at 32 warps.
    # The CE kernel is memory-bound over the [chunk, V] tile (2 streaming passes), so the tunable
    # levers are BLOCK_V (block width over V) and num_warps. Defaults below are tuned on the A40
    # by benchmark_flce.py; override via _CE_CFG for sweeps.
    # ``chunk_mult`` is THE measured speed lever (A40, real Qwen3.5 V=248320): Liger locks
    # its chunk to ~[N,H] memory
    # (``inc_factor=ceil(V/H)``), which under-feeds cuBLAS — the grad_weight GEMM has inner dim
    # K=chunk≈256, far below cuBLAS's efficient range. chalk widens the chunk ``chunk_mult`` x:
    # K grows, the GEMMs get dramatically more efficient, and the live ``[chunk, V]`` tile (the only
    # per-chunk allocation) grows only modestly vs the persistent weight/grad buffers. MEASURED on
    # A40 @ H=4096/N=8192: mult 1->2->4->8 = 1.00x->1.48x->1.97x->2.19x speed at +0.0%/+1.0%/+3.0%/
    # +7.1% peak mem vs Liger. At that OLD ~152k vocab **mult=4 was the A40 knee** (~2x speed at +3%
    # peak; mult=8 +4% mem for 0.2x more speed) — but the CURRENT default is **mult=16** (bumped from 8
    # in #60; see the SUPERSEDED block below for the 8->16 and 16-vs-32 GPU-verified history).
    # num_warps/BLOCK_V did NOT move the needle (the
    # CE Triton kernel is only ~2% of the wall; the shared cuBLAS GEMMs dominate), so they stay at
    # Liger's values. Set ``chunk_mult=1`` for exact Liger memory parity (1.00x speed, 1.000x mem).
    #
    # chunk_mult = 8 (chalk's "winning kernel on every arch"). The original mult=4 was tuned on the
    # A40 at the OLD assumed vocab (~152k). At the REAL Qwen3.5/3.6 vocab (V=248320) mult=4 LOSES the
    # small-token (4096) case on EVERY arch tested — RTX 5090/Blackwell sm_120 (0.78x), H100/Hopper
    # sm_90 (0.74x), A100 sm_80, AND A40 sm_86 (0.79x). A BIGGER chunk feeds the chunked
    # grad-weight/grad-hidden cuBLAS GEMMs efficiently. MEASURED via dev/flce_sweep across the full
    # {V=151936,248320} x {tok=4096,8192} grid on RTX 5090 + H100 (and the V248320/4096 decision shape
    # on A100 + A40): mult 4->6->8 monotonically faster, and mult=8 WINS EVERY shape on speed
    # (5090 1.02-1.25x, H100 1.03-1.39x, A100 1.05-1.34x, A40 1.06x at the loser shape) while staying
    # 2.4-4.7x UNDER eager peak memory (the +chunk tile is negligible vs the persistent [V,H]
    # grad_weight accumulator -- true only while chunk stays small, i.e. mult<=8; see SUPERSEDED).
    # The SAME value wins on all four arches -> no per-arch split needed for FLCE.
    #
    # SUPERSEDED (2026-07): the default is now **mult=16**, not 8. Two GPU-verified updates:
    #   (#60) 8->16: chunk_mult=16 beat 8 at every tok on A100 (sm_80) + H100 (sm_90) at V=248320 --
    #         the mult=8 sweep above only compared chunk_mult<=8, never 16.
    #   (2026-07 B200 sweep) 16-vs-32 on A100 + H100 + B200 (sm_100), V=248320, direct fwd+bwd A/B,
    #         order-reversed, 8 as a losing sanity anchor: mult=32 is 1.02-1.12x faster than 16 (largest
    #         at small tok -- 1.08x/1.07x/1.12x @tok=2048 on A100/H100/B200 -- shrinking to ~1.02x @16384),
    #         saturating at 32 (48 and 64 tie it). BUT for these shapes next_pow2(cdiv(N,inc_factor))
    #         == N/32, so mult>=32 drives chunk==N: a SINGLE chunk == full [N,V] logit materialization,
    #         ~2x the [chunk,V] transient of the 2-chunk mult=16 (at tok=16384 that transient EXCEEDS the
    #         [V,H] grad_weight -- no longer "negligible"). That trades away flce's core memory
    #         optimization for ~1-12% speed, so we KEEP mult=16 as the memory/speed knee. Callers with
    #         memory slack (or small V / small tok, where the full tile is cheap) can opt into the speed
    #         per-call via ``_chunk_mult=32``. (chunk_mult only changes tiling, never the math;
    #         numerically identical to any mult -- self_test passes on all of sm_80/sm_90/sm_100.)
    _CE_CFG = dict(_DEFAULT_CE_CFG)
    # grad_weight accumulator dtype. ``None`` -> accumulate in the weight dtype (bf16), matching
    # Liger's default ``accum_dtype=None`` peak-memory footprint. Set to ``torch.float32`` for a
    # more accurate (but +2 bytes/elt -> +memory) accumulator. A fp32 accumulator would push peak
    # OVER Liger (the persistent [V,H] accumulator doubles) AND was ~18% SLOWER in profiling (the
    # +HBM traffic of the wider in-place add), so bf16 is both the parity-memory and faster choice.
    _ACC_CFG = {"dtype": None}

    def _ce_launch_cfg(V: int):
        block = min(_CE_CFG["BLOCK_V"], triton.next_power_of_2(V))
        return block, _CE_CFG["num_warps"]

    def _chunk_size(N: int, H: int, V: int, mult: int = 1) -> int:
        # Peak extra memory ~ [chunk, V]. Liger's heuristic keeps that ~ [N, H]: inc_factor =
        # ceil(V/H), chunk = next_pow2(ceil(N / inc_factor)). Same formula -> same memory envelope.
        # ``mult`` (>=1) scales the chunk up: bigger chunks amortize launch/kernel overhead and feed
        # cuBLAS larger GEMMs (faster) at the cost of a proportionally larger live [chunk, V] tile.
        inc_factor = triton.cdiv(V, H)
        cs = triton.next_power_of_2(triton.cdiv(N, inc_factor)) * max(1, int(mult))
        return max(1, min(cs, N))

    class _FLCEFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, labels, ignore_index, reduction, label_smoothing, chunk_mult):
            # hidden [N,H], weight [V,H], labels [N] (int64). Compute loss + (in forward, since CE
            # is the last layer) the grads wrt hidden and weight, chunked over N so only one
            # [chunk, V] logits tile is live at a time.
            device = hidden.device
            N, H = hidden.shape
            V = weight.shape[0]
            hidden = hidden if hidden.is_contiguous() else hidden.contiguous()
            weight = weight if weight.is_contiguous() else weight.contiguous()
            labels = labels if labels.is_contiguous() else labels.contiguous()

            need_grad = hidden.requires_grad or weight.requires_grad
            grad_hidden = torch.zeros_like(hidden) if need_grad else None
            # grad_weight accumulator. Match Liger's default memory footprint: accumulate in the
            # WEIGHT dtype (bf16) so the persistent [V,H] accumulator is 2 bytes/elt (Liger's
            # ``accum_dtype=None`` path), not fp32 (which would be +2GB at V=248320/H=4096 and push
            # peak over Liger). The per-chunk [V,H] product is formed in the accumulator dtype and
            # only widened to fp32 when the accumulator IS fp32 (the bf16 path never materializes a
            # fp32 [V,H] temporary — it would be downcast on the add anyway; see the loop below).
            acc_dtype = _ACC_CFG["dtype"] or weight.dtype
            grad_weight = torch.zeros_like(weight, dtype=acc_dtype) if (need_grad and weight.requires_grad) else None

            loss_1d = torch.zeros(N, dtype=torch.float32, device=device)
            n_non_ignore = float((labels != ignore_index).sum().item())
            # all-ignored batch -> zero loss, zero grads (avoid divide-by-zero in the kernel).
            denom = n_non_ignore if n_non_ignore > 0 else 1.0

            # resolve the chunk multiplier (per-call override, else the tuned module default).
            cm = _CE_CFG["chunk_mult"] if chunk_mult is None else chunk_mult
            chunk_size = _chunk_size(N, H, V, cm)
            BLOCK_V, num_warps = _ce_launch_cfg(V)

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                h_c = hidden[start:end]  # [c, H]
                logits = torch.matmul(h_c, weight.t())  # [c, V] cuBLAS, input dtype
                logits = logits if logits.is_contiguous() else logits.contiguous()
                y_c = labels[start:end]
                loss_c = loss_1d[start:end]
                rows = end - start
                with torch.cuda.device(device):
                    _ce_kernel[(rows,)](
                        logits,
                        logits.stride(0),
                        y_c,
                        loss_c,
                        denom,
                        ignore_index,
                        # V is a tl.constexpr meta-parameter (the kernel loops ``range(0, V, BLOCK_V)``
                        # at compile time), so it MUST be passed as a KEYWORD meta arg, not a
                        # positional runtime arg.
                        V=V,
                        label_smoothing=label_smoothing,
                        reduction=reduction,
                        HAS_GRAD=need_grad,
                        BLOCK_V=BLOCK_V,
                        num_warps=num_warps,
                    )
                if need_grad:
                    # logits now holds d_logits ([c, V], already mean-scaled by the kernel).
                    grad_hidden[start:end] = torch.matmul(logits, weight).to(grad_hidden.dtype)
                    if grad_weight is not None:
                        # Accumulate the per-chunk product into ``grad_weight`` in its OWN dtype.
                        # Only widen the [V,H] product to fp32 when the accumulator is actually
                        # fp32 — then the wider temporary buys real precision. When the accumulator
                        # is bf16 (the default), a fp32 ``.float()`` materialization of the [V,H]
                        # product is pure waste (it's immediately downcast on the in-place add, so
                        # it can't improve accuracy) AND a huge per-chunk temporary, so keep the
                        # product in the accumulator dtype directly.
                        prod = torch.matmul(logits.t(), h_c.to(logits.dtype))
                        if acc_dtype == torch.float32:
                            prod = prod.float()
                        grad_weight += prod
                del logits

            loss = loss_1d.sum()
            if grad_weight is not None and grad_weight.dtype != weight.dtype:
                grad_weight = grad_weight.to(weight.dtype)
            # ``save_for_backward`` only accepts tensors (a ``None`` raises). Either grad can be
            # ``None``: the no-grad path (need_grad False -> both None) or the frozen-weight path
            # (weight.requires_grad False -> grad_weight None). Track which grads exist on ctx and
            # save ONLY the real tensors, reconstructing ``None`` for the absent ones in backward().
            ctx.has_grad_hidden = grad_hidden is not None
            ctx.has_grad_weight = grad_weight is not None
            to_save = tuple(t for t in (grad_hidden, grad_weight) if t is not None)
            ctx.save_for_backward(*to_save)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            # CE is the last op: grads wrt logits were computed in forward, so backward just scales
            # the saved grad_hidden / grad_weight by the incoming grad_output (usually 1.0).
            # Only the grads that actually existed were saved (forward dropped any None before
            # save_for_backward), so unpack positionally per the ctx flags and rebuild None for the
            # absent ones — returning a grad in the SAME position as each forward input.
            saved = list(ctx.saved_tensors)
            grad_hidden = saved.pop(0) if ctx.has_grad_hidden else None
            grad_weight = saved.pop(0) if ctx.has_grad_weight else None
            if grad_hidden is not None and not (grad_output.numel() == 1 and float(grad_output.detach()) == 1.0):
                go = grad_output.to(grad_hidden.dtype)
                grad_hidden = grad_hidden * go
                if grad_weight is not None:
                    grad_weight = grad_weight * grad_output.to(grad_weight.dtype)
            # 7 forward inputs: hidden, weight, labels, ignore_index, reduction, label_smoothing,
            # chunk_mult -> grads for (hidden, weight) then None for the 5 non-tensor inputs.
            return grad_hidden, grad_weight, None, None, None, None, None

    def flce_fn(hidden, weight, labels, ignore_index=-100, reduction="mean", label_smoothing=0.0, _chunk_mult=None):
        # _chunk_mult=None -> use the tuned module default (_CE_CFG["chunk_mult"]); pass an int to
        # override (e.g. 1 for exact Liger memory parity, or a sweep value from the benchmark).
        return _FLCEFunction.apply(hidden, weight, labels, ignore_index, reduction, label_smoothing, _chunk_mult)

    # expose the tuning knobs so the benchmark can sweep configs without rebuilding.
    flce_fn._CE_CFG = _CE_CFG  # type: ignore[attr-defined]
    flce_fn._ACC_CFG = _ACC_CFG  # type: ignore[attr-defined]
    flce_fn._chunk_size = _chunk_size  # type: ignore[attr-defined]
    return flce_fn


def _eager_flce(hidden, weight, labels, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    """Naive reference: materialize logits with ``F.linear`` then ``F.cross_entropy`` (the exact
    math chalk's chunked kernel must reproduce). fp32 self-test oracle."""
    import torch.nn.functional as F

    logits = F.linear(hidden, weight)
    return F.cross_entropy(
        logits.float(),
        labels,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def _self_test(flce_fn) -> None:
    """Live-GPU numeric + autograd parity vs the naive ``F.linear`` + ``F.cross_entropy`` reference:
    loss, grad_hidden, grad_weight. fp32 loss rel-err < 1e-3; grads rel-err < 2e-3. Also exercises
    ignore_index and label_smoothing. Raises on mismatch so the caller keeps the Liger/eager path.
    Smaller-than-real V (so the test is fast) but the same chunked code path."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA for flce self-test")
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)

    def rel(a, b):
        return (a - b).norm().item() / (b.norm().item() + 1e-9)

    # (N, H, V): a couple of shapes incl. V > H (forces multi-chunk) and small for speed.
    for N, H, V in ((512, 2560, 8000), (1024, 1024, 4096)):
        for ls in (0.0, 0.1):
            hidden = torch.randn(N, H, device=dev, dtype=torch.float32, generator=gen) * 0.2
            weight = torch.randn(V, H, device=dev, dtype=torch.float32, generator=gen) * 0.05
            labels = torch.randint(0, V, (N,), device=dev, generator=gen)
            # mark ~10% of rows ignored
            ign = torch.rand(N, device=dev, generator=gen) < 0.1
            labels = labels.masked_fill(ign, -100)

            hr = hidden.clone().requires_grad_(True)
            wr = weight.clone().requires_grad_(True)
            ref = _eager_flce(hr, wr, labels, label_smoothing=ls)
            ref.backward()
            dh_ref, dw_ref = hr.grad.clone(), wr.grad.clone()

            hc = hidden.clone().requires_grad_(True)
            wc = weight.clone().requires_grad_(True)
            got = flce_fn(hc, wc, labels, label_smoothing=ls)
            got.backward()
            dh_got, dw_got = hc.grad.clone(), wc.grad.clone()

            r_loss = abs(got.item() - ref.item()) / (abs(ref.item()) + 1e-9)
            r_dh, r_dw = rel(dh_got, dh_ref), rel(dw_got, dw_ref)
            if not (r_loss < 1e-3 and r_dh < 2e-3 and r_dw < 2e-3):
                raise RuntimeError(
                    f"flce self-test failed at N={N} H={H} V={V} ls={ls}: loss={r_loss:.2e} dh={r_dh:.2e} dw={r_dw:.2e}"
                )


def load_flce():
    """Return ``flce_fn`` if the kernel builds and passes its live-GPU self-test; otherwise
    ``None`` (keep the Liger/eager path). Never raises — any failure (no torch/triton, no CUDA,
    compile/self-test error) -> ``None``.

    Routes through ``load_entry`` so a verified ``chalk/ops/arch/<arch>/flce.py`` can dispatch:
    that loader is the ONLY consulter of the arch tree, and building ``_build_kernels`` directly
    made every flce overlay unreachable on every arch regardless of how well it scored.
    ``load_entry`` already self-tests the entry it returns (arch OR portable), so ``load_kernel``
    omits ``self_test`` here — a second one would only double startup validation latency (mirrors
    ``load_conv`` / ``load_lora``)."""
    from chalk.ops.arch import load_entry
    from chalk.ops.arch import load_kernel

    return load_kernel(
        "flce",
        "fused-linear-CE (chunked LM-head+CE, fwd+bwd) enabled",
        "fused-linear-CE disabled",
        build=lambda: load_entry("flce", _self_test, portable=_build_kernels),
    )


def _causal_lm_loss(
    flce_fn,
    hidden_states,
    lm_head_weight,
    labels,
    *,
    ignore_index=-100,
    num_items_in_batch=None,
    shift_labels=None,
    label_smoothing=0.0,
):
    """Replicate ``LigerForCausalLMLoss``: shift labels (token<n predicts n), flatten to
    ``[N,H]``/``[N]``, run the chunked FLCE, and (for grad-accum) divide a ``sum`` reduction by
    ``num_items_in_batch``. Returns the scalar loss. ``num_items_in_batch`` mirrors Liger: when
    set, reduce with "sum" then divide (so multi-microbatch accumulation is exact)."""
    import torch.nn.functional as F

    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    # Flatten leading (batch, seq) dims to [N, H] using the tensor's OWN trailing dim. Do NOT
    # derive H from config/lm_head: on Qwen3.5/3.6 the model's hidden_states dim (e.g. 2560) can
    # differ from both config.text_config.hidden_size and lm_head.weight.shape[1] (e.g. 2048, the
    # lm_head INPUT dim after the model's own out-projection), so a config/lm_head-derived value
    # gives a wrong reshape ("shape '[-1, 2048]' is invalid for input of size ..."). The trailing
    # dim is authoritative — which is why _causal_lm_loss takes no hidden_size argument.
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    shift_labels = shift_labels.view(-1).to(hidden_states.device)
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = flce_fn(
        hidden_states,
        lm_head_weight,
        shift_labels,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    if reduction == "sum" and num_items_in_batch is not None:
        loss = loss / num_items_in_batch
    return loss


def install_qwen35_flce(model=None, run_benchmark: bool = False) -> bool:
    """Patch the Qwen3.5/3.6 causal-LM loss path so the LM head + cross-entropy run through chalk's
    chunked fused-linear-CE (NEVER materializing the full ``[N, V]`` logits) — IFF the live-GPU
    self-test passes. Mirrors Liger's FLCE patch (it swaps ``Qwen3_5ForCausalLM.forward`` for an
    ``lce_forward`` that routes the head+CE through its kernel); chalk does the same with this one.

    Install-on-call (the Liger model): calling this IS the opt-in (no env flag). With ``model``
    given, only that instance's ``forward`` is bound (so a non-causal-LM instance is left alone);
    without ``model`` the ``Qwen3_5ForCausalLM.forward`` CLASS method is patched (every instance).
    Either way the patched forward runs the model body normally and replaces ONLY the
    logits+loss computation with the kernel (during TRAINING, when ``labels`` are present and the
    full logits aren't needed for return).

    SAFE NO-OP CONDITIONS (any -> return False, leave the model untouched):
      * kernel disabled / build / self-test failure (``load_flce`` -> None);
      * no qwen3_5/3_6 modeling module with a ``*ForCausalLM`` class is importable.

    A FAILED (re)install is non-destructive. ``RESULT`` is only rewritten on the success path.
    Returns True iff the kernel was installed."""
    fn = load_flce()
    if fn is None:
        return False

    import importlib

    from types import MethodType

    # Collect importable text-only and VL causal-LM classes for qwen3_5 / qwen3_5_moe / qwen3_6 /
    # qwen3_6_moe. The Qwen3.5 HF checkpoints flash trains are natively multimodal and load as
    # *ForConditionalGeneration; they still run a causal-LM text loss over ``lm_head``.
    targets = []  # (label, module, cls)
    for mod_name in ("qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_6_moe"):
        try:
            mod = importlib.import_module(f"transformers.models.{mod_name}.modeling_{mod_name}")
        except Exception:
            continue
        # Transformers keeps the underscore+digit casing, e.g. qwen3_5 -> Qwen3_5ForCausalLM.
        cls_names = {
            "qwen3_5": ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"),
            "qwen3_5_moe": ("Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"),
            "qwen3_6": ("Qwen3_6ForCausalLM", "Qwen3_6ForConditionalGeneration"),
            "qwen3_6_moe": ("Qwen3_6MoeForCausalLM", "Qwen3_6MoeForConditionalGeneration"),
        }[mod_name]
        for cls_name in cls_names:
            cls = getattr(mod, cls_name, None)
            if cls is not None and hasattr(cls, "forward"):
                targets.append((f"{mod_name}.{cls_name}", mod, cls))
    if not targets:
        print("[flce] no qwen3_5/3_6 ForCausalLM class to patch; keeping eager/Liger", flush=True)
        return False

    def _make_forward(orig_forward):
        def forward(self, *args, **kwargs):
            # Pull the args the loss path needs; everything else flows to the model body. We accept
            # the standard HF causal-LM signature positionally-or-by-keyword via a small shim.
            labels = kwargs.pop("labels", None)
            shift_labels = kwargs.pop("shift_labels", None)
            num_items_in_batch = kwargs.pop("num_items_in_batch", None)
            return_dict = kwargs.pop("return_dict", None)
            logits_to_keep = kwargs.pop("logits_to_keep", 0)

            # Resolve return_dict the way HF does: the explicit arg if given, else the model's
            # config.use_return_dict (defaults True). Honoring this matters for the fused path's
            # return shape below — a caller with return_dict=False expects a tuple, not a
            # ModelOutput, per HF's CausalLM contract.
            use_return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)

            # Decide whether to take the fused (skip-logits) path: training, labels present, on CUDA.
            # A NON-INT ``logits_to_keep`` (an index tensor) selects an arbitrary, non-contiguous set
            # of positions; we cannot generically re-align the (shifted) labels to such a gather, so
            # fall back to HF's original forward in that case rather than risk a label/position
            # mismatch. The int case (keep the last N positions) IS handled below by slicing the
            # shifted labels to the same window.
            # NOTE: test ``isinstance(int)`` and NEVER ``logits_to_keep == 0`` here — a torch index
            # tensor compares elementwise, so ``== 0`` would raise "ambiguous truth value". An int
            # (incl. the 0 default) takes the fused path; a non-int index tensor falls back below.
            keep_n = logits_to_keep if isinstance(logits_to_keep, int) else 0
            want_fused = (
                (labels is not None or shift_labels is not None)
                and getattr(self, "training", False)
                and self.lm_head.weight.is_cuda
                and isinstance(logits_to_keep, int)
            )

            def _delegate():
                # Re-populate the kwargs we popped above and hand the call back to HF's original
                # (materialize-logits) forward, untouched. Shared by every non-fused exit below.
                if labels is not None:
                    kwargs["labels"] = labels
                if shift_labels is not None:
                    kwargs["shift_labels"] = shift_labels
                if num_items_in_batch is not None:
                    kwargs["num_items_in_batch"] = num_items_in_batch
                if return_dict is not None:
                    kwargs["return_dict"] = return_dict
                kwargs["logits_to_keep"] = logits_to_keep
                return orig_forward(self, *args, **kwargs)

            if not want_fused:
                # Eval / no-labels / CPU -> the original (materialize-logits) forward, untouched.
                return _delegate()

            # Run the model body (everything except lm_head + loss). The base model is at
            # self.model for Qwen3.5/3.6 causal LMs. The fused path needs the body's HIDDEN STATES
            # (dim == lm_head input). For some Qwen3.5/3.6 *ForConditionalGeneration wrappers
            # self.model(...)[0] is NOT the hidden states the fused CE expects (a different trailing
            # dim — e.g. an MTP/VL wrapper), so we (a) prefer the explicit hidden_states the output
            # exposes, and (b) if none matches the lm_head input dim, FALL BACK to the correct eager
            # forward rather than mis-reshape / re-project (garbage).
            _hin = self.lm_head.weight.shape[1]
            # Force the body to return a ModelOutput (not a tuple): a return_dict=False caller would
            # otherwise make self.model(...) return a tuple, and getattr(tuple, "past_key_values"/
            # "hidden_states"/"attentions", None) below is ALWAYS None — silently dropping those
            # outputs. We still honor the caller's use_return_dict for THIS wrapper's own return.
            outputs = self.model(*args, **{**kwargs, "return_dict": True})
            hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            if hidden_states.shape[-1] != _hin:
                # Try the model output's hidden_states stack (last layer) — the real body hidden states.
                hs_stack = getattr(outputs, "hidden_states", None)
                if hs_stack:
                    for cand in reversed(hs_stack):
                        if hasattr(cand, "shape") and cand.shape[-1] == _hin:
                            hidden_states = cand
                            break
            # SAFETY: still no hidden-state tensor matching the lm_head input dim -> FALL BACK to the
            # original (correct) forward instead of producing garbage or crashing.
            if hidden_states.shape[-1] != _hin:
                return _delegate()
            slice_idx = slice(-keep_n, None) if keep_n else slice(None)
            kept = hidden_states[:, slice_idx, :]

            # logits_to_keep>0 keeps only the LAST ``keep_n`` hidden positions, so the labels must be
            # sliced to the SAME window or the fused loss would pair kept logits with the wrong
            # labels. Mirror HF's ``Qwen3*ForCausalLM.forward`` + ``ForCausalLMLoss``: SHIFT first
            # (token<n predicts n -> ``shift_labels[..., j] = labels[..., j+1]``, last position
            # padded to ignore_index), THEN keep the last ``keep_n`` shifted positions so they line
            # up with the kept hidden states. Doing the shift BEFORE the window slice is what keeps
            # each kept position paired with its own next-token target (the final kept position's
            # target is the padded ignore, exactly as in the full-sequence case). We hand the
            # already-shifted+sliced labels to ``_causal_lm_loss`` via ``shift_labels`` so its own
            # (full-sequence) shift is bypassed for this windowed case. keep_n==0 -> unchanged.
            ls_for_loss = shift_labels
            labels_for_loss = labels
            if keep_n:
                import torch.nn.functional as F

                if ls_for_loss is None:
                    padded = F.pad(labels, (0, 1), value=-100)
                    ls_for_loss = padded[..., 1:]
                ls_for_loss = ls_for_loss[..., -keep_n:].contiguous()
                labels_for_loss = None  # shift_labels is now authoritative for the kept window

            loss = _causal_lm_loss(
                fn,
                kept,
                self.lm_head.weight,
                labels_for_loss,
                num_items_in_batch=num_items_in_batch,
                shift_labels=ls_for_loss,
                label_smoothing=getattr(self.config, "label_smoothing", 0.0) or 0.0,
            )

            # Return in the same container the original forward would (logits left as None — the
            # whole point is to not materialize them during training), HONORING return_dict like
            # HF's Qwen3*ForCausalLM.forward: a ModelOutput when use_return_dict, else the tuple
            # form. The tuple mirrors HF's CausalLM layout — fields in declaration order
            # (loss, logits, past_key_values, hidden_states, attentions) with None entries dropped
            # and loss first when present — which is exactly ModelOutput.to_tuple().
            from transformers.modeling_outputs import CausalLMOutputWithPast

            output = CausalLMOutputWithPast(
                loss=loss,
                logits=None,
                past_key_values=getattr(outputs, "past_key_values", None),
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )
            return output if use_return_dict else output.to_tuple()

        return forward

    def _instance_patch_candidates(model):
        """Yield candidate instances to patch, preferring a PEFT wrapper's underlying HF model.

        Patching ``PeftModelForCausalLM.forward`` bypasses PEFT's own forward hooks and, for Qwen3.5
        VL, ``self.model`` resolves to the full conditional-generation module rather than the body
        that returns pre-lm_head hidden states. That makes FLCE fall back to eager full-logits loss.
        Patching ``get_base_model()`` instead lets PEFT call the patched base from inside its normal
        hook context, so LoRA stays active and FLCE receives the correct hidden states.
        """
        seen = set()
        getter = getattr(model, "get_base_model", None)
        if callable(getter):
            try:
                base = getter()
            except Exception:
                base = None
            if base is not None:
                seen.add(id(base))
                yield base, f" via {type(model).__name__}"
        if id(model) not in seen:
            yield model, ""

    patched = []
    if model is not None:
        # Bind to this instance only (leave the class — and other instances — untouched). Prefer the
        # unwrapped PEFT base model when present; if the unwrap fails or has an unexpected shape, fall
        # back to the object the caller supplied. Guard first: only patch a real causal-LM (an instance
        # of a target class, OR at least carrying the `forward`/`model`/`lm_head`/`config` the fused
        # forward needs). A non-model object (e.g. a test sentinel) is a safe no-op (return False)
        # rather than a raise — mirroring the other installers' "touch nothing if it's not the right
        # shape" contract.
        target_classes = tuple(cls for _, _, cls in targets)
        patch_target = None
        via = ""
        for cand, cand_via in _instance_patch_candidates(model):
            is_target = isinstance(cand, target_classes)
            has_shape = all(hasattr(cand, a) for a in ("forward", "model", "lm_head", "config")) and hasattr(
                type(cand), "forward"
            )
            if is_target or has_shape:
                patch_target = cand
                via = cand_via
                break
        if patch_target is None:
            print("[flce] given object is not a qwen3_5/3_6 causal LM; keeping eager/Liger", flush=True)
            return False
        orig = type(patch_target).forward
        # The bound forward falls back to the class's original forward (with explicit self) on the
        # eval / no-labels / CPU path.
        patch_target.forward = MethodType(_make_forward(orig), patch_target)
        patched.append(f"instance:{type(patch_target).__name__}{via}")
    else:
        for label, _mod, cls in targets:
            cls.forward = _make_forward(cls.forward)
            patched.append(label)

    RESULT.clear()
    RESULT.update({"installed": True, "self_test": "passed", "patched": patched})
    print(f"[flce] fused-linear-CE installed on {patched} (self-test passed)", flush=True)
    if run_benchmark:
        try:
            _benchmark(fn)
        except Exception as e:
            RESULT["bench_error"] = f"{type(e).__name__}: {e}"
            print(f"[flce][bench] skipped: {e}", flush=True)
    return True


def _benchmark(flce_fn, *, hidden=4096, vocab=248320, seqs=(4096, 8192, 16384), iters=20) -> None:
    """Diagnostic sweep naive vs the fused kernel (fwd+bwd) across token counts. Records the
    per-seq curve (ms + peak GB) in RESULT['sweep']. Never raises out of install (caller guards)."""
    import torch

    dev, dt = "cuda", torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(0)

    def run_fused(h, w, y):
        loss = flce_fn(h, w, y)
        loss.backward()

    sweep = []
    for seq in seqs:
        h0 = torch.randn(seq, hidden, device=dev, dtype=dt, generator=gen) * 0.2
        w0 = torch.randn(vocab, hidden, device=dev, dtype=dt, generator=gen) * 0.05
        y0 = torch.randint(0, vocab, (seq,), device=dev, generator=gen)

        def make(h0=h0, w0=w0, y0=y0):
            return h0.clone().requires_grad_(True), w0.clone().requires_grad_(True), y0

        for _ in range(5):
            h, w, y = make()
            run_fused(h, w, y)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            h, w, y = make()
            run_fused(h, w, y)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        sweep.append({"seq": seq, "fused_ms": round(ms, 4), "peak_gb": round(peak_gb, 3)})
        RESULT["sweep"] = list(sweep)


if __name__ == "__main__":  # manual self-test / smoke
    import torch

    if torch.cuda.is_available():
        try:
            f = _build_kernels()
            _self_test(f)
            print("flce self-test: PASS")
        except Exception as e:
            print(f"flce self-test: FAIL ({e})")
    else:
        print("flce: requires CUDA (skipped)")
