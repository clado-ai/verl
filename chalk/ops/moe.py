"""Portable top-k grouped MoE expert-GEMM reference + loader for Qwen3.5-MoE.

The Qwen3.5-MoE sparse block routes each token to its ``top_k`` experts and runs a per-expert
SwiGLU MLP over just the routed tokens, then scatters the router-weighted expert outputs back:

    routing = softmax(router_logits, fp32)            # [T, E]
    topk_w, topk_i = topk(routing, top_k)             # [T, top_k]
    topk_w /= topk_w.sum(-1)                           # norm_topk_prob (Qwen3.5 default)
    for each selected (token, expert):
        h = x @ gate_up_proj[e].T                      # [., 2*inter]   (fused gate+up)
        gate, up = h.chunk(2, -1)
        act = silu(gate) * up                          # [., inter]
        o = act @ down_proj[e].T                       # [., hidden]
    y[token] += topk_w * o                             # router-weighted scatter-add

This module is the harness/production side of the ``moe_grouped_gemm`` autoresearch cell: it
supplies the fp32 eager reference (``_eager_grouped_gemm`` — the oracle math), the PRODUCTION
portable kernel (``_build_kernels`` -> ``grouped_gemm_fn``: a loop-free batched-``bmm`` grouped GEMM,
~50x the naive 128-expert loop on an H100 at the Qwen3.5-MoE shape, arch-agnostic so every arch
benefits, output in the input dtype), a live-GPU fwd+bwd self-test (``_self_test``), and a
self-test-gated loader (``load_moe`` via ``load_entry``) mirroring ``chalk.ops.swiglu``. The arch
tree (``chalk/ops/arch/<arch>/moe_grouped_gemm.py``) can hold a per-arch TUNED win; until one beats
this portable kernel, ``load_entry`` falls back to the batched-bmm kernel here.

Tensor contract (matches Qwen3MoeExperts' 3D params, so each grad-carrying weight is a separate
named float tensor the verifier's backward check can track):
    x            [T, hidden]
    gate_up_proj [E, 2*inter, hidden]                 # fused gate+up per expert
    down_proj    [E, hidden, inter]
    router_logits[T, E]
    top_k        int
    -> y         [T, hidden]

Router top-k SELECTION is discrete / non-differentiable: ``topk_i`` is used only to index and
carries no gradient. Gradients flow through ``x`` / ``gate_up_proj`` / ``down_proj`` and the
softmax ROUTER WEIGHTS ``topk_w`` (which are renormalized by ``norm_topk_prob``). The reference is
computed per-expert via gather -> compute -> ``index_add`` scatter so no [T, E, inter] tensor is
materialized (keeps the fp32 oracle within memory at the fuzz upper bounds).
"""

from __future__ import annotations

# Elements per tile in the self-test's relative-L2 comparison. 32Mi elements is 128 MiB of fp32,
# so the two upcast tiles and their difference together stay a rounding error against the tensors
# being compared, while keeping the reduction long enough that per-tile launch cost is irrelevant.
_REL_CHUNK_ELEMS = 32 * 1024 * 1024


def _eager_grouped_gemm(x, gate_up_proj, down_proj, router_logits, top_k):
    """Exact fp32 reference for the top-k grouped MoE expert MLP (the self-test / autoresearch
    oracle). All math in fp32; router top-k selection detached; ``norm_topk_prob`` renormalization
    applied. Per-expert gather/scatter so no [T, E, inter] intermediate is materialized. Returns
    ``y`` [T, hidden] in fp32."""
    import torch
    import torch.nn.functional as F

    xf = x.float()
    gup = gate_up_proj.float()  # [E, 2*inter, hidden]
    dwn = down_proj.float()  # [E, hidden, inter]
    n_tokens, hidden = xf.shape
    n_experts = gup.shape[0]
    k = int(top_k)

    routing = F.softmax(router_logits.float(), dim=-1)  # [T, E]
    topk_w, topk_i = torch.topk(routing, k, dim=-1)  # [T, k]; indices detached
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)  # norm_topk_prob renormalize

    y = xf.new_zeros((n_tokens, hidden))
    for e in range(n_experts):
        sel = topk_i == e  # [T, k] bool
        if not bool(sel.any()):
            continue
        tok_idx, slot_idx = sel.nonzero(as_tuple=True)  # [n_e]; integer, no grad
        h = xf[tok_idx] @ gup[e].t()  # [n_e, 2*inter]
        g, u = h.chunk(2, dim=-1)
        o = (F.silu(g) * u) @ dwn[e].t()  # [n_e, hidden]
        w = topk_w[tok_idx, slot_idx].unsqueeze(-1)  # [n_e, 1] router weight (carries grad)
        y = y.index_add(0, tok_idx, o * w)
    return y


def _build_kernels():
    """Build the portable grouped-MoE kernel. Returns ``grouped_gemm_fn(x, gate_up_proj, down_proj,
    router_logits, top_k) -> y`` or raises on any import problem (the caller treats a raise as "keep
    eager"). This is the PRODUCTION portable MoE kernel — arch-agnostic (torch ``bmm`` / scatter, no
    per-arch Triton), so every arch benefits — and the numeric contract for the arch tree.

    LOOP-FREE batched-bmm design (replaces the naive 128-expert Python loop, ~50x faster on an H100
    at the Qwen3.5-MoE shape): every ``(token, slot)`` assignment is routed into a padded per-expert
    bucket ``buf`` [E, cap, H] with ONE scatter, then BOTH expert projections run as a SINGLE batched
    GEMM (``torch.bmm`` over the expert axis) — one big, well-utilised kernel instead of 128 tiny
    ones. The math is unchanged from the eager reference: fp32 router softmax + ``norm_topk_prob``
    renorm, fp32 SiLU, bf16 expert GEMMs. Padded rows are zeros that contribute nothing
    (``SiLU(0)=0`` -> 0 after the down-proj) and are dropped on the gather-back. The scatter-back is a
    DETERMINISTIC per-token reduction over the k slots (an invert-sort + reshape-sum, no atomic
    ``index_add``), so forward-determinism holds. All ops are autograd-native, so fwd+bwd and the
    grads wrt x / gate_up_proj / down_proj / router_logits (router top-k SELECTION detached) hold —
    the fwd+bwd self-test below validates this path before it is dispatched.

    The expert MLPs run in the input dtype (bf16); the router softmax + renormalization run in fp32
    (Qwen3.5's router precision), then the top-k weights are cast to the input dtype for the scatter.
    Pure torch bmm/scatter, so it is DEVICE-FOLLOWING (no explicit device guard needed: every tensor
    is allocated on ``x.device``; no Triton launch to bind)."""
    import torch
    import torch.nn.functional as F

    def grouped_gemm_fn(x, gate_up_proj, down_proj, router_logits, top_k):
        n_tokens, hidden = x.shape
        n_experts = gate_up_proj.shape[0]
        k = int(top_k)

        # Empty batch: nothing to route (also guards the ``cap = counts.max()`` host sync below,
        # which would read from an empty tensor). Return a correctly-shaped zero output.
        if n_tokens == 0:
            return x.new_zeros((0, hidden))

        routing = F.softmax(router_logits.float(), dim=-1)  # fp32 router softmax
        topk_w, topk_i = torch.topk(routing, k, dim=-1)  # [N, k]; indices detached (discrete)
        topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(x.dtype)  # norm_topk_prob (k==1 -> 1)

        flat_e = topk_i.reshape(-1)  # [N*k] expert id per slot
        flat_w = topk_w.reshape(-1)  # [N*k] router weight per slot (carries grad)
        tok_ids = (
            torch.arange(n_tokens, device=x.device).unsqueeze(1).expand(n_tokens, k).reshape(-1)
        )  # [N*k] token row per slot

        # Group slots by expert; compute each slot's rank WITHIN its expert bucket.
        order = torch.argsort(flat_e, stable=True)
        se = flat_e[order]  # sorted expert ids
        counts = torch.bincount(flat_e, minlength=n_experts)  # [E]
        cap = int(counts.max().item())  # bucket capacity (one host sync; n_tokens>0 -> N*k>=1 -> >=1)

        # Degenerate routing (many of the N*k slots landing on one expert) drives cap -> ~N*k, which
        # would blow the padded [E, cap, H] buffer to E x the active rows (E x its balanced size) and
        # can OOM. When the padded footprint would BOTH waste a lot vs the active working set AND be
        # large in absolute terms, take the memory-safe per-expert reference path instead (gather ->
        # compute -> scatter, peak memory ~ the largest single bucket, no E-way padding). Balanced
        # routing keeps cap*E ~ N*k, so every realistic shape (incl. the near-uniform grid inputs)
        # still runs the fast batched-bmm path; this only guards the pathological tail.
        active = flat_e.numel()  # N*k active (token, slot) rows
        if cap * n_experts > 8 * active and cap * n_experts * hidden > (1 << 26):
            return _eager_grouped_gemm(x, gate_up_proj, down_proj, router_logits, top_k).to(x.dtype)

        starts = torch.cumsum(counts, 0) - counts  # [E] start offset of each expert's group
        within = torch.arange(order.numel(), device=x.device) - starts[se]  # [N*k] rank in bucket

        # Scatter gathered token rows into a padded [E, cap, H] buffer (padding stays zero). The
        # (se, within) indices are unique by construction, so this index_put is deterministic and
        # autograd-differentiable (grad flows back to x through the gather).
        buf = x.new_zeros((n_experts, cap, hidden))
        buf[se, within] = x[tok_ids[order]]

        # Both expert projections as single batched GEMMs over the expert axis.
        h = torch.bmm(buf, gate_up_proj.transpose(1, 2))  # [E, cap, 2*inter]
        g, u = h.chunk(2, dim=-1)
        # SiLU in fp32 (sigmoid needs fp32), cast back to the input dtype for the up-multiply.
        a = F.silu(g.float()).to(x.dtype) * u  # [E, cap, inter]
        o = torch.bmm(a, down_proj.transpose(1, 2))  # [E, cap, hidden]

        # Gather each slot's output back, apply the router weight.
        o_slot = o[se, within]  # [N*k, hidden] in expert-sorted order
        o_slot = o_slot * flat_w[order].unsqueeze(-1)

        # Invert the sort to (token, slot) order, then a deterministic sum over the k slots.
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        return o_slot[inv].view(n_tokens, k, hidden).sum(dim=1)  # [N, hidden]

    return grouped_gemm_fn


def _self_test(grouped_gemm_fn, *, hidden=2560, inter=1536, n_experts=128, top_k=8) -> None:
    """Live-GPU numeric + autograd parity vs the exact fp32 eager grouped-MoE MLP: forward AND the
    grads wrt x / gate_up_proj / down_proj / router_logits. Raises on mismatch so the caller keeps
    the eager path. Defaults to the Qwen3.5-MoE shape (hidden=2560, inter=1536, E=128, top_k=8).

    Mirrors the rmsnorm/swiglu self-tests: run the reduced-dtype path (bf16 when supported, else
    fp16) and compare against an fp32 reference computed on upcast leaves, with the error measured
    in fp32 against a bf16-sized tolerance (~2e-2 over a full fwd+bwd)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA for moe grouped-gemm self-test")
    dev = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tol = 2e-2
    gen = torch.Generator(device=dev).manual_seed(0)
    scale = hidden**-0.5

    # One shape per call, so every tensor this check allocates is a local that dies on return. The
    # loop below previously ran inline, which kept iteration 1's weights, clones and grads bound
    # while iteration 2 allocated its own -- and rebinding does not help, since ``gate_up = scale *
    # torch.randn(...)`` fully evaluates the right side before the old 1.88 GiB tensor is dropped.
    # A nested function is used rather than an end-of-body ``del`` list because the del list has to
    # be maintained by hand: a later edit that adds a tensor and forgets the del silently restores
    # the leak, whereas scope exit cannot be forgotten.
    # ``experts`` and ``k`` are parameters rather than closure reads of ``n_experts``/``top_k`` so
    # the loop below can vary them. Both are RUNTIME dims for every implementation of this op -- the
    # portable kernel reads E as ``gate_up_proj.shape[0]`` and k as ``int(top_k)``, and the sm90
    # overlay passes E as a ``tl.constexpr`` argument -- so a check that only ever runs one value
    # cannot tell a generic kernel from one that baked in 128 or 8.
    def _check_shape(n_tokens: int, experts: int, k: int) -> None:
        x = torch.randn(n_tokens, hidden, device=dev, dtype=dtype, generator=gen)
        gate_up = scale * torch.randn(experts, 2 * inter, hidden, device=dev, dtype=dtype, generator=gen)
        down = scale * torch.randn(experts, hidden, inter, device=dev, dtype=dtype, generator=gen)
        router = torch.randn(n_tokens, experts, device=dev, dtype=dtype, generator=gen)
        # Cotangent from the SAME seeded ``gen`` (not the default RNG) so the gate is deterministic
        # regardless of the caller's global RNG state.
        grad = torch.randn(n_tokens, hidden, device=dev, dtype=dtype, generator=gen)

        # ORACLE IN fp32: upcast the leaves, run fwd+bwd in fp32, compare in fp32 (bf16-toleranced).
        #
        # The oracle is retired to plain tensors and its graph dropped BEFORE the candidate runs.
        # At the default Qwen3.5-MoE shape the two weight tensors are 2.81 GiB in bf16 and 5.62 GiB
        # upcast, so holding the fp32 leaves live across the candidate's backward costs 5.62 GiB
        # that nothing reads again: the comparison below needs the oracle's grad VALUES, never the
        # leaves or the graph that produced them. Dropping them here is what keeps this check inside
        # a 24GB card, where it previously OOMed -- and because load_entry gates dispatch on the
        # self-test, an OOM here is not a skipped check but a silent fallback to eager.
        #
        # ``.detach()`` is needed on the OUTPUT only. A ``.grad`` is already a detached leaf
        # (grad_fn None, requires_grad False), so detaching one is a no-op; the output is the last
        # live edge into the graph, and keeping it undetached keeps every leaf reachable through
        # that graph even after the ``del`` below.
        xr = x.float().clone().requires_grad_(True)
        gur = gate_up.float().clone().requires_grad_(True)
        dnr = down.float().clone().requires_grad_(True)
        rlr = router.float().clone().requires_grad_(True)
        ref = _eager_grouped_gemm(xr, gur, dnr, rlr, k)
        ref.backward(grad.float())
        ref_y, ref_dx = ref.detach(), xr.grad
        ref_dgu, ref_ddn = gur.grad, dnr.grad
        ref_drl = rlr.grad
        del ref, xr, gur, dnr, rlr

        xc = x.clone().requires_grad_(True)
        guc = gate_up.clone().requires_grad_(True)
        dnc = down.clone().requires_grad_(True)
        rlc = router.clone().requires_grad_(True)
        # The clones are the last readers of the bf16 sources -- ``.clone()`` gives each its own
        # storage, and everything below compares GRADS, never the sources. Releasing them here
        # rather than at scope exit is load-bearing: the two weight tensors are 2.81 GiB together
        # and would otherwise stay resident straight through the fp32 upcast in the comparison.
        del gate_up, down, x, router
        got = grouped_gemm_fn(xc, guc, dnc, rlc, k)
        torch.cuda.synchronize()
        got.backward(grad)
        torch.cuda.synchronize()

        # Same argument one step further: a leaf is dead once its ``.grad`` is bound to a local, and
        # the four clones are another 2.81 GiB. ``.detach()`` on the output drops the last edge into
        # the candidate's graph so nothing holds the leaves after this line.
        #
        # Measured on an A5000: these two dels take the self-test's peak from 19.29 to 9.18 GiB, and
        # they are what carries it through ``backward``, which without them died needing 1.88 GiB
        # with 19.72 GiB genuinely allocated.
        #
        # They are not sufficient on their own. What this del returns is the four bf16 clones,
        # 2.81 GiB but as four separate blocks of at most 1.88 GiB (only the two weight grads are
        # large), and the comparison below then wants a single contiguous 3.75 GiB fp32 copy of the
        # gate_up grad, which no such block can serve. That is why ``rel`` tiles rather than
        # calling chalk.utils.rel_l2: the peak had to come down AND the largest single request had
        # to come down, and only one of those is a lifetime problem.
        #
        # An OOM here is NOT a skipped check: load_entry gates dispatch on the self-test, so it
        # silently falls back to eager and the fused kernel never runs at all.
        got_y = got.detach()
        got_dx, got_dgu = xc.grad, guc.grad
        got_ddn, got_drl = dnc.grad, rlc.grad
        del got, xc, guc, dnc, rlc, grad

        def rel(a: torch.Tensor, b: torch.Tensor) -> float:
            """Relative L2 error, accumulated over flat tiles so peak scales with the tile.

            Same quantity as chalk.utils.rel_l2, computed without ever materialising a full fp32
            copy. That matters only here: this self-test compares the largest tensors in the repo.
            rel_l2 upcasts BOTH arguments and then materialises their difference, so comparing the
            1.88 GiB bf16 gate_up grad wants three 3.75 GiB fp32 blocks at once. On a 24GB card
            that is exactly where this died -- "tried to allocate 3.75 GiB, 2.45 GiB free" with only
            12.22 GiB actually allocated and 8.41 GiB reserved-but-unallocated. Live memory was
            never the problem; no free block was contiguous enough to serve a single 3.75 GiB
            request, and the largest block the dels above release is the 1.88 GiB gate_up. Freeing
            more cannot fix a request that no block can satisfy, so bound the request instead.

            Accumulating in fp64 makes this strictly more accurate than the fp32 original, not a
            tolerance concession. The two disagree by at most 1.2e-04 relative on bf16 and 5.4e-06
            on fp32, measured against rel_l2 over 1e6- and 4e6-element inputs at 1, ~40 and ~500
            tiles; the spread tracks dtype rather than tile count, i.e. it is the fp64 accumulation
            and not the tiling. Both bounds are ~2 orders inside the 2e-2 this check gates on.

            Deliberately local. rel_l2 has nine call sites across five ops, none of which handle
            tensors near this size, and widening its contract for one caller is not this change.
            """
            af, bf = a.reshape(-1), b.reshape(-1)
            num = torch.zeros((), dtype=torch.float64, device=af.device)
            den = torch.zeros((), dtype=torch.float64, device=af.device)
            for i in range(0, af.numel(), _REL_CHUNK_ELEMS):
                x = af[i : i + _REL_CHUNK_ELEMS].float()
                y = bf[i : i + _REL_CHUNK_ELEMS].float()
                num += (x - y).pow(2).sum().double()
                den += y.pow(2).sum().double()
            return float(num.sqrt().item() / (den.sqrt().item() + 1e-9))

        checks = [
            ("y", got_y, ref_y),
            ("dx", got_dx, ref_dx),
            ("dgate_up", got_dgu, ref_dgu),
            ("ddown", got_ddn, ref_ddn),
            ("drouter", got_drl, ref_drl),
        ]
        for nm, gv, rv in checks:
            r = rel(gv, rv)
            if not (r < tol):
                raise RuntimeError(
                    f"moe grouped-gemm self-test failed at n={n_tokens} E={experts} k={k} "
                    f"dtype={dtype}: {nm}={r:.2e} (tol={tol:.0e})"
                )

    # E varies DOWNWARD, not upward. Every weight this check allocates is [E, ., .], so E scales the
    # dominant term linearly: at the Qwen3.5-MoE E=128 the two weights are 2.81 GiB in bf16 and the
    # whole check peaks at 18.09 GiB (measured on an A5000), which is already most of the 24GB card
    # #135 had to fit it inside. Doubling E would add 2.81 GiB of bf16 plus its 5.62 GiB fp32 oracle
    # upcast, landing near 26.5 GiB and putting the check back where it OOMed -- and an OOM here is
    # not a skipped check but a silent fallback to eager, since load_entry gates dispatch on it. A
    # second, SMALLER E falsifies the same thing (a kernel that baked in 128 fails at any other
    # value) and costs nothing at the peak, since the cases run sequentially inside a scope that
    # frees between them -- measured: adding the E=16 case moved the peak by 0.0 GiB.
    #
    # 16 rather than 8: top_k is 8, and at E==top_k the router's topk selects EVERY expert, so all
    # buckets are exactly full and the padded [E, cap, H] buffer never carries padding. E=16 makes
    # topk pick a strict subset with uneven bucket occupancy, which is the shape the scatter/gather
    # path actually has to get right.
    #
    # k varies on that same third case rather than in a fourth one, so it costs no GPU time at all.
    # k is a runtime dim too (``k = int(top_k)``; it sizes topk, the [N*k] slot arrays, and `active`
    # in the eager-fallback guard), and pinning it at 8 left it as unfalsifiable as E was before the
    # case above. k=4 of E=16 is a STRICTER subset than k=8 of 16, so it sharpens rather than dilutes
    # what this case was added to test, and it keeps k <= E, which torch.topk requires.
    for n_tokens, experts, k in ((128, n_experts, top_k), (512, n_experts, top_k), (128, 16, 4)):
        _check_shape(n_tokens, experts, k)
        # Return the freed blocks to the driver between shapes. Python has dropped the references by
        # here, but torch's caching allocator keeps the segments reserved, and a reserved-but-unused
        # segment of the wrong size does not satisfy the next shape's allocation.
        torch.cuda.empty_cache()


def load_moe():
    """Return ``grouped_gemm_fn`` if the kernel builds + passes its live-GPU self-test, else None
    (keep the eager path). Never raises. Consults the per-arch tuned tree (a verified win, TUNED)
    via ``load_entry``, falling back to the portable seed above; the returned entry is always
    self-test gated (fwd+bwd parity vs the fp32 oracle) before it is dispatched."""
    from chalk.ops.arch import load_entry
    from chalk.ops.arch import load_kernel

    # ``load_entry`` already self-tests the entry it returns (arch OR portable), so load_kernel
    # omits ``self_test`` here (mirrors ``load_conv``): a second self-test would only double
    # startup validation latency.
    return load_kernel(
        "moe_grouped_gemm",
        "fused grouped MoE expert GEMM (fwd+bwd) enabled",
        "disabled",
        build=lambda: load_entry("moe_grouped_gemm", _self_test, portable=_build_kernels),
    )


if __name__ == "__main__":  # manual self-test / smoke
    import torch

    if torch.cuda.is_available():
        try:
            f = _build_kernels()
            _self_test(f)
            print("moe grouped-gemm self-test: PASS")
        except Exception as e:
            print(f"moe grouped-gemm self-test: FAIL ({e})")
    else:
        print("moe grouped-gemm: requires CUDA (skipped)")
