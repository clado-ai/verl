"""moe_grouped_gemm@sm90 -- chalk autoresearch kernel (one file per layer, per arch).

Cell: moe_grouped_gemm@sm90
Entry: grouped_gemm_fn(x, gate_up_proj, down_proj, router_logits, top_k) -> y   (direction: fwd+bwd)
Oracle: chalk.ops.moe._eager_grouped_gemm   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the arch-tuned kernel this file already ships

STATUS: TUNED. Verified against the portable chalk kernel on a real H100 80GB HBM3, memory_ratio
1.000 (neutral, no memory regression), zero cheat flags. Measured under the post-#121/#147
verifier; entry_id 992de1be3233.

SPEEDUP is 3.4581x: fwd+bwd, paired and alternating over 5 rounds (min 3.4567, max 3.4587) at the
gate shape, which is tokens=4096 hidden=2560 inter=1536 E=128 top_k=8. That shape is not a choice;
it is what the verifier times. ``_run_candidate`` calls ``_largest(dev_shapes)`` under
``timing.shape_cost``, and for this cell DEV_TOKENS=(2048, 4096) resolves to the 4096 rung at cost
16.49T. The gate's own 1.6897x was FORWARD-ONLY, so it understated the kernel over a training step,
where the backward it also replaces is the larger half. Alternating the arms within each round
rather than running them in blocks keeps clock drift from being attributed to either.

The figure is a GATE-SHAPE claim, not a production claim, and the distinction is load-bearing here
because the win is partly expert-padding-driven. fwd+bwd across token counts at hidden 2560 / inter
1536 / E=128 / top_k=8: 5.155x at 64 tokens (padding_ratio 2.25), 5.150x at 128 (2.125), 5.170x at
512 (1.563), 4.156x at 2048 (1.305), 3.458x at 4096 (1.129). Portable pads a [E, cap, H] buffer
whose waste shrinks as tokens grow, so no single scalar describes this kernel across shapes --
quote SPEEDUP as what it is, one shape's ratio, and re-measure before repeating it about any other.

Two cautions this measurement earned, both recorded because they cost real GPU time:

Do not read the fwd+bwd ladder off the forward-only one. Forward-only spans 18.165x at 64 tokens
down to 3.444x at 4096, a 5.3x swing; fwd+bwd spans 5.155x to 3.458x, a 1.5x swing. Backward is a
larger and far less padding-sensitive share of a training step, so it flattens the curve. The two
ladders happen to converge at the gate shape (3.444 forward-only vs 3.458 fwd+bwd), which is a
coherence check, not a license to substitute one for the other at any smaller shape.

Build the inputs OUTSIDE the timed region. An earlier probe of this same kernel read 2.9506x
because it constructed four fresh tensors inside its CUDA event window. That setup cost is identical
in both arms, so it does not cancel in a ratio: it adds a constant to numerator and denominator and
drags the result toward 1.0. Every rung above clones its leaves before starting the clock.

Router scale does not matter here, which is worth stating because it looks like it should. Near-
uniform routing (logits scaled 0.02, the early-training regime) versus unscaled randn moves the
ratio by under 0.3% at every rung: 5.1551 vs 5.1463 at 64 tokens, 3.4581 vs 3.4546 at 4096. Both
constructions produce the same expert-load histogram at these shapes, so the padding advantage the
kernel rides on is unchanged. Forward agreement against portable stays within 0.0045-0.0079 rel
across all ten cells, against a 0.02 tolerance.

The header names THIS file, not portable, and the two are not in conflict: the figure was measured
when no sm90 overlay existed, so portable was the anchor. Adopting the file moves the anchor:
``chalk.ops.moe.load_moe`` routes through ``load_entry``, so production now dispatches this kernel
on sm90 and the verifier resolves it as the cell's CURRENT. The next author must beat this, not
portable -- pricing a new target against portable would re-inflate the anchor the #94 fix removed.

Provenance: 18 passing sm90 verdict files for this cell resolve through ``archive.json`` to ONE
distinct entry (``entry_id`` is the source sha16), so this is one kernel verified 18 times, not 18
candidates. The bar it clears is already the loop-free batched-bmm design, NOT the naive 128-expert
Python loop, so the margin is over a strong baseline rather than a strawman.
"""


def build():
    import torch
    import triton
    import triton.language as tl

    from chalk.ops.moe import _build_kernels

    # this kernel is built on torch._grouped_mm, which chalk's declared floor (torch>=2.1.2,
    # setup.py) long predates. raise before build() returns so load_entry falls back to portable
    # on the cheap import-time check rather than after paying for a full live-GPU self test.
    if not hasattr(torch, "_grouped_mm"):
        raise LookupError("torch._grouped_mm unavailable (needs a newer torch); using portable")

    @triton.jit
    def _selected_softmax_fwd_kernel(
        logits_ptr,
        weights_ptr,
        topk: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = cols < topk

        logits = tl.load(
            logits_ptr + row * topk + cols,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        has_nan = (
            tl.sum(
                (logits != logits).to(tl.int32),
                axis=0,
            )
            != 0
        )
        row_max = tl.max(logits, axis=0)
        row_max = tl.where(has_nan, float("nan"), row_max)

        numerators = tl.exp(logits - row_max)
        denominator = tl.sum(numerators, axis=0)
        weights = numerators / denominator

        tl.store(
            weights_ptr + row * topk + cols,
            weights,
            mask=mask,
        )

    @triton.jit
    def _selected_softmax_bwd_kernel(
        grad_weights_ptr,
        weights_ptr,
        grad_logits_ptr,
        topk: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = cols < topk

        weights = tl.load(
            weights_ptr + row * topk + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_weights = tl.load(
            grad_weights_ptr + row * topk + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        correction = tl.sum(
            weights * grad_weights,
            axis=0,
        )
        grad_logits = weights * (grad_weights - correction)

        tl.store(
            grad_logits_ptr + row * topk + cols,
            grad_logits,
            mask=mask,
        )

    @triton.jit
    def _histogram_kernel(
        experts_ptr,
        counts_ptr,
        routes,
        block: tl.constexpr,
    ):
        route_offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = route_offsets < routes
        experts = tl.load(
            experts_ptr + route_offsets,
            mask=mask,
            other=0,
        ).to(tl.int32)
        tl.atomic_add(counts_ptr + experts, 1, mask=mask)

    @triton.jit
    def _prefix_kernel(
        counts_ptr,
        offsets_ptr,
        experts: tl.constexpr,
        block: tl.constexpr,
    ):
        indices = tl.arange(0, block)
        mask = indices < experts
        counts = tl.load(
            counts_ptr + indices,
            mask=mask,
            other=0,
        )
        ends = tl.cumsum(counts, axis=0)
        starts = ends - counts

        tl.store(offsets_ptr + indices, ends, mask=mask)
        tl.store(counts_ptr + indices, starts, mask=mask)

    @triton.jit
    def _assign_routes_kernel(
        experts_ptr,
        cursors_ptr,
        order_ptr,
        inverse_ptr,
        routes,
        block: tl.constexpr,
    ):
        original_routes = tl.program_id(0) * block + tl.arange(0, block)
        mask = original_routes < routes

        experts = tl.load(
            experts_ptr + original_routes,
            mask=mask,
            other=0,
        ).to(tl.int32)
        sorted_routes = tl.atomic_add(
            cursors_ptr + experts,
            1,
            mask=mask,
        )

        tl.store(
            order_ptr + sorted_routes,
            original_routes,
            mask=mask,
        )
        tl.store(
            inverse_ptr + original_routes,
            sorted_routes,
            mask=mask,
        )

    @triton.jit
    def _gather_x_kernel(
        x_ptr,
        order_ptr,
        sorted_x_ptr,
        routes,
        hidden: tl.constexpr,
        topk: tl.constexpr,
        block_n: tl.constexpr,
    ):
        sorted_route = tl.program_id(0).to(tl.int64)
        block_id = tl.program_id(1)

        route_mask = sorted_route < routes
        # order holds int32, so both the program id and the value read out of it are widened
        # before they scale a row stride: routes * hidden passes 2**31 at ~105k tokens.
        original_route = tl.load(
            order_ptr + sorted_route,
            mask=route_mask,
            other=0,
        ).to(tl.int64)
        token = original_route // topk

        cols = block_id * block_n + tl.arange(0, block_n)
        mask = route_mask & (cols < hidden)

        values = tl.load(
            x_ptr + token * hidden + cols,
            mask=mask,
            other=0.0,
        )
        tl.store(
            sorted_x_ptr + sorted_route * hidden + cols,
            values,
            mask=mask,
        )

    @triton.jit
    def _unpack_x_bwd_kernel(
        grad_sorted_x_ptr,
        inverse_ptr,
        grad_x_ptr,
        tokens,
        hidden: tl.constexpr,
        topk: tl.constexpr,
        block_n: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        block_id = tl.program_id(1)

        cols = block_id * block_n + tl.arange(0, block_n)
        mask = (token < tokens) & (cols < hidden)
        accumulator = tl.zeros((block_n,), dtype=tl.float32)

        for choice in tl.static_range(0, topk):
            original_route = token * topk + choice
            sorted_route = tl.load(inverse_ptr + original_route).to(tl.int64)
            values = tl.load(
                grad_sorted_x_ptr + sorted_route * hidden + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += values

        tl.store(
            grad_x_ptr + token * hidden + cols,
            accumulator,
            mask=mask,
        )

    @triton.jit
    def _weighted_activation_fwd_kernel(
        gate_up_ptr,
        flat_weights_ptr,
        order_ptr,
        output_ptr,
        inter: tl.constexpr,
        block: tl.constexpr,
    ):
        # gate_up_stride is the largest stride in this file, so this is the first expression to
        # overflow: at qwen3.5-moe dims it passes 2**31 at ~87k tokens, inside a training batch.
        sorted_route = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * block + tl.arange(0, block)
        mask = cols < inter
        gate_up_stride = inter * 2

        original_route = tl.load(order_ptr + sorted_route).to(tl.int64)
        weight = tl.load(flat_weights_ptr + original_route).to(tl.float32)

        gate = tl.load(
            gate_up_ptr + sorted_route * gate_up_stride + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            gate_up_ptr + sorted_route * gate_up_stride + inter + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        sigmoid_gate = tl.sigmoid(gate)
        activated = gate * sigmoid_gate * up * weight

        tl.store(
            output_ptr + sorted_route * inter + cols,
            activated,
            mask=mask,
        )

    @triton.jit
    def _weighted_activation_bwd_kernel(
        grad_output_ptr,
        gate_up_ptr,
        flat_weights_ptr,
        order_ptr,
        grad_gate_up_ptr,
        grad_flat_weights_ptr,
        inter: tl.constexpr,
        block: tl.constexpr,
    ):
        sorted_route = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * block + tl.arange(0, block)
        mask = cols < inter
        gate_up_stride = inter * 2

        original_route = tl.load(order_ptr + sorted_route).to(tl.int64)
        weight = tl.load(flat_weights_ptr + original_route).to(tl.float32)

        gate = tl.load(
            gate_up_ptr + sorted_route * gate_up_stride + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            gate_up_ptr + sorted_route * gate_up_stride + inter + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_output = tl.load(
            grad_output_ptr + sorted_route * inter + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        sigmoid_gate = tl.sigmoid(gate)
        silu_gate = gate * sigmoid_gate
        silu_derivative = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        weighted_grad = grad_output * weight

        tl.store(
            grad_gate_up_ptr + sorted_route * gate_up_stride + cols,
            weighted_grad * up * silu_derivative,
            mask=mask,
        )
        tl.store(
            grad_gate_up_ptr + sorted_route * gate_up_stride + inter + cols,
            weighted_grad * silu_gate,
            mask=mask,
        )

        weight_terms = tl.where(
            mask,
            grad_output * silu_gate * up,
            0.0,
        )
        grad_weight = tl.sum(weight_terms, axis=0)
        # grad_flat_weights reduces over the whole inter axis, which is now split across column
        # programs, so each tile must accumulate rather than store. the caller zeroes the buffer.
        tl.atomic_add(
            grad_flat_weights_ptr + original_route,
            grad_weight,
        )

    @triton.jit
    def _route_sum_fwd_kernel(
        routes_ptr,
        inverse_ptr,
        output_ptr,
        tokens,
        hidden: tl.constexpr,
        topk: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        token_offsets = tl.program_id(0).to(tl.int64) * block_m + tl.arange(0, block_m)
        hidden_offsets = tl.program_id(1) * block_n + tl.arange(0, block_n)

        token_mask = token_offsets < tokens
        hidden_mask = hidden_offsets < hidden
        mask = token_mask[:, None] & hidden_mask[None, :]
        accumulator = tl.zeros(
            (block_m, block_n),
            dtype=tl.float32,
        )

        for choice in tl.static_range(0, topk):
            original_routes = token_offsets * topk + choice
            sorted_routes = tl.load(
                inverse_ptr + original_routes,
                mask=token_mask,
                other=0,
            ).to(tl.int64)
            values = tl.load(
                routes_ptr + sorted_routes[:, None] * hidden + hidden_offsets[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += values

        tl.store(
            output_ptr + token_offsets[:, None] * hidden + hidden_offsets[None, :],
            accumulator,
            mask=mask,
        )

    @triton.jit
    def _route_sum_bwd_kernel(
        grad_output_ptr,
        inverse_ptr,
        grad_routes_ptr,
        tokens,
        hidden: tl.constexpr,
        topk: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        token_offsets = tl.program_id(0).to(tl.int64) * block_m + tl.arange(0, block_m)
        hidden_offsets = tl.program_id(1) * block_n + tl.arange(0, block_n)

        token_mask = token_offsets < tokens
        hidden_mask = hidden_offsets < hidden
        mask = token_mask[:, None] & hidden_mask[None, :]

        grad_values = tl.load(
            grad_output_ptr + token_offsets[:, None] * hidden + hidden_offsets[None, :],
            mask=mask,
            other=0.0,
        )

        for choice in tl.static_range(0, topk):
            original_routes = token_offsets * topk + choice
            sorted_routes = tl.load(
                inverse_ptr + original_routes,
                mask=token_mask,
                other=0,
            ).to(tl.int64)
            tl.store(
                grad_routes_ptr + sorted_routes[:, None] * hidden + hidden_offsets[None, :],
                grad_values,
                mask=mask,
            )

    class _SelectedSoftmax(torch.autograd.Function):
        @staticmethod
        def forward(ctx, selected_logits):
            rows, topk = selected_logits.shape
            weights = torch.empty_like(selected_logits)

            block = triton.next_power_of_2(topk)
            with torch.cuda.device(selected_logits.device):
                _selected_softmax_fwd_kernel[(rows,)](
                    selected_logits,
                    weights,
                    topk=topk,
                    block=block,
                    num_warps=1,
                )

            ctx.save_for_backward(weights)
            ctx.topk = topk
            return weights

        @staticmethod
        def backward(ctx, grad_weights):
            (weights,) = ctx.saved_tensors
            grad_weights = grad_weights.contiguous()
            grad_logits = torch.empty_like(weights)

            rows = weights.shape[0]
            block = triton.next_power_of_2(ctx.topk)
            with torch.cuda.device(weights.device):
                _selected_softmax_bwd_kernel[(rows,)](
                    grad_weights,
                    weights,
                    grad_logits,
                    topk=ctx.topk,
                    block=block,
                    num_warps=1,
                )
            return grad_logits

    class _BucketPack(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            x,
            flat_experts,
            expert_count,
            topk,
        ):
            tokens, hidden = x.shape
            routes = flat_experts.numel()

            counts = torch.zeros(
                (expert_count,),
                device=x.device,
                dtype=torch.int32,
            )
            offsets = torch.empty_like(counts)

            with torch.cuda.device(x.device):
                histogram_block = 256
                _histogram_kernel[(triton.cdiv(routes, histogram_block),)](
                    flat_experts,
                    counts,
                    routes,
                    block=histogram_block,
                    num_warps=4,
                )

                # the scan is a single program, so its block must cover every expert: a lane that does
                # not exist never initializes offsets[e] and never turns counts[e] into a cursor, and
                # _grouped_mm then reads uninitialized offsets. size it from expert_count, do not
                # hardcode the 128 of the qwen3.5-moe shape.
                expert_block = triton.next_power_of_2(expert_count)
                _prefix_kernel[(1,)](
                    counts,
                    offsets,
                    experts=expert_count,
                    block=expert_block,
                    num_warps=4,
                )

                order = torch.empty(
                    (routes,),
                    device=x.device,
                    dtype=torch.int32,
                )
                inverse = torch.empty_like(order)

                assign_block = 256
                _assign_routes_kernel[(triton.cdiv(routes, assign_block),)](
                    flat_experts,
                    counts,
                    order,
                    inverse,
                    routes,
                    block=assign_block,
                    num_warps=4,
                )

                sorted_x = torch.empty(
                    (routes, hidden),
                    device=x.device,
                    dtype=x.dtype,
                )
                gather_block = 1024
                _gather_x_kernel[
                    (
                        routes,
                        triton.cdiv(hidden, gather_block),
                    )
                ](
                    x,
                    order,
                    sorted_x,
                    routes,
                    hidden=hidden,
                    topk=topk,
                    block_n=gather_block,
                    num_warps=8,
                )

            ctx.save_for_backward(inverse)
            ctx.tokens = tokens
            ctx.hidden = hidden
            ctx.topk = topk
            ctx.mark_non_differentiable(
                inverse,
                order,
                offsets,
            )

            return sorted_x, inverse, order, offsets

        @staticmethod
        def backward(
            ctx,
            grad_sorted_x,
            grad_inverse,
            grad_order,
            grad_offsets,
        ):
            (inverse,) = ctx.saved_tensors
            grad_sorted_x = grad_sorted_x.contiguous()

            grad_x = torch.empty(
                (ctx.tokens, ctx.hidden),
                device=grad_sorted_x.device,
                dtype=grad_sorted_x.dtype,
            )

            block_n = 1024
            with torch.cuda.device(grad_sorted_x.device):
                _unpack_x_bwd_kernel[
                    (
                        ctx.tokens,
                        triton.cdiv(ctx.hidden, block_n),
                    )
                ](
                    grad_sorted_x,
                    inverse,
                    grad_x,
                    ctx.tokens,
                    hidden=ctx.hidden,
                    topk=ctx.topk,
                    block_n=block_n,
                    num_warps=8,
                )

            return grad_x, None, None, None

    class _WeightedActivation(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            gate_up,
            flat_weights,
            order,
        ):
            rows = gate_up.shape[0]
            inter = gate_up.shape[1] // 2
            output = torch.empty(
                (rows, inter),
                device=gate_up.device,
                dtype=gate_up.dtype,
            )

            block = min(2048, triton.next_power_of_2(inter)) if inter else 1
            with torch.cuda.device(gate_up.device):
                _weighted_activation_fwd_kernel[
                    (
                        rows,
                        triton.cdiv(inter, block),
                    )
                ](
                    gate_up,
                    flat_weights,
                    order,
                    output,
                    inter=inter,
                    block=block,
                    num_warps=8,
                )

            ctx.save_for_backward(
                gate_up,
                flat_weights,
                order,
            )
            ctx.inter = inter
            return output

        @staticmethod
        def backward(ctx, grad_output):
            gate_up, flat_weights, order = ctx.saved_tensors
            grad_output = grad_output.contiguous()

            inter = ctx.inter
            grad_gate_up = torch.empty_like(gate_up)
            # the weight gradient is atomically accumulated across column tiles, so it must start
            # at zero and carry enough precision to sum inter terms: accumulate in fp32, then cast.
            grad_weight_accum = torch.zeros(
                flat_weights.shape,
                device=flat_weights.device,
                dtype=torch.float32,
            )

            block = min(2048, triton.next_power_of_2(inter)) if inter else 1
            with torch.cuda.device(gate_up.device):
                _weighted_activation_bwd_kernel[
                    (
                        gate_up.shape[0],
                        triton.cdiv(inter, block),
                    )
                ](
                    grad_output,
                    gate_up,
                    flat_weights,
                    order,
                    grad_gate_up,
                    grad_weight_accum,
                    inter=inter,
                    block=block,
                    num_warps=8,
                )

            return grad_gate_up, grad_weight_accum.to(flat_weights.dtype), None

    class _RouteSum(torch.autograd.Function):
        @staticmethod
        def forward(ctx, routes, inverse, topk):
            route_count, hidden = routes.shape
            tokens = route_count // topk
            output = torch.empty(
                (tokens, hidden),
                device=routes.device,
                dtype=routes.dtype,
            )

            block_m = 4
            block_n = 512
            grid = (
                triton.cdiv(tokens, block_m),
                triton.cdiv(hidden, block_n),
            )
            with torch.cuda.device(routes.device):
                _route_sum_fwd_kernel[grid](
                    routes,
                    inverse,
                    output,
                    tokens,
                    hidden=hidden,
                    topk=topk,
                    block_m=block_m,
                    block_n=block_n,
                    num_warps=8,
                )

            ctx.save_for_backward(inverse)
            ctx.tokens = tokens
            ctx.hidden = hidden
            ctx.topk = topk
            return output

        @staticmethod
        def backward(ctx, grad_output):
            (inverse,) = ctx.saved_tensors
            grad_output = grad_output.contiguous()

            grad_routes = torch.empty(
                (ctx.tokens * ctx.topk, ctx.hidden),
                device=grad_output.device,
                dtype=grad_output.dtype,
            )

            block_m = 4
            block_n = 512
            grid = (
                triton.cdiv(ctx.tokens, block_m),
                triton.cdiv(ctx.hidden, block_n),
            )
            with torch.cuda.device(grad_output.device):
                _route_sum_bwd_kernel[grid](
                    grad_output,
                    inverse,
                    grad_routes,
                    ctx.tokens,
                    hidden=ctx.hidden,
                    topk=ctx.topk,
                    block_m=block_m,
                    block_n=block_n,
                    num_warps=8,
                )

            return grad_routes, None, None

    # this overlay is built on torch._grouped_mm, which accepts bf16 operands ONLY: an fp32 block
    # raises "Expected mat_a to be BFloat16 matrix got Float" rather than computing. the portable
    # kernel is pure bmm/scatter and handles any float dtype, so the dtype is a limit of THIS file,
    # not of the op -- and load_entry's contract is that an arch file is "a pure, self-validated
    # speedup overlay on top of each op's portable kernel". an overlay that raises on an input the
    # op supports is not an overlay, it is a narrowing.
    #
    # the raise is reachable in production. chalk/transformers/moe.py's `owns` guard admits a block
    # on structure alone (3D stacked expert params, integer top_k, norm_topk_prob, cuda, no shared
    # expert) with no dtype predicate, so an fp32 MoE block satisfies every predicate, gets OWNED,
    # and then dies here instead of delegating to the stock forward. fixing it there would key the
    # SHARED guard on this file's private limitation and disable the fast path on the four arches
    # whose portable kernel handles fp32 correctly, so the narrowing is repaired where it lives.
    portable_fn = _build_kernels()

    def grouped_gemm_fn(
        x,
        gate_up_proj,
        down_proj,
        router_logits,
        top_k,
    ):
        # torch._grouped_mm accepts bf16 only, so an fp32 or fp16 block reaches it and RAISES.
        # an arch file is a speedup overlay on top of the op's portable kernel, so an input the op
        # supports must still work here: delegate instead of narrowing what production accepts.
        # bf16 is also the only dtype _self_test exercises on sm90 (is_bf16_supported() is true, so
        # the fp16 branch never runs), so anything else would be unverified even where it ran.
        #
        # delegating rather than widening `owns` is deliberate. `owns` is shared by all five arches
        # and portable handles fp32 correctly, so keying it on this file's private _grouped_mm
        # limitation would disable fp32 MoE acceleration on the four arches that never had it.
        #
        # verified on an H100 80GB HBM3: fp32 now returns float32 at 1e-06 rel-l2 on y and all four
        # grads, matching portable exactly; fp16 rel-l2 is byte-identical to portable, which is what
        # proves the delegation actually fires rather than the overlay quietly handling it. mixed
        # dtypes (fp32 x with bf16 weights, and the reverse) raise the SAME error on both paths, so
        # this predicate rejects nothing portable would have accepted.
        if not (x.dtype is gate_up_proj.dtype is down_proj.dtype is torch.bfloat16):
            return portable_fn(x, gate_up_proj, down_proj, router_logits, top_k)

        x_contiguous = x.contiguous()
        gate_up_contiguous = gate_up_proj.contiguous()
        down_contiguous = down_proj.contiguous()

        expert_count = gate_up_contiguous.shape[0]

        # the op contract fixes both layouts: gate_up_proj is [e, 2*inter, hidden] and
        # down_proj is [e, hidden, inter]. the portable kernel transposes both
        # unconditionally, so match it instead of sniffing which axis is hidden.
        gate_up_matrix = gate_up_contiguous.transpose(1, 2)
        down_matrix = down_contiguous.transpose(1, 2)

        selected_logits, route_experts = torch.topk(
            router_logits,
            top_k,
            dim=-1,
            sorted=False,
        )
        route_weights = _SelectedSoftmax.apply(selected_logits)
        flat_weights = route_weights.reshape(-1)

        (
            sorted_x,
            inverse,
            order,
            offsets,
        ) = _BucketPack.apply(
            x_contiguous,
            route_experts.reshape(-1),
            expert_count,
            top_k,
        )

        gate_up = torch._grouped_mm(
            sorted_x,
            gate_up_matrix,
            offsets,
        )
        activated = _WeightedActivation.apply(
            gate_up,
            flat_weights,
            order,
        )
        routed_output = torch._grouped_mm(
            activated,
            down_matrix,
            offsets,
        )

        return _RouteSum.apply(
            routed_output,
            inverse,
            top_k,
        )

    return grouped_gemm_fn


TUNED = True
SPEEDUP = 3.4581
SPEEDUP_ANCHOR = "the portable chalk kernel, fwd+bwd at the gate shape"
