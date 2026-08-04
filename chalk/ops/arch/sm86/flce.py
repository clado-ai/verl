"""flce@sm86 — chalk autoresearch kernel (one file per layer, per arch).

Cell: flce@sm86
Entry: flce_fn(hidden, lm_head_weight, labels, ignore_index=-100, reduction='mean',
       label_smoothing=0.0) -> loss   (direction: fwd+bwd)
Oracle: chalk.ops.flce._eager_flce   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the arch-tuned kernel this file already ships   target speedup: up to 6.0x

STATUS: ADOPTED — 1.07x floor vs the PORTABLE chalk kernel, and that floor is the honest number: the
geomean across the six production shapes is 1.26x, but the SPEEDUP constant below reports the WORST
shape, because that is the delta a user is guaranteed rather than the one they get on the best draw.

The header's baseline and that figure name different kernels on purpose. vs-portable is the delta a
user actually gets, since portable is what ``load_entry`` falls back to when this file is absent. But
since #99 the verifier's resolver is arch-aware, so once this file ships TUNED it is what production
dispatches on sm86 and what a NEW candidate is anchored against — the bar to beat moves here, while
the measured figure stays vs-portable and stays true, because it was taken when no overlay shipped.

Selected by production dispatch (``TUNED = True``): ``chalk.ops.flce.load_flce`` routes through
``load_entry("flce", _self_test, ...)``, which runs the op's own live-GPU parity check against its
fp32 oracle before returning this entry, so this file only dispatches when it matches on loss,
grad_hidden and grad_weight. The signature above is load-bearing: production calls the entry with
``ignore_index``/``reduction``/``label_smoothing`` as KEYWORDS (``_causal_lm_loss``) and
``_self_test`` passes ``label_smoothing``, so a 3-arg entry raises TypeError at argument binding,
``load_entry`` swallows it, and the overlay ships INERT while claiming a verified win — the #112/PR
#99 defect shape. ``test/test_correctness_entry_kwargs.py`` grades exactly this.

MEASURED on a real sm86 RTX A5000 (torch 2.8.0+cu129) against the portable kernel imported from the
PINNED tree, not the pod image — a baseline read from the wrong tree makes the ratio unattributable.
Six production shapes (hidden/vocab from the qwen_hybrid family this op ships against), bf16 I/O,
~10% ignore_index rows, timed as fwd+bwd with CUDA events. A/B slot order is cancelled with
sqrt(fwd_ratio * rev_ratio): pooling both orders into one median leaves a bias large enough to flip
a ranking. Portable microseconds over this kernel's, so >1 is a win:

    shape                        rep 1    rep 2    rep 3
    t1024_h2560_v248320          1.5323   1.5317   1.5288
    t2048_h1024_v248320          1.3537   1.3516   1.3515
    t2048_h2048_v130560          1.2322   1.2309   1.2308
    t2048_h2560_v248320          1.2079   1.2083   1.2075
    t2048_h2048_v248320          1.2006   1.1968   1.1976
    t4096_h2560_v248320          1.0704   1.0702   1.0704   <- the floor
    geomean                      1.2582   1.2570   1.2565

Three independent reps because one does not separate a real ratio from run-to-run drift (portable
itself moves 8-10% between runs on this card). Every shape is positive in all three and the ranking
is stable. The win shrinks as tokens grow — 1.53x at 1024 tokens down to 1.07x at 4096 — because
this kernel's edge is per-chunk work that amortizes, so do NOT extrapolate it past 4096 tokens.

MEMORY: forward peak is 1.62x-2.41x better across the same six shapes (worst 1.62x at
t2048_h2048_v130560, best 2.41x at t4096_h2560_v248320). That is TRANSIENT peak only. What forward
still HOLDS when it returns is byte-identical to portable on all six shapes (ratio exactly 1.000),
because 98.4%-99.6% of the retained bytes are the [vocab, hidden] weight-gradient accumulator
``pre_grad_weight``, which both kernels must carry to backward at full weight size and neither can
shrink. So this file lowers the high-water mark a concurrent allocation has to fit under; it does
NOT free memory for the rest of the step. No retained-memory claim is made.

Math: chunked over tokens with a per-chunk logits budget (1<<28 elements), so the [tokens, vocab]
matrix is never materialized whole — the same tradeoff the portable kernel makes deliberately, and
the one the retired sm90 sibling took the wrong side of. The forward Triton kernel does an online
max/log-sum-exp scan over vocab in fp32 and, for the reducing reductions, overwrites the logits
chunk in place with dlogits, so the backward is two GEMMs against a buffer that already exists.
``reduction="none"`` instead saves the LSE row and recomputes logits in backward, since there is no
single upstream scalar to fold in at forward time. NaNs are propagated rather than swallowed: a NaN
anywhere in a row forces that row's max to the NaN so the loss carries it, matching eager.

Degenerate batches follow the PORTABLE contract, not eager's. An empty or fully-ignored batch has no
valid rows; eager ``F.cross_entropy`` returns NaN there, but ``src/chalk/ops/flce.py`` deliberately
clamps its denominator (``denom = n_non_ignore if n_non_ignore > 0 else 1.0``) so one fully-masked
microbatch cannot poison a training run, and this file matches that: 0 loss and 0 grads. The weight
gradient uses ``torch.zeros if rows == 0 else torch.empty`` for the same reason — the per-chunk loop
is its only writer, and at zero rows that loop never runs, so an unconditional ``empty`` returns
uninitialized garbage that reads as a plausible gradient (the #171 defect shape). The condition is
load-bearing on speed, not style: an earlier revision of this file was A/B'd with and without an
unconditional zeroing, and paying it on EVERY call cost ~1.5% end-to-end (that build's worst-shape
floor 1.0379 -> 1.0302, geomean 1.1833 -> 1.1656 — a paired delta on a body that predates the table
above, so read the 1.5%, not the absolutes). rows==0 is the only case where the choice is
observable, so it is paid only there.

That defect is invisible to every other check in this file's path. Measured: the pre-fix ``empty``
body passes ``_self_test`` AND the sum-reduction check, and its empty-batch grad still reads 0.0 —
a freshly allocated CUDA page is essentially always zero, so the unwritten buffer returns exactly
the correct answer by luck. It only fails once the caching allocator is dirtied first; a control arm
built that way returns the planted sentinel as a gradient. Neither the production ``_self_test`` nor
the autoresearch gate generates an empty batch, so nothing upstream would catch it.

Measured by the chalk autoresearch verifier on a real sm86 GPU. build() returns the entry callable.
"""

TUNED = True
# the worst production shape (t4096_h2560_v248320), not the geomean (1.26). load_entry prints this
# to the user, so it reports the delta they are guaranteed. three independent reps put that shape at
# 1.0704, 1.0702 and 1.0704, so the floor is 1.07.
SPEEDUP = 1.07
SPEEDUP_ANCHOR = "the portable chalk kernel"


def build():
    import torch
    import triton
    import triton.language as tl

    SCAN_BLOCK = 1024
    POINT_BLOCK = 256

    # Bound the temporary logits matrix while retaining a large token dimension
    # for efficient cuBLAS tensor-core GEMMs.
    LOGITS_ELEMENT_BUDGET = 1 << 28

    @triton.jit
    def _loss_and_optional_dlogits(
        logits_ptr,
        labels_ptr,
        losses_ptr,
        lse_ptr,
        valid_count_ptr,
        row_start,
        chunk_rows,
        vocab,
        logits_stride_row,
        ignore_index,
        epsilon,
        REDUCTION_MODE: tl.constexpr,
        MAKE_DLOGITS: tl.constexpr,
        SAVE_LSE: tl.constexpr,
        DO_SMOOTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        local_row = tl.program_id(0)
        global_row = row_start + local_row
        row_valid = local_row < chunk_rows

        label = tl.load(
            labels_ptr + global_row,
            mask=row_valid,
            other=ignore_index,
        )
        active = row_valid & (label != ignore_index)

        offsets = tl.arange(0, BLOCK)

        maximum = -float("inf")
        nan_count = 0
        nan_value = 0.0

        start = 0
        while start < vocab:
            columns = start + offsets
            mask = row_valid & (columns < vocab)

            values = tl.load(
                logits_ptr + local_row * logits_stride_row + columns,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)

            is_nan = values != values
            nan_count += tl.sum(
                tl.where(mask & is_nan, 1, 0),
                axis=0,
            )
            nan_value += tl.sum(
                tl.where(mask & is_nan, values, 0.0),
                axis=0,
            )

            clean_values = tl.where(is_nan, -float("inf"), values)
            block_maximum = tl.max(clean_values, axis=0)
            maximum = tl.maximum(maximum, block_maximum)

            start += BLOCK

        maximum = tl.where(nan_count != 0, nan_value, maximum)

        exponential_sum = 0.0
        logit_sum = 0.0

        start = 0
        while start < vocab:
            columns = start + offsets
            mask = row_valid & (columns < vocab)

            values = tl.load(
                logits_ptr + local_row * logits_stride_row + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            exponentials = tl.exp(values - maximum)
            exponential_sum += tl.sum(
                tl.where(mask, exponentials, 0.0),
                axis=0,
            )

            if DO_SMOOTH:
                logit_sum += tl.sum(
                    tl.where(mask, values, 0.0),
                    axis=0,
                )

            start += BLOCK

        lse = maximum + tl.log(exponential_sum)

        label_in_range = active & (label >= 0) & (label < vocab)
        target = tl.load(
            logits_ptr + local_row * logits_stride_row + label,
            mask=label_in_range,
            other=0.0,
        ).to(tl.float32)

        if DO_SMOOTH:
            loss = lse - (1.0 - epsilon) * target - epsilon * logit_sum / vocab
        else:
            loss = lse - target

        loss = tl.where(active, loss, 0.0)
        tl.store(losses_ptr + global_row, loss, mask=row_valid)

        if SAVE_LSE:
            tl.store(lse_ptr + global_row, lse, mask=row_valid)

        if MAKE_DLOGITS:
            if REDUCTION_MODE == 2:
                count = tl.load(valid_count_ptr).to(tl.float32)
                reduction_scale = tl.where(
                    count > 0.0,
                    1.0 / count,
                    0.0,
                )
            else:
                reduction_scale = 1.0

            start = 0
            while start < vocab:
                columns = start + offsets
                mask = row_valid & (columns < vocab)

                values = tl.load(
                    logits_ptr + local_row * logits_stride_row + columns,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)

                gradient = tl.exp(values - lse)
                if DO_SMOOTH:
                    gradient -= epsilon / vocab

                gradient -= tl.where(
                    columns == label,
                    1.0 - epsilon,
                    0.0,
                )

                gradient = tl.where(
                    active,
                    gradient * reduction_scale,
                    0.0,
                )

                tl.store(
                    logits_ptr + local_row * logits_stride_row + columns,
                    gradient,
                    mask=mask,
                )

                start += BLOCK

    @triton.jit
    def _recomputed_logits_to_dlogits(
        logits_ptr,
        labels_ptr,
        lse_ptr,
        grad_output_ptr,
        row_start,
        chunk_rows,
        vocab,
        logits_stride_row,
        ignore_index,
        epsilon,
        DO_SMOOTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        local_row = tl.program_id(0)
        block_id = tl.program_id(1)

        columns = block_id * BLOCK + tl.arange(0, BLOCK)
        global_row = row_start + local_row
        mask = (local_row < chunk_rows) & (columns < vocab)

        logits = tl.load(
            logits_ptr + local_row * logits_stride_row + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        lse = tl.load(
            lse_ptr + global_row,
            mask=local_row < chunk_rows,
            other=0.0,
        ).to(tl.float32)
        label = tl.load(
            labels_ptr + global_row,
            mask=local_row < chunk_rows,
            other=ignore_index,
        )
        upstream = tl.load(
            grad_output_ptr + global_row,
            mask=local_row < chunk_rows,
            other=0.0,
        ).to(tl.float32)

        gradient = tl.exp(logits - lse)
        if DO_SMOOTH:
            gradient -= epsilon / vocab

        gradient -= tl.where(
            columns == label,
            1.0 - epsilon,
            0.0,
        )

        gradient = tl.where(
            label != ignore_index,
            gradient * upstream,
            0.0,
        )

        tl.store(
            logits_ptr + local_row * logits_stride_row + columns,
            gradient,
            mask=mask,
        )

    class _FLCE(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            hidden,
            lm_head_weight,
            labels,
            ignore_index,
            reduction_mode,
            label_smoothing,
        ):
            hidden_shape = hidden.shape
            label_shape = labels.shape

            hidden_2d = hidden.reshape(-1, hidden.shape[-1])
            labels_1d = labels.reshape(-1).contiguous()

            rows = hidden_2d.shape[0]
            hidden_size = hidden_2d.shape[1]
            vocab = lm_head_weight.shape[0]

            if lm_head_weight.ndim != 2:
                raise RuntimeError("lm_head_weight must be two-dimensional")
            if lm_head_weight.shape[1] != hidden_size:
                raise RuntimeError("hidden and lm_head_weight dimensions differ")
            if labels_1d.numel() != rows:
                raise RuntimeError("labels and hidden token counts differ")

            rows_per_chunk = max(
                128,
                LOGITS_ELEMENT_BUDGET // max(vocab, 1),
            )
            rows_per_chunk = max(
                128,
                (rows_per_chunk // 128) * 128,
            )
            rows_per_chunk = min(rows, rows_per_chunk)

            smoothing = float(label_smoothing)
            do_smooth = smoothing != 0.0
            # precomputing dlogits in forward only pays off if a backward will consume them. when
            # neither input requires grad -- a frozen LM head scored for loss only, or an
            # inference_mode caller -- autograd builds no node at all, so the third full-vocab
            # read-modify-write pass over every [chunk, vocab] tile is written and never read.
            # this is exactly portable's HAS_GRAD=need_grad gate: needs_input_grad tracks
            # requires_grad on both tensor inputs, so the two agree in every grad mode. without
            # it a frozen-input caller is SLOWER on sm86 than before this overlay shipped, since
            # sm86 used to dispatch portable. note it does NOT fire under a bare torch.no_grad()
            # with trainable inputs, where needs_input_grad is still (True, True) -- portable does
            # not skip there either, and matching portable is the bar. the loss-only branch below
            # computes bit-identical losses (the loss math never reads REDUCTION_MODE, which only
            # scales dlogits), so this is a pure work skip, not a numerics change.
            precompute_gradients = reduction_mode != 0 and (ctx.needs_input_grad[0] or ctx.needs_input_grad[1])

            losses = torch.empty(
                rows,
                device=hidden.device,
                dtype=torch.float32,
            )

            if reduction_mode == 2:
                valid_count = (labels_1d != int(ignore_index)).sum()
            else:
                valid_count = losses

            if precompute_gradients:
                lse = losses

                need_hidden = ctx.needs_input_grad[0]
                need_weight = ctx.needs_input_grad[1]

                if need_hidden:
                    pre_grad_hidden = torch.empty(
                        (rows, hidden_size),
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                else:
                    pre_grad_hidden = torch.empty(
                        0,
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )

                if need_weight:
                    # the chunk loop below is the only writer of this buffer, and its FIRST
                    # iteration overwrites the whole thing (torch.mm(..., out=)) rather than
                    # accumulating, so empty is safe whenever the loop runs at all. at rows==0
                    # the loop never runs and empty would return uninitialized garbage that
                    # reads as a plausible gradient (the #171 defect shape), so that case -- and
                    # only that case -- pays for zeros. allocating zeros unconditionally costs a
                    # full [vocab, hidden] memset per call (~1.2 GB at v248320_h2560) and
                    # measured ~1.5% of end-to-end fwd+bwd on sm86.
                    _alloc = torch.zeros if rows == 0 else torch.empty
                    pre_grad_weight = _alloc(
                        lm_head_weight.shape,
                        device=lm_head_weight.device,
                        dtype=lm_head_weight.dtype,
                    )
                else:
                    pre_grad_weight = torch.empty(
                        0,
                        device=lm_head_weight.device,
                        dtype=lm_head_weight.dtype,
                    )

                first_weight_chunk = True
                row_start = 0

                while row_start < rows:
                    chunk_rows = min(
                        rows_per_chunk,
                        rows - row_start,
                    )
                    hidden_chunk = hidden_2d.narrow(
                        0,
                        row_start,
                        chunk_rows,
                    )

                    logits = torch.empty(
                        (chunk_rows, vocab),
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    torch.mm(
                        hidden_chunk,
                        lm_head_weight.transpose(0, 1),
                        out=logits,
                    )

                    # triton validates the pointers it is handed against the ACTIVE cuda device,
                    # not against the device the tensors actually live on, so a launch made while
                    # a different device is current faults instead of running. the portable kernel
                    # guards every launch this way (src/chalk/ops/flce.py); an overlay that skips
                    # it is only correct as long as callers never switch device around the op.
                    with torch.cuda.device(hidden.device):
                        _loss_and_optional_dlogits[(chunk_rows,)](
                            logits,
                            labels_1d,
                            losses,
                            lse,
                            valid_count,
                            row_start,
                            chunk_rows,
                            vocab,
                            logits.stride(0),
                            int(ignore_index),
                            smoothing,
                            REDUCTION_MODE=reduction_mode,
                            MAKE_DLOGITS=True,
                            SAVE_LSE=False,
                            DO_SMOOTH=do_smooth,
                            BLOCK=SCAN_BLOCK,
                            num_warps=8,
                        )

                    if need_hidden:
                        torch.mm(
                            logits,
                            lm_head_weight,
                            out=pre_grad_hidden.narrow(
                                0,
                                row_start,
                                chunk_rows,
                            ),
                        )

                    if need_weight:
                        if first_weight_chunk:
                            torch.mm(
                                logits.transpose(0, 1),
                                hidden_chunk,
                                out=pre_grad_weight,
                            )
                            first_weight_chunk = False
                        else:
                            pre_grad_weight.addmm_(
                                logits.transpose(0, 1),
                                hidden_chunk,
                                beta=1.0,
                                alpha=1.0,
                            )

                    del logits
                    row_start += chunk_rows

                ctx.precomputed = True
                ctx.hidden_shape = hidden_shape
                ctx.need_hidden = need_hidden
                ctx.need_weight = need_weight
                # these go through save_for_backward, not plain ctx attributes, for three reasons.
                # a plain attribute is a live reference the engine cannot release, so AccumulateGrad
                # cannot take ownership of the returned buffer and clones the whole [vocab, hidden]
                # accumulator every step. it is also invisible to saved_tensors_hooks/save_on_cpu,
                # so an offloading trainer cannot move it. and dropping the attribute in backward to
                # regain the steal is what made the second .backward() of a retain_graph=True graph
                # read a None buffer. a SavedVariable gets all three at once: the engine releases it
                # after backward when the graph is NOT retained (so the steal happens), and holds it
                # when it IS retained (so backward is re-entrant). portable does the same.
                # both are always real tensors here -- the un-needed side is allocated size 0 above
                # -- so neither the save nor the unpack needs a presence flag.
                ctx.save_for_backward(pre_grad_hidden, pre_grad_weight)
            else:
                lse = torch.empty(
                    rows,
                    device=hidden.device,
                    dtype=torch.float32,
                )

                row_start = 0
                while row_start < rows:
                    chunk_rows = min(
                        rows_per_chunk,
                        rows - row_start,
                    )
                    hidden_chunk = hidden_2d.narrow(
                        0,
                        row_start,
                        chunk_rows,
                    )

                    logits = torch.empty(
                        (chunk_rows, vocab),
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    torch.mm(
                        hidden_chunk,
                        lm_head_weight.transpose(0, 1),
                        out=logits,
                    )

                    with torch.cuda.device(hidden.device):
                        _loss_and_optional_dlogits[(chunk_rows,)](
                            logits,
                            labels_1d,
                            losses,
                            lse,
                            valid_count,
                            row_start,
                            chunk_rows,
                            vocab,
                            logits.stride(0),
                            int(ignore_index),
                            smoothing,
                            REDUCTION_MODE=0,
                            MAKE_DLOGITS=False,
                            SAVE_LSE=True,
                            DO_SMOOTH=do_smooth,
                            BLOCK=SCAN_BLOCK,
                            num_warps=8,
                        )

                    del logits
                    row_start += chunk_rows

                ctx.precomputed = False
                ctx.save_for_backward(
                    hidden_2d,
                    lm_head_weight,
                    labels_1d,
                    lse,
                )
                ctx.hidden_shape = hidden_shape
                ctx.ignore_index = int(ignore_index)
                ctx.label_smoothing = smoothing
                ctx.rows_per_chunk = rows_per_chunk

            if reduction_mode == 0:
                return losses.reshape(label_shape)
            if reduction_mode == 1:
                return losses.sum()
            # match the portable kernel: an empty or fully-ignored batch has no valid rows,
            # so report 0 loss (and 0 grads, which the in-kernel scale guard already gives)
            # instead of 0/0 = NaN. eager cross_entropy returns NaN here, but chalk's shipped
            # src/chalk/ops/flce.py deliberately clamps the denominator and production depends
            # on that: one fully-masked microbatch must not poison a whole training run.
            return losses.sum() / valid_count.clamp(min=1)

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.precomputed:
                grad_hidden = None
                grad_weight = None

                # scale out of place and keep the buffers, matching the portable kernel. scaling
                # in place and then dropping the saved refs makes the second .backward() of a
                # retain_graph=True graph read a stale buffer, which portable serves correctly.
                # at grad_output == 1.0 -- what a plain loss.backward() passes -- both multiplies
                # are skipped entirely, so the default path is cheaper than an unconditional mul_.
                unit = grad_output.numel() == 1 and float(grad_output.detach()) == 1.0

                pre_grad_hidden, pre_grad_weight = ctx.saved_tensors

                if ctx.need_hidden:
                    if not unit:
                        pre_grad_hidden = pre_grad_hidden * grad_output.to(pre_grad_hidden.dtype)
                    grad_hidden = pre_grad_hidden.reshape(ctx.hidden_shape)

                if ctx.need_weight:
                    grad_weight = pre_grad_weight
                    if not unit:
                        grad_weight = grad_weight * grad_output.to(grad_weight.dtype)

                return (
                    grad_hidden,
                    grad_weight,
                    None,
                    None,
                    None,
                    None,
                )

            (
                hidden_2d,
                lm_head_weight,
                labels_1d,
                lse,
            ) = ctx.saved_tensors

            rows, hidden_size = hidden_2d.shape
            vocab = lm_head_weight.shape[0]

            need_hidden = ctx.needs_input_grad[0]
            need_weight = ctx.needs_input_grad[1]

            if not need_hidden and not need_weight:
                return None, None, None, None, None, None

            grad_output_flat = grad_output.reshape(-1).contiguous()

            if need_hidden:
                grad_hidden_2d = torch.empty(
                    (rows, hidden_size),
                    device=hidden_2d.device,
                    dtype=hidden_2d.dtype,
                )
            else:
                grad_hidden_2d = None

            if need_weight:
                # same reasoning as the forward path: the first chunk overwrites via out=, so
                # only the rows==0 case (where the loop never runs) needs a zeroed buffer.
                _alloc = torch.zeros if rows == 0 else torch.empty
                grad_weight = _alloc(
                    lm_head_weight.shape,
                    device=lm_head_weight.device,
                    dtype=lm_head_weight.dtype,
                )
            else:
                grad_weight = None

            first_weight_chunk = True
            row_start = 0

            while row_start < rows:
                chunk_rows = min(
                    ctx.rows_per_chunk,
                    rows - row_start,
                )
                hidden_chunk = hidden_2d.narrow(
                    0,
                    row_start,
                    chunk_rows,
                )

                logits = torch.empty(
                    (chunk_rows, vocab),
                    device=hidden_2d.device,
                    dtype=hidden_2d.dtype,
                )
                torch.mm(
                    hidden_chunk,
                    lm_head_weight.transpose(0, 1),
                    out=logits,
                )

                with torch.cuda.device(hidden_2d.device):
                    _recomputed_logits_to_dlogits[
                        (
                            chunk_rows,
                            triton.cdiv(vocab, POINT_BLOCK),
                        )
                    ](
                        logits,
                        labels_1d,
                        lse,
                        grad_output_flat,
                        row_start,
                        chunk_rows,
                        vocab,
                        logits.stride(0),
                        ctx.ignore_index,
                        ctx.label_smoothing,
                        DO_SMOOTH=ctx.label_smoothing != 0.0,
                        BLOCK=POINT_BLOCK,
                        num_warps=8,
                    )

                if need_hidden:
                    torch.mm(
                        logits,
                        lm_head_weight,
                        out=grad_hidden_2d.narrow(
                            0,
                            row_start,
                            chunk_rows,
                        ),
                    )

                if need_weight:
                    if first_weight_chunk:
                        torch.mm(
                            logits.transpose(0, 1),
                            hidden_chunk,
                            out=grad_weight,
                        )
                        first_weight_chunk = False
                    else:
                        grad_weight.addmm_(
                            logits.transpose(0, 1),
                            hidden_chunk,
                            beta=1.0,
                            alpha=1.0,
                        )

                del logits
                row_start += chunk_rows

            grad_hidden = None
            if need_hidden:
                grad_hidden = grad_hidden_2d.reshape(ctx.hidden_shape)

            return (
                grad_hidden,
                grad_weight,
                None,
                None,
                None,
                None,
            )

    def flce_fn(
        hidden,
        lm_head_weight,
        labels,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
    ):
        if reduction == "none":
            reduction_mode = 0
        elif reduction == "sum":
            reduction_mode = 1
        elif reduction == "mean":
            reduction_mode = 2
        else:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")

        smoothing = float(label_smoothing)
        if smoothing < 0.0 or smoothing > 1.0:
            raise ValueError("label_smoothing must be between 0 and 1")

        return _FLCE.apply(
            hidden,
            lm_head_weight,
            labels,
            int(ignore_index),
            reduction_mode,
            smoothing,
        )

    return flce_fn
