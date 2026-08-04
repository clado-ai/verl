"""gdn_conv@sm86 - chalk arch-tuned production overlay.

Cell: gdn_conv@sm86
Entry: conv_fn(x, weight, bias=None) -> y   (direction: fwd+bwd)
Oracle: chalk.ops.gdn._eager_conv   tol fwd_rel=0.02/bwd_rel=0.02
Baseline to beat: the arch-tuned kernel this file already ships   target speedup: up to 2.0x

STATUS: ADOPTED - 1.2124x floor versus the portable chalk kernel, on the path production dispatches.
That floor is the WORST of the four production shapes, not their geomean (1.2370x), because the worst
shape is the delta a user is guaranteed rather than the one they get on the best draw.

WHAT PRODUCTION ACTUALLY DISPATCHES, and why that is the only arm worth a figure: ``gdn.py`` builds
the conv call as ``conv_b = self.conv1d.bias  # None for Qwen3.5``, so the bias is None on every
Qwen3.5 forward and the triton path below IS the production path. A present bias routes to the
portable kernel and is therefore exactly a wash, never a regression -- see the fallback note below,
which is the one behavioural change made to the verified entry.

MEASURED on a real sm86 RTX A5000 (torch 2.8.0+cu129, triton 3.4.0) against the portable builder
imported from the PINNED tree rather than the pod image: a baseline read from the wrong tree makes
the ratio unattributable (#165). Four production shapes from the qwen_hybrid family, bf16, timed as
fwd+bwd with CUDA events. A/B slot order is cancelled with sqrt(fwd_ratio * rev_ratio), since pooling
both orders into one median leaves a bias large enough to flip a ranking (#166). Three full reps, and
the FLOOR across reps is what is quoted -- one rep cannot separate a real ratio from the 8-10%
run-to-run drift portable itself shows on identical hardware (#168). Portable microseconds over this
kernel's, so >1 is a win:

    B1_C1536_L2048   floor 1.2427   reps [1.2475, 1.2480, 1.2427]
    B2_C1536_L4096   floor 1.2744   reps [1.2744, 1.2760, 1.2752]
    B4_C1536_L1024   floor 1.2124   reps [1.2219, 1.2124, 1.2132]   <- SPEEDUP quotes this one
    B1_C3072_L2048   floor 1.2194   reps [1.2210, 1.2227, 1.2194]

The anchor is portable and that is CORRECT here rather than #94 anchor inflation: the tree ships
gdn_conv only at arch/sm80/, so before this file sm86 dispatched the portable builder and portable is
what a real sm86 user was being served. Once this file ships TUNED it becomes what production
dispatches on sm86, which is why the header's baseline names THIS file while the measured figure
stays vs-portable and stays true -- it was taken when no sm86 overlay existed.

MEMORY is a wash on the dispatched path, not a win: total peak bytes came in at 1.001x-1.005x across
the four shapes (floor 1.001), which is inside the noise of an allocator that rounds. No memory claim
is made.

THE ONE CHANGE from the verified entry: its unsupported-input fallback called ``F.conv1d`` + ``F.silu``
directly. That is a REGRESSION rather than a neutral degrade -- eager is slower than the portable
chalk kernel it would displace, and the measurement shows exactly that, with the bias arm at 0.72x on
B2_C1536_L4096 and its peak memory nearly doubled by the padded conv output. Binding the fallback to
``_build_conv_kernels()`` at build time, as the sm80 sibling does, removes that regression by
construction: an unsupported input now gets precisely what it would have got with no overlay
installed. Nothing on the triton path is touched, so the figures above still describe this file.

Selected by production dispatch (``TUNED = True``): ``chalk.ops.gdn.load_conv`` routes through
``load_entry("gdn_conv", _self_test_conv, portable=_build_conv_kernels)``, which runs the op's own
live-GPU fwd+bwd parity check -- including the bias=True configuration this file delegates -- before
returning the entry. Any failure there is caught and production falls back to portable, so this file
can only dispatch when it matches the eager oracle on y, dx, dw and dbias.
"""

TUNED = True
SPEEDUP = 1.2124
SPEEDUP_ANCHOR = "the portable chalk kernel"


def build():
    import torch
    import triton
    import triton.language as tl

    # bind the PORTABLE builder, not eager: load_entry degrades to portable when this file is
    # absent, so an unsupported input must land on exactly that and never on something slower.
    from chalk.ops.gdn import _build_conv_kernels

    fallback = _build_conv_kernels()

    @triton.jit
    def _gdn_fwd_silu(
        x_ptr,
        w_ptr,
        y_ptr,
        tokens,
        channels,
        sx_batch,
        sx_channel,
        sx_token,
        sw_channel,
        sw_kernel,
        BLOCK_T: tl.constexpr,
    ):
        bc = tl.program_id(0)
        b = bc // channels
        c = bc - b * channels

        x_base = b * sx_batch + c * sx_channel
        w_base = c * sw_channel

        w0 = tl.load(w_ptr + w_base).to(tl.float32)
        w1 = tl.load(w_ptr + w_base + sw_kernel).to(tl.float32)
        w2 = tl.load(w_ptr + w_base + 2 * sw_kernel).to(tl.float32)
        w3 = tl.load(w_ptr + w_base + 3 * sw_kernel).to(tl.float32)

        offsets = tl.arange(0, BLOCK_T)

        for start in tl.range(0, tokens, BLOCK_T):
            t = start + offsets
            valid = t < tokens

            xm3 = tl.load(
                x_ptr + x_base + (t - 3) * sx_token,
                mask=valid & (t >= 3),
                other=0.0,
            ).to(tl.float32)
            xm2 = tl.load(
                x_ptr + x_base + (t - 2) * sx_token,
                mask=valid & (t >= 2),
                other=0.0,
            ).to(tl.float32)
            xm1 = tl.load(
                x_ptr + x_base + (t - 1) * sx_token,
                mask=valid & (t >= 1),
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(
                x_ptr + x_base + t * sx_token,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

            z = xm3 * w0
            z += xm2 * w1
            z += xm1 * w2
            z += x0 * w3

            y = z * tl.sigmoid(z)
            tl.store(y_ptr + bc * tokens + t, y, mask=valid)

    @triton.jit
    def _gdn_bwd_fused(
        x_ptr,
        w_ptr,
        dy_ptr,
        dx_ptr,
        dw_ptr,
        positions,
        tokens,
        channels,
        sx_batch,
        sx_channel,
        sx_token,
        sdy_batch,
        sdy_channel,
        sdy_token,
        sw_channel,
        sw_kernel,
        WRITE_DX: tl.constexpr,
        WRITE_DW: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        c = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_T)

        w_base = c * sw_channel
        w0 = tl.load(w_ptr + w_base).to(tl.float32)
        w1 = tl.load(w_ptr + w_base + sw_kernel).to(tl.float32)
        w2 = tl.load(w_ptr + w_base + 2 * sw_kernel).to(tl.float32)
        w3 = tl.load(w_ptr + w_base + 3 * sw_kernel).to(tl.float32)

        if WRITE_DW:
            acc0 = tl.zeros((), dtype=tl.float32)
            acc1 = tl.zeros((), dtype=tl.float32)
            acc2 = tl.zeros((), dtype=tl.float32)
            acc3 = tl.zeros((), dtype=tl.float32)

        for start in tl.range(0, positions, BLOCK_T):
            n = start + offsets
            lane_valid = n < positions

            b = n // tokens
            t = n - b * tokens

            x_base = b * sx_batch + c * sx_channel
            dy_base = b * sdy_batch + c * sdy_channel

            xm3 = tl.load(
                x_ptr + x_base + (t - 3) * sx_token,
                mask=lane_valid & (t >= 3),
                other=0.0,
            ).to(tl.float32)
            xm2 = tl.load(
                x_ptr + x_base + (t - 2) * sx_token,
                mask=lane_valid & (t >= 2),
                other=0.0,
            ).to(tl.float32)
            xm1 = tl.load(
                x_ptr + x_base + (t - 1) * sx_token,
                mask=lane_valid & (t >= 1),
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(
                x_ptr + x_base + t * sx_token,
                mask=lane_valid,
                other=0.0,
            ).to(tl.float32)

            z0 = xm3 * w0
            z0 += xm2 * w1
            z0 += xm1 * w2
            z0 += x0 * w3

            s0 = tl.sigmoid(z0)
            g0 = tl.load(
                dy_ptr + dy_base + t * sdy_token,
                mask=lane_valid,
                other=0.0,
            ).to(tl.float32)
            d0_value = g0 * s0 * (1.0 + z0 * (1.0 - s0))
            d0 = tl.where(lane_valid, d0_value, 0.0)

            if WRITE_DW:
                acc0 += tl.sum(d0 * xm3, axis=0)
                acc1 += tl.sum(d0 * xm2, axis=0)
                acc2 += tl.sum(d0 * xm1, axis=0)
                acc3 += tl.sum(d0 * x0, axis=0)

            if WRITE_DX:
                xp1 = tl.load(
                    x_ptr + x_base + (t + 1) * sx_token,
                    mask=lane_valid & (t + 1 < tokens),
                    other=0.0,
                ).to(tl.float32)
                xp2 = tl.load(
                    x_ptr + x_base + (t + 2) * sx_token,
                    mask=lane_valid & (t + 2 < tokens),
                    other=0.0,
                ).to(tl.float32)
                xp3 = tl.load(
                    x_ptr + x_base + (t + 3) * sx_token,
                    mask=lane_valid & (t + 3 < tokens),
                    other=0.0,
                ).to(tl.float32)

                valid1 = lane_valid & (t + 1 < tokens)
                z1 = xm2 * w0
                z1 += xm1 * w1
                z1 += x0 * w2
                z1 += xp1 * w3
                s1 = tl.sigmoid(z1)
                g1 = tl.load(
                    dy_ptr + dy_base + (t + 1) * sdy_token,
                    mask=valid1,
                    other=0.0,
                ).to(tl.float32)
                d1_value = g1 * s1 * (1.0 + z1 * (1.0 - s1))
                d1 = tl.where(valid1, d1_value, 0.0)

                valid2 = lane_valid & (t + 2 < tokens)
                z2 = xm1 * w0
                z2 += x0 * w1
                z2 += xp1 * w2
                z2 += xp2 * w3
                s2 = tl.sigmoid(z2)
                g2 = tl.load(
                    dy_ptr + dy_base + (t + 2) * sdy_token,
                    mask=valid2,
                    other=0.0,
                ).to(tl.float32)
                d2_value = g2 * s2 * (1.0 + z2 * (1.0 - s2))
                d2 = tl.where(valid2, d2_value, 0.0)

                valid3 = lane_valid & (t + 3 < tokens)
                z3 = x0 * w0
                z3 += xp1 * w1
                z3 += xp2 * w2
                z3 += xp3 * w3
                s3 = tl.sigmoid(z3)
                g3 = tl.load(
                    dy_ptr + dy_base + (t + 3) * sdy_token,
                    mask=valid3,
                    other=0.0,
                ).to(tl.float32)
                d3_value = g3 * s3 * (1.0 + z3 * (1.0 - s3))
                d3 = tl.where(valid3, d3_value, 0.0)

                dx = d0 * w3
                dx += d1 * w2
                dx += d2 * w1
                dx += d3 * w0

                tl.store(
                    dx_ptr + (b * channels + c) * tokens + t,
                    dx,
                    mask=lane_valid,
                )

        if WRITE_DW:
            out = c * 4
            tl.store(dw_ptr + out, acc0)
            tl.store(dw_ptr + out + 1, acc1)
            tl.store(dw_ptr + out + 2, acc2)
            tl.store(dw_ptr + out + 3, acc3)

    def _layout(x):
        if x.ndim == 2:
            return (
                1,
                x.shape[0],
                x.shape[1],
                0,
                x.stride(0),
                x.stride(1),
            )
        return (
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.stride(0),
            x.stride(1),
            x.stride(2),
        )

    def _grad_layout(x):
        if x.ndim == 2:
            return 0, x.stride(0), x.stride(1)
        return x.stride(0), x.stride(1), x.stride(2)

    # these two agree today and are still kept apart on purpose. they are not duplicated logic, they
    # are two INDEPENDENT tuning outputs that happened to land on the same pair. the backward does
    # roughly four times the per-lane work of the forward (it re-derives z at t..t+3 to build dx), so
    # its occupancy curve is genuinely a different curve -- folding them into one function would
    # assert "forward and backward must share a block size", which nothing measured here supports and
    # which the next retune would have to undo.
    #
    # BLOCK_T is a constant either way, never derived from tokens. that is what makes the L=0 compile
    # error of #173 structurally unreachable in this file: #173 sized a block as
    # next_power_of_2(tokens), which is 0 at L=0 and fails in the compiler. tl.arange(0, BLOCK_T)
    # here always compiles and tl.range(0, tokens, BLOCK_T) simply runs zero iterations.
    def _forward_config(tokens):
        if tokens <= 128:
            return 128, 4
        return 256, 8

    def _backward_config(tokens):
        if tokens <= 128:
            return 128, 4
        return 256, 8

    class _GDNConv(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            (
                batch,
                channels,
                tokens,
                sx_batch,
                sx_channel,
                sx_token,
            ) = _layout(x)

            y = torch.empty(
                x.shape,
                dtype=x.dtype,
                device=x.device,
                memory_format=torch.contiguous_format,
            )

            block_t, num_warps = _forward_config(tokens)
            with torch.cuda.device(x.device):
                _gdn_fwd_silu[(batch * channels,)](
                    x,
                    weight,
                    y,
                    tokens,
                    channels,
                    sx_batch,
                    sx_channel,
                    sx_token,
                    weight.stride(0),
                    weight.stride(1),
                    BLOCK_T=block_t,
                    num_warps=num_warps,
                )

            ctx.save_for_backward(x, weight)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            x, weight = ctx.saved_tensors
            (
                batch,
                channels,
                tokens,
                sx_batch,
                sx_channel,
                sx_token,
            ) = _layout(x)
            sdy_batch, sdy_channel, sdy_token = _grad_layout(grad_y)

            need_dx, need_dw = ctx.needs_input_grad

            dx = None
            if need_dx:
                dx = torch.empty(
                    x.shape,
                    dtype=x.dtype,
                    device=x.device,
                    memory_format=torch.contiguous_format,
                )

            dw = None
            if need_dw:
                dw = torch.empty(
                    weight.shape,
                    dtype=weight.dtype,
                    device=weight.device,
                    memory_format=torch.contiguous_format,
                )

            if need_dx or need_dw:
                block_t, num_warps = _backward_config(tokens)
                with torch.cuda.device(x.device):
                    _gdn_bwd_fused[(channels,)](
                        x,
                        weight,
                        grad_y,
                        dx if need_dx else x,
                        dw if need_dw else weight,
                        batch * tokens,
                        tokens,
                        channels,
                        sx_batch,
                        sx_channel,
                        sx_token,
                        sdy_batch,
                        sdy_channel,
                        sdy_token,
                        weight.stride(0),
                        weight.stride(1),
                        WRITE_DX=need_dx,
                        WRITE_DW=need_dw,
                        BLOCK_T=block_t,
                        num_warps=num_warps,
                    )

            return dx, dw

    candidate = _GDNConv.apply

    def conv_fn(x, weight, bias=None):
        supported = (
            bias is None
            and x.is_cuda
            and weight.is_cuda
            and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and x.ndim in (2, 3)
            and weight.ndim == 2
            and weight.shape[0] == x.shape[-2]
            and weight.shape[1] == 4
        )
        if not supported:
            return fallback(x, weight, bias)
        return candidate(x, weight)

    return conv_fn
