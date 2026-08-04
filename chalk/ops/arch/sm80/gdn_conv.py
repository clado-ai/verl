"""gdn_conv@sm80 - chalk arch-tuned production overlay.

Cell: gdn_conv@sm80
Entry: conv_fn(x, weight, bias=None) -> y
Oracle: chalk.ops.gdn._eager_conv
STATUS: ADOPTED (TUNED)

verified by the chalk autoresearch verifier versus the shipped chalk kernel on a real A100
(sm80) gpu: speed 1.231x, memory 1.001x. correctness, generalization, timing,
roofline, and anti-cheat gates passed. production dispatch selects this overlay only on sm80 and
only after the op's live-gpu self-test passes; otherwise it uses the shipped portable kernel.
anchor provenance: gdn_conv has been registered in the current-anchor registry since a1d2940
(2026-07-04), predating the #86 eager-anchor fix.

Production contract: qwen3.5 depthwise causal convolution with kernel width four, optional bias, channel-first or channel-last layout handling, and forward plus input, weight, and bias gradients.
"""

TUNED = True
SPEEDUP = 1.231
SPEEDUP_ANCHOR = "the portable chalk kernel"
MEMORY_SPEEDUP = 1.001


def build():
    import torch
    import triton
    import triton.language as tl

    from chalk.ops.gdn import _build_conv_kernels

    fallback = _build_conv_kernels()

    @triton.jit
    def _fwd_cl_stream(
        X,
        W,
        BIAS,
        Y,
        T,
        C,
        XSB,
        XST,
        XSC,
        WSC,
        WSK,
        BS,
        YSB,
        YST,
        YSC,
        HAS_BIAS: tl.constexpr,
        CHUNK: tl.constexpr,
        BC: tl.constexpr,
    ):
        start = tl.program_id(0) * CHUNK
        c = tl.program_id(1) * BC + tl.arange(0, BC)
        b = tl.program_id(2)
        cm = c < C

        w0 = tl.load(W + c * WSC + 0 * WSK, mask=cm, other=0.0).to(tl.float32)
        w1 = tl.load(W + c * WSC + 1 * WSK, mask=cm, other=0.0).to(tl.float32)
        w2 = tl.load(W + c * WSC + 2 * WSK, mask=cm, other=0.0).to(tl.float32)
        w3 = tl.load(W + c * WSC + 3 * WSK, mask=cm, other=0.0).to(tl.float32)

        bias = tl.zeros((BC,), tl.float32)
        if HAS_BIAS:
            bias = tl.load(BIAS + c * BS, mask=cm, other=0.0).to(tl.float32)

        xa = tl.load(
            X + b * XSB + (start - 3) * XST + c * XSC,
            mask=cm & (start >= 3),
            other=0.0,
        ).to(tl.float32)
        xb = tl.load(
            X + b * XSB + (start - 2) * XST + c * XSC,
            mask=cm & (start >= 2),
            other=0.0,
        ).to(tl.float32)
        xc = tl.load(
            X + b * XSB + (start - 1) * XST + c * XSC,
            mask=cm & (start >= 1),
            other=0.0,
        ).to(tl.float32)

        for i in tl.range(0, CHUNK, loop_unroll_factor=1):
            t = start + i
            valid = cm & (t < T)

            xd = tl.load(
                X + b * XSB + t * XST + c * XSC,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

            z = xa * w0 + xb * w1 + xc * w2 + xd * w3 + bias
            y = z * tl.sigmoid(z)

            tl.store(
                Y + b * YSB + t * YST + c * YSC,
                y,
                mask=valid,
            )

            xa = xb
            xb = xc
            xc = xd

    @triton.jit
    def _fwd_cf(
        X,
        W,
        BIAS,
        Y,
        T,
        XSB,
        XSC,
        XST,
        WSC,
        WSK,
        BS,
        YSB,
        YSC,
        YST,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        c = tl.program_id(1)
        b = tl.program_id(2)
        mask = t < T

        w0 = tl.load(W + c * WSC + 0 * WSK).to(tl.float32)
        w1 = tl.load(W + c * WSC + 1 * WSK).to(tl.float32)
        w2 = tl.load(W + c * WSC + 2 * WSK).to(tl.float32)
        w3 = tl.load(W + c * WSC + 3 * WSK).to(tl.float32)

        base = b * XSB + c * XSC
        x0 = tl.load(
            X + base + (t - 3) * XST,
            mask=mask & (t >= 3),
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            X + base + (t - 2) * XST,
            mask=mask & (t >= 2),
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            X + base + (t - 1) * XST,
            mask=mask & (t >= 1),
            other=0.0,
        ).to(tl.float32)
        x3 = tl.load(
            X + base + t * XST,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        z = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3
        if HAS_BIAS:
            z += tl.load(BIAS + c * BS).to(tl.float32)

        tl.store(
            Y + b * YSB + c * YSC + t * YST,
            z * tl.sigmoid(z),
            mask=mask,
        )

    @triton.jit
    def _dz_from_window_cl(
        DY,
        xa,
        xb,
        xc,
        xd,
        w0,
        w1,
        w2,
        w3,
        bias,
        q,
        c,
        cm,
        b,
        T,
        DYSB,
        DYST,
        DYSC,
        BC: tl.constexpr,
    ):
        valid = cm & (q < T)
        z = xa * w0 + xb * w1 + xc * w2 + xd * w3 + bias
        dy = tl.load(
            DY + b * DYSB + q * DYST + c * DYSC,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        s = tl.sigmoid(z)
        dz = dy * s * (1.0 + z * (1.0 - s))
        return tl.where(valid, dz, 0.0)

    @triton.jit
    def _bwd_cl_stream(
        X,
        W,
        BIAS,
        DY,
        DX,
        ACC,
        T,
        C,
        XSB,
        XST,
        XSC,
        WSC,
        WSK,
        BS,
        DYSB,
        DYST,
        DYSC,
        DXSB,
        DXST,
        DXSC,
        HAS_BIAS: tl.constexpr,
        DO_X: tl.constexpr,
        DO_W: tl.constexpr,
        DO_B: tl.constexpr,
        CHUNK: tl.constexpr,
        BC: tl.constexpr,
    ):
        start = tl.program_id(0) * CHUNK
        end = tl.minimum(start + CHUNK, T)
        c = tl.program_id(1) * BC + tl.arange(0, BC)
        b = tl.program_id(2)
        cm = c < C

        w0 = tl.load(W + c * WSC + 0 * WSK, mask=cm, other=0.0).to(tl.float32)
        w1 = tl.load(W + c * WSC + 1 * WSK, mask=cm, other=0.0).to(tl.float32)
        w2 = tl.load(W + c * WSC + 2 * WSK, mask=cm, other=0.0).to(tl.float32)
        w3 = tl.load(W + c * WSC + 3 * WSK, mask=cm, other=0.0).to(tl.float32)

        bias = tl.zeros((BC,), tl.float32)
        if HAS_BIAS:
            bias = tl.load(BIAS + c * BS, mask=cm, other=0.0).to(tl.float32)

        g0 = tl.zeros((BC,), tl.float32)
        g1 = tl.zeros((BC,), tl.float32)
        g2 = tl.zeros((BC,), tl.float32)
        g3 = tl.zeros((BC,), tl.float32)
        gb = tl.zeros((BC,), tl.float32)

        xa = tl.load(
            X + b * XSB + (start - 3) * XST + c * XSC,
            mask=cm & (start >= 3),
            other=0.0,
        ).to(tl.float32)
        xb = tl.load(
            X + b * XSB + (start - 2) * XST + c * XSC,
            mask=cm & (start >= 2),
            other=0.0,
        ).to(tl.float32)
        xc = tl.load(
            X + b * XSB + (start - 1) * XST + c * XSC,
            mask=cm & (start >= 1),
            other=0.0,
        ).to(tl.float32)
        xd = tl.load(
            X + b * XSB + start * XST + c * XSC,
            mask=cm & (start < T),
            other=0.0,
        ).to(tl.float32)

        d0 = _dz_from_window_cl(
            DY,
            xa,
            xb,
            xc,
            xd,
            w0,
            w1,
            w2,
            w3,
            bias,
            start,
            c,
            cm,
            b,
            T,
            DYSB,
            DYST,
            DYSC,
            BC=BC,
        )

        if DO_W:
            core0 = cm & (start < end)
            g0 += tl.where(core0, d0 * xa, 0.0)
            g1 += tl.where(core0, d0 * xb, 0.0)
            g2 += tl.where(core0, d0 * xc, 0.0)
            g3 += tl.where(core0, d0 * xd, 0.0)
        if DO_B:
            gb += tl.where(cm & (start < end), d0, 0.0)

        xa = xb
        xb = xc
        xc = xd
        q1 = start + 1
        xd = tl.load(
            X + b * XSB + q1 * XST + c * XSC,
            mask=cm & (q1 < T),
            other=0.0,
        ).to(tl.float32)
        d1 = _dz_from_window_cl(
            DY,
            xa,
            xb,
            xc,
            xd,
            w0,
            w1,
            w2,
            w3,
            bias,
            q1,
            c,
            cm,
            b,
            T,
            DYSB,
            DYST,
            DYSC,
            BC=BC,
        )

        if DO_W:
            core1 = cm & (q1 < end)
            g0 += tl.where(core1, d1 * xa, 0.0)
            g1 += tl.where(core1, d1 * xb, 0.0)
            g2 += tl.where(core1, d1 * xc, 0.0)
            g3 += tl.where(core1, d1 * xd, 0.0)
        if DO_B:
            gb += tl.where(cm & (q1 < end), d1, 0.0)

        xa = xb
        xb = xc
        xc = xd
        q2 = start + 2
        xd = tl.load(
            X + b * XSB + q2 * XST + c * XSC,
            mask=cm & (q2 < T),
            other=0.0,
        ).to(tl.float32)
        d2 = _dz_from_window_cl(
            DY,
            xa,
            xb,
            xc,
            xd,
            w0,
            w1,
            w2,
            w3,
            bias,
            q2,
            c,
            cm,
            b,
            T,
            DYSB,
            DYST,
            DYSC,
            BC=BC,
        )

        if DO_W:
            core2 = cm & (q2 < end)
            g0 += tl.where(core2, d2 * xa, 0.0)
            g1 += tl.where(core2, d2 * xb, 0.0)
            g2 += tl.where(core2, d2 * xc, 0.0)
            g3 += tl.where(core2, d2 * xd, 0.0)
        if DO_B:
            gb += tl.where(cm & (q2 < end), d2, 0.0)

        for i in tl.range(0, CHUNK, loop_unroll_factor=1):
            t = start + i
            q = t + 3

            xa = xb
            xb = xc
            xc = xd
            xd = tl.load(
                X + b * XSB + q * XST + c * XSC,
                mask=cm & (q < T),
                other=0.0,
            ).to(tl.float32)

            d3 = _dz_from_window_cl(
                DY,
                xa,
                xb,
                xc,
                xd,
                w0,
                w1,
                w2,
                w3,
                bias,
                q,
                c,
                cm,
                b,
                T,
                DYSB,
                DYST,
                DYSC,
                BC=BC,
            )

            if DO_X:
                dx = d0 * w3 + d1 * w2 + d2 * w1 + d3 * w0
                tl.store(
                    DX + b * DXSB + t * DXST + c * DXSC,
                    dx,
                    mask=cm & (t < end),
                )

            if DO_W:
                core = cm & (q < end)
                g0 += tl.where(core, d3 * xa, 0.0)
                g1 += tl.where(core, d3 * xb, 0.0)
                g2 += tl.where(core, d3 * xc, 0.0)
                g3 += tl.where(core, d3 * xd, 0.0)

            if DO_B:
                gb += tl.where(cm & (q < end), d3, 0.0)

            d0 = d1
            d1 = d2
            d2 = d3

        if DO_W:
            tl.atomic_add(ACC + 0 * C + c, g0, mask=cm)
            tl.atomic_add(ACC + 1 * C + c, g1, mask=cm)
            tl.atomic_add(ACC + 2 * C + c, g2, mask=cm)
            tl.atomic_add(ACC + 3 * C + c, g3, mask=cm)

        if DO_B:
            tl.atomic_add(ACC + 4 * C + c, gb, mask=cm)

    @triton.jit
    def _bwd_cf_persistent(
        X,
        W,
        BIAS,
        DY,
        DX,
        GW,
        GB,
        B,
        T,
        C,
        XSB,
        XSC,
        XST,
        WSC,
        WSK,
        BS,
        DYSB,
        DYSC,
        DYST,
        DXSB,
        DXSC,
        DXST,
        HAS_BIAS: tl.constexpr,
        DO_X: tl.constexpr,
        DO_W: tl.constexpr,
        DO_B: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        off = tl.arange(0, BLOCK)
        lane = off & 31
        warp_in_block = off // 32
        leader = lane == 0

        w0 = tl.load(W + c * WSC + 0 * WSK).to(tl.float32)
        w1 = tl.load(W + c * WSC + 1 * WSK).to(tl.float32)
        w2 = tl.load(W + c * WSC + 2 * WSK).to(tl.float32)
        w3 = tl.load(W + c * WSC + 3 * WSK).to(tl.float32)

        bias = 0.0
        if HAS_BIAS:
            bias = tl.load(BIAS + c * BS).to(tl.float32)

        g0 = 0.0
        g1 = 0.0
        g2 = 0.0
        g3 = 0.0
        gb = 0.0

        for b in tl.range(0, B, loop_unroll_factor=1):
            for base_t in tl.range(0, T, BLOCK, loop_unroll_factor=1):
                t = base_t + off
                mask = t < T
                xbase = b * XSB + c * XSC

                x0 = tl.load(
                    X + xbase + (t - 3) * XST,
                    mask=mask & (t >= 3),
                    other=0.0,
                ).to(tl.float32)
                x1 = tl.load(
                    X + xbase + (t - 2) * XST,
                    mask=mask & (t >= 2),
                    other=0.0,
                ).to(tl.float32)
                x2 = tl.load(
                    X + xbase + (t - 1) * XST,
                    mask=mask & (t >= 1),
                    other=0.0,
                ).to(tl.float32)
                x3 = tl.load(
                    X + xbase + t * XST,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)

                z = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + bias
                dy = tl.load(
                    DY + b * DYSB + c * DYSC + t * DYST,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                s = tl.sigmoid(z)
                dz_raw = dy * s * (1.0 + z * (1.0 - s))
                dz = tl.where(mask, dz_raw, 0.0)

                dz1 = tl.inline_asm_elementwise(
                    "shfl.sync.down.b32 $0, $1, 1, 0x1f, 0xffffffff;",
                    "=r,r",
                    [dz],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
                dz2 = tl.inline_asm_elementwise(
                    "shfl.sync.down.b32 $0, $1, 2, 0x1f, 0xffffffff;",
                    "=r,r",
                    [dz],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
                dz3 = tl.inline_asm_elementwise(
                    "shfl.sync.down.b32 $0, $1, 3, 0x1f, 0xffffffff;",
                    "=r,r",
                    [dz],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )

                if DO_X:
                    dx = dz * w3
                    dx += tl.where(lane <= 30, dz1 * w2, 0.0)
                    dx += tl.where(lane <= 29, dz2 * w1, 0.0)
                    dx += tl.where(lane <= 28, dz3 * w0, 0.0)

                    tl.store(
                        DX + b * DXSB + c * DXSC + t * DXST,
                        dx,
                        mask=mask,
                    )

                    tl.debug_barrier()

                    q = base_t + warp_in_block * 32
                    corr_mask = leader & (q < T)

                    p1 = q - 1
                    ptr1 = DX + b * DXSB + c * DXSC + p1 * DXST
                    old1 = tl.load(
                        ptr1,
                        mask=corr_mask & (q >= 1),
                        other=0.0,
                    ).to(tl.float32)
                    corr1 = dz * w2 + dz1 * w1 + dz2 * w0
                    tl.store(
                        ptr1,
                        old1 + corr1,
                        mask=corr_mask & (q >= 1),
                    )

                    p2 = q - 2
                    ptr2 = DX + b * DXSB + c * DXSC + p2 * DXST
                    old2 = tl.load(
                        ptr2,
                        mask=corr_mask & (q >= 2),
                        other=0.0,
                    ).to(tl.float32)
                    corr2 = dz * w1 + dz1 * w0
                    tl.store(
                        ptr2,
                        old2 + corr2,
                        mask=corr_mask & (q >= 2),
                    )

                    p3 = q - 3
                    ptr3 = DX + b * DXSB + c * DXSC + p3 * DXST
                    old3 = tl.load(
                        ptr3,
                        mask=corr_mask & (q >= 3),
                        other=0.0,
                    ).to(tl.float32)
                    tl.store(
                        ptr3,
                        old3 + dz * w0,
                        mask=corr_mask & (q >= 3),
                    )

                    tl.debug_barrier()

                if DO_W:
                    g0 += tl.sum(tl.where(mask, dz * x0, 0.0), axis=0)
                    g1 += tl.sum(tl.where(mask, dz * x1, 0.0), axis=0)
                    g2 += tl.sum(tl.where(mask, dz * x2, 0.0), axis=0)
                    g3 += tl.sum(tl.where(mask, dz * x3, 0.0), axis=0)

                if DO_B:
                    gb += tl.sum(tl.where(mask, dz, 0.0), axis=0)

        if DO_W:
            tl.store(GW + c * 4 + 0, g0)
            tl.store(GW + c * 4 + 1, g1)
            tl.store(GW + c * 4 + 2, g2)
            tl.store(GW + c * 4 + 3, g3)

        if DO_B:
            tl.store(GB + c, gb)

    @triton.jit
    def _finish_grads(
        ACC,
        GW,
        GB,
        C,
        DO_W: tl.constexpr,
        DO_B: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = c < C

        if DO_W:
            tl.store(
                GW + c * 4 + 0,
                tl.load(ACC + 0 * C + c, mask=mask, other=0.0),
                mask=mask,
            )
            tl.store(
                GW + c * 4 + 1,
                tl.load(ACC + 1 * C + c, mask=mask, other=0.0),
                mask=mask,
            )
            tl.store(
                GW + c * 4 + 2,
                tl.load(ACC + 2 * C + c, mask=mask, other=0.0),
                mask=mask,
            )
            tl.store(
                GW + c * 4 + 3,
                tl.load(ACC + 3 * C + c, mask=mask, other=0.0),
                mask=mask,
            )

        if DO_B:
            tl.store(
                GB + c,
                tl.load(ACC + 4 * C + c, mask=mask, other=0.0),
                mask=mask,
            )

    def _layout(x, channels):
        if x.ndim == 2:
            if x.shape[0] == channels:
                return (
                    True,
                    1,
                    x.shape[1],
                    0,
                    x.stride(0),
                    x.stride(1),
                )
            return (
                False,
                1,
                x.shape[0],
                0,
                x.stride(1),
                x.stride(0),
            )

        if x.shape[1] == channels:
            return (
                True,
                x.shape[0],
                x.shape[2],
                x.stride(0),
                x.stride(1),
                x.stride(2),
            )

        return (
            False,
            x.shape[0],
            x.shape[1],
            x.stride(0),
            x.stride(2),
            x.stride(1),
        )

    def _logical_strides(tensor, channel_first):
        if tensor.ndim == 2:
            if channel_first:
                return 0, tensor.stride(0), tensor.stride(1)
            return 0, tensor.stride(1), tensor.stride(0)

        if channel_first:
            return tensor.stride(0), tensor.stride(1), tensor.stride(2)
        return tensor.stride(0), tensor.stride(2), tensor.stride(1)

    class _GDNConv(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias):
            ctx.set_materialize_grads(False)

            channels = weight.shape[0]
            cf, batch, tokens, xsb, xsc, xst = _layout(x, channels)

            y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            ysb, ysc, yst = _logical_strides(y, cf)

            wsc = weight.stride(0)
            wsk = weight.stride(-1)
            has_bias = bias is not None
            bs = bias.stride(0) if has_bias else 0
            bias_arg = bias if has_bias else x

            if cf:
                with torch.cuda.device(x.device):
                    _fwd_cf[(triton.cdiv(tokens, 256), channels, batch)](
                        x,
                        weight,
                        bias_arg,
                        y,
                        tokens,
                        xsb,
                        xsc,
                        xst,
                        wsc,
                        wsk,
                        bs,
                        ysb,
                        ysc,
                        yst,
                        HAS_BIAS=has_bias,
                        BLOCK=256,
                        num_warps=8,
                    )
            else:
                fwd_chunk = 16 if tokens <= 256 else 32
                with torch.cuda.device(x.device):
                    _fwd_cl_stream[
                        (
                            triton.cdiv(tokens, fwd_chunk),
                            triton.cdiv(channels, 64),
                            batch,
                        )
                    ](
                        x,
                        weight,
                        bias_arg,
                        y,
                        tokens,
                        channels,
                        xsb,
                        xst,
                        xsc,
                        wsc,
                        wsk,
                        bs,
                        ysb,
                        yst,
                        ysc,
                        HAS_BIAS=has_bias,
                        CHUNK=fwd_chunk,
                        BC=64,
                        num_warps=2,
                    )

            ctx.save_for_backward(x, weight, bias)
            ctx.cf = cf
            ctx.batch = batch
            ctx.tokens = tokens
            ctx.channels = channels
            ctx.x_strides = (xsb, xsc, xst)
            ctx.w_strides = (wsc, wsk)
            ctx.bias_stride = bs
            ctx.has_bias = has_bias
            return y

        @staticmethod
        def backward(ctx, dy):
            if dy is None:
                return None, None, None

            x, weight, bias = ctx.saved_tensors
            need_x, need_w, need_bias_input = ctx.needs_input_grad
            need_b = need_bias_input and ctx.has_bias

            if not (need_x or need_w or need_b):
                return None, None, None

            cf = ctx.cf
            batch = ctx.batch
            tokens = ctx.tokens
            channels = ctx.channels
            xsb, xsc, xst = ctx.x_strides
            wsc, wsk = ctx.w_strides
            bs = ctx.bias_stride
            dysb, dysc, dyst = _logical_strides(dy, cf)
            bias_arg = bias if ctx.has_bias else x

            dx = None
            if need_x:
                dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
                dxsb, dxsc, dxst = _logical_strides(dx, cf)
            else:
                dxsb = dxsc = dxst = 0

            gw = None
            if need_w:
                gw = torch.empty(
                    weight.shape,
                    dtype=weight.dtype,
                    device=weight.device,
                )

            gb = None
            if need_b:
                gb = torch.empty(
                    bias.shape,
                    dtype=bias.dtype,
                    device=bias.device,
                )

            if cf:
                with torch.cuda.device(x.device):
                    _bwd_cf_persistent[(channels,)](
                        x,
                        weight,
                        bias_arg,
                        dy,
                        dx if need_x else x,
                        gw if need_w else x,
                        gb if need_b else x,
                        batch,
                        tokens,
                        channels,
                        xsb,
                        xsc,
                        xst,
                        wsc,
                        wsk,
                        bs,
                        dysb,
                        dysc,
                        dyst,
                        dxsb,
                        dxsc,
                        dxst,
                        HAS_BIAS=ctx.has_bias,
                        DO_X=need_x,
                        DO_W=need_w,
                        DO_B=need_b,
                        BLOCK=256,
                        num_warps=8,
                    )
            else:
                if need_w or need_b:
                    acc = torch.zeros(
                        (5, channels),
                        dtype=torch.float32,
                        device=x.device,
                    )
                else:
                    acc = torch.empty((1,), dtype=torch.float32, device=x.device)

                if tokens <= 256:
                    chunk = 32
                elif tokens <= 1024:
                    chunk = 64
                else:
                    chunk = 128

                with torch.cuda.device(x.device):
                    _bwd_cl_stream[
                        (
                            triton.cdiv(tokens, chunk),
                            triton.cdiv(channels, 64),
                            batch,
                        )
                    ](
                        x,
                        weight,
                        bias_arg,
                        dy,
                        dx if need_x else x,
                        acc,
                        tokens,
                        channels,
                        xsb,
                        xst,
                        xsc,
                        wsc,
                        wsk,
                        bs,
                        dysb,
                        dyst,
                        dysc,
                        dxsb,
                        dxst,
                        dxsc,
                        HAS_BIAS=ctx.has_bias,
                        DO_X=need_x,
                        DO_W=need_w,
                        DO_B=need_b,
                        CHUNK=chunk,
                        BC=64,
                        num_warps=2,
                    )

                if need_w or need_b:
                    with torch.cuda.device(x.device):
                        _finish_grads[(triton.cdiv(channels, 256),)](
                            acc,
                            gw if need_w else x,
                            gb if need_b else x,
                            channels,
                            DO_W=need_w,
                            DO_B=need_b,
                            BLOCK=256,
                            num_warps=4,
                        )

            return dx, gw, gb

    candidate = _GDNConv.apply

    def conv_fn(x, weight, bias=None):
        supported = weight.shape[-1] == 4
        if not supported:
            return fallback(x, weight, bias)
        return candidate(x, weight, bias)

    return conv_fn
