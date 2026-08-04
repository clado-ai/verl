"""rope@sm80 — chalk autoresearch kernel (one file per layer, per arch).

Cell: rope@sm80  (NVIDIA A100, Ampere)
Entry: apply_fn(q, k, cos, sin, unsqueeze_dim=1) -> (q_out, k_out)   (direction: fwd+bwd)
Oracle: chalk.ops.rope._eager_apply   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32

Non-interleaved (GPT-NeoX) partial-rotary RoPE:
  rot = cos.shape[-1] (<= head_dim); half = rot//2
  x1 = t[:half], x2 = t[half:rot]        (cos/sin split the same way, both halves loaded)
  out[:half]    = x1*cos1 - x2*sin1
  out[half:rot] = x2*cos2 + x1*sin2
  out[rot:]     = t[rot:]                 (pass-through tail)
cos/sin are [B, T, rot] and broadcast over heads.

Design (A100). The GPU rotation is only ~60us on the largest dev shape, so the FULL fwd+bwd step is
dominated by CPU dispatch / kernel-launch / autograd Python overhead, NOT HBM. Two things drive the
design:

  * ONE kernel launch handles BOTH q and k in the forward, and ONE in the backward — the grid's head
    axis spans B*(NHq+NHk) virtual rows; a uniform (program-level) branch routes each row to the q or
    k buffer. This halves the launch count (4 -> 2) vs a per-tensor kernel. The rotation is inlined in
    both arms (a @triton.jit device helper does NOT compile under the verifier's exec sandbox).
  * STRIDE-AWARE loads (no .contiguous() on q/k or the grad tensors) — on this dispatch-bound step
    every torch op dispatch counts, and forcing contiguity added ~8 pointless dispatches per step.
    Reading native strides also lets us consume the [B,H,T,D] head-major layout directly, coalesced,
    with NO transpose/copy (Liger reinterprets the same layout and gets far fewer, huge programs).
  * cos/sin (tiny) are made contiguous once in fwd and reused (already contiguous) in bwd.
  * The analytic backward reuses the SAME kernel with the transposed (orthogonal-rotation) formula
    and the saved cos/sin — no q/k saved, no autograd recompute.
  * fp32 math, bf16 I/O; pass-through tail copied exactly. A lean fixed config (no autotune) keeps
    per-call dispatch minimal where GPU time is negligible.
Verified by the chalk autoresearch verifier on a real sm80 GPU. build() returns the entry callable.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fused(
    Xq,
    Xk,
    OXq,
    OXk,
    COS,
    SIN,
    q_bs,
    q_hs,
    q_ts,
    q_ds,
    k_bs,
    k_hs,
    k_ts,
    k_ds,
    cos_bs,
    cos_ts,
    T,
    NHq,
    NHk,
    ROT,
    HALF,
    PASS,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HAS_PASS: tl.constexpr,
    IS_BWD: tl.constexpr,
):
    pid_t = tl.program_id(0)
    r = tl.program_id(1)  # virtual head row over q THEN k
    NHT = NHq + NHk
    b = r // NHT
    hr = r % NHT
    is_q = hr < NHq

    t0 = pid_t * BLOCK_T
    trow = t0 + tl.arange(0, BLOCK_T)
    tmask = trow < T
    d = tl.arange(0, BLOCK_D)
    dmask = d < HALF
    p = tl.arange(0, BLOCK_P)
    m = tmask[:, None] & dmask[None, :]

    cbase = b * cos_bs
    c1 = trow[:, None] * cos_ts + d[None, :]
    c2 = trow[:, None] * cos_ts + (HALF + d[None, :])
    cos1 = tl.load(COS + cbase + c1, mask=m, other=0.0).to(tl.float32)
    sin1 = tl.load(SIN + cbase + c1, mask=m, other=0.0).to(tl.float32)
    cos2 = tl.load(COS + cbase + c2, mask=m, other=0.0).to(tl.float32)
    sin2 = tl.load(SIN + cbase + c2, mask=m, other=0.0).to(tl.float32)

    if is_q:
        base = b * q_bs + hr * q_hs
        o1 = trow[:, None] * q_ts + d[None, :] * q_ds
        o2 = trow[:, None] * q_ts + (HALF + d[None, :]) * q_ds
        x1 = tl.load(Xq + base + o1, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(Xq + base + o2, mask=m, other=0.0).to(tl.float32)
        if IS_BWD:
            y1 = x1 * cos1 + x2 * sin2
            y2 = x2 * cos2 - x1 * sin1
        else:
            y1 = x1 * cos1 - x2 * sin1
            y2 = x2 * cos2 + x1 * sin2
        tl.store(OXq + base + o1, y1, mask=m)
        tl.store(OXq + base + o2, y2, mask=m)
        if HAS_PASS:
            pmask = tmask[:, None] & (p < PASS)[None, :]
            op = trow[:, None] * q_ts + (ROT + p[None, :]) * q_ds
            tl.store(OXq + base + op, tl.load(Xq + base + op, mask=pmask, other=0.0), mask=pmask)
    else:
        base = b * k_bs + (hr - NHq) * k_hs
        o1 = trow[:, None] * k_ts + d[None, :] * k_ds
        o2 = trow[:, None] * k_ts + (HALF + d[None, :]) * k_ds
        x1 = tl.load(Xk + base + o1, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(Xk + base + o2, mask=m, other=0.0).to(tl.float32)
        if IS_BWD:
            y1 = x1 * cos1 + x2 * sin2
            y2 = x2 * cos2 - x1 * sin1
        else:
            y1 = x1 * cos1 - x2 * sin1
            y2 = x2 * cos2 + x1 * sin2
        tl.store(OXk + base + o1, y1, mask=m)
        tl.store(OXk + base + o2, y2, mask=m)
        if HAS_PASS:
            pmask = tmask[:, None] & (p < PASS)[None, :]
            op = trow[:, None] * k_ts + (ROT + p[None, :]) * k_ds
            tl.store(OXk + base + op, tl.load(Xk + base + op, mask=pmask, other=0.0), mask=pmask)


# Lean fixed launch config. GPU time here is ~60us of the ~hundreds-of-us step, so per-call autotune
# (key hash + dict lookup + config bench) is pure overhead; a good static config wins on dispatch.
_BLOCK_T = 32
_NUM_WARPS = 4
_NUM_STAGES = 3


def _run(q, k, cos, sin, is_bwd):
    # Stride-aware: NO .contiguous() on q/k or the grad tensors (each such call is a wasted dispatch
    # on this dispatch-bound step). empty_like preserves the input layout, and we index outputs with
    # the same strides, so a non-contiguous input yields a correctly-valued output in its own layout.
    B, NHq, T, HD = q.shape
    NHk = k.shape[1]
    ROT = cos.shape[-1]
    HALF = ROT // 2
    PASS = HD - ROT
    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    qs = q.stride()
    ks = k.stride()
    _rope_fused[(triton.cdiv(T, _BLOCK_T), B * (NHq + NHk))](
        q,
        k,
        oq,
        ok,
        cos,
        sin,
        qs[0],
        qs[1],
        qs[2],
        qs[3],
        ks[0],
        ks[1],
        ks[2],
        ks[3],
        cos.stride(0),
        cos.stride(1),
        T,
        NHq,
        NHk,
        ROT,
        HALF,
        PASS,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=triton.next_power_of_2(max(HALF, 1)),
        BLOCK_P=triton.next_power_of_2(max(PASS, 1)),
        HAS_PASS=PASS > 0,
        IS_BWD=is_bwd,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return oq, ok


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        # cos/sin are tiny; make them contiguous ONCE so the kernel indexes them with a plain
        # [bs, ts, 1] stride, and reuse the contiguous copies in backward (no re-contiguous there).
        cos = cos.contiguous()
        sin = sin.contiguous()
        # Pin the launch to the input device: under multi-GPU device_map the patched layer's q/k may
        # be on a device other than Triton's current one (the portable kernel binds the device too).
        with torch.cuda.device(q.device):
            oq, ok = _run(q, k, cos, sin, False)
        ctx.save_for_backward(cos, sin)
        ctx.shapes = (q.shape, k.shape)
        ctx.meta = (q.dtype, q.device)
        return oq, ok

    @staticmethod
    def backward(ctx, gq, gk):
        cos, sin = ctx.saved_tensors
        if gq is None and gk is None:
            return None, None, None, None
        qshape, kshape = ctx.shapes
        dtype, device = ctx.meta
        # Zero-fill any unused-output side to run the fused backward, but return None (not a zero
        # tensor) for that input to honor autograd's grad contract.
        gq_none, gk_none = gq is None, gk is None
        if gq_none:
            gq = torch.zeros(qshape, dtype=dtype, device=device)
        if gk_none:
            gk = torch.zeros(kshape, dtype=dtype, device=device)
        with torch.cuda.device(device):
            dq, dk = _run(gq, gk, cos, sin, True)
        return (None if gq_none else dq), (None if gk_none else dk), None, None


# Verified sm80 (A100) win from the hive autoresearch grid (wave 2): 3.28x eager / 1.37x vs
# Liger rope / 4.0x vs the portable current — fuses q+k into ONE fwd + ONE bwd Triton launch with
# an analytic transposed backward, stride-aware to drop the per-step .contiguous() dispatches.
# load_entry selects it only if build() also passes the op's live-GPU rope self-test.
TUNED = True
SPEEDUP = 1.37
# 1.37 is the LIGER arm of the three figures above, and Liger is not what production falls back to --
# load_entry degrades to the op's portable kernel, so vs-portable (4.0x above) is the delta a user
# actually gets. The two differ by ~3x, so printing 1.37 as a vs-portable claim would be false in the
# understating direction. Restating it means re-measuring vs portable, not copying the 4.0x comment:
# that figure is one wave-2 run, and portable drifts run-to-run on identical hardware.
SPEEDUP_ANCHOR = None


def build():
    def rope_fn(q, k, cos, sin):
        return _RoPE.apply(q, k, cos, sin)

    # Production (apply_rotary_pos_emb) passes unsqueeze_dim; Qwen3.5/3.6 always use 1 (the standard
    # [B, 1, S, D] cos/sin broadcast this kernel is built + self-tested for). Accept + assert it so
    # the entry matches the portable signature and load_entry can drop it in.
    def apply_fn(q, k, cos, sin, unsqueeze_dim=1):
        # The class patch is global, so this entry must carry the SAME production guards the portable
        # kernel has: route CPU/offloaded tensors, a non-default unsqueeze_dim, and any shape the fused
        # kernel can't index safely back to the exact eager path (the win kernel is validated only for
        # the standard CUDA [B, H, T, D] / unsqueeze_dim=1 layout).
        from chalk.ops.rope import _eager_apply
        from chalk.ops.rope import _shape_fallback

        if (
            unsqueeze_dim != 1
            or not (q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda)
            or _shape_fallback(q, k, cos, sin)
        ):
            return _eager_apply(q, k, cos, sin, unsqueeze_dim)
        return rope_fn(q, k, cos, sin)

    return apply_fn
