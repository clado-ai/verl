"""rope@sm90 — chalk autoresearch kernel (one file per layer, per arch).

Cell: rope@sm90  (NVIDIA H100/H200, Hopper)
Entry: apply_fn(q, k, cos, sin, unsqueeze_dim=1) -> (q_out, k_out)   (direction: fwd+bwd)
Oracle: chalk.ops.rope._eager_apply   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32

Non-interleaved (GPT-NeoX) partial-rotary RoPE; cos/sin are [B, T, rot] broadcast over heads. One
program per (b, h, t) row rotates the two rotary halves AND copies its own pass-through tail
(rot:head_dim) in-kernel — there is NO host-side ``.copy_`` of the tail. The prior sm90 win (PR #33)
copied the ~192-wide tail on the host before the launch; on the production Qwen3.5 shape (head_dim=256,
rot=64) that strided host memcpy is a whole extra kernel over the dominant memory region and made the
"tuned" kernel LOSE to the portable kernel (which fuses the tail). This version fuses the tail into the
rotary kernel: the rotary lanes use a tight BLOCK_H (= next_pow2(half)) block while the tail uses its own
BLOCK_T (= next_pow2(head_dim-rot)) block, so the rotary math keeps low register pressure (vs the portable
kernel, which sizes its single block to the *tail* width and pays that register pressure on the rotary
lanes too). Rotary math is byte-identical to PR #33. The analytic backward reuses the same kernel with the
transposed (orthogonal-rotation) formula and the saved cos/sin — no q/k saved, no autograd recompute; the
tail gradient passes straight through (dx[rot:] = g[rot:]).

Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm90 GPU. Adopted into production dispatch (TUNED=True); load_entry selects it only
if build() also passes the op's live-GPU rope self-test, else falls back to the portable kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    X,
    OUT,
    COS,
    SIN,
    heads,
    t,
    head_dim,
    rot,
    half,
    tail,
    BACKWARD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    ht = heads * t
    b_idx = pid // ht
    r = pid % ht
    t_idx = r % t
    row_off = pid * head_dim
    cos_off = (b_idx * t + t_idx) * rot
    offs = tl.arange(0, BLOCK_H)
    mask = offs < half
    x1 = tl.load(X + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X + row_off + half + offs, mask=mask, other=0.0).to(tl.float32)
    c1 = tl.load(COS + cos_off + offs, mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(COS + cos_off + half + offs, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(SIN + cos_off + offs, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(SIN + cos_off + half + offs, mask=mask, other=0.0).to(tl.float32)
    if BACKWARD:
        o1 = x1 * c1 + x2 * s2
        o2 = x2 * c2 - x1 * s1
    else:
        o1 = x1 * c1 - x2 * s1
        o2 = x2 * c2 + x1 * s2
    tl.store(OUT + row_off + offs, o1, mask=mask)
    tl.store(OUT + row_off + half + offs, o2, mask=mask)
    # Pass-through tail [rot:head_dim], copied in-kernel (no host memcpy). Pure byte copy — no
    # upcast/rotation — so the tail is bit-identical to the input tail (fwd) or the upstream grad
    # (bwd, which passes straight through an untouched pass-through dim).
    if HAS_TAIL:
        poffs = tl.arange(0, BLOCK_T)
        pmask = poffs < tail
        xp = tl.load(X + row_off + rot + poffs, mask=pmask, other=0.0)
        tl.store(OUT + row_off + rot + poffs, xp, mask=pmask)


def _apply(x, cos, sin, backward):
    b, heads, t, head_dim = x.shape
    rot = cos.shape[-1]
    half = rot // 2
    tail = head_dim - rot
    xc = x.contiguous()
    out = torch.empty_like(xc)
    cosc = cos.contiguous()
    sinc = sin.contiguous()
    BLOCK_H = triton.next_power_of_2(half)
    has_tail = tail > 0
    BLOCK_T = triton.next_power_of_2(tail) if has_tail else 1
    grid = (b * heads * t,)
    _rope_kernel[grid](
        xc,
        out,
        cosc,
        sinc,
        heads,
        t,
        head_dim,
        rot,
        half,
        tail,
        BACKWARD=backward,
        BLOCK_H=BLOCK_H,
        HAS_TAIL=has_tail,
        BLOCK_T=BLOCK_T,
        # num_warps=2 is the sm90-tuned launch for this tiny-work, bandwidth-bound kernel: the
        # rotary lanes are few (half=next_pow2 of rot/2) and the tail is a plain copy, so 2 warps
        # keep occupancy high without the scheduling overhead that made 4/8 warps LOSE to portable.
        num_warps=2,
    )
    return out


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.save_for_backward(cos, sin)
        # Pin the launch to the input device: under multi-GPU device_map the patched layer's q/k may
        # live on a device other than Triton's current one (the portable kernel binds the device too).
        with torch.cuda.device(q.device):
            q_out = _apply(q, cos, sin, False)
            k_out = _apply(k, cos, sin, False)
        return q_out, k_out

    @staticmethod
    def backward(ctx, gq, gk):
        cos, sin = ctx.saved_tensors
        # autograd passes None for an unused output grad; return None for that input rather than
        # run the kernel on a None tensor. If both are None there is nothing to compute.
        if gq is None and gk is None:
            return None, None, None, None
        dev = (gq if gq is not None else gk).device
        with torch.cuda.device(dev):
            grad_q = _apply(gq, cos, sin, True) if gq is not None else None
            grad_k = _apply(gk, cos, sin, True) if gk is not None else None
        return grad_q, grad_k, None, None


# Verified sm90 win vs the chalk PORTABLE rope kernel (not vs eager): 1.07x median fwd+bwd on the
# production Qwen3.5 shape (B=1, q_heads=16, k_heads=4, head_dim=256, rot=64, seq=8192, bf16),
# robust across both timing orderings, byte-exact parity. load_entry selects it only if build()
# also passes the op's live-GPU rope self-test, else falls back to portable.
TUNED = True
SPEEDUP = 1.07
SPEEDUP_ANCHOR = "the portable chalk kernel"


def build():
    # Portable chalk kernel, built once and reused, for any shape the fused kernel can't index
    # safely — we degrade to the op's PORTABLE kernel, never to torch eager.
    _portable: list = []

    def _portable_entry():
        if not _portable:
            from chalk.ops.rope import _build_kernels

            _portable.append(_build_kernels())
        return _portable[0]

    def apply_fn(q, k, cos, sin, unsqueeze_dim=1):
        # The apply_rotary_pos_emb patch is global, so this entry carries the SAME production guards
        # the portable kernel has: route a non-default unsqueeze_dim, CPU/offloaded tensors, and any
        # shape the fused kernel can't index safely to the portable chalk kernel (validated for
        # those). The win kernel is validated for the standard CUDA [B, H, T, D] / unsqueeze_dim=1
        # layout only.
        from chalk.ops.rope import _shape_fallback

        if (
            unsqueeze_dim != 1
            or not (q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda)
            or _shape_fallback(q, k, cos, sin)
        ):
            return _portable_entry()(q, k, cos, sin, unsqueeze_dim)
        return _RoPE.apply(q, k, cos, sin)

    return apply_fn
