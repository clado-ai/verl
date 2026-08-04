"""Custom fused Triton RoPE kernel for Qwen3.5/3.6 full-attention layers.

Liger ships a RoPE kernel but its qwen3_5 patcher *refuses* to apply it
(``raise NotImplementedError`` "due to hybrid attention: Gated DeltaNet + Gated
Attention") — the GDN layers don't call ``apply_rotary_pos_emb`` at all, only the
full-attention layers do, and Liger's blanket patch couldn't target just those.

This module sidesteps that by monkeypatching the module-level
``transformers.models.qwen3_5.modeling_qwen3_5.apply_rotary_pos_emb`` function
itself — which ONLY the ``Qwen3_5Attention`` layers call — so the GDN path is
untouched. The HF eager version is heavily unfused: ``rotate_half`` allocates a
full ``cat([-x2, x1])`` tensor and the rotation is ~8 separate elementwise
kernels + intermediates per attention layer. We fuse the whole rotation into one
Triton kernel (forward + backward), eliminating those launches/allocations.

Correctness is gated by a live-GPU numeric self-test (loss + grad vs the eager
reference within tolerance); ANY import/compile/self-test failure leaves the
eager path untouched — correctness over speed. Enablement is install-on-call,
mirroring Liger (``apply_liger_kernel_to_qwen3``): ``install_qwen35_rope()``
patches the kernel in — there is no env flag. Calling it IS the opt-in, so the
kernel is active exactly when a consumer wires chalk in, and absent otherwise.

Semantics matched exactly to modeling_qwen3_5.apply_rotary_pos_emb:
  rotate_half(x) = cat(-x[d/2:], x[:d/2])            # GPT-NeoX / non-interleaved
  q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin); q_pass kept as-is
with rotary_dim = cos.shape[-1] possibly < head_dim (the tail is passed through),
cos/sin of shape [batch, seq, rotary_dim] (broadcast over heads).
"""

from __future__ import annotations

# Populated by install_qwen35_rope/benchmark so the worker can fold the measured speedup
# into metrics.json's `notes` (RunPod doesn't persist worker stdout, but metrics.json is
# always uploaded). Empty {} means the kernel was not engaged this run.
RESULT: dict = {}


def _build_kernels():
    """Import torch/triton and define the fused RoPE forward+backward kernels + the
    autograd Function. Returns ``apply_fn`` (HF-signature drop-in) or raises on any
    import/compile problem (the caller treats a raise as "keep eager")."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_fwd_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        H_T,
        T,
        head_dim,
        rotary_dim,
        half,
        x_row_stride,
        cs_row_stride,
        BLOCK: tl.constexpr,
    ):
        # one program == one [head_dim] vector for a single (batch, head, token)
        pid = tl.program_id(0)
        b = pid // H_T
        t = pid % T
        cs_row = b * T + t

        offs = tl.arange(0, BLOCK)
        mask_half = offs < half
        # the two rotary halves of this row
        x1 = tl.load(x_ptr + pid * x_row_stride + offs, mask=mask_half, other=0.0)
        x2 = tl.load(x_ptr + pid * x_row_stride + half + offs, mask=mask_half, other=0.0)
        cos1 = tl.load(cos_ptr + cs_row * cs_row_stride + offs, mask=mask_half, other=0.0)
        sin1 = tl.load(sin_ptr + cs_row * cs_row_stride + offs, mask=mask_half, other=0.0)
        cos2 = tl.load(cos_ptr + cs_row * cs_row_stride + half + offs, mask=mask_half, other=0.0)
        sin2 = tl.load(sin_ptr + cs_row * cs_row_stride + half + offs, mask=mask_half, other=0.0)
        # rotate_half: out1 = x1*cos1 - x2*sin1 ; out2 = x2*cos2 + x1*sin2
        out1 = x1 * cos1 - x2 * sin1
        out2 = x2 * cos2 + x1 * sin2
        tl.store(out_ptr + pid * x_row_stride + offs, out1, mask=mask_half)
        tl.store(out_ptr + pid * x_row_stride + half + offs, out2, mask=mask_half)
        # pass-through tail [rotary_dim : head_dim]
        if head_dim > rotary_dim:
            poffs = tl.arange(0, BLOCK)
            pmask = poffs < (head_dim - rotary_dim)
            xp = tl.load(x_ptr + pid * x_row_stride + rotary_dim + poffs, mask=pmask, other=0.0)
            tl.store(out_ptr + pid * x_row_stride + rotary_dim + poffs, xp, mask=pmask)

    @triton.jit
    def _rope_bwd_kernel(
        g_ptr,
        cos_ptr,
        sin_ptr,
        dx_ptr,
        H_T,
        T,
        head_dim,
        rotary_dim,
        half,
        g_row_stride,
        cs_row_stride,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // H_T
        t = pid % T
        cs_row = b * T + t
        offs = tl.arange(0, BLOCK)
        mask_half = offs < half
        g1 = tl.load(g_ptr + pid * g_row_stride + offs, mask=mask_half, other=0.0)
        g2 = tl.load(g_ptr + pid * g_row_stride + half + offs, mask=mask_half, other=0.0)
        cos1 = tl.load(cos_ptr + cs_row * cs_row_stride + offs, mask=mask_half, other=0.0)
        sin1 = tl.load(sin_ptr + cs_row * cs_row_stride + offs, mask=mask_half, other=0.0)
        cos2 = tl.load(cos_ptr + cs_row * cs_row_stride + half + offs, mask=mask_half, other=0.0)
        sin2 = tl.load(sin_ptr + cs_row * cs_row_stride + half + offs, mask=mask_half, other=0.0)
        # transpose of the forward (orthogonal rotation):
        #   dx1 = g1*cos1 + g2*sin2 ; dx2 = -g1*sin1 + g2*cos2
        dx1 = g1 * cos1 + g2 * sin2
        dx2 = -g1 * sin1 + g2 * cos2
        tl.store(dx_ptr + pid * g_row_stride + offs, dx1, mask=mask_half)
        tl.store(dx_ptr + pid * g_row_stride + half + offs, dx2, mask=mask_half)
        if head_dim > rotary_dim:
            poffs = tl.arange(0, BLOCK)
            pmask = poffs < (head_dim - rotary_dim)
            gp = tl.load(g_ptr + pid * g_row_stride + rotary_dim + poffs, mask=pmask, other=0.0)
            tl.store(dx_ptr + pid * g_row_stride + rotary_dim + poffs, gp, mask=pmask)

    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return max(p, 1)

    def _rope_one(x, cos, sin, forward: bool):
        # x: [B, H, T, D] (contiguous), cos/sin: [B, T, rotary_dim]
        B, H, T, D = x.shape
        rotary_dim = cos.shape[-1]
        half = rotary_dim // 2
        x = x.contiguous()
        out = torch.empty_like(x)
        xf = x.view(B * H * T, D)
        of = out.view(B * H * T, D)
        cosf = cos.contiguous().view(B * T, rotary_dim)
        sinf = sin.contiguous().view(B * T, rotary_dim)
        BLOCK = _next_pow2(max(half, D - rotary_dim, 1))
        grid = (B * H * T,)
        kern = _rope_fwd_kernel if forward else _rope_bwd_kernel
        # Triton launches against the *current* CUDA device, but under multi-GPU device_map the
        # layer's tensors may live on a different device than the one active when the installer
        # self-tested (e.g. self-test on cuda:0, this layer on cuda:1). Bind the current device to
        # the input tensor's device for the launch so the kernel runs where the pointers point —
        # otherwise it launches on the wrong device and faults on the cross-device pointers.
        with torch.cuda.device(x.device):
            kern[grid](
                xf,
                cosf,
                sinf,
                of,
                H * T,
                T,
                D,
                rotary_dim,
                half,
                xf.stride(0),
                cosf.stride(0),
                BLOCK=BLOCK,
                # One program per (b,h,t) doing only ~head_dim elements over a HUGE B*H*T grid
                # (~262k programs at 8192 tok) -> overhead/occupancy-bound, not compute-bound. A
                # direct fwd+bwd A/B on A100 (Qwen partial-rotary hd128/rot32) showed num_warps=2
                # beats Triton's implicit default (~4) by ~1.07x (persists under order-reversal),
                # while num_warps=8 over-subscribes and LOSES ~22%. Fewer warps/block => more
                # concurrent blocks => higher occupancy for this tiny-per-program kernel.
                num_warps=2,
            )
        return out

    class _RoPEFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, cos, sin):
            ctx.save_for_backward(cos, sin)
            q_embed = _rope_one(q, cos, sin, forward=True)
            k_embed = _rope_one(k, cos, sin, forward=True)
            return q_embed, k_embed

        @staticmethod
        def backward(ctx, gq, gk):
            cos, sin = ctx.saved_tensors
            # autograd passes None for an unused output grad; return None for that input
            # instead of calling .contiguous() on None (which would AttributeError).
            dq = _rope_one(gq.contiguous(), cos, sin, forward=False) if gq is not None else None
            dk = _rope_one(gk.contiguous(), cos, sin, forward=False) if gk is not None else None
            return dq, dk, None, None

    def apply_fn(q, k, cos, sin, unsqueeze_dim=1):
        # cos/sin arrive as [B, T, rotary_dim]; HF unsqueezes to broadcast over heads.
        # Our kernel broadcasts over heads internally, so use cos/sin as-is ([B,T,rd]).
        # The fused kernel is hard-wired to the default unsqueeze_dim=1 layout: q/k are
        # [B, H, T, D] (head at dim 1, token at dim 2) and cos/sin broadcast over heads.
        # A caller passing a different unsqueeze_dim (e.g. unsqueeze_dim=2 with q/k shaped
        # [B, T, H, D]) wants the angles broadcast along a different axis — _shape_fallback's
        # batch/seq checks would then validate the *head* axis as the token axis and the kernel
        # would rotate by head index. Route any non-default unsqueeze_dim to eager.
        # Fall back to eager when inputs aren't on CUDA (the patch is global, so a later CPU
        # path — e.g. CPU offload — would otherwise try to launch a CUDA Triton kernel on CPU
        # tensors) or for any shape we can't index safely (see _shape_fallback).
        if (
            unsqueeze_dim != 1
            or not (q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda)
            or _shape_fallback(q, k, cos, sin)
        ):
            return _eager_apply(q, k, cos, sin, unsqueeze_dim)
        return _RoPEFunction.apply(q, k, cos, sin)

    return apply_fn


def _shape_fallback(q, k, cos, sin) -> bool:
    """True iff q/k/cos/sin shapes can't be indexed safely by the fused kernel, so the
    caller must route to eager. Device-agnostic (the CUDA check lives in apply_fn) so it
    is unit-testable on CPU.

    The kernel processes each (batch, head, token) row independently, reading the two
    rotary halves at [offs] / [half + offs] and a pass-through tail of head_dim-rotary_dim;
    cos/sin are indexed per (batch, token) only — broadcast over heads internally — so the
    head COUNT of q and k is irrelevant and may differ. That is exactly the grouped-query
    attention (GQA) case (q has num_attention_heads, k has the smaller num_key_value_heads
    before repeat_kv): differing head counts are safe and must NOT force fallback.

    We fall back only on conditions that would actually mis-index or read out of bounds:
      * non-4D q or non-3D cos (interleaved / odd ranks we don't model);
      * odd rotary_dim (cos.shape[-1]); the halves split must be exact;
      * rotary_dim > q's head_dim, which would read past q's last-dim tail;
      * k's head_dim (last dim) != q's head_dim — a head_dim mismatch (NOT a head-count
        mismatch) would read past k's tail; q and k legitimately share head_dim in GQA;
      * cos/sin batch or seq not matching q's (and thus k's) batch/seq: _rope_one indexes
        cos/sin as view(B*T, rotary_dim) keyed on each tensor's own batch/seq, so a
        broadcast layout (batch 1 vs B, or a seq mismatch) would mis-index the angles;
      * k's batch/seq differing from q's: k reuses the same cos/sin, so it must share q's
        batch and seq (only the head count may differ);
      * sin not matching cos.
    """
    return (
        q.dim() != 4
        or k.dim() != 4
        or cos.dim() != 3
        or (cos.shape[-1] % 2) != 0
        or cos.shape[-1] > q.shape[-1]
        or k.shape[-1] != q.shape[-1]
        or k.shape[0] != q.shape[0]
        or k.shape[2] != q.shape[2]
        or cos.shape[0] != q.shape[0]
        or cos.shape[1] != q.shape[2]
        or sin.shape != cos.shape
    )


def _eager_apply(q, k, cos, sin, unsqueeze_dim=1):
    """The exact HF reference (used as the self-test oracle and the shape fallback)."""
    import torch

    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    rd = cos_u.shape[-1]
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]

    def rh(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = torch.cat([(q_rot * cos_u) + (rh(q_rot) * sin_u), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos_u) + (rh(k_rot) * sin_u), k_pass], dim=-1)
    return q_embed, k_embed


def self_test(
    apply_fn,
    *,
    head_dim=128,
    rotary_dim=128,
    q_heads=4,
    k_heads=4,
    batch=2,
    seq=64,
    dtype=None,
) -> bool:
    """Numeric parity of the fused kernel vs eager HF apply_rotary_pos_emb, on the
    live GPU: forward q/k AND backward dq/dk. Returns True iff within tolerance."""
    import torch

    if not torch.cuda.is_available():
        return False
    dtype = dtype or torch.bfloat16
    B, T = batch, seq
    dev = "cuda"
    # Drive the self-test's random tensors from a *local* generator so that simply enabling the
    # kernel doesn't perturb the caller's global RNG (and thus downstream dropout / sampling /
    # weight init) — callers often seed training before installing kernels.
    gen = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(B, q_heads, T, head_dim, device=dev, dtype=dtype, generator=gen, requires_grad=True)
    k = torch.randn(B, k_heads, T, head_dim, device=dev, dtype=dtype, generator=gen, requires_grad=True)
    pos = torch.arange(T, device=dev)
    inv = 1.0 / (10000 ** (torch.arange(0, rotary_dim // 2, device=dev).float() / (rotary_dim // 2)))
    ang = pos[:, None].float() * inv[None, :]
    emb = torch.cat([ang, ang], dim=-1)  # [T, rotary_dim], duplicated halves (standard RoPE)
    cos = emb.cos()[None].expand(B, T, rotary_dim).to(dtype).contiguous()
    sin = emb.sin()[None].expand(B, T, rotary_dim).to(dtype).contiguous()

    qe_ref, ke_ref = _eager_apply(q, k, cos, sin)
    (qe_ref.float().square().mean() + ke_ref.float().square().mean()).backward()
    dq_ref, dk_ref = q.grad.clone(), k.grad.clone()
    q.grad = None
    k.grad = None

    qe, ke = apply_fn(q, k, cos, sin)
    (qe.float().square().mean() + ke.float().square().mean()).backward()
    dq, dk = q.grad.clone(), k.grad.clone()

    def close(a, b, atol=2e-2, rtol=2e-2):
        return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)

    ok = close(qe, qe_ref) and close(ke, ke_ref) and close(dq, dq_ref) and close(dk, dk_ref)
    if not ok:
        print(
            "[rope] self-test FAILED "
            f"(head_dim={head_dim} rotary_dim={rotary_dim} q_heads={q_heads} k_heads={k_heads} "
            f"batch={batch} seq={seq} fwd_q={close(qe, qe_ref)} fwd_k={close(ke, ke_ref)} "
            f"bwd_q={close(dq, dq_ref)} bwd_k={close(dk, dk_ref)}) -> keeping eager",
            flush=True,
        )
    return ok


def benchmark(apply_fn, *, head_dim=128, rotary_dim=128, n_heads=16, seqs=(1024, 2048, 4096, 8192), iters=50) -> None:
    """Sweep eager-HF vs the fused kernel (forward+backward) across sequence lengths on the
    live GPU. RoPE cost scales with seq, so this shows whether the fused win grows with scale.
    Records the per-seq curve in RESULT['sweep'] and mirrors the seq=4096 point into the
    top-level fields. Diagnostic only — never raises."""
    import torch

    try:
        if not torch.cuda.is_available():
            return
        dev, dt, B = "cuda", torch.bfloat16, 1
        # Local generator so the diagnostic sweep never perturbs the caller's global RNG.
        gen = torch.Generator(device=dev).manual_seed(0)

        def bench_one(seq):
            pos = torch.arange(seq, device=dev)
            inv = 1.0 / (10000 ** (torch.arange(0, rotary_dim // 2, device=dev).float() / (rotary_dim // 2)))
            ang = pos[:, None].float() * inv[None, :]
            emb = torch.cat([ang, ang], dim=-1)
            cos = emb.cos()[None].expand(B, seq, rotary_dim).to(dt).contiguous()
            sin = emb.sin()[None].expand(B, seq, rotary_dim).to(dt).contiguous()

            def run(fn):
                q = torch.randn(B, n_heads, seq, head_dim, device=dev, dtype=dt, generator=gen, requires_grad=True)
                k = torch.randn(B, n_heads, seq, head_dim, device=dev, dtype=dt, generator=gen, requires_grad=True)
                qe, ke = fn(q, k, cos, sin)
                (qe.float().square().mean() + ke.float().square().mean()).backward()

            for _ in range(5):  # warmup (Triton JIT + autotune)
                run(_eager_apply)
                run(apply_fn)
            torch.cuda.synchronize()

            def timed(fn):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()
                for _ in range(iters):
                    run(fn)
                e.record()
                torch.cuda.synchronize()
                return s.elapsed_time(e) / iters  # ms/iter

            te, tk = timed(_eager_apply), timed(apply_fn)
            return {
                "seq": seq,
                "eager_ms": round(te, 4),
                "kernel_ms": round(tk, 4),
                "speedup": round(te / tk if tk > 0 else 0.0, 3),
            }

        sweep = []
        RESULT["head_dim"] = head_dim
        RESULT["heads"] = n_heads
        for seq in seqs:
            r = bench_one(seq)
            sweep.append(r)
            print(
                f"[rope][bench] head_dim={head_dim} heads={n_heads} seq={seq} fwd+bwd: "
                f"eager={r['eager_ms']:.3f}ms kernel={r['kernel_ms']:.3f}ms -> {r['speedup']:.2f}x",
                flush=True,
            )
            # Publish after EACH seq so a later-length failure (e.g. 8192 OOM once the model is
            # loaded) keeps the timings already computed for shorter sequences, rather than
            # discarding the whole sweep and recording only bench_error.
            RESULT["sweep"] = list(sweep)
            # Mirror the seq closest to 4096 into the top-level fields (back-compat with run #1).
            primary = min(sweep, key=lambda r: abs(r["seq"] - 4096))
            RESULT.update({k: primary[k] for k in ("seq", "eager_ms", "kernel_ms", "speedup")})
    except Exception as e:
        RESULT["bench_error"] = f"{type(e).__name__}: {e}"
        print(f"[rope][bench] skipped: {e}", flush=True)


def _self_test_entry(fn) -> None:
    for cfg in (
        {"head_dim": 128, "rotary_dim": 128, "q_heads": 4, "k_heads": 4},
        {"head_dim": 256, "rotary_dim": 64, "q_heads": 16, "k_heads": 4},
    ):
        if not self_test(fn, **cfg):
            raise RuntimeError(f"rope self-test failed for {cfg}")


def load_rope():
    """Resolve the same validated arch-or-portable rope entry production installs."""
    try:
        from chalk.ops.arch import load_entry

        return load_entry("rope", _self_test_entry, portable=_build_kernels)
    except Exception as e:
        print(f"[rope] kernel build failed ({type(e).__name__}: {e}); keeping eager", flush=True)
        return None


def install_qwen35_rope(run_benchmark: bool = False) -> bool:
    """Patch ``apply_rotary_pos_emb`` in the qwen3_5/qwen3_6 (dense + MoE) modeling modules
    with the fused Triton kernel — IFF the live-GPU self-test passes.

    Install-on-call (the Liger model): calling this function IS the opt-in — there is no
    env flag. It patches the module-level function only the full-attention layers call, so
    the GDN layers are untouched. Never raises: any failure (no GPU, build/compile error,
    self-test mismatch) leaves the eager path in place. Returns True iff the kernel was
    installed.

    ``run_benchmark`` defaults to False: the diagnostic sweep does warmups + 50 fwd/bwd
    iterations up to seq=8192, which would add large GPU startup cost (and OOM risk) on every
    training/serving boot. Pass run_benchmark=True only for explicit benchmarking runs.

    A FAILED (re)install is non-destructive: every early/False return below leaves both the
    monkeypatches and ``RESULT`` exactly as they were on entry. So if a long-lived worker
    installs successfully and a later re-install fails (build/self-test/module-lookup error),
    the previously-good fused patch stays live and ``RESULT`` keeps reporting that good install
    — return value, metrics, and the actually-active function never disagree. ``RESULT`` is
    only ever cleared/rewritten on the success path, just before returning True."""

    # the production loader already fails closed on build or live-gpu validation errors, so repeating
    # its self-tests here would double startup work without strengthening the install decision.
    apply_fn = load_rope()
    if apply_fn is None:
        return False

    # Build the set of modules to patch BEFORE mutating any of them or RESULT, so a "nothing to
    # patch" outcome returns False without having disturbed a prior good install. Qwen3.5/3.6
    # ship separate dense and MoE modeling modules; the MoE full-attention layers call their own
    # module-level apply_rotary_pos_emb (identical non-interleaved semantics), so patch both. The
    # loop tolerates modules that don't exist in the installed transformers.
    targets = []
    for mod_name in ("qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_6_moe"):
        try:
            import importlib

            mod = importlib.import_module(f"transformers.models.{mod_name}.modeling_{mod_name}")
        except Exception:
            continue
        if hasattr(mod, "apply_rotary_pos_emb"):
            targets.append((mod_name, mod))
    if not targets:
        print("[rope] no qwen3_5/3_6 modeling module to patch; keeping eager", flush=True)
        return False

    # Past the last failure point: commit. Apply the patches and (re)publish RESULT together, so
    # the global state flips from "prior install (or not engaged)" to "this install" atomically.
    patched = []
    for mod_name, mod in targets:
        mod.apply_rotary_pos_emb = apply_fn
        patched.append(mod_name)
    RESULT.clear()
    RESULT.update({"installed": True, "self_test": "passed", "patched": patched})
    print(f"[rope] fused Triton RoPE installed on {patched} (self-test passed)", flush=True)
    if run_benchmark:
        benchmark(apply_fn)
    return True
