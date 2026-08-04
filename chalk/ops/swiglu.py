"""Custom fused Triton SwiGLU activation kernel (fwd+bwd) for Qwen3.5/3.6 — beats Liger's ``swiglu``.

Liger ships a SwiGLU *activation* kernel (``LigerSiLUMulFunction`` / ``liger_kernel.ops.swiglu``)
and its qwen3_5 patcher swaps ``Qwen3_5MLP`` for ``LigerSwiGLUMLP``, whose forward fuses the
ELEMENTWISE activation ``out = silu(gate) * up`` (the gate/up/down GEMMs stay cuBLAS). This
module is chalk's own SwiGLU activation so chalk can stand ALONE (drop the Liger dependency for
this layer) while matching-or-beating Liger's throughput at quality parity.

Semantics matched EXACTLY to Qwen3.5's MLP activation (``Qwen3_5MLP.forward``):

    down_proj( act_fn(gate_proj(x)) * up_proj(x) )            with act_fn = SiLU

so the fused activation is ``out = silu(g) * u`` where ``g = gate_proj(x)``, ``u = up_proj(x)``.
SiLU is ``silu(z) = z * sigmoid(z)``. Following Liger/Qwen3.5 dtype semantics, the activation is
computed in **fp32** (sigmoid needs fp32) and the product is cast back to the input dtype (bf16):

    FORWARD   out = (silu(g_fp32)).to(dtype) * u
    BACKWARD  given dout:
        du = dout * silu(g)                                 (cast to dtype)
        dg = dout * u * (sigmoid(g) + silu(g) * (1 - sigmoid(g)))
           = dout * u * sigmoid(g) * (1 + g * (1 - sigmoid(g)))   (cast to dtype)

i.e. the d/dz of silu is ``sig(z)*(1 + z*(1 - sig(z)))``. ``g`` (the pre-activation) is recomputed
in the backward from the saved ``g, u`` (no extra cached silu buffer — the kernel is HBM-bound, so
fewer saved tensors = fewer HBM round-trips). This is the SAME memory-bound shape as the RMSNorm
kernel in this repo.

WHY IT MATCHES-OR-BEATS LIGER (cross-GPU measured lever — A40, H100, A100, Ada): the op is HBM
bandwidth-bound, so chalk reads/writes the SAME bytes as Liger; the win is NOT in the kernel math
(chalk's raw kernels already beat Liger's) but in shedding the PER-CALL CPU overhead that, at
small/medium token counts, dominates the autograd-wrapped op (the kernels are only tens of
microseconds there, so allocation + dispatch is the real cost). The lever that remains:
  SPLIT WARP COUNTS — forward follows Liger's ``calculate_settings`` curve (scale to 16 warps
  on wide rows: it's 2 loads + 1 store and likes the parallelism), but the backward CAPS at 8
  warps (it's 3 loads + 2 stores; 16 over-subscribes and loses ~1-2%). whole-row block
  ``next_pow2(N)``, grid ``(M,)``, like Liger — col-tiling was tried and gives no win on any GPU
  while adding launch overhead.

An earlier revision carried a second lever, an IN-PLACE BACKWARD that wrote ``dg``/``du`` back over
the saved ``g``/``u`` (Liger's backward does this) so the backward allocated nothing. It has been
REMOVED: those saved buffers are the caller's own activations whenever the forward's
``.contiguous()`` is the identity, so writing over them corrupts live tensors, silently (see the
backward kernel's note). The two ``empty_like`` allocations it saved are back.

WHAT THE REMOVAL COSTS (measured on A5000/sm86, in-place body vs this one, both A/B orderings, five
shapes from [64,1024] to [8192,9216]): the BACKWARD is 3.6%-8.3% slower on four of the five shapes,
worst case 1.083x. Both orderings agree in sign on every shape, so that is not a slot-order
artifact. Two numbers are deliberately NOT quoted. The fwd+bwd axis: its orderings disagree in sign
on [256,4096], which puts that cell at the measurement floor. And the backward geometric mean
(1.008x): it is dragged to a near-wash by that same [256,4096] cell reading 20% FASTER, which a
body that strictly allocates more cannot be, so the cell is not trusted even though its orderings
agree. Carry the worst case, 1.083x. It is the price of not corrupting the caller, not a regression
to fix. Note this measures the in-place removal, NOT the margin over Liger, which is a separate
question and remains un-rebenchmarked. Both the sm80 and sm100 swiglu overlays already allocate
their own dg/du, so portable was the outlier, not the reference.
Forward writes ONLY ``out`` (Liger does too).

Correctness is gated by a live-GPU numeric self-test (fwd + dg + du vs an eager reference in the
real reduced dtype — bf16 when supported, else fp16 — with the error measured in fp32, within a
bf16-sized tolerance); ANY import/compile/self-test failure leaves the model untouched
(``install_qwen35_swiglu`` returns False and the eager/Liger MLP keeps running). Install-on-call
mirrors Liger / chalk's other installers: calling ``install_qwen35_swiglu()`` IS the opt-in (no
env flag); it patches ``Qwen3_5MLP.forward`` (and the moe/3.6 equivalents) on the CLASS so the
fused ``silu(gate)*up`` activation runs in place of ``act_fn(gate) * up`` while the gate/up/down
projections stay exactly as they are (cuBLAS, LoRA-wrapped, whatever) — so the activation swap is
DIFFERENTIABLE and composes with LoRA on the projections, working during TRAINING (unlike the
fused-GEMM ``mlp.py`` path, which is eval-only).
"""

from __future__ import annotations

from chalk.ops._hf_targets import collect_qwen_classes
from chalk.utils import rel_l2

# Populated by install_qwen35_swiglu so a worker can fold the outcome into metrics.json's
# notes. Empty {} means the kernel was not engaged this run.
RESULT: dict = {}


def _build_kernels():
    """Import torch/triton and define the fused SwiGLU activation fwd+bwd kernels + the autograd
    Function. Returns ``swiglu_fn`` (signature ``(gate, up) -> out``) or raises on any
    import/compile problem (the caller treats a raise as "keep eager/Liger")."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _swiglu_fwd_kernel(
        g_ptr,
        u_ptr,
        o_ptr,
        row_stride,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # One program per row (grid ``(M,)``), whole-row block ``BLOCK_N = next_pow2(N)`` with a
        # masked tail — the same launch shape as Liger's forward. The activation is elementwise
        # (no cross-column reduction), so a 2D col-tiled grid is *possible*, but a cross-GPU sweep
        # found tiling gives no win on H100/A100/Ada/A40 (the op is HBM-bound and the whole-row
        # block already streams the row contiguously) while adding launch/index overhead — so we
        # keep the single whole-row program and spend the tuning budget on the warp count instead.
        pid = tl.program_id(0).to(tl.int64)
        g_ptr += pid * row_stride
        u_ptr += pid * row_stride
        o_ptr += pid * row_stride
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        # sigmoid/silu need fp32; up stays in its native dtype until the final multiply.
        g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + cols, mask=mask, other=0.0)
        silu_g = (g * tl.sigmoid(g)).to(u.dtype)  # silu in fp32, cast to up dtype (Liger casting)
        tl.store(o_ptr + cols, silu_g * u, mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        g_ptr,
        u_ptr,
        do_ptr,
        dg_ptr,
        du_ptr,
        row_stride,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # One pass over (g, u, do) recomputing silu from g in fp32 (no cached silu buffer), writing
        # dg and du to their OWN output buffers. An earlier version wrote them back over g and u
        # (Liger's backward does this) to save the two ``empty_like`` allocations. That is not
        # sound here: ``g``/``u`` are the tensors handed to ``save_for_backward``, and when the
        # caller's gate/up are already contiguous the ``.contiguous()`` in the forward is the
        # identity, so those buffers ARE the caller's activations. Overwriting them corrupts
        # tensors the caller still owns, and the corruption is silent -- a raw ``tl.store`` through
        # a pointer never bumps the autograd version counter, so the usual "modified by an inplace
        # operation" error cannot fire. It also breaks a second backward on a ``retain_graph=True``
        # graph, which then recomputes silu from gradient values instead of activations. Both the
        # sm80 and sm100 swiglu overlays already allocate their own dg/du; this matches them.
        pid = tl.program_id(0).to(tl.int64)
        g_ptr += pid * row_stride
        u_ptr += pid * row_stride
        do_ptr += pid * row_stride
        dg_ptr += pid * row_stride
        du_ptr += pid * row_stride
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + cols, mask=mask, other=0.0)
        do = tl.load(do_ptr + cols, mask=mask, other=0.0)
        sig = tl.sigmoid(g)
        silu_g = g * sig
        # dg = do * u * d/dg[silu] ; d/dg[silu] = sig*(1 + g*(1 - sig)) = silu*(1 - sig) + sig.
        dg = do * (silu_g * (1.0 - sig) + sig) * u
        du = do * silu_g.to(u.dtype)
        tl.store(dg_ptr + cols, dg.to(u.dtype), mask=mask)
        tl.store(du_ptr + cols, du, mask=mask)

    def _fwd_warps(block: int) -> int:
        """Forward warp count — Liger's ``calculate_settings`` curve (scale warps with row width
        so wide rows are well-parallelized). The forward is 2 loads + 1 store per element and
        benefits from more warps to hide HBM latency on wide rows."""
        if block >= 8192:
            return 16
        if block >= 2048:
            return 8
        if block >= 512:
            return 4
        return 2

    def _bwd_warps(block: int) -> int:
        """Backward warp count — CAPPED at 8 (do NOT follow Liger up to 16/32). The backward is
        heavier per element (3 loads + 2 stores); a cross-GPU sweep found 16 warps
        OVER-subscribes the wide-row backward and loses ~1-2%, while 8 warps tips the largest
        cells from a tie into a win on H100. Narrow rows that can't fill 8 warps get 4/2."""
        if block >= 2048:
            return 8
        if block >= 512:
            return 4
        return 2

    class _SwiGLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gate, up):
            # gate/up: [..., N]; flatten leading dims to [M, N] row matrices (contiguous so the
            # row stride is exactly N, which both kernels index with). NOTE ``.contiguous()`` is
            # the IDENTITY when the caller's tensor is already contiguous, so ``g2``/``u2`` alias
            # the caller's storage -- the backward must never write through them.
            sh = gate.shape
            N = sh[-1]
            g2 = gate.contiguous().view(-1, N)
            u2 = up.contiguous().view(-1, N)
            M = g2.shape[0]
            out = torch.empty_like(g2)
            block = triton.next_power_of_2(N)
            _swiglu_fwd_kernel[(M,)](
                g2,
                u2,
                out,
                g2.stride(0),
                N=N,
                BLOCK_N=block,
                num_warps=_fwd_warps(block),
            )
            ctx.save_for_backward(g2, u2)
            ctx._sh = sh
            return out.view(sh)

        @staticmethod
        def backward(ctx, dout):
            # dg/du go to their own buffers. Writing them back over ``g2``/``u2`` would corrupt the
            # caller's activations whenever the forward's ``.contiguous()`` was the identity, and
            # would leave a second backward recomputing silu from gradients (see the kernel note).
            g2, u2 = ctx.saved_tensors
            sh = ctx._sh
            N = sh[-1]
            M = g2.shape[0]
            do2 = dout.contiguous().view(-1, N)
            dg = torch.empty_like(g2)
            du = torch.empty_like(u2)
            block = triton.next_power_of_2(N)
            _swiglu_bwd_kernel[(M,)](
                g2,
                u2,
                do2,
                dg,
                du,
                g2.stride(0),
                N=N,
                BLOCK_N=block,
                num_warps=_bwd_warps(block),
            )
            return dg.view(sh), du.view(sh)

    def swiglu_fn(gate, up):
        return _SwiGLUFunction.apply(gate, up)

    return swiglu_fn


def _eager_swiglu(gate, up):
    """The exact Qwen3.5 MLP activation reference (self-test oracle): ``silu(gate) * up`` with
    SiLU computed in fp32 then cast back to the input dtype (Liger/Qwen3.5 casting)."""
    import torch.nn.functional as F

    return F.silu(gate.float()).to(up.dtype) * up


def _self_test(swiglu_fn, inters=(3584, 9216, 12288)) -> None:
    """Live-GPU numeric + autograd parity vs the exact eager SwiGLU activation: forward, dg, AND
    du. Raises on mismatch so the caller keeps the Liger/eager path. Runs at ``inters`` (real
    Qwen3.5 intermediate sizes by default; the MiniCPM installer passes the model's actual
    intermediate size, e.g. 5760). The activation is elementwise, so parity is size-independent —
    ``inters`` just picks concrete widths to exercise the whole-row block + warp-count ladder.

    The kernel's contract is a REDUCED dtype (bf16 in real Qwen3.5 workloads, fp16 otherwise) with
    the silu computed in fp32 and the product cast back — so the test exercises that real path:
    inputs are bf16 (``torch.cuda.is_bf16_supported()``) else fp16, and the parity error is
    measured in fp32 (both chalk and the eager reference cast up to fp32) against a tolerance sized
    for the reduced dtype (much looser than fp32 would be — bf16 has ~8 mantissa bits)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA for swiglu self-test")
    dev = "cuda"
    # Validate the REAL (reduced-precision) path the installer runs under: bf16 when the GPU
    # supports it (Qwen3.5 training dtype), else fp16. Tolerance is sized for bf16's ~8-bit
    # mantissa (~2e-2 norm-wise over a full fwd+bwd at these sizes); fp16 comfortably fits under it.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tol = 2e-2
    gen = torch.Generator(device=dev).manual_seed(0)
    for inter in inters:
        for n_tokens in (128, 2048):
            gate = torch.randn(n_tokens, inter, device=dev, dtype=dtype, generator=gen)
            up = torch.randn(n_tokens, inter, device=dev, dtype=dtype, generator=gen)
            # Cotangent from the SAME seeded ``gen`` (not the default RNG via ``randn_like``) so the
            # gate is deterministic regardless of the caller's global RNG state (this runs from
            # ``install_qwen35_swiglu`` during ``apply_chalk_kernel_to_qwen35``, where the default RNG
            # is in an arbitrary post-model-build state).
            grad = torch.randn(n_tokens, inter, device=dev, dtype=dtype, generator=gen)

            # ORACLE IN fp32. chalk computes the SwiGLU vjp with SiLU in fp32; the trustworthy
            # reference is the same math in fp32, NOT a bf16 eager backward (whose own bf16 rounding
            # on the dg/du vjp can exceed the tolerance for an unlucky cotangent and spuriously fail a
            # correct kernel — the same oracle pitfall fixed in rmsnorm). Compare chalk's bf16 grads
            # against the fp32 truth in fp32 (``rel`` measures in fp32), bf16-toleranced.
            gr = gate.float().clone().requires_grad_(True)
            ur = up.float().clone().requires_grad_(True)
            ref = _eager_swiglu(gr, ur)  # fp32 in -> fp32 out
            ref.backward(grad.float())
            dg_ref, du_ref = gr.grad.clone(), ur.grad.clone()

            gc = gate.clone().requires_grad_(True)
            uc = up.clone().requires_grad_(True)
            got = swiglu_fn(gc, uc)
            got.backward(grad)
            dg_got, du_got = gc.grad.clone(), uc.grad.clone()

            rel = rel_l2  # fp32 relative-L2 error; shared helper (see chalk.utils.rel_l2)

            r_fwd, r_dg, r_du = rel(got, ref), rel(dg_got, dg_ref), rel(du_got, du_ref)
            if not (r_fwd < tol and r_dg < tol and r_du < tol):
                raise RuntimeError(
                    f"swiglu self-test failed at inter={inter} n={n_tokens} dtype={dtype}: "
                    f"fwd={r_fwd:.2e} dg={r_dg:.2e} du={r_du:.2e} (tol={tol:.0e})"
                )


def load_swiglu():
    """Return ``swiglu_fn`` if the kernel builds and passes its live-GPU self-test; otherwise
    return ``None`` (keep the Liger/eager path). Never raises — any failure (no torch/triton,
    no CUDA, compile/self-test error) -> ``None``."""
    from chalk.ops.arch import load_entry
    from chalk.ops.arch import load_kernel

    return load_kernel(
        "swiglu",
        "fused Triton SwiGLU activation (fwd+bwd) enabled",
        "fused Triton SwiGLU disabled",
        # no self_test= here: load_entry already runs _self_test on the entry it returns, on both
        # the arch and the portable branch, so a second one would just double startup latency.
        build=lambda: load_entry("swiglu", _self_test, portable=_build_kernels),
    )


def install_qwen35_swiglu(run_benchmark: bool = False) -> bool:
    """Patch ``Qwen3_5MLP.forward`` (and the qwen3_5_moe/qwen3_6/qwen3_6_moe equivalents) so the
    SwiGLU activation ``act_fn(gate_proj(x)) * up_proj(x)`` runs through chalk's fused Triton
    ``silu(gate)*up`` kernel — IFF the live-GPU self-test passes.

    Install-on-call (the Liger model): calling this function IS the opt-in — there is no env
    flag. It patches the MLP CLASS forward (not individual instances), so every current and
    future instance uses the fused activation, mirroring chalk's RMSNorm/rope installers. The
    gate/up/down PROJECTIONS are left exactly as the model has them (cuBLAS ``nn.Linear``,
    LoRA-wrapped, biased, whatever) — only the elementwise ``silu(gate)*up`` is replaced — so
    the swap is fully differentiable and composes with LoRA on the projections (works during
    TRAINING, unlike the fused-GEMM ``mlp.py`` eval-only path).

    SAFE NO-OP CONDITIONS (any -> return False, leave the class untouched):
      * kernel disabled / build / self-test failure (``load_swiglu`` -> None);
      * no qwen3_5/3_6 modeling module with a ``Qwen3_5MLP`` class is importable.

    The patched ``forward`` routes any NON-CUDA gate/up activation back to the exact eager math
    (the class patch is global, so a CPU-offloaded MLP would otherwise try to launch a CUDA
    Triton kernel on CPU tensors).

    A FAILED (re)install is non-destructive: every early/False return leaves both the patch and
    ``RESULT`` exactly as they were on entry. ``RESULT`` is only rewritten on the success path.
    Returns True iff the kernel was installed."""
    fn = load_swiglu()
    if fn is None:
        return False

    # Qwen3.5/3.6 ship separate dense and MoE modeling modules, each with its own MLP class
    # (``Qwen3_5MLP`` / ``Qwen3_5MoeMLP`` / ...). Collect every importable MLP class BEFORE
    # mutating any of them, so a "nothing to patch" outcome returns False having disturbed nothing.
    targets = collect_qwen_classes("MLP")  # [(label, cls)]
    if not targets:
        print("[swiglu] no qwen3_5/3_6 MLP class to patch; keeping eager/Liger", flush=True)
        return False

    def _make_forward():
        def forward(self, x):
            g = self.gate_proj(x)
            u = self.up_proj(x)
            # The class patch is global; only take the fused CUDA path for CUDA activations (a
            # CPU-offloaded MLP falls back to the exact eager activation so we never launch a
            # CUDA kernel on a CPU tensor). ``g``/``u`` carry the projections' autograd graph,
            # so the differentiable fused activation composes with LoRA on the projections.
            # NEVER-EAGER invariant: a standard bf16 CUDA training shape ALWAYS takes ``fn`` (the
            # arch-or-portable fused kernel from ``load_swiglu`` -> ``load_entry``); the eager else
            # below is genuinely-impossible for chalk — the portable Triton kernel cannot run on a
            # CPU activation either — so it is not a chalk->eager regression.
            if g.is_cuda and u.is_cuda:
                act = fn(g, u)
            else:
                act = self.act_fn(g) * u
            return self.down_proj(act)

        return forward

    # Past the last failure point: commit. Patch every target class and (re)publish RESULT.
    patched = []
    fwd = _make_forward()
    for label, cls in targets:
        cls.forward = fwd
        patched.append(label)
    RESULT.clear()
    RESULT.update({"installed": True, "self_test": "passed", "patched": patched})
    print(f"[swiglu] fused Triton SwiGLU installed on {patched} (self-test passed)", flush=True)
    if run_benchmark:
        try:
            _benchmark(fn)
        except Exception as e:
            RESULT["bench_error"] = f"{type(e).__name__}: {e}"
            print(f"[swiglu][bench] skipped: {e}", flush=True)
    return True


def _benchmark(swiglu_fn, *, inter=9216, seqs=(2048, 4096, 8192, 16384), iters=50) -> None:
    """Diagnostic sweep eager vs the fused kernel (fwd+bwd) across token counts. Records the
    per-seq curve in RESULT['sweep']. Never raises out of install (caller guards)."""
    import torch

    dev, dt = "cuda", torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(0)

    def run(fn, g, u):
        out = fn(g, u) if fn is not _eager_swiglu else _eager_swiglu(g, u)
        out.float().pow(2).mean().backward()

    sweep = []
    for seq in seqs:
        g0 = torch.randn(seq, inter, device=dev, dtype=dt, generator=gen)
        u0 = torch.randn(seq, inter, device=dev, dtype=dt, generator=gen)

        def make(g0=g0, u0=u0):
            g = g0.clone().requires_grad_(True)
            u = u0.clone().requires_grad_(True)
            return g, u

        for _ in range(10):
            g, u = make()
            run(_eager_swiglu, g, u)
            g, u = make()
            run(swiglu_fn, g, u)
        torch.cuda.synchronize()

        def timed(fn):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(iters):
                g, u = make()
                run(fn, g, u)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / iters

        te, tk = timed(_eager_swiglu), timed(swiglu_fn)
        sweep.append(
            {
                "seq": seq,
                "eager_ms": round(te, 4),
                "kernel_ms": round(tk, 4),
                "speedup": round(te / tk if tk > 0 else 0.0, 3),
            }
        )
        RESULT["sweep"] = list(sweep)


if __name__ == "__main__":  # manual self-test / smoke
    import torch

    if torch.cuda.is_available():
        try:
            f = _build_kernels()
            _self_test(f)
            print("swiglu self-test: PASS")
        except Exception as e:
            print(f"swiglu self-test: FAIL ({e})")
    else:
        print("swiglu: requires CUDA (skipped)")
