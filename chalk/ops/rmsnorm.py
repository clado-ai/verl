"""Custom fused Triton RMSNorm kernel (fwd+bwd) for Qwen3.5/3.6 — beats Liger's ``rms_norm``.

Liger ships an RMSNorm kernel (``LigerRMSNorm`` / ``liger_kernel.ops.rms_norm``) and its
qwen3_5 patcher swaps ``Qwen3_5RMSNorm`` for it. This module is chalk's own RMSNorm so chalk
can stand ALONE (drop the Liger dependency for this layer) while matching-or-beating Liger's
throughput at quality parity.

Semantics matched EXACTLY to ``modeling_qwen3_5.Qwen3_5RMSNorm.forward`` (Gemma-style casting):

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (hidden_states * (1.0 + self.weight.float())).type_as(hidden_states)

i.e. the variance + normalize happen in fp32, the normalized activation is multiplied by
``(1 + weight)`` while STILL in fp32, and the cast back to the input dtype is the LAST op.
Qwen3.5's RMSNorm uses the Gemma-style ``(1 + weight)`` offset (the weight is a zero-centered
delta), and the weight-multiply stays in fp32 with the dtype cast happening last — this is NOT
the Llama path (which casts to the input dtype BEFORE a plain ``weight`` multiply). This module
now supports BOTH conventions: ``_build_kernels(gemma=True)`` (the Qwen3.5 default, unchanged) and
``_build_kernels(gemma=False)`` (the Llama / MiniCPM plain-weight, cast-before path) — selected by
a single ``gemma`` flag threaded through the kernels, the eager oracle and the self-test.

RMSNorm is memory-bound, so the kernel is one HBM pass per row:
  * FORWARD: one program == one row (token). Vectorized fp32 load of the whole hidden vector
    (``BLOCK_N = next_pow2(hidden)``), single-pass sum-of-squares (no Welford), ``rstd`` cached
    to a per-row ``[M]`` buffer for the backward, the ``(1+weight)`` multiply applied in fp32
    with the dtype cast last.
  * BACKWARD: dx (the RMSNorm jacobian) and dweight. dx is one pass per row reusing the cached
    ``rstd`` (never recomputed). dweight is accumulated the way Liger does it — each program
    owns a partial ``[GROUP, N]`` buffer (rows striped across ``GROUP`` programs), then a cheap
    PyTorch ``sum(0)`` reduces the partials — atomics over a shared ``[N]`` vector across all
    rows are slow, so we avoid them.

LAUNCH CONFIG IS ARCH-AWARE (this is what makes it portable, not GPU-overfit), in the spirit of
Liger's ``calculate_settings`` (``liger_kernel/ops/utils.py``):
  * ``num_warps`` is picked from ``BLOCK_N = next_pow2(hidden)`` exactly like Liger
    (4 / 8 / 16 / 32 as the row widens), so each row's vectorized load is parallelized to match
    its width on any device.
  * the backward dweight ``GROUP`` (number of striped partials = number of programs) is sized
    off the DEVICE'S SM COUNT, not a fixed constant. Liger uses one program per SM
    (``grid = (sm_count,)``); that is great for WIDE rows (the per-row work amortizes the long
    serial row-loop and the ``[sm_count, N]`` reduction stays tiny) but starves NARROW rows on a
    big GPU — with few programs each must grind a very long serial row-loop at low occupancy
    (measured: 1 program/SM is ~1.7x SLOWER than chalk for hidden=1024/tok=16384 on the A100).
    So chalk targets a width-aware number of *waves* of programs over the SMs: more waves for
    narrow rows (cheap per row → need more concurrent rows to fill the machine), fewer for wide
    rows (expensive per row → 2 waves already saturate and the smaller reduction wins). This is
    a size/capability formula (``waves(BLOCK_N) * sm_count``), so it adapts to any GPU's SM
    count and any hidden size rather than hardcoding a launch grid for one device — it beats both
    a fixed ``min(M, 256)`` cap and Liger's one-program-per-SM across the A100 sweep while keeping
    chalk's existing wins on H100/Ada/A40 (which the fixed cap had already won).

Correctness is gated by a live-GPU numeric self-test (fwd + dx + dweight vs an eager reference in
the real reduced dtype — bf16 when supported, else fp16 — with the error measured in fp32, within
a bf16-sized tolerance); ANY import/compile/self-test failure leaves the class untouched
(``install_qwen35_rmsnorm`` returns False and the eager/Liger ``Qwen3_5RMSNorm`` keeps running).
Install-on-call mirrors Liger / chalk's other installers: calling ``install_qwen35_rmsnorm()``
IS the opt-in (no env flag); it patches ``Qwen3_5RMSNorm.forward`` on the CLASS so every
instance picks up the fused kernel (mirroring how ``rope`` patches the module-level function).
"""

from __future__ import annotations

from chalk.ops._hf_targets import collect_qwen_classes
from chalk.utils import rel_l2

# Populated by install_qwen35_rmsnorm so a worker can fold the outcome into metrics.json's
# notes. Empty {} means the kernel was not engaged this run.
RESULT: dict = {}


def _build_kernels(gemma: bool = True):
    """Import torch/triton and define the fused RMSNorm fwd+bwd kernels + the autograd
    Function. Returns ``rmsnorm_fn`` (signature ``(x, weight, eps) -> y``) or raises on any
    import/compile problem (the caller treats a raise as "keep eager/Liger").

    ``gemma`` selects the weight/casting convention (baked into the kernels as a ``tl.constexpr``):

    * ``True`` (Qwen3.5 default): normalize in fp32, multiply by ``(1 + weight)`` in fp32, cast to
      the input dtype LAST (Gemma-style).
    * ``False`` (Llama / MiniCPM): normalize in fp32, cast BACK to the input dtype FIRST, then
      multiply by the PLAIN ``weight`` (no ``1+`` offset) — MiniCPM's ``rms_layernorm``.

    The only math difference is the weight coefficient (``(1+w)`` vs ``w``) and the cast ordering;
    the rstd jacobian and ``dweight = dy*xhat`` are identical for both."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        rstd_ptr,
        x_row_stride,
        y_row_stride,
        N,
        eps,
        BLOCK_N: tl.constexpr,
        IS_GEMMA: tl.constexpr,
    ):
        # one program == one row (token); load the whole hidden vector at once
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        # single-pass sum-of-squares (no Welford): RMSNorm has no mean-subtraction.
        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(rstd_ptr + row, rstd)
        xhat = x * rstd
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if IS_GEMMA:
            # GEMMA-style: normalize in fp32, multiply by (1+weight) in fp32, cast to dtype LAST.
            y = ((1.0 + w) * xhat).to(y_ptr.dtype.element_ty)
        else:
            # LLAMA-style (MiniCPM): cast the normalized activation BACK to the input dtype FIRST
            # (the llama cast-before), then multiply by the PLAIN weight (no 1+ offset). The
            # weight-multiply is done in fp32 (as torch does for bf16*bf16 internally) and the
            # result cast to the input dtype on store — matching ``h=(h*rstd).to(dtype); h*weight``.
            xc = xhat.to(y_ptr.dtype.element_ty).to(tl.float32)
            y = (w * xc).to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + row * y_row_stride + cols, y, mask=mask)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        x_ptr,
        w_ptr,
        dy_ptr,
        rstd_ptr,
        dx_ptr,
        dw_partial_ptr,  # [GROUP, N] fp32 partial dweight, one slot per program
        x_row_stride,
        dy_row_stride,
        dx_row_stride,
        dw_partial_row_stride,
        M,
        N,
        rows_per_prog: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_GEMMA: tl.constexpr,
    ):
        # each program strides over a contiguous chunk of rows and accumulates its slice of the
        # dweight partial locally (avoids atomics over a shared [N] vector across all rows).
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dw_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # ``rows_per_prog`` is a constexpr meta-param, but ``pid`` (and hence ``row_start``) is a
        # runtime value, so ``row_start``/``row_end`` are runtime bounds. Triton requires the
        # dynamic-bounds loop iterator ``tl.range`` for a loop over runtime values — a plain
        # Python ``range`` over ``tl`` values does not compile. (``tl.static_range`` is only for
        # compile-time-constant bounds, which these are not.)
        row_start = pid * rows_per_prog
        row_end = tl.minimum(row_start + rows_per_prog, M)
        for row in tl.range(row_start, row_end):
            x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr + row * dy_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.load(rstd_ptr + row)
            xhat = x * rstd  # fp32 normalized activation
            # d(y)/d(xhat) is the weight coefficient: (1+w) for gemma, plain w for llama.
            if IS_GEMMA:
                c = dy * (1.0 + w)
            else:
                # llama: eager computes the grad wrt the CAST-rounded activation in the reduced
                # dtype, so the local dy*w dx coefficient is rounded before the RMSNorm jacobian —
                # match it (consistent with the rounded llama dweight below).
                c = (dy * w).to(x_ptr.dtype.element_ty).to(tl.float32)
            # dx_j = rstd * (c_j - xhat_j * mean_k(c_k * xhat_k))
            s = tl.sum(c * xhat, axis=0) / N
            dx = rstd * (c - xhat * s)
            tl.store(dx_ptr + row * dx_row_stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)
            # dweight = d(y)/d(weight)*dy. gemma: y=(1+w)*xhat -> xhat (fp32). llama: y=w*round(xhat)
            # -> eager accumulates from the CAST-rounded activation, so match it (else the bf16
            # dweight drifts from eager by the forward cast rounding).
            if IS_GEMMA:
                dw_acc += dy * xhat
            else:
                dw_acc += dy * xhat.to(x_ptr.dtype.element_ty).to(tl.float32)
        tl.store(dw_partial_ptr + pid * dw_partial_row_stride + cols, dw_acc, mask=mask)

    def _next_pow2(n: int) -> int:
        return triton.next_power_of_2(n)

    def _warps_for(block_n: int) -> int:
        # num_warps from the row width, mirroring Liger's ``calculate_settings``
        # (``liger_kernel/ops/utils.py``): a memory-bound row is loaded as one ``BLOCK_N``-wide
        # vector, so wider rows get more warps to parallelize that load. Liger's ladder is
        # 4 (default) / 8 (>=2048) / 16 (>=8192) / 32 (>=32768); we keep a 2-warp rung below 512
        # for the tiny hidden sizes chalk also sees (sub-512 rows would otherwise over-subscribe
        # 4 warps). This is purely size-driven, so it is correct on any GPU.
        if block_n >= 32768:
            return 32
        if block_n >= 8192:
            return 16
        if block_n >= 2048:
            return 8
        if block_n >= 512:
            return 4
        return 2

    def _bwd_grid(M: int, block_n: int, device) -> tuple:
        """Arch-aware launch geometry for the dweight backward: return ``(GROUP, rows_per_prog)``
        where ``GROUP`` is the number of striped ``[N]`` partials (== number of programs) and each
        program reduces ``rows_per_prog`` contiguous rows.

        ``GROUP`` is sized off the DEVICE'S SM COUNT (``multi_processor_count``) — this is what
        keeps the kernel portable instead of overfit to one GPU. Liger launches exactly one
        program per SM; chalk launches a width-aware number of *waves* over the SMs:

            waves = 4 if BLOCK_N < 2048    # narrow rows: cheap per row, need more concurrency
                    3 if BLOCK_N < 4096    # medium rows
                    2 otherwise            # wide rows: 2 waves saturate; smaller reduction wins

        ``GROUP = min(M, waves * sm_count)`` (never more programs than rows). It is then tightened
        to ``cdiv(M, rows_per_prog)`` so the grid exactly covers M rows with no empty program and
        the ``[GROUP, N]`` reduction buffer is as small as the row-loop allows."""
        try:
            sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        except Exception:
            sm_count = 64  # conservative fallback if the device can't be queried
        if block_n < 2048:
            waves = 4
        elif block_n < 4096:
            waves = 3
        else:
            waves = 2
        GROUP = min(M, max(1, waves * sm_count))
        rows_per_prog = triton.cdiv(M, GROUP)
        GROUP = triton.cdiv(M, rows_per_prog)  # tighten so no program is empty
        return GROUP, rows_per_prog

    class _RMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            # x: [..., N]; flatten the leading dims to a [M, N] row matrix.
            sh = x.shape
            N = sh[-1]
            x2 = x.reshape(-1, N)
            x2 = x2 if x2.is_contiguous() else x2.contiguous()
            M = x2.shape[0]
            w = weight if weight.is_contiguous() else weight.contiguous()
            y = torch.empty_like(x2)
            rstd = torch.empty((M,), device=x2.device, dtype=torch.float32)
            BLOCK_N = _next_pow2(N)
            # an empty batch has no rows to normalize, and launching would ask triton for a
            # zero-sized grid; the empty y allocated above is already the whole answer.
            if M > 0:
                with torch.cuda.device(x2.device):
                    _rmsnorm_fwd_kernel[(M,)](
                        x2,
                        w,
                        y,
                        rstd,
                        x2.stride(0),
                        y.stride(0),
                        N,
                        eps,
                        BLOCK_N=BLOCK_N,
                        IS_GEMMA=gemma,
                        num_warps=_warps_for(BLOCK_N),
                    )
            ctx.save_for_backward(x2, w, rstd)
            ctx._sh = sh
            ctx._eps = eps
            return y.reshape(sh)

        @staticmethod
        def backward(ctx, dy):
            x2, w, rstd = ctx.saved_tensors
            sh = ctx._sh
            N = sh[-1]
            M = x2.shape[0]
            dy2 = dy.reshape(-1, N)
            dy2 = dy2 if dy2.is_contiguous() else dy2.contiguous()
            dx = torch.empty_like(x2)
            BLOCK_N = _next_pow2(N)
            # Stripe rows across an ARCH-AWARE number of programs so each reduces many rows into
            # ONE [N] partial; then sum the partials (a [GROUP, N] -> [N] reduction). This is
            # Liger's dweight strategy (cheap partials + a final reduce, never atomics over a
            # shared [N] vector across all M rows) but with the program count sized off the
            # device SM count + row width (see ``_bwd_grid``) instead of a fixed ``min(M, 256)``
            # cap that under-fills big GPUs for wide rows and starves narrow rows. ``GROUP`` is
            # already tightened so ``GROUP * rows_per_prog >= M`` covers every row.
            #
            # An empty batch contributes nothing to any gradient. dx is already empty at the
            # caller's shape, but dweight is a full [N] row that autograd accumulates into, so it
            # has to be a real zeroed tensor rather than an empty one. Returning here also keeps
            # ``_bwd_grid`` off a zero row count, where its ``cdiv(M, GROUP)`` divides by zero.
            if M == 0:
                return dx.reshape(sh), torch.zeros_like(w), None
            GROUP, rows_per_prog = _bwd_grid(M, BLOCK_N, x2.device)
            dw_partial = torch.empty((GROUP, N), device=x2.device, dtype=torch.float32)
            with torch.cuda.device(x2.device):
                _rmsnorm_bwd_kernel[(GROUP,)](
                    x2,
                    w,
                    dy2,
                    rstd,
                    dx,
                    dw_partial,
                    x2.stride(0),
                    dy2.stride(0),
                    dx.stride(0),
                    dw_partial.stride(0),
                    M,
                    N,
                    # ``rows_per_prog`` is a ``tl.constexpr`` meta-parameter, so it must be passed
                    # as a KEYWORD meta argument (passing it positionally as a runtime arg raises a
                    # Triton launch error). The grid is ``(GROUP,)`` with ``GROUP`` already tightened
                    # so ``GROUP * rows_per_prog >= M`` covers every row.
                    rows_per_prog=rows_per_prog,
                    BLOCK_N=BLOCK_N,
                    IS_GEMMA=gemma,
                    num_warps=_warps_for(BLOCK_N),
                )
            dweight = dw_partial.sum(0).to(w.dtype)
            return dx.reshape(sh), dweight, None

    def rmsnorm_fn(x, weight, eps):
        return _RMSNormFunction.apply(x, weight, eps)

    return rmsnorm_fn


def _eager_rmsnorm(x, weight, eps, gemma: bool = True):
    """Exact HF RMSNorm reference (self-test oracle), selectable by convention:

    * ``gemma=True`` (Qwen3.5 default): fp32 var/normalize, multiply by ``(1 + weight)`` in fp32,
      then cast to the input dtype LAST (Gemma-style casting).
    * ``gemma=False`` (Llama / MiniCPM): fp32 var/normalize, cast BACK to the input dtype, THEN
      multiply by the PLAIN ``weight`` (no ``1+`` offset) — exactly MiniCPM's ``rms_layernorm``:
      ``h=(h*rsqrt(var+eps)).to(dtype); return h*weight``.
    """
    import torch

    input_dtype = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    if gemma:
        return ((1.0 + weight.float()) * h).to(input_dtype)
    return h.to(input_dtype) * weight


def _self_test(rmsnorm_fn, gemma: bool = True, hiddens=(1024, 2560, 4096)) -> None:
    """Live-GPU numeric + autograd parity vs the exact eager RMSNorm math: forward, dx, AND
    dweight. Raises on mismatch so the caller keeps the Liger/eager path. Runs at ``hiddens``
    (real Qwen3.5 sizes by default; the MiniCPM installer passes the model's actual dims, e.g.
    hidden=2304). ``gemma`` picks the oracle convention (Gemma ``(1+weight)`` vs Llama plain-weight).

    The kernel's contract is a REDUCED dtype (bf16 in real Qwen3.5 workloads, fp16 otherwise) with
    the variance/normalize AND the ``(1+weight)`` multiply in fp32 and the dtype cast LAST — so the
    test exercises that real path: inputs (and weight) are bf16 (``torch.cuda.is_bf16_supported()``)
    else fp16, and the parity error is measured in fp32 (both chalk and the eager reference cast up
    to fp32) against a tolerance sized for the reduced dtype (much looser than fp32 would be — bf16
    has ~8 mantissa bits)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA for rmsnorm self-test")
    dev, eps = "cuda", 1e-6
    # Validate the REAL (reduced-precision) path the installer runs under: bf16 when the GPU
    # supports it (Qwen3.5 training dtype), else fp16. Tolerance is sized for bf16's ~8-bit
    # mantissa (~2e-2 norm-wise over a full fwd+bwd at these sizes); fp16 comfortably fits under it.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tol = 2e-2
    gen = torch.Generator(device=dev).manual_seed(0)
    for hidden in hiddens:
        for n_tokens in (128, 2048):
            x = torch.randn(n_tokens, hidden, device=dev, dtype=dtype, generator=gen)
            # Qwen3.5 RMSNorm applies (1+weight) (weight is a zero-centered delta); the test
            # weight tensor is arbitrary — parity vs the oracle holds for any weight.
            weight = (1.0 + 0.02 * torch.randn(hidden, device=dev, dtype=torch.float32, generator=gen)).to(dtype)
            # Upstream cotangent for the vjp. Drawn from the SAME seeded ``gen`` (not the default
            # RNG via ``randn_like``) so the self-test is deterministic regardless of the caller's
            # global RNG state — when this runs from ``install_qwen35_rmsnorm`` during
            # ``apply_chalk_kernel_to_qwen35`` the default RNG is in an arbitrary post-model-build
            # state, and a ``randn_like`` cotangent would make the gate non-reproducible.
            g = torch.randn(n_tokens, hidden, device=dev, dtype=dtype, generator=gen)

            # ORACLE IN fp32 (not bf16). The kernel's contract is "fp32-internal math, bf16 I/O", and
            # chalk's backward does its reductions in fp32 — so the trustworthy reference is the SAME
            # math in fp32. A bf16 eager backward is NOT a valid oracle here: its own bf16 rounding on
            # the dx vjp can reach ~8e-2 for an unlucky cotangent (measured: bf16-eager dx is 7.8e-2
            # off fp64 while chalk is 1.6e-3 off fp64), which would spuriously FAIL a correct kernel.
            # We upcast inputs to fp32 for the reference, run fwd+bwd in fp32, and compare chalk's
            # bf16 output against it in fp32 (``rel`` already measures in fp32). This is the standard
            # reduced-precision-kernel test pattern (cf. Liger): chalk vs an fp32 truth, bf16-toleranced.
            xr = x.float().clone().requires_grad_(True)
            wr = weight.float().clone().requires_grad_(True)
            ref = _eager_rmsnorm(xr, wr, eps, gemma=gemma)  # fp32 in -> fp32 out
            ref.backward(g.float())
            dx_ref, dw_ref = xr.grad.clone(), wr.grad.clone()

            xc = x.clone().requires_grad_(True)
            wc = weight.clone().requires_grad_(True)
            got = rmsnorm_fn(xc, wc, eps)
            # Synchronize across the fused forward/backward. The forward kernel writes the per-row
            # ``rstd`` buffer the backward reads; with a large model resident (the apply-on-real-model
            # path) the two launches can race under the default async stream, so a sync on each side
            # of the backward keeps the one-shot gate's measurement honest. (The production autograd
            # path is stream-ordered by torch; this guard is only for the self-test.)
            torch.cuda.synchronize()
            got.backward(g)
            torch.cuda.synchronize()
            dx_got, dw_got = xc.grad.clone(), wc.grad.clone()

            rel = rel_l2  # fp32 relative-L2 error; shared helper (see chalk.utils.rel_l2)

            r_fwd, r_dx, r_dw = rel(got, ref), rel(dx_got, dx_ref), rel(dw_got, dw_ref)
            if not (r_fwd < tol and r_dx < tol and r_dw < tol):
                raise RuntimeError(
                    f"rmsnorm self-test failed (gemma={gemma}) at hidden={hidden} n={n_tokens} "
                    f"dtype={dtype}: fwd={r_fwd:.2e} dx={r_dx:.2e} dw={r_dw:.2e} (tol={tol:.0e})"
                )


def load_rmsnorm(gemma: bool = True):
    """Return ``rmsnorm_fn`` if the kernel builds and passes its live-GPU self-test; otherwise
    return ``None`` (keep the Liger/eager path). Never raises — any failure (no torch/triton,
    no CUDA, compile/self-test error) -> ``None``.

    ``gemma`` selects the convention (see ``_build_kernels``), and each convention has its OWN
    overlay tree under its own op-id, because a kernel tuned for one is numerically wrong for the
    other: ``arch/<arch>/rmsnorm.py`` hardcodes the Gemma ``(1+w)`` offset and is graded against the
    Gemma oracle, ``arch/<arch>/rmsnorm_llama.py`` against ``_eager_rmsnorm(gemma=False)``. Neither
    tree may be pointed at the other's convention, so both the op-id and the self-test below are
    derived from ``gemma`` rather than fixed."""
    from chalk.ops.arch import load_entry
    from chalk.ops.arch import load_kernel

    # NOT ``_self_test``: its ``gemma`` parameter defaults to True, so an ungated pass-through would
    # grade a llama kernel against the Gemma oracle and could not fail it.
    def _st(f):
        _self_test(f, gemma=gemma)

    op = "rmsnorm" if gemma else "rmsnorm_llama"
    conv = "gemma" if gemma else "llama"
    return load_kernel(
        "rmsnorm",
        f"fused Triton RMSNorm (fwd+bwd, {conv}) enabled",
        "fused Triton RMSNorm disabled",
        build=lambda: load_entry(op, _st, portable=lambda: _build_kernels(gemma=gemma)),
    )


def install_qwen35_rmsnorm(run_benchmark: bool = False) -> bool:
    """Patch ``Qwen3_5RMSNorm.forward`` (and the qwen3_5_moe/qwen3_6/qwen3_6_moe equivalents)
    with chalk's fused Triton RMSNorm — IFF the live-GPU self-test passes.

    Install-on-call (the Liger model): calling this function IS the opt-in — there is no env
    flag. It patches the RMSNorm CLASS (not individual instances), so every current and future
    instance uses the fused kernel, mirroring how ``rope`` patches the module-level function.

    SAFE NO-OP CONDITIONS (any -> return False, leave the class untouched):
      * kernel disabled / build / self-test failure (``load_rmsnorm`` -> None);
      * no qwen3_5/3_6 modeling module with an ``Qwen3_5RMSNorm`` class is importable.

    The patched ``forward`` reads ``self.variance_epsilon`` (the HF attr; falls back to ``eps``)
    and routes any NON-CUDA input back to the original eager forward (the class patch is global,
    so a CPU-offloaded norm would otherwise try to launch a CUDA Triton kernel on CPU tensors).

    A FAILED (re)install is non-destructive: every early/False return leaves both the patch and
    ``RESULT`` exactly as they were on entry. ``RESULT`` is only rewritten on the success path.
    Returns True iff the kernel was installed."""
    fn = load_rmsnorm()
    if fn is None:
        return False

    # Qwen3.5/3.6 ship separate dense and MoE modeling modules, each with its own RMSNorm class
    # (``Qwen3_5RMSNorm`` / ``Qwen3_5MoeRMSNorm`` / ...). Collect every importable RMSNorm class
    # BEFORE mutating any of them, so a "nothing to patch" outcome returns False having disturbed
    # nothing.
    targets = collect_qwen_classes("RMSNorm")  # [(label, cls)]
    if not targets:
        print("[rmsnorm] no qwen3_5/3_6 RMSNorm class to patch; keeping eager/Liger", flush=True)
        return False

    def _make_forward():
        def forward(self, hidden_states):
            # Resolve eps ONCE for both the fused and eager paths. Use an explicit ``is None`` check,
            # NOT ``getattr(...) or ...``: a valid eps of 0.0 is falsy, so the ``or`` form would
            # wrongly fall back to the default. HF stores it as ``variance_epsilon`` (fall back to
            # ``eps``, then 1e-6).
            eps = getattr(self, "variance_epsilon", None)
            if eps is None:
                eps = getattr(self, "eps", 1e-6)
            eps = float(eps)
            # The class patch is global; only take the fused CUDA path for CUDA inputs whose
            # last dim matches the weight (a CPU-offloaded / mis-shaped call falls back to the
            # exact eager math so we never launch a CUDA kernel on a CPU tensor). NEVER-EAGER
            # invariant: a standard bf16 CUDA training shape ALWAYS takes ``fn`` (the arch-or-
            # portable fused kernel from ``load_rmsnorm`` -> ``load_entry``); the eager branch below
            # is genuinely-impossible for chalk — the portable Triton kernel likewise cannot run on
            # a CPU / mis-shaped call — so it is not a chalk->eager regression.
            if hidden_states.is_cuda and self.weight.is_cuda and hidden_states.shape[-1] == self.weight.numel():
                return fn(hidden_states, self.weight, eps)
            # Fall back to the SAME eager oracle the self-test grades against (Gemma ``(1+weight)``
            # convention) so the fallback can never silently drift from the kernel's reference math.
            return _eager_rmsnorm(hidden_states, self.weight, eps)

        return forward

    # Past the last failure point: commit. Patch every target class and (re)publish RESULT.
    patched = []
    fwd = _make_forward()
    for label, cls in targets:
        cls.forward = fwd
        patched.append(label)
    RESULT.clear()
    RESULT.update({"installed": True, "self_test": "passed", "patched": patched})
    print(f"[rmsnorm] fused Triton RMSNorm installed on {patched} (self-test passed)", flush=True)
    if run_benchmark:
        try:
            _benchmark(fn)
        except Exception as e:
            RESULT["bench_error"] = f"{type(e).__name__}: {e}"
            print(f"[rmsnorm][bench] skipped: {e}", flush=True)
    return True


def _benchmark(rmsnorm_fn, *, hidden=2560, seqs=(2048, 4096, 8192, 16384), iters=50) -> None:
    """Diagnostic sweep eager vs the fused kernel (fwd+bwd) across token counts. Records the
    per-seq curve in RESULT['sweep']. Never raises out of install (caller guards)."""
    import torch

    dev, dt = "cuda", torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(0)

    def run(fn, x, w):
        y = fn(x, w, 1e-6) if fn is not _eager_rmsnorm else _eager_rmsnorm(x, w, 1e-6)
        y.float().pow(2).mean().backward()

    sweep = []
    for seq in seqs:
        x0 = torch.randn(seq, hidden, device=dev, dtype=dt, generator=gen)
        w0 = 1.0 + 0.02 * torch.randn(hidden, device=dev, dtype=dt, generator=gen)

        def make(x0=x0, w0=w0):
            x = x0.clone().requires_grad_(True)
            w = w0.clone().requires_grad_(True)
            return x, w

        for _ in range(10):
            x, w = make()
            run(_eager_rmsnorm, x, w)
            x, w = make()
            run(rmsnorm_fn, x, w)
        torch.cuda.synchronize()

        def timed(fn):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(iters):
                x, w = make()
                run(fn, x, w)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / iters

        te, tk = timed(_eager_rmsnorm), timed(rmsnorm_fn)
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
            print("rmsnorm self-test: PASS")
        except Exception as e:
            print(f"rmsnorm self-test: FAIL ({e})")
    else:
        print("rmsnorm: requires CUDA (skipped)")
