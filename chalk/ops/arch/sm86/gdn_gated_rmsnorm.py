"""gdn_gated_rmsnorm@sm86 — chalk autoresearch kernel (one file per layer, per arch).

Cell: gdn_gated_rmsnorm@sm86
Entry: gnorm_fn(x, weight, gate, eps) -> y   (direction: fwd+bwd)
Oracle: chalk.ops.gdn._eager_gated_rmsnorm   tol fwd/bwd rel=2e-2 @ bf16 I/O, fp32 oracle
Baseline to beat: the arch-tuned kernel this file already ships   target speedup: up to 1.3x

STATUS: ADOPTED — 1.03x floor vs the PORTABLE chalk kernel on an sm86 A5000 (RTX A5000, torch
2.8.0+cu129, triton 3.4.0), measured on the four production self-test shapes (M in {2048, 16384} x
N in {128, 256}, bf16). The baseline is portable, not eager: ``gdn_gated_rmsnorm`` is registered in
the verifier's chalk resolvers, and portable is also what ``load_entry`` falls back to when this file
is absent — so vs-portable is the delta a user actually gets. It is not, however, the bar the header
sets, and the two differ on purpose: since #99 that resolver is arch-aware, so on sm86 it returns
THIS file and anchors a new candidate here rather than on portable. Selected by production dispatch
(``TUNED = True``); ``chalk.ops.gdn.load_gated_rmsnorm`` routes through
``load_entry("gdn_gated_rmsnorm", ...)`` and guards it with ``_self_test_gated_rmsnorm``, so this file
only runs when it matches the fp32 oracle on fwd, dx, dw and dz. ``build()`` returns the entry
callable ``gnorm_fn(x, weight, gate, eps) -> y``.

MEASUREMENT METHOD matters more than the number here, because three ways of timing this op give three
different answers. Wall-clock A/B on the entry callable carries ~12ms of fixed dispatch overhead
against 0.4-0.6ms of real work, which compresses every ratio toward 1.0. ``do_bench`` around
``y.backward()`` spends 700-900us per step in autograd graph traversal against 30-130us of device
work, which drags every ratio below 1.0. Neither measures the kernel. The figures below are total GPU
KERNEL SELF-TIME under ``torch.profiler`` with ``ProfilerActivity.CUDA`` only, driving the backward
with ``torch.autograd.grad`` (no ``.grad`` accumulation), summed over ``self_device_time_total`` and
divided by iteration count — portable microseconds over this kernel's microseconds, so >1 is a win:

    shape         run 1    run 2    run 3   portable us (run 1 / 2 / 3)
    N128 M2048     2.337    2.266    2.340    31.54 /  34.86 /  31.47
    N128 M16384    1.240    1.195    1.233    75.06 /  80.73 /  74.91
    N256 M2048     2.027    1.947    2.029    38.02 /  40.96 /  38.01
    N256 M16384    1.047    1.036    1.043   129.40 / 141.97 / 129.63
    geomean        1.575    1.529    1.572

Three independent runs, because a single one does not separate a real ratio from run-to-run drift:
note that portable itself moved 8-10% between run 1 and run 2 on the same card. The ranking is stable
and every shape stays positive in all three, but the widest shape lands at 1.047, 1.036 and 1.043, so
the floor the SPEEDUP constant reports is 1.03 — the lowest across replications, not the best draw.
Launch count per fwd+bwd step drops from portable's 4 to 3. The point estimate does NOT resolve finer
than the table: this file must not be re-ranked against sibling candidates on speed alone — it was
selected on properties that do resolve (below).

Why THIS kernel and not the higher-scoring siblings: the backward has NO atomics. It accumulates
each program's weight-gradient slice into a private ``partial_dw`` row and reduces with a second
kernel, so dw is bit-reproducible across runs and across launch order, and a repeated backward
recomputes correctly (``partial_dw`` is allocated per call, not reused). Candidates scoring higher on
the gate were disqualified for real defects: one aliases and mutates the caller's cotangent buffer,
one corrupts dw on a second backward, and four hardcode BLOCK=128 and so normalize over half the row
at N=256 — they fail the production self-test at exactly sqrt(2). No memory claim is made: the gate
measured this cell's memory ratio at 1.000.

Math: forward normalizes in fp32 over the true H (``H`` is a constexpr, never a fixed block), applies
weight then silu(gate), and saves rstd for the backward. Loads are fully strided (``sx_row``/``sx_col``
and friends), so a non-contiguous input is handled. Rank is normalized the way the portable kernel
does it — collapse to ``(-1, H)`` on entry, restore the caller's shape on the way out — because the
kernels index a (row, col) pair and would otherwise read ``stride(1)`` off the wrong axis of an N-d
tensor. Production always passes 2D and the self-test only generates 2D, so nothing downstream would
catch that; the entry accepts the same ranks portable does rather than silently miscomputing. The
backward recomputes the gate and its derivative rather than saving them, trading a little arithmetic
for the activation memory it would otherwise hold.
"""

import torch
import triton
import triton.language as tl

TUNED = True
# the worst production shape, not the geomean (1.52-1.58). load_entry prints this to the user, so it
# reports the delta they are guaranteed rather than the one they get on the best shape. three
# independent runs put that worst shape (N256 M16384) at 1.047, 1.036 and 1.043, so the floor is 1.03.
SPEEDUP = 1.03
SPEEDUP_ANCHOR = "the portable chalk kernel"


@triton.jit
def _gated_rmsnorm_fwd(
    x_ptr,
    weight_ptr,
    z_ptr,
    y_ptr,
    rstd_ptr,
    n_rows,
    sx_row,
    sx_col,
    sw,
    sz_row,
    sz_col,
    eps,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    group = tl.program_id(0)
    rows = group * GROUP_M + tl.arange(0, GROUP_M)
    cols = tl.arange(0, BLOCK_H)

    row_mask = rows < n_rows
    col_mask = cols < H
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(x_ptr + rows[:, None] * sx_row + cols[None, :] * sx_col, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + rows[:, None] * sz_row + cols[None, :] * sz_col, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols * sw, mask=col_mask, other=0.0).to(tl.float32)

    mean_square = tl.sum(x * x, axis=1) * (1.0 / H)
    rstd = tl.rsqrt(mean_square + eps)
    gate = z * tl.sigmoid(z)
    y = x * rstd[:, None] * weight[None, :] * gate

    tl.store(y_ptr + rows[:, None] * H + cols[None, :], y, mask=mask)
    tl.store(rstd_ptr + rows, rstd, mask=row_mask)


@triton.jit
def _gated_rmsnorm_bwd_partial(
    x_ptr,
    weight_ptr,
    z_ptr,
    rstd_ptr,
    dy_ptr,
    dx_ptr,
    dz_ptr,
    partial_dw_ptr,
    n_rows,
    n_groups,
    sx_row,
    sx_col,
    sw,
    sz_row,
    sz_col,
    sdy_row,
    sdy_col,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < H

    weight = tl.load(weight_ptr + cols * sw, mask=col_mask, other=0.0).to(tl.float32)

    # private per-program accumulator: no atomics, so dw is order-independent and reproducible
    dw_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for group in tl.range(pid, n_groups, NUM_PROGRAMS):
        rows = group * GROUP_M + tl.arange(0, GROUP_M)
        row_mask = rows < n_rows
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(x_ptr + rows[:, None] * sx_row + cols[None, :] * sx_col, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + rows[:, None] * sz_row + cols[None, :] * sz_col, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + rows[:, None] * sdy_row + cols[None, :] * sdy_col, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        sigmoid_z = tl.sigmoid(z)
        gate = z * sigmoid_z
        weighted_dy = dy * weight[None, :]
        projected = weighted_dy * gate

        # d/dx of rmsnorm: the projected cotangent minus x's share of the reduction
        dot = tl.sum(projected * x, axis=1)
        rstd_2d = rstd[:, None]
        correction = dot[:, None] * rstd_2d * rstd_2d * rstd_2d * (1.0 / H)
        dx = projected * rstd_2d - x * correction

        # d/dz of silu: sigmoid(z) * (1 + z * (1 - sigmoid(z)))
        gate_grad = sigmoid_z * (1.0 + z * (1.0 - sigmoid_z))
        dz = weighted_dy * x * rstd_2d * gate_grad

        dw_contribution = dy * x * rstd_2d * gate
        dw_contribution = tl.where(mask, dw_contribution, 0.0)
        dw_acc += tl.sum(dw_contribution, axis=0)

        tl.store(dx_ptr + rows[:, None] * H + cols[None, :], dx, mask=mask)
        tl.store(dz_ptr + rows[:, None] * H + cols[None, :], dz, mask=mask)

    # one store per program into a private row; the reduce kernel sums them
    tl.store(partial_dw_ptr + pid * H + cols, dw_acc, mask=col_mask)


@triton.jit
def _reduce_weight_grad(
    partial_dw_ptr,
    dw_ptr,
    n_partials,
    H: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    col_start = tl.program_id(0) * BLOCK_C
    partial_offsets = tl.arange(0, BLOCK_P)[:, None]
    col_offsets = col_start + tl.arange(0, BLOCK_C)[None, :]

    mask = (partial_offsets < n_partials) & (col_offsets < H)
    values = tl.load(partial_dw_ptr + partial_offsets * H + col_offsets, mask=mask, other=0.0).to(tl.float32)
    result = tl.sum(values, axis=0)

    output_cols = col_start + tl.arange(0, BLOCK_C)
    tl.store(dw_ptr + output_cols, result, mask=output_cols < H)


class _GatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, z, eps):
        shape = x.shape
        hidden = shape[-1]
        # collapse to 2D like the portable kernel: the strided loads below index (row, col), so an
        # N-d input must be flattened first or stride(1) would name the wrong axis
        x2 = x.reshape(-1, hidden)
        z2 = z.reshape(-1, hidden)
        n_rows = x2.shape[0]
        block_h = triton.next_power_of_2(hidden)
        eps_value = float(eps)

        y = torch.empty((n_rows, hidden), device=x.device, dtype=x.dtype)
        rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

        group_m = 4
        n_groups = triton.cdiv(n_rows, group_m)

        # an empty batch has nothing to normalize, and launching anyway would ask triton for a
        # zero-sized grid; the allocated empty outputs above are already the right answer
        if n_rows > 0:
            # pin the launch to the input's device: under a sharded device_map x can live on a
            # device other than the process current one, and triton would otherwise build the
            # kernel against the wrong context and fault on foreign pointers
            with torch.cuda.device(x2.device):
                _gated_rmsnorm_fwd[(n_groups,)](
                    x2,
                    weight,
                    z2,
                    y,
                    rstd,
                    n_rows,
                    x2.stride(0),
                    x2.stride(1),
                    weight.stride(0),
                    z2.stride(0),
                    z2.stride(1),
                    eps_value,
                    H=hidden,
                    BLOCK_H=block_h,
                    GROUP_M=group_m,
                    num_warps=4,
                    num_stages=1,
                )

        ctx.save_for_backward(x2, weight, z2, rstd)
        ctx.input_shape = shape
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x, weight, z, rstd = ctx.saved_tensors
        shape = ctx.input_shape
        hidden = x.shape[1]
        n_rows = x.shape[0]
        block_h = triton.next_power_of_2(hidden)

        dy2 = dy.reshape(-1, hidden)
        dx = torch.empty((n_rows, hidden), device=x.device, dtype=x.dtype)
        dz = torch.empty((n_rows, hidden), device=z.device, dtype=z.dtype)

        # 16 rows per group, not the forward's 4. at M=16384 a group of 4 leaves each of the 256
        # programs serially walking 16 iterations of 4 rows, which under-fills a 64-SM card and made
        # the backward slower than portable on the widest shape. 16 measured fastest on every
        # production shape; raising the program cap instead is worse and grows partial_dw linearly.
        group_m = 16
        n_groups = triton.cdiv(n_rows, group_m)
        num_programs = min(n_groups, 256)

        # an empty batch contributes nothing to any gradient. dw must still be a real zeroed tensor
        # (autograd accumulates into it), while dx and dz are already empty at the caller's shape.
        if n_rows == 0:
            dw = torch.zeros(weight.shape, device=weight.device, dtype=weight.dtype)
            return dx.reshape(shape), dw, dz.reshape(shape), None

        # allocated per call, so a repeated backward recomputes instead of accumulating into a stale buffer
        partial_dw = torch.empty((num_programs, hidden), device=weight.device, dtype=torch.float32)

        dw = torch.empty(weight.shape, device=weight.device, dtype=weight.dtype)
        block_p = triton.next_power_of_2(num_programs)
        # columns per reduce program. this sets the reduce grid to cdiv(hidden, block_c), so it is
        # the same under-fill tradeoff as group_m above, one kernel over. swept 4..128 on both
        # production widths: 8 is the geomean best (1.529 vs 1.523 at 4, 1.526 at 16), and it falls
        # off past 32 as the grid stops filling the card (1.410 at 128, where N256 turns negative).
        block_c = 8

        # same device pin as the forward: both launches must run in the input's context
        with torch.cuda.device(x.device):
            _gated_rmsnorm_bwd_partial[(num_programs,)](
                x,
                weight,
                z,
                rstd,
                dy2,
                dx,
                dz,
                partial_dw,
                n_rows,
                n_groups,
                x.stride(0),
                x.stride(1),
                weight.stride(0),
                z.stride(0),
                z.stride(1),
                dy2.stride(0),
                dy2.stride(1),
                H=hidden,
                BLOCK_H=block_h,
                GROUP_M=group_m,
                NUM_PROGRAMS=num_programs,
                num_warps=4,
                num_stages=1,
            )

            _reduce_weight_grad[(triton.cdiv(hidden, block_c),)](
                partial_dw,
                dw,
                num_programs,
                H=hidden,
                BLOCK_P=block_p,
                BLOCK_C=block_c,
                num_warps=4,
                num_stages=1,
            )

        return dx.reshape(shape), dw, dz.reshape(shape), None


def build():
    def gnorm_fn(x, weight, gate, eps):
        return _GatedRMSNorm.apply(x, weight, gate, eps)

    return gnorm_fn
