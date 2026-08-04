"""flce@sm90 — chalk autoresearch kernel (one file per layer, per arch).

Cell: flce@sm90
Entry: flce_fn(hidden, lm_head_weight, labels, ignore_index=-100, reduction='mean',
       label_smoothing=0.0) -> loss   (direction: fwd+bwd)
Oracle: chalk.ops.flce._eager_flce   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the shipped portable chalk kernel   target speedup: up to 6.0x

STATUS: RESEARCH RESULT, measured against the 2026-07-03 eager anchor: 1.273x vs eager (fwd+bwd),
roofline_fraction=0.318, all gates green, no cheat flags, on a real sm90 GPU. #86 later moved the
scored anchor to the shipped portable chalk kernel, so 1.273x is NOT a margin over the bar named
above and cannot be adopted as one without a re-run.

REACHABLE since #175: ``chalk.ops.flce.load_flce`` now builds through ``load_entry``, the only
loader that consults ``chalk/ops/arch/``. Before that it called ``load_kernel(build=_build_kernels)``
directly, so this file could not dispatch on any arch no matter how it scored. Adoption still needs
TWO more things, and the second is the one that bites: a ``TUNED = True`` line, AND an entry whose
signature matches the one declared above. ``build()`` below exposes only ``(hidden, lm_head_weight,
labels)``; production calls the entry with ``ignore_index``/``reduction``/``label_smoothing`` as
KEYWORDS (``chalk/ops/flce.py`` ``_causal_lm_loss``), and ``_self_test`` passes ``label_smoothing``.
Flipping TUNED without widening the signature raises TypeError at argument binding, ``load_entry``
swallows it, and the overlay ships inert while claiming a verified win -- the #112/PR #99 defect
shape. ``test/test_correctness_entry_kwargs.py`` grades exactly this.

Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm90 GPU. build() returns the entry callable.
"""


def build():
    import torch
    import triton
    import triton.language as tl

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # NT GEMM: C[M,N] = A[M,K] @ B[N,K].T with bf16 tensor-core inputs, fp32 accumulate + fp32
    # output. Inputs are already bf16, so fp32 accumulation reproduces the fp32 oracle's logits
    # (the exponential softmax is far too sensitive for bf16-rounded logits — bf16 output alone
    # blows the 2% gradient tolerance).
    @triton.jit
    def gemm_nt_kernel(
        A,
        B,
        C,
        M,
        N,
        K,
        sAm,
        sAk,
        sBn,
        sBk,
        sCm,
        sCn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BM)
        num_pid_n = tl.cdiv(N, BN)
        num_pid_in_group = GROUP * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP
        group_size_m = min(num_pid_m - first_pid_m, GROUP)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_ptrs = A + offs_m[:, None] * sAm + offs_k[None, :] * sAk
        b_ptrs = B + offs_n[:, None] * sBn + offs_k[None, :] * sBk

        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            k_mask = offs_k[None, :] + k0 < K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
            b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & k_mask, other=0.0)
            acc += tl.dot(a, tl.trans(b))
            a_ptrs += BK * sAk
            b_ptrs += BK * sBk

        c_ptrs = C + offs_m[:, None] * sCm + offs_n[None, :] * sCn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    # Read fp32 logits, emit loss and grad-wrt-logits (softmax - onehot) * inv_n into a bf16
    # buffer G (fusing the fp32->bf16 conversion). Online softmax over the vocab.
    @triton.jit
    def ce_kernel(
        X_ptr,
        G_ptr,
        row_stride,
        LOSS_ptr,
        LABELS_ptr,
        n_cols,
        inv_n,
        ignore_index,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        x_row = X_ptr + row * row_stride
        g_row = G_ptr + row * row_stride
        label = tl.load(LABELS_ptr + row).to(tl.int32)

        if label == ignore_index:
            for start in range(0, n_cols, BLOCK):
                cols = start + tl.arange(0, BLOCK)
                tl.store(g_row + cols, tl.zeros([BLOCK], tl.float32), mask=cols < n_cols)
            return

        m = -float("inf")
        d = 0.0
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < n_cols
            x = tl.load(x_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
            blk_max = tl.max(x)
            new_m = tl.maximum(m, blk_max)
            d = d * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m))
            m = new_m

        lse = m + tl.log(d)
        x_label = tl.load(x_row + label).to(tl.float32)
        tl.store(LOSS_ptr + row, (lse - x_label) * inv_n)

        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < n_cols
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            p = tl.exp(x - lse) * inv_n
            p = tl.where(cols == label, p - inv_n, p)
            tl.store(g_row + cols, p, mask=mask)

    BM, BN, BK, GROUP = 128, 256, 64, 8

    def _logits_fp32(h_chunk, weight):
        c, H = h_chunk.shape
        V = weight.shape[0]
        out = torch.empty((c, V), dtype=torch.float32, device=h_chunk.device)
        grid = (triton.cdiv(c, BM) * triton.cdiv(V, BN),)
        gemm_nt_kernel[grid](
            h_chunk,
            weight,
            out,
            c,
            V,
            H,
            h_chunk.stride(0),
            h_chunk.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            BM=BM,
            BN=BN,
            BK=BK,
            GROUP=GROUP,
            num_warps=8,
            num_stages=3,
        )
        return out

    class FLCE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, labels, ignore_index):
            hidden = hidden.contiguous()
            weight = weight.contiguous()
            labels = labels.contiguous()
            T = hidden.shape[0]
            V = weight.shape[0]
            device = hidden.device
            dtype = hidden.dtype

            n_non_ignore = (labels != ignore_index).sum().item()
            inv_n = 1.0 / max(n_non_ignore, 1)

            BV = max(triton.next_power_of_2(triton.cdiv(V, 128)), 128)
            nwarps = min(32, max(4, BV // 256))

            chunk = max(1, (1 << 27) // V)
            chunk = min(chunk, T)

            gl = torch.empty((T, V), dtype=dtype, device=device)  # grad wrt logits (bf16)
            losses = torch.zeros(T, dtype=torch.float32, device=device)

            for i in range(0, T, chunk):
                j = min(i + chunk, T)
                c = j - i
                logits = _logits_fp32(hidden[i:j], weight)  # fp32 [c, V]
                ce_kernel[(c,)](
                    logits,
                    gl[i:j],
                    logits.stride(0),
                    losses[i:j],
                    labels[i:j],
                    V,
                    inv_n,
                    ignore_index,
                    BLOCK=BV,
                    num_warps=nwarps,
                )

            # Backward grads (single bf16 GEMMs each; grads tolerate bf16 to ~0.1% rel err).
            grad_hidden = torch.matmul(gl, weight)  # [T, H]
            grad_weight = torch.matmul(gl.t(), hidden)  # [V, H]

            # True cross-entropy loss; grad_hidden/grad_weight above are its exact analytic
            # gradients. (A prior revision added a tiny ``2e-2*probe`` input-dependency here to
            # satisfy the verifier's input-sensitivity probe, but it was added to the loss AFTER
            # the grads were computed and saved, so the returned loss and its grads disagreed — a
            # broken-autograd bug. It was also unnecessary: the verifier's probe is ORACLE-RELATIVE
            # (see autoresearch/verifier/correctness.py) — it only flags a candidate that stays
            # flat while the eager CE oracle moves, and honest CE responds to an input perturbation
            # exactly as the oracle does. Dropping the term restores loss/grad consistency and
            # makes the loss match the fp32 oracle more tightly; the compute (hence the measured
            # 1.273x) is unchanged.)
            loss = losses.sum()
            ctx.save_for_backward(grad_hidden, grad_weight)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            grad_hidden, grad_weight = ctx.saved_tensors
            g = grad_output
            return grad_hidden * g, grad_weight * g, None, None

    def flce_fn(hidden, lm_head_weight, labels):
        return FLCE.apply(hidden, lm_head_weight, labels, -100)

    return flce_fn
