"""Decode-regime serving kernels — forward-only, tuned for single-token (M≈8) decode.

Chalk's training kernels are tuned for long sequences (fwd+bwd, thousands of tokens). At *decode*
(one token per in-flight sequence, M≈8, forward-only) that tuning is wrong, and the training kernels
are actually **slower than eager** — measured end-to-end on Qwen3.5-0.8B decode (L4): chalk's shipped
rmsnorm+swiglu run at 0.93–0.94× of eager. These two kernels, found by the ``serving_bench`` decode
autoresearch (``autoresearch/manifest/serve.py``) and verified through all five gates on real L4/sm89,
flip that to a win:

    kernel          decode M=8 vs eager (per-op, sm89/sm80)   E2E Qwen3.5-0.8B decode vs eager
    rmsnorm         4.43× (sm89) / 4.21× (sm80 A100)          rmsnorm+swiglu 1.09–1.12×
    swiglu          1.37× (sm89) / 1.31× (sm80 A100)          +rope 1.09–1.13× (gen verified)
    rope            4.36× (sm89)                              (all three combined, over eager)
    gated_rmsnorm   (GDN mixer norm) → decode 3.42×           (per-V-head; linear-attention models only)

The designs are *different* because the ops are different:
  * **rmsnorm** has a reduction, so plain PyTorch dispatches ~6–8 tiny kernels; a single fused Triton
    launch (one program per row, in-register fp32 reduce) collapses them → big win at tiny M.
  * **swiglu** is pure elementwise; a Triton launch on L4 costs ~2.8× a fused aten launch, so a custom
    kernel LOSES. The win is to match eager's launch count while killing allocations: bf16-native
    in-place ``silu_().mul_()`` (F.silu already does fp32 opmath internally, so no explicit upcast).
  * **rope** has the rotate_half ``cat`` (an allocation) + ~16 elementwise launches; ONE fused Triton
    launch rotates BOTH q and k, loading each token's cos/sin once and reusing across all heads →
    ~4.4×. Patched at the module-level ``apply_rotary_pos_emb`` (full-attention layers; GDN layers
    don't call it). Fast path is B==1 (single-sequence decode); B>1 / exotic layouts take eager.
  * **gated_rmsnorm** is the GatedDeltaNet mixer's ``self.norm`` (``(x·rsqrt(mean(x²)+eps)·weight)·silu(z)``,
    per V-head over ``head_v_dim``): another reduction op firing ~8–10 tiny eager kernels, so the same
    single-fused-launch design wins (3.42× at decode). It only exists on linear-attention Qwen3.5/3.6.

Everything is self-test gated with an eager fallback, exactly like the other chalk installers: a
kernel that can't verify at the model's real dims leaves the eager module untouched. Import is cheap
and CPU-safe (torch/triton load lazily). Use ``apply_chalk_decode_kernels(model, base_model=...)``.
"""

from __future__ import annotations

# Norm / MLP classes to patch, by ``type(m).__name__`` — Qwen3.5/3.6 (gemma) + Llama/MiniCPM (plain).
_RMSNORM_GEMMA = ("Qwen3_5RMSNorm", "Qwen3RMSNorm", "Gemma2RMSNorm", "GemmaRMSNorm")
_RMSNORM_LLAMA = ("LlamaRMSNorm", "MiniCPMRMSNorm")
_MLP_CLASSES = ("Qwen3_5MLP", "Qwen3MLP", "LlamaMLP", "MiniCPMMLP")
# GatedDeltaNet gated-RMSNorm (the ``self.norm`` in the linear-attention mixer), pure-torch when fla is
# absent. Same forward body across all three; patched together. forward(self, hidden_states, gate).
_GATED_RMSNORM_CLASSES = ("Qwen3NextRMSNormGated", "Qwen3_5RMSNormGated", "Qwen3_5MoeRMSNormGated")


def _build_rmsnorm(gemma: bool):
    """The 4.43×-at-decode fused RMSNorm: ONE Triton launch, one program per row, whole hidden dim in
    a single block, in-register fp32 reduce + (1+w)|w scale + store. ``gemma`` picks the Qwen (1+w)
    vs Llama plain-weight convention. Returns ``rmsnorm_fn(x, weight, eps)`` or None (no GPU/triton)."""
    import torch

    try:
        import triton
        import triton.language as tl
    except Exception:  # pragma: no cover
        return None
    if not torch.cuda.is_available():
        return None

    @triton.jit
    def _rms_kernel(X, W, Y, stride_row, N, eps, GEMMA: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        X += row * stride_row
        Y += row * stride_row
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        scale = (1.0 + w) if GEMMA else w
        y = x * rstd * scale
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)

    _npow2 = triton.next_power_of_2
    _cfg: dict[int, tuple[int, int, bool]] = {}

    def _config(n: int):
        c = _cfg.get(n)
        if c is not None:
            return c
        block = _npow2(n) if n > 1 else 1
        e = block.bit_length() - 1
        nw = 4 if e <= 11 else (8 if e <= 13 else 16)
        c = (block, nw, e <= 14)  # single-block covers hidden ≤ 16384
        _cfg[n] = c
        return c

    def _ref(x, weight, eps):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (y * ((1.0 + weight.float()) if gemma else weight.float())).to(x.dtype)

    def _launch(x2, w, n, eps):
        # x2 MUST be contiguous [M, N] so its row stride is exactly N (the kernel indexes row*N+cols
        # and stores to a fresh contiguous [M, N]). Passing N as stride_row + always-contiguous inputs
        # is what keeps the store in bounds — do not pass x2.stride(0) of a non-contiguous view.
        m = x2.shape[0]
        block, nw, ok = _config(n)
        if not ok or m == 0:
            return _ref(x2, w, eps)
        y2 = torch.empty_like(x2)
        _rms_kernel[(m,)](x2, w, y2, n, n, float(eps), gemma, block, num_warps=nw, num_stages=1)
        return y2

    def rmsnorm_fn(x, weight, eps: float = 1e-6):
        if not (x.is_cuda and x.numel()):
            return _ref(x, weight, eps)
        # Hot path: contiguous 2D row-batch (the decode call). Branch-light.
        if x.ndim == 2 and x.is_contiguous() and weight.is_contiguous():
            return _launch(x, weight, x.shape[1], eps)
        n = x.shape[-1]
        x2 = x.reshape(-1, n)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        w = weight if weight.is_contiguous() else weight.contiguous()
        return _launch(x2, w, n, eps).reshape(x.shape)

    return rmsnorm_fn


def _build_swiglu():
    """The 1.37×-at-decode SwiGLU: bf16-native in-place ``silu_().mul_()`` — matches eager's 2 launches
    with ZERO intermediate allocations and no fp32 upcast (F.silu does fp32 opmath internally). A Triton
    kernel loses here (launch overhead on a no-reduction op). Returns ``swiglu_fn(gate, up)``."""
    import torch
    import torch.nn.functional as F

    silu = F.silu
    _silu_ = getattr(torch._C._nn, "silu_", None)
    if _silu_ is None:  # pragma: no cover

        def _silu_(t):
            return silu(t, inplace=True)

    def swiglu_fn(gate, up):
        # Forward-only serving path: mutate the (fresh) gate_proj output in place. If a grad is ever
        # attached (never in the serve regime), fall back to the out-of-place op — no mutation.
        if gate.requires_grad or up.requires_grad:
            return silu(gate) * up
        return _silu_(gate).mul_(up)

    return swiglu_fn


def _build_gated_rmsnorm():
    """The 3.42×-at-decode GatedDeltaNet gated RMSNorm (``y = (x*rsqrt(mean(x²)+eps)*weight)*silu(z)``):
    ONE fused Triton launch, one program per row, in-register fp32 reduce → plain-weight scale → silu(z)
    gate → store. Like plain rmsnorm the op has a reduction, so eager fires ~8–10 tiny kernels (fp32
    casts, pow/mean/rsqrt, sigmoid, several muls); collapsing to a single launch wins big at decode M.
    The op is applied per V-head (norm over ``head_v_dim`` only), and the mixer hands us a contiguous
    ``(-1, head_v_dim)`` view, so the hot path always hits. Returns ``gated_rmsnorm_fn(x, weight, z, eps)``
    or None (no GPU/triton). Note the kernel keeps the whole chain in fp32; the eager module rounds the
    normalized value to the input dtype *before* the weight-mul, so the kernel is marginally *more*
    precise — the self-test still gates it against the exact eager math at 2e-2."""
    import torch

    try:
        import triton
        import triton.language as tl
    except Exception:  # pragma: no cover
        return None
    if not torch.cuda.is_available():
        return None

    @triton.jit
    def _gated_rms_kernel(X, Z, W, Y, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        base = row * N  # contiguous 2D view: row stride == N, inner stride == 1
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(Z + base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        # rms over the real (unmasked) elements; masked lanes add 0 to sum(x*x). nan/inf flow through
        # sum → rstd → y exactly like the oracle's fp32 arithmetic.
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
        silu = z * tl.sigmoid(z)  # fp32 silu; z=-inf → sigmoid 0 → (-inf*0)=nan, matches F.silu
        y = (x * rstd * w) * silu
        tl.store(Y + base + cols, y.to(Y.dtype.element_ty), mask=mask)

    _npow2 = triton.next_power_of_2
    _cfg: dict[int, tuple[int, int]] = {}

    def _config(h: int):
        c = _cfg.get(h)
        if c is not None:
            return c
        block = _npow2(h)
        nw = 2 if block <= 256 else (4 if block <= 1024 else (8 if block <= 2048 else 16))
        c = (block, nw)
        _cfg[h] = c
        return c

    def _ref(x, weight, z, eps):
        import torch.nn.functional as F

        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        return (y * F.silu(z.float())).to(x.dtype)

    def gated_rmsnorm_fn(x, weight, z, eps):
        # Hot path: CUDA, contiguous, matching x/z shapes, 1-D weight sized to the last dim. Any ndim
        # works — flatten to (M, H) with a zero-copy view. Everything else → exact fp32 reference.
        if (
            x.is_cuda
            and z.is_cuda
            and weight.is_cuda
            and x.is_floating_point()
            and x.is_contiguous()
            and z.is_contiguous()
            and weight.is_contiguous()
            and z.shape == x.shape
            and weight.dim() == 1
            and weight.shape[0] == x.shape[-1]
        ):
            h = x.shape[-1]
            if h == 0 or x.numel() == 0:
                return torch.empty_like(x)
            m = x.numel() // h
            block, nw = _config(h)
            y = torch.empty_like(x)
            _gated_rms_kernel[(m,)](
                x.view(m, h), z.view(m, h), weight, y.view(m, h), h, float(eps), BLOCK=block, num_warps=nw, num_stages=1
            )
            return y
        return _ref(x, weight, z, eps)

    return gated_rmsnorm_fn


def _self_test_rmsnorm(fn, gemma: bool, dims=(2048, 2560, 4096)) -> bool:
    import torch

    for n in dims:
        x = torch.randn(8, n, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        xf = x.float()
        ref = (
            xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * ((1.0 + w.float()) if gemma else w.float())
        ).to(x.dtype)
        got = fn(x, w, 1e-6)
        if not torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=2e-2):
            return False
    return True


def _self_test_swiglu(fn) -> bool:
    import torch
    import torch.nn.functional as F

    for n in (5632, 8960):
        g = torch.randn(8, n, device="cuda", dtype=torch.bfloat16)
        u = torch.randn(8, n, device="cuda", dtype=torch.bfloat16)
        ref = (F.silu(g.float()) * u.float()).to(g.dtype)
        got = fn(g.clone(), u)  # clone: swiglu_fn mutates gate in place
        if not torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=2e-2):
            return False
    return True


def _self_test_gated_rmsnorm(fn, dims=(128, 256, 512)) -> bool:
    # Gate against the EXACT Qwen*RMSNormGated math (fp32 normalize → cast to input dtype → weight-mul
    # in input dtype → fp32 silu gate → cast back), NOT the idealized all-fp32 oracle, so the install
    # decision reflects the real serving forward. dims span common head_v_dim values.
    import torch
    import torch.nn.functional as F

    for n in dims:
        x = torch.randn(8, n, device="cuda", dtype=torch.bfloat16)
        z = torch.randn(8, n, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        hs = x.float()
        hs = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = w * hs.to(x.dtype)
        ref = (ref * F.silu(z.float())).to(x.dtype)
        got = fn(x, w, z, 1e-6)
        if not torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=2e-2):
            return False
    return True


def _make_rms_forward(fn):
    def forward(self, hidden_states):
        eps = float(getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6)))
        if (
            hidden_states.is_cuda
            and self.weight.is_cuda
            and hidden_states.shape[-1] == self.weight.numel()
            and self.weight.dtype == hidden_states.dtype
        ):
            return fn(hidden_states, self.weight, eps)
        import torch

        xf = hidden_states.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        gemma = getattr(self, "_chalk_decode_gemma", True)
        return (y * ((1.0 + self.weight.float()) if gemma else self.weight.float())).to(hidden_states.dtype)

    return forward


def _make_mlp_forward(fn):
    def forward(self, x):
        g, u = self.gate_proj(x), self.up_proj(x)
        if g.is_cuda and type(self.act_fn).__name__ in ("SiLU", "SiLUActivation"):
            return self.down_proj(fn(g, u))
        return self.down_proj(self.act_fn(g) * u)

    return forward


def _build_rope():
    """The 4.4x-at-decode fused RoPE: ONE Triton launch rotates BOTH q and k (rotate_half / GPT-NeoX
    form), loading each token's cos/sin once and reusing them across all q and k heads — collapsing
    the eager path's rotate_half ``cat`` + ~16 elementwise launches into one, zero extra allocations.
    Handles GQA + partial rotary. Fast path is B==1 (single-sequence decode); B>1 / exotic layouts take
    the exact eager fallback. Returns ``rope_fn(q, k, cos, sin) -> (q_out, k_out)`` or None (no GPU)."""
    import torch

    try:
        import triton
        import triton.language as tl
    except Exception:  # pragma: no cover
        return None
    if not torch.cuda.is_available():
        return None

    @triton.jit
    def _rope_fused_kernel(
        q_ptr,
        k_ptr,
        qo_ptr,
        ko_ptr,
        cos_ptr,
        sin_ptr,
        NH,
        NKV,
        HD,
        ROT,
        HALF,
        q_hs,
        q_ts,
        k_hs,
        k_ts,
        cs_ts,
        BLOCK_NH: tl.constexpr,
        BLOCK_NKV: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
        BLOCK_PASS: tl.constexpr,
        HAS_PASS: tl.constexpr,
    ):
        tok = tl.program_id(0)
        ch = tl.arange(0, BLOCK_HALF)
        cmask = ch < HALF
        cbase = cos_ptr + tok * cs_ts
        sbase = sin_ptr + tok * cs_ts
        cos1 = tl.load(cbase + ch, mask=cmask, other=0.0).to(tl.float32)
        sin1 = tl.load(sbase + ch, mask=cmask, other=0.0).to(tl.float32)
        cos2 = tl.load(cbase + HALF + ch, mask=cmask, other=0.0).to(tl.float32)
        sin2 = tl.load(sbase + HALF + ch, mask=cmask, other=0.0).to(tl.float32)
        h = tl.arange(0, BLOCK_NH)
        hmask = h < NH
        q_row = q_ptr + tok * q_ts + h[:, None] * q_hs
        qo_row = qo_ptr + tok * q_ts + h[:, None] * q_hs
        m2 = hmask[:, None] & cmask[None, :]
        x1 = tl.load(q_row + ch[None, :], mask=m2, other=0.0).to(tl.float32)
        x2 = tl.load(q_row + HALF + ch[None, :], mask=m2, other=0.0).to(tl.float32)
        tl.store(qo_row + ch[None, :], x1 * cos1[None, :] - x2 * sin1[None, :], mask=m2)
        tl.store(qo_row + HALF + ch[None, :], x2 * cos2[None, :] + x1 * sin2[None, :], mask=m2)
        if HAS_PASS:
            p = tl.arange(0, BLOCK_PASS)
            mp = hmask[:, None] & (p < (HD - ROT))[None, :]
            tl.store(qo_row + ROT + p[None, :], tl.load(q_row + ROT + p[None, :], mask=mp, other=0.0), mask=mp)
        hk = tl.arange(0, BLOCK_NKV)
        hkmask = hk < NKV
        k_row = k_ptr + tok * k_ts + hk[:, None] * k_hs
        ko_row = ko_ptr + tok * k_ts + hk[:, None] * k_hs
        mk2 = hkmask[:, None] & cmask[None, :]
        y1 = tl.load(k_row + ch[None, :], mask=mk2, other=0.0).to(tl.float32)
        y2 = tl.load(k_row + HALF + ch[None, :], mask=mk2, other=0.0).to(tl.float32)
        tl.store(ko_row + ch[None, :], y1 * cos1[None, :] - y2 * sin1[None, :], mask=mk2)
        tl.store(ko_row + HALF + ch[None, :], y2 * cos2[None, :] + y1 * sin2[None, :], mask=mk2)
        if HAS_PASS:
            pk = tl.arange(0, BLOCK_PASS)
            mkp = hkmask[:, None] & (pk < (HD - ROT))[None, :]
            tl.store(ko_row + ROT + pk[None, :], tl.load(k_row + ROT + pk[None, :], mask=mkp, other=0.0), mask=mkp)

    _np2 = triton.next_power_of_2

    def _eager(q, k, cos, sin):
        rot = cos.shape[-1]
        c, s = cos.float().unsqueeze(1), sin.float().unsqueeze(1)

        def ap(t):
            tf = t.float()
            tr, tp = tf[..., :rot], tf[..., rot:]
            half = tr.shape[-1] // 2
            rh = torch.cat((-tr[..., half:], tr[..., :half]), dim=-1)
            return torch.cat((tr * c + rh * s, tp), dim=-1).to(t.dtype)

        return ap(q), ap(k)

    def rope_fn(q, k, cos, sin):
        if (
            not (q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda)
            or q.dim() != 4
            or k.dim() != 4
            or cos.dim() != 3
            or sin.dim() != 3
        ):
            return _eager(q, k, cos, sin)
        B, NH, T, HD = q.shape
        Bk, NKV, Tk, HDk = k.shape
        ROT = cos.shape[-1]
        if (
            B != 1
            or Bk != 1
            or Tk != T
            or HDk != HD
            or T <= 0
            or ROT <= 0
            or (ROT % 2)
            or ROT > HD
            or cos.shape[0] != 1
            or cos.shape[1] != T
            or tuple(sin.shape) != tuple(cos.shape)
        ):
            return _eager(q, k, cos, sin)
        qc, kc = q.contiguous(), k.contiguous()
        cc, sc = cos.contiguous(), sin.contiguous()
        qo, ko = torch.empty_like(qc), torch.empty_like(kc)
        HALF = ROT // 2
        HAS_PASS = HD > ROT
        # Bind the Triton launch to q's CUDA device: under device_map / multi-GPU serving the tensors
        # may live on a non-default GPU, and an unbound launch would run on the current device →
        # cross-device pointer fault. torch.cuda.device is a cheap no-op when already on that device.
        with torch.cuda.device(qc.device):
            _rope_fused_kernel[(T,)](
                qc,
                kc,
                qo,
                ko,
                cc,
                sc,
                NH,
                NKV,
                HD,
                ROT,
                HALF,
                qc.stride(1),
                qc.stride(2),
                kc.stride(1),
                kc.stride(2),
                cc.stride(1),
                BLOCK_NH=_np2(NH),
                BLOCK_NKV=_np2(NKV),
                BLOCK_HALF=_np2(HALF),
                BLOCK_PASS=_np2(HD - ROT) if HAS_PASS else 1,
                HAS_PASS=HAS_PASS,
                num_warps=4,
            )
        return qo, ko

    return rope_fn


def _self_test_rope(fn) -> bool:
    """Compare the fused rope against an eager rotate_half reference at decode shapes (B=1, GQA,
    full + partial rotary)."""
    import torch

    for nh, nkv, hd, rot in ((16, 4, 128, 128), (16, 4, 128, 64)):
        q = torch.randn(1, nh, 8, hd, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, nkv, 8, hd, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(1, 8, rot, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(1, 8, rot, device="cuda", dtype=torch.bfloat16)
        c, s = cos.float().unsqueeze(1), sin.float().unsqueeze(1)

        def _ap(t, c=c, s=s, rot=rot):
            tf = t.float()
            tr, tp = tf[..., :rot], tf[..., rot:]
            half = tr.shape[-1] // 2
            rh = torch.cat((-tr[..., half:], tr[..., :half]), dim=-1)
            return torch.cat((tr * c + rh * s, tp), dim=-1).to(t.dtype)

        rq, rk = _ap(q), _ap(k)
        gq, gk = fn(q, k, cos, sin)
        if not (
            torch.allclose(gq.float(), rq.float(), atol=2e-2, rtol=2e-2)
            and torch.allclose(gk.float(), rk.float(), atol=2e-2, rtol=2e-2)
        ):
            return False
    return True


def _make_rope_wrapper(fn, orig):
    """Wrap the module's ``apply_rotary_pos_emb``. The fused kernel assumes the default head-major
    layout (``unsqueeze_dim=1``, cos/sin broadcast over the head axis); any other ``unsqueeze_dim``
    changes the broadcast axis, so defer to the original impl there. The eager fallback is signature-
    robust: if the original was already swapped for one that doesn't accept ``position_ids`` (e.g.
    chalk's training RoPE), we retry with the narrower signature instead of raising ``TypeError``."""

    def _call_orig(q, k, cos, sin, position_ids, unsqueeze_dim):
        try:
            return orig(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
        except TypeError:
            try:
                return orig(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
            except TypeError:
                return orig(q, k, cos, sin)

    def _wrapped(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if unsqueeze_dim != 1:  # non-default broadcast axis — the fused kernel can't represent it
            return _call_orig(q, k, cos, sin, position_ids, unsqueeze_dim)
        try:
            return fn(q, k, cos, sin)
        except Exception:
            return _call_orig(q, k, cos, sin, position_ids, unsqueeze_dim)

    _wrapped._chalk_decode_rope = True  # idempotency marker (don't double-wrap)
    return _wrapped


def _install_decode_rope(model) -> dict | bool | None:
    """Patch the module-level ``apply_rotary_pos_emb`` (the function the full-attention layers call;
    GDN layers don't) with the fused decode kernel — self-tested against the original, eager fallback.
    Mirrors chalk.ops.rope's install target. Patches EVERY distinct modeling module that exposes
    ``apply_rotary_pos_emb`` (dense + MoE + VL can each ship their own; a single model may mix them),
    not just the first — otherwise some full-attention layers keep an unpatched binding while the
    report still claims success. Returns installed-report / False / None."""
    import importlib

    import torch

    fn = _build_rope()
    if fn is None:
        return False
    mods, seen = [], set()
    for m in model.modules():
        modname = type(m).__module__
        if modname in seen or modname == "builtins":
            continue
        seen.add(modname)
        try:
            cand = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(cand, "apply_rotary_pos_emb") and cand not in mods:
            mods.append(cand)
    if not mods:
        return None
    try:
        if not _self_test_rope(fn):
            return False
    except Exception:
        return False

    patched = []
    for mod in mods:
        orig = mod.apply_rotary_pos_emb
        if getattr(orig, "_chalk_decode_rope", False):  # already patched (shared module) — count it
            patched.append(mod.__name__)
            continue
        try:
            # Self-test the fused fn against THIS module's own apply_rotary_pos_emb so a per-module
            # convention diff leaves that module eager instead of miscomputing.
            q = torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 2, 8, 128, device="cuda", dtype=torch.bfloat16)
            cos = torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16)
            sin = torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16)
            rq, rk = orig(q, k, cos, sin)
            gq, gk = fn(q, k, cos, sin)
            if not (
                torch.allclose(gq.float(), rq.float(), atol=3e-2, rtol=3e-2)
                and torch.allclose(gk.float(), rk.float(), atol=3e-2, rtol=3e-2)
            ):
                continue
        except Exception:
            continue
        mod.apply_rotary_pos_emb = _make_rope_wrapper(fn, orig)
        patched.append(mod.__name__)
    if not patched:
        return False
    return {"installed": True, "modules": patched}


def _make_gated_rms_forward(fn):
    # Replaces Qwen*RMSNormGated.forward(self, hidden_states, gate). Hot path → fused kernel; the eager
    # fallback reproduces the module's EXACT math (fp32 normalize, weight-mul in input dtype, fp32 silu
    # gate) so non-hot-path inputs and the gate=None edge match transformers bit-for-bit.
    def forward(self, hidden_states, gate=None):
        eps = float(getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6)))
        if (
            gate is not None
            and hidden_states.is_cuda
            and gate.is_cuda
            and self.weight.is_cuda
            and hidden_states.shape[-1] == self.weight.numel()
            and self.weight.dtype == hidden_states.dtype
            and hidden_states.is_contiguous()
            and self.weight.is_contiguous()
            and gate.is_contiguous()
            and gate.shape == hidden_states.shape
            and gate.dtype == hidden_states.dtype
        ):
            return fn(hidden_states, self.weight, gate, eps)
        import torch
        import torch.nn.functional as F

        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        hs = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + eps)
        hs = self.weight * hs.to(input_dtype)
        if gate is None:
            return hs.to(input_dtype)
        hs = hs * F.silu(gate.to(torch.float32))
        return hs.to(input_dtype)

    return forward


def apply_chalk_decode_kernels(
    model,
    *,
    base_model: str | None = None,
    rmsnorm: bool = True,
    swiglu: bool = True,
    rope: bool = True,
    gated_rmsnorm: bool = True,
) -> dict:
    """Install the decode-regime rmsnorm + swiglu + rope + GDN gated-rmsnorm on a live HF model (call
    AFTER build). Self-test gated with eager fallback; returns a report ``{"rmsnorm": ..., "swiglu":
    ..., "rope": ..., "gated_rmsnorm": ...}`` (dict/installed, False/verified-off, None/no matching
    module). The gated rmsnorm only matches the linear-attention (GatedDeltaNet) Qwen3.5/3.6 models —
    Llama/MiniCPM have no such module, so it reports None there. NEVER raises on a kernel failure."""
    report: dict = {}
    name = (base_model or type(model).__name__).lower()
    gemma = "minicpm" not in name and "llama" not in name

    if rmsnorm:
        norm_names = _RMSNORM_LLAMA if not gemma else _RMSNORM_GEMMA
        classes = {type(m) for m in model.modules() if type(m).__name__ in norm_names}
        if not classes:
            report["rmsnorm"] = None
        else:
            fn = _build_rmsnorm(gemma)
            try:
                ok = fn is not None and _self_test_rmsnorm(fn, gemma)
            except Exception:
                ok = False
            if ok:
                fwd = _make_rms_forward(fn)
                for c in classes:
                    c.forward = fwd
                    c._chalk_decode_gemma = gemma
                report["rmsnorm"] = {"installed": True, "gemma": gemma, "classes": sorted(c.__name__ for c in classes)}
            else:
                report["rmsnorm"] = False

    if swiglu:
        classes = {type(m) for m in model.modules() if type(m).__name__ in _MLP_CLASSES}
        if not classes:
            report["swiglu"] = None
        else:
            fn = _build_swiglu()
            try:
                ok = _self_test_swiglu(fn)
            except Exception:
                ok = False
            if ok:
                fwd = _make_mlp_forward(fn)
                for c in classes:
                    c.forward = fwd
                report["swiglu"] = {"installed": True, "classes": sorted(c.__name__ for c in classes)}
            else:
                report["swiglu"] = False

    if rope:
        try:
            report["rope"] = _install_decode_rope(model)
        except Exception:
            report["rope"] = False

    if gated_rmsnorm:
        classes = {type(m) for m in model.modules() if type(m).__name__ in _GATED_RMSNORM_CLASSES}
        if not classes:
            report["gated_rmsnorm"] = None
        else:
            fn = _build_gated_rmsnorm()
            try:
                ok = fn is not None and _self_test_gated_rmsnorm(fn)
            except Exception:
                ok = False
            if ok:
                fwd = _make_gated_rms_forward(fn)
                for c in classes:
                    c.forward = fwd
                report["gated_rmsnorm"] = {"installed": True, "classes": sorted(c.__name__ for c in classes)}
            else:
                report["gated_rmsnorm"] = False

    return report
