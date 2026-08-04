# chalk decode serving kernels — autoresearch + E2E results

`chalk.serving.decode.apply_chalk_decode_kernels` ships kernels tuned for the **decode / serving**
regime (one token per in-flight sequence, M≈8, forward-only), found by the `serving_bench` decode
autoresearch (`autoresearch/manifest/serve.py`) and validated end-to-end. All numbers are real Modal
L4 (sm89) runs.

## Why chalk needed decode-specific kernels

chalk's training kernels are tuned for long sequences (fwd+bwd). At decode that tuning is wrong — the
training kernels are **slower than eager**. Per-op, forward-only, at M=8 on sm89 (verified through all
five verifier gates: sandbox / correctness / generalization / timing / roofline):

| op | chalk *training* kernel | **decode kernel (this module)** | why they differ |
|---|---|---|---|
| rmsnorm | 1.4× | **4.43×** | rmsnorm has a reduction → one fused Triton launch collapses PyTorch's ~6–8 tiny kernels |
| swiglu | **0.24×** (harmful) | **1.37×** | swiglu is pure elementwise → a Triton launch *loses*; win = bf16-native in-place `silu_().mul_()`, 2 launches, 0 allocations |
| gated_rmsnorm | — | **3.42×** | GatedDeltaNet mixer `self.norm`: `(x·rsqrt(mean(x²)+eps)·weight)·silu(z)` per V-head — another reduction op firing ~8–10 tiny eager kernels → one fused launch collapses them |

### gated_rmsnorm — GatedDeltaNet mixer norm (linear-attention models only)

Qwen3.5/3.6 are hybrid GatedDeltaNet + attention. The GDN mixer applies a **gated RMSNorm** to its core
output (`self.norm(core_attn_out, z)` → `(x·rsqrt(mean(x²,-1)+eps)·weight)·silu(z)`, normalized per
V-head over `head_v_dim`). With `fla` absent the runtime path is the pure-torch `Qwen*RMSNormGated`
class, which fires ~8–10 tiny eager kernels (fp32 casts, pow/mean/rsqrt, sigmoid, several muls). The
decode kernel collapses that to a **single fused Triton launch** (one program per row, in-register fp32
reduce → plain-weight scale → `silu(z)` gate → store): **3.42× at decode M=8 on L4/sm89**, verified
through all five gates. Patches `Qwen3NextRMSNormGated` / `Qwen3_5RMSNormGated` / `Qwen3_5MoeRMSNormGated`.
Self-test-gated against the *exact* eager math (which rounds the normalized value to the input dtype
before the weight-mul — the fused kernel keeps it fp32, so it is marginally more precise), 2e-2 tol.
Llama/MiniCPM have no such module → reported `None` there.

## E2E confirmation (Qwen3.5-0.8B, full-model decode, tok/s)

Patching only these two layers, everything else identical:

| variant | batch 1 | batch 8 |
|---|---|---|
| eager | 36.0 | 272.8 |
| chalk **training** kernels | 33.7 (0.94×) | 253.0 (0.93×) |
| chalk **decode** kernels (this module) | **39.1 (1.09×)** | **306.4 (1.12×)** |

So chalk's decode kernels flip serving-decode from a **0.93× regression to a 1.09–1.12× win** over
eager, and are **1.16–1.21× faster than chalk's training kernels**. The net E2E gain is smaller than
the per-op speedups because rmsnorm+swiglu are only part of the decode step (attention / GDN /
projections dominate) — but it is a real, positive, verified serving improvement.

## Reproduce

Per-op autoresearch: `autoresearch/manifest/serve.py` builds forward-only decode contracts; the loop
generates + verifies candidates on L4/sm89. E2E: patch `apply_chalk_decode_kernels` into an HF model
and measure `generate` throughput vs eager and vs `chalk.transformers` kernels.

_Scope: the rmsnorm kernel supports both gemma (Qwen, `(1+w)`) and llama (MiniCPM/Llama, plain `w`)
conventions; swiglu is convention-agnostic. E2E validated on Qwen3.5-0.8B; the same pattern applies to
the other dense serving tiers (follow-up: E2E on 2B/4B/9B + MiniCPM5)._
