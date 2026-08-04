"""Install chalk's fused Gated-DeltaNet (GDN) epilogue kernel (fwd+bwd).

The Qwen3.5/3.6 GDN block runs three classes of op around its linear-attention scan:
  * the ``chunk_gated_delta_rule`` SCAN — fla's chunk-parallel kernel (+ tilelang on Hopper) is
    genuine SOTA; chalk leaves it on fla (the fallback is fla, NOT eager);
  * the short depthwise CAUSAL CONV + SiLU — runs EAGER (``F.silu(F.conv1d(...))``) whenever
    ``causal_conv1d`` is absent, which the flash worker deliberately does (causal_conv1d ships
    sdist-only and fails to build at cold-start). chalk REPLACES this eager conv with one fused
    Triton conv+SiLU kernel — ~1.05–1.32x on the conv, ~1.14–1.15x on the whole GDN block (H100);
  * the GATE (``beta``/``g``) — stays eager: a fused gate kernel was built + benchmarked and LOSES
    (the tensors are tiny so the path is autograd-dispatch bound, not compute bound — even
    torch.compile is slower than eager). Kept as a documented negative result in ``chalk.ops.gdn``.

``install_qwen35_gdn`` patches the GDN CLASS ``forward`` so the eager conv runs on chalk's kernel.
It is conservative: it only owns the multi-token, non-cached, non-packed training path and only
when ``causal_conv1d_fn is None``; every other path delegates to the original forward unchanged.
Self-test gated — any build/self-test failure leaves the eager path untouched.
"""

from chalk.ops.gdn import install_qwen35_gdn

__all__ = ["install_qwen35_gdn"]
