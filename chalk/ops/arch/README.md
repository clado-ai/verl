# chalk per-architecture kernel tree — one folder per GPU arch, one file per layer

This tree is BOTH the shipped per-architecture kernels AND the live targets of the chalk
autoresearch grid. Kernels are **arch-gated, not card-gated** (a cell is per SM architecture), so
the grid is materialized as source: one folder per architecture, one file per layer (op) inside.

```
src/chalk/ops/arch/
  sm80/   ampere (A100)      — rmsnorm swiglu rope embedding lora flce gdn_* grpo_chunked_loss qkv moe_grouped_gemm
  sm86/   ampere (consumer)  — "
  sm89/   ada                — + fp8_base
  sm90/   hopper             — + fp8_base gdn_scan
  sm100/  blackwell (B200)   — + fp8_base
  sm120/  blackwell (consumer)— + fp8_base
```

Each `<arch>/<op>.py` defines `build()` returning the entry callable for that cell (e.g.
`rmsnorm_fn(x, weight, eps) -> y`). The file header records the cell id, entry signature, oracle,
tolerance, baseline, and target speedup.

## How production selects an arch kernel (dispatch)

`chalk/ops/arch/__init__.py::load_entry(op, self_test, portable=...)` is the dispatch each
`chalk.ops.<layer>.load_<layer>()` calls. On the running GPU it:

1. maps `torch.cuda.get_device_capability()` → arch string (`sm80` … `sm120`);
2. imports `chalk.ops.arch.<arch>.<op>` and selects it **only if** it is flagged a verified win
   (`TUNED = True`) **and** its `build()` passes the op's live-GPU numeric self-test vs the fp32
   oracle;
3. otherwise falls back to the op's **portable** kernel.

So this tree is a pure, self-validated _speedup overlay_: a seed (no `TUNED`), a missing file, a
broken build, or a numerically-wrong kernel can never degrade production — it only ever
_accelerates_ an op when a verified arch-specific kernel exists. A win file carries
`TUNED = True`, `SPEEDUP = <verified x>`, and `SPEEDUP_ANCHOR = <the baseline that figure divides
by>` set by the autoresearch promotion step.

`SPEEDUP` never gates selection — the three conditions above do — so its only consumer is the
dispatch line printed on load. That makes `SPEEDUP_ANCHOR` load-bearing for honesty rather than
behavior: a ratio means nothing without the baseline it divided by, and production always falls
back to the **portable** kernel (never eager, never Liger), so vs-portable is the only delta a
user actually receives. An overlay whose baseline has since been retired sets
`SPEEDUP_ANCHOR = None` and dispatch prints the figure as _unrestated_ instead of _verified_. The
strong word is opt-in: omitting the field reads as the weak claim, so a new overlay that forgets
it under-claims rather than inheriting a guarantee nobody checked.

### When is an arch file actually adopted?

A `VERIFIED` header records an autoresearch _research result_ — a kernel that beat eager on a real GPU of that arch. It is **not** the same as being live in production. On a given branch an arch file is adopted by `load_entry` only when **all three** hold:

1. the op's `load_<layer>()` is actually wired to call `load_entry` (the dispatch hook exists);
2. the arch file defines `TUNED = True`; and
3. its `build()` returns an entry whose signature matches that op's production `_self_test` (so the parity check can even run).

If any one is missing, `load_entry` falls back to the op's portable kernel — a `VERIFIED` header alone changes nothing. **Wired today** (their `load_*()` calls `load_entry`): `swiglu`, `rmsnorm`, `gdn_gate`, `gdn_conv`. The arch files on this branch are seeds/research results with no `TUNED` (and, for `flce`/`embedding`/`gdn_gated_rmsnorm`, entry signatures that still diverge from production, so their dispatch is intentionally reverted here); the verified wins — with `TUNED = True` and entry-signature alignment — land in the per-op kernel PRs.

## How a file is improved (autoresearch)

Each file starts as a **seed**: a numerically-correct eager reference (`TUNED` unset, ~1.0×). A
per-cell autoresearch agent — driven on the self-hosted [hive](https://github.com/rllm-org/hive)
server, one agent per `(arch, layer)` — then optimizes it in place, iterating

    author (Triton kernel) → verify on a real GPU of the file's arch → reward → refine

against the reward-hacking-proof verifier (`autoresearch/verifier/`): correctness, generalization
(secret + fuzz shapes), timing (clock-locked paired samples vs the CURRENT shipped chalk path when
one exists, else eager; median/IQR/CI plus a significance test against the effect-size floor),
roofline (physics ceiling), and a static anti-cheat scan. Only a kernel that passes **every** gate
with no cheat flags is written back here — with `TUNED = True`, the verified `SPEEDUP` and the
`SPEEDUP_ANCHOR` it was measured against, the
median+CI/significance verdict, and the GPU it was measured on recorded in the header. The harness
that does all this (verifier, manifest, contract, run loop, hive glue) lives under `autoresearch/` at
the repo root and is **not** shipped in the wheel; only this `src/chalk/ops/arch/` tree is.
