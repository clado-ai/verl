# Deleted arch cells — proven not to beat portable (graduate-or-delete)

Per the graduate-or-delete rule: the per-arch tree is a **speedup overlay** on the portable
kernels. A cell earns its place only by beating portable on real hardware (`load_entry` selects it
only when `TUNED=True` and it passes the live self-test; otherwise dispatch falls back to portable).
Cells proven **not** to beat portable are dead weight and are removed here. Deletion is
production-neutral: `load_entry` already fell back to portable for every one of these (none was
`TUNED=True`), and it tolerates a missing cell (ImportError → portable). Each cell can be
re-seeded from the portable kernel if a genuinely new approach is attempted.

Evidence for the on-box A/B rows: `benchmark/results/arch_vs_portable_20260706.json` (PR #71),
100 reps, order-reversed, self-test-gated, ratio = portable_ms/arch_ms (>1 = arch faster).

## Proven-bad cells removed (16)

| cell | why removed | evidence |
|---|---|---|
| rmsnorm@sm80 | `TUNED=False` — demoted; ~0.69× vs the improved portable | prior on-box (PR #52) |
| gdn_gated_rmsnorm@sm80 | seed; the best-known candidate (ported 2.07×-vs-eager win) A/B'd **0.90×** vs portable | A/B 2026-07-06 |
| rmsnorm@sm86 | A/B **0.84×** vs portable (1.06× only at tiny tokens, 0.75× at scale) | A/B 2026-07-06 |
| swiglu@sm86 | A/B ~1.00× (parity, geomean 1.004 min 0.979) — no real win | A/B 2026-07-06 |
| rope@sm86 | A/B **0.95×** vs portable | A/B 2026-07-06 |
| gdn_gated_rmsnorm@sm86 | `TUNED=False` — 4.02× vs eager but **0.91×** vs the row-tiled portable at M=196608 | prior on-box (dbeae66) |
| rmsnorm@sm89 | A/B **0.89×** vs portable | A/B 2026-07-06 |
| swiglu@sm89 | A/B ~1.00× (geomean 0.998) — no win | A/B 2026-07-06 |
| rope@sm89 | A/B **0.98×** (1.007× on GQA prod shape, far below any bar) | A/B 2026-07-06 |
| qkv@sm89 | `TUNED=False` — verifier heads-major layout inflated the grade; on-box 0.69–0.99× | PR #51 |
| rmsnorm@sm90 | A/B **0.55×** vs portable (badly slower at scale) | A/B 2026-07-06 |
| swiglu@sm90 | `STATUS: NO WIN` — autograd.Function Python overhead ~2× eager's C++ nodes; eager-passthrough | dev header |
| qkv@sm90 | `TUNED=False` — BLOCK_M=256 backward register-spills at head_dim=256 | PR #51 |
| fp8_base@sm90 | `STATUS: NO WIN` — fp8 accelerates only the fwd GEMM; bwd must stay bf16 on Hopper (structural) | dev header |
| rope@sm100 | A/B **0.89×** — the host-side pass-through tail copy dominates on the GQA prod shape (192-wide tail) | A/B 2026-07-06 |
| fp8_base@sm100 | `STATUS: NO WIN` — quant/transpose overhead exceeds the scaled_mm saving; bwd bf16 (structural) | dev header |

**Not deleted — a validated (if modest) win kept as a candidate:** `rmsnorm@sm100` beats portable
at *every* shape on B200 (geomean 1.037×, min 1.023×, no regression across 4k–196k tokens) but sits
below the 1.05 promotion gate. It is a real beats-portable result, not dead weight — retained as a
confirmation-run candidate (see the results JSON), not removed.

## sm120 (consumer Blackwell) removed wholesale

sm120 is a documented **non-target**: Triton 3.4.0 miscompiles on consumer Blackwell (~70% forward
error on correct kernels), so every cell there fails its live self-test and `load_entry` falls to
portable regardless. No sm120 cell can be validated or adopted, so the whole `sm120/` arch dir is
removed (untestable → cannot validate → delete). Re-seed if sm120 becomes a real target and Triton
is fixed.

## gdn_conv (all arches) — tested by evolve, disproven, removed

The `gdn_conv` cells were never-targeted stubs. A generation-1 hive evolve loop on `gdn_conv@sm100`
(B200) explored 4 distinct strategies (fused-epilogue, autotune, weight-in-registers,
backward-fusion) — all converged to the same lever: **sequence-tiling to raise SM occupancy**, since
the portable one-program-per-(batch,channel) launch under-occupies the 148 SMs at the verifier's
narrow dev shape (C=768 / Qwen3.5-0.8B). The hive verifier graded the best at 1.09× — but that was a
**timing artifact**: a follow-up C-sweep A/B (order-reversed, `min_ratio` = worst of two passes)
showed the "win" does **not reproduce** and, crucially, **regresses at production channel counts**:

| C (conv_dim) | model | min_ratio vs portable |
|---|---|---|
| 768 | 0.8B | 0.99 (parity — no win at its home shape) |
| 1536 | 2B | 1.03–1.04 (noise) |
| 3072 | ~4B | **0.955 (regresses ~4.5%)** |
| 6144 | ~35B | 1.07–1.14 (noise) |

Root cause: `gdn_conv` fwd+bwd is a tiny (~12 MB), **launch-latency-bound** workload; single-shot
timing is order/warmup-noise-dominated (pass1 read 1.67× where pass2 read 0.99×). At production
wide-C the portable already saturates the SMs, so the occupancy lever has nothing to fix — matching
the documented "portable gdn_conv near-ceiling (num_warps formula confirmed A100/H100/B200)" finding.
The evolve exhausted the plausible strategies with no real win, so all 5 `gdn_conv` cells are removed
(production runs the portable kernel via `load_entry`, unchanged). Re-seed only for a fundamentally
different (non-occupancy) approach. Lesson: the hive verifier's single-shot timing needs
order-reversed multi-shape confirmation before trust on launch-bound ops.
