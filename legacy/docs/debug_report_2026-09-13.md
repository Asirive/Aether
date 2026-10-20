# Debug Report — 2026-09-13

Full training-pipeline audit after four consecutive full-size runs died between step 600 and step 2035.
Everything below is backed by the run logs in `runs/aether_200m_dead_v1/` and by executed probes.

## Summary of root causes

| # | Root cause | Effect | Fix |
|---|-----------|--------|-----|
| 1 | Spike guard never had an absolute grad-norm threshold (`grad_spike_threshold` was dead code) | A 19,523-norm gradient was clipped to 1.0 and applied at step ~1700, destroying the model (loss 5.73 → 12.57, worse than random init 11.76) | `_optimizer_step` now skips when the pre-clip norm exceeds the threshold |
| 2 | Loss-EMA only updated on non-skipped steps | After the damage, every step was skipped forever (loss 12.57 > 1.5 × frozen EMA 5.8); guard log shows `grad_norm=0.000` for 335 consecutive steps | EMA tracks slowly during skips; after `max_consecutive_skips` it re-seeds and arms a forced clipped update |
| 3 | bf16 optimizer state silently rounded away updates | `A_log` 0.0 %, RMSNorm 0.0 %, `mask_logits` 2.3 %, embedding 87.5 % of elements changed per step — the two novel mechanisms (learned SSM timescales, learned FSA frequency bins) never trained | fp32 master weights + bf16 autocast compute |
| 4 | Hillis–Steele scan materialised a `(bsz, seq_len, N, N)` tensor | ~14,000 `as_strided` calls per micro-batch; CPU launch-bound, 3–16 % MFU | Vector-only doubling scan (only `(N, N)` powers), 7× fewer ops |
| 5 | LM head materialised `(B, L, 128256)` logits + a contiguous shift copy + its gradient | ~2 GB of the 8.57 GB budget; forcing fp32 master weights pushed the allocator into shared memory (5.2× slowdown) | Chunked LM head + loss (`_chunked_causal_loss`), measured −0.92 GB |

## 1. The collapse (step ~1700, run of 10–11 Aug)

```
step 1650  loss=5.7298  grad_norm=1.018     healthy
step 1700  loss=8.1553  grad_norm=19523.755 explosion (applied, clipped to 1.0)
step 1750  loss=12.5733 grad_norm=0.000     335 steps skipped, model destroyed
```

The loss rose *before* the gradient exploded (5.73 → 8.16), so the gradient spike is a
symptom of an already-damaged state. Once the loss exceeded 1.5 × EMA the guard skipped
every subsequent update and never recovered, because the EMA was frozen by construction.
The guard's *only* documented escape (`grad_spike_threshold = 100`) was never referenced
anywhere in the code.

## 2. The learning-capacity bug (measured)

Adam's first step has `|Δw| = lr` exactly, which makes it a clean probe of bf16 rounding.
With `lr = 1.2e-4` and bf16 parameters, one real-data step changed:

| group | params | before | after fp32 master weights |
|---|---|---|---|
| `A_log` (PolyMamba timescales) | 384 | 0.0 % | 10.2 % |
| RMSNorm gains | 28,224 | 0.0 % | 100 % |
| `mask_logits` (FSA bin selection) | 3,456 | 2.3 % | 100 % |
| `B`/`C` (SSM maps) | 1,179,648 | 74.8 % | 100 % |
| token embedding | 73,875,456 | 87.5 % | 100 % |
| linear weights | 111,527,424 | 100 % | 100 % |

Cause: bf16 has 8 mantissa bits, so `ULP(3.86) = 1.5e-2` and `ULP(1.0) = 3.9e-3`, while the
update is `1.2e-4`. For |w| ≳ 0.015 the update rounds to zero. This supersedes PAF-3.
Table-wide: **99.8 % of parameters now change per step** (was 88 % with the two headline
components frozen at 0 %).

## 3. Throughput

| configuration | tok/s (L=1024, mb=2) | note |
|---|---|---|
| before (bf16 params, old scan) | ~3,800 (when not stalling) | 600 ↔ 4150 oscillation in the same run |
| old scan, fp32 master weights | 735 | allocator spill |
| new scan, fp32 master weights | 3,833 | |
| new scan + chunked head, fp32 | **3,900–3,976** | measured in the real trainer |
| bf16 reference | 4,150–4,255 | for comparison |

Plan-level consequence: 6.55 B tokens at 3,900 tok/s = **19.4 days** of continuous training,
not the 5–6 days budgeted. That gap, not the architecture, is the binding constraint.

## 4. Verification

Every change was audited by independent verification agents that reproduced each claim by
executing code. They found and I fixed:

- the chunked loss flattened batch and sequence together, training the model to predict the
  first token of the *next* document from the last token of the previous one (batch > 1);
- the anti-freeze force update bypassed the finite-grad check and applied a **nan** update,
  making every parameter nan;
- the anti-freeze counters were unreachable when the loss itself was nan.

Regression tests added: chunked-vs-full-logits equivalence (batch > 1, multi-chunk, −100
labels), short/all-masked edge cases, `_optimizer_step` contract (open window / healthy /
spike / nan), forced-update safety, and the skip-state bookkeeping.

`python -m pytest tests/ -q` → **113 passed**.

## 5. Honest limitations

- A permanently non-finite gradient regime still cannot be trained out of; the run now stops
  loudly after `wedged_skip_limit` consecutive skips instead of spinning forever.
- `grade()` on the released checkpoints: none of the pre-13-Sep checkpoints are usable
  (`runs/aether_200m_dead_v1/final.pt` scores 12.56 nats, worse than random init).
- The 6.55 B-token target needs ~19 days; the plan's 5–6 day estimate assumed full MFU on
  the measured 29.48 TFLOPS matmul, which no real training step reaches.
