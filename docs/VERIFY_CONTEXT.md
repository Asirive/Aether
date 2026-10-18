# Architecture verification context (15 Sep 2026, round 1)

Shared briefing for the verification agents. Read this, then investigate your assigned angle.

## The model

`AETHER-200M` at `C:\Users\haziq\Documents\niga`. A 186,615,276-param LM:
`d_model=576, d_inner=1536, n_layers=24, n_heads=9, d_k=64, vocab=128256` (Llama-3 tokenizer,
tied embeddings, fp32 master weights + bf16 autocast).

Block layout (`aether/model.py::AetherBlock`): layers **even** = `PolyMamba2`, **odd** =
`FreqSparseAttention` (CC-FSA); every block has a SwiGLU FFN; pre-norm residuals.
A single `MoDRouter` (`aether/components/mod_router.py`) runs at layers **4, 8, 12, 16, 20**
with keep ratios `(0.5, 0.7, 0.85, 1.0, 1.0)`; the gate it computes is **reused** for every
subsequent block until the next routing point, applied as `x = gate*block(x) + (1-gate)*x`.

Training: FineWeb-Edu `sample-10BT`, seq_len 1024, micro_batch 2, grad_accum 32 (65,536
token batch), AdamW(fused, fp32 params), lr 1.2e-4 with 1000-step warmup then cosine,
grad clip 1.0, gradient checkpointing on.

## What just happened (the reason for this audit)

Four full runs died the same way: healthy to ~1200 steps (loss ~5.7, grad norm ~1.0), then a
slow grad-norm climb (1.0 → 9.5 @1300 → 23 → 46 @1400), then every step skipped forever.
A 10-agent investigation concluded the cause was the MoD router reading the **un-normalised**
residual stream (gradient ∝ ‖x‖²/T). Four fixes shipped:

1. `mod_router.py`: `F.rms_norm` on the router input (+learned `route_norm_weight`),
   `temperature` 0.1 → 1.0, and an ST-MoE z-loss `1e-2 * mean(score²)` folded into the
   returned routing loss.
2. `polymamba.py`: `dt` clamp made two-sided (`_compute_dt_bounds`).
3. `fsa.py` + `ops/topk_soft.py`: `soft_topk_mask` returns the indices it actually selected
   and the FSA gathers coefficients AND weights with those.
4. `optim.py`: `embed_lr_scale` → the tied embedding sits in its own param group at **0.1×**
   the base LR (config `optim.embed_lr_scale: 0.1`).

## The result, measured

| step | run2 (before fixes) loss | run3 (after fixes) loss |
|---|---|---|
| 100 | 10.65 | 10.87 |
| 200 | **9.34** | **10.53** |
| 300 | ~8.8 | 10.11 |
| 400 | ~8.3 | 9.71 |

- Grad norm is now **1.5–1.9** (was 0.8–1.0) but the collapse is **absent**: **zero**
  `[spike-guard]` events through step 440, versus a collapse at step 1274 before.
- Throughput unchanged (~3,800 tok/s).
- run3 was stopped at step 440 (never reached the 1300 region).

So: the collapse symptom is gone AND the model learns ~1.2 nats slower. Both facts matter.

## Prime suspect for the regression

`embed_lr_scale: 0.1`. The tied embedding IS the LM head; early training is dominated by
learning unigram/bigram statistics, which live in the embedding + head. A 10× LR cut there
directly slows the loss curve.

## Reference numbers (measured on this exact data + tokenizer)

- random init CE `ln(128256)` = 11.76
- empirical unigram entropy = 7.71 nats
- empirical bigram conditional entropy = 4.45 nats
- a ~200M model at ~6.5B tokens on FineWeb-Edu should reach ~3.0–3.2 nats

## HARD RESOURCE RULES (the user's PC has crashed twice under load)

- **CPU ONLY.** No GPU, no `torch.cuda`, no training runs, no `CUDA_VISIBLE_DEVICES` games.
- `export OMP_NUM_THREADS=1` and call `torch.set_num_threads(1)`.
- Every experiment under ~3 minutes; tiny tensors (`d_model ≤ 64`, `≤ 4` layers, `seq ≤ 64`)
  unless your brief says otherwise.
- Do NOT load `runs/aether_200m_run3/rolling.pt` (2.1 GB) unless your brief explicitly allows it.
- Do NOT launch `aether.training.pretrain` or any long job.

## Historical evidence on disk (read-only, text)

- `runs/aether_200m/train.log` — the four August runs (419 KB, 8 `start step=` boundaries).
- `runs/aether_200m_run2/train.log` — the 13–15 Sep run that died at step 1670.
- `runs/aether_200m_run3/train.log` — the post-fix run stopped at step 440.
- `docs/debug_report_2026-09-13.md`, `AGENTS.md` §PAF-8 and §PAF-9 — prior findings.

## Your deliverable (keep it tight)

1. **VERDICT** on your assigned question: CONFIRMED BUG / NOT A BUG / INCONCLUSIVE.
2. **EVIDENCE**: exact numbers from code you actually executed, with `file:line`.
3. **MINIMAL FIX** if you found a bug (exact file + function + the change).
4. **CONFIDENCE** 0–100 % and what would falsify your answer.
