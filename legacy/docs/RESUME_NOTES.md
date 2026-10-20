# RESUME NOTES — read this first when picking the project back up

Last updated: **2026-09-15**. Written so a fresh session (or a human) can resume with no
context loss. Everything here was verified against the live filesystem on that date.

---

## 0. The root cause was found and fixed (15 Sep) — see AGENTS.md PAF-9

Ten parallel investigators analysed the collapse that killed all four full runs. The cause
was **the MoD router reading the un-normalised residual stream**: its gradient scales as
`‖x‖²/T` with nothing bounding it (no input norm, no z-loss, no weight decay, T=0.1 ≈ a step
function, one shared `route_proj` across 5 routing points with contradictory keep ratios).
Measured: router gradient **461 → 4879** as ‖x‖ grew 8 → 511; `route_proj` hot in **100 %** of
822 diagnostic lines and **64 %** of the entire pre-clip grad norm in one run; 31–58 % of
tokens flipped their hard route *every step*.

Disproved along the way (do not re-investigate): PolyMamba SSM divergence (ρ(A) = 0.950000 in
all 12 layers, zero inf/NaN in 861 tensors), mixed precision, the LR schedule (C¹-flat), and
the data pipeline.

Fixes shipped: router input RMSNorm + temperature 1.0 + ST-MoE z-loss; two-sided `dt` clamp;
FSA gathers with the indices it actually selected; `embed_lr_scale: 0.1`. Verified: router
gradient growth over a 64× input sweep **10.6× → 0.91×** (flat). 116 tests pass.

**Checkpoint compatibility:** the router fix adds `mod_router.route_norm_weight`, so pre-15-Sep
checkpoints will not load with `strict=True`.

## 1. What is running right now

A validation run (started 15 Sep) is live, detached, in `runs/aether_200m_run3/`. It exists to
answer one question: **does training survive past step 1300**, where all four previous runs
died. Check it with:

```powershell
Get-Process python* -ErrorAction SilentlyContinue | Select-Object Id, StartTime, CPU
Get-Content runs\aether_200m_run3\train.log -Tail 20
```

Success criterion: `step 1350 loss=... grad_norm=...` with a grad norm still near 1.0. Any
`[spike-guard] skipped update ... reason=grad-norm` in the 1250–1450 range means the fix is
incomplete.

- Output dir: `runs/aether_200m_run3/`
- Checkpoints: `rolling.pt` (fp32, ~2.2 GB) every `save_every_steps` = **100** steps
- Expected throughput: ~3,900–4,200 tok/s at L=1024, micro_batch 2

**If the machine was shut down, the run stopped with it.** Re-launch per §3;
`--resume auto` picks up `rolling.pt` if one exists.

## 2. What was fixed in the 13–14 Sep session

Five independent defects killed every previous full run. Details + evidence in
`docs/debug_report_2026-09-13.md`; the durable summary is AGENTS.md **PAF-8** (which
supersedes the optimizer guidance in PAF-3).

| # | Defect | Evidence | Fix |
|---|--------|----------|-----|
| 1 | `grad_spike_threshold` was dead code | a 19,523-norm gradient was clipped to 1.0 and applied at step ~1700; loss 5.73 → 12.57 (worse than random 11.76) | `_optimizer_step` measures the pre-clip norm via `clip_grad_norm_(params, inf)` and skips above threshold |
| 2 | Guard death-spiral | EMA only updated on applied steps → 335 consecutive skips at `grad_norm=0.000` | EMA tracks during skips, re-seeds after `max_consecutive_skips`, arms a forced clipped update; `wedged_skip_limit` stops a hopeless run loudly |
| 3 | bf16 optimizer state rounded updates to zero | one real step at lr=1.2e-4 changed `A_log` **0.0 %**, RMSNorm **0.0 %**, `mask_logits` **2.3 %**, embedding 87.5 % of elements | **fp32 master weights + bf16 autocast**; now 99.8 % of parameters change per step |
| 4 | Scan was CPU-launch-bound | ~14,000 `as_strided` calls per micro-batch | vector-only doubling scan over the `(N, N)` powers |
| 5 | LM head materialised ~2 GB of logits | blocked fp32 master weights (735 tok/s) | `_chunked_causal_loss` → −0.92 GB → 3,900–3,976 tok/s |

Also: `device="cuda"` silently resolved to the **integrated GPU**, which hard-crashes with
a 0xC0000005 access violation. This broke `scripts/benchmark.py`, all five ablation
scripts, and the whole test suite. `aether/utils/device.py::select_device()` now picks the
discrete RX 7600 and calls `torch.cuda.set_device`, so bare `"cuda"` resolves correctly;
`tests/conftest.py` applies it for the suite.

Three independent verification agents audited the fixes and found five further bugs **in
my own fixes** (cross-document loss pairs, a NaN force-update that corrupted every
parameter, unreachable counters, lying logs, dead code). All fixed with regression tests.

## 3. How to resume

```powershell
cd C:\Users\haziq\Documents\niga

# (only if a run is not already alive)
$env:PATH = "C:\Program Files\AMD\ROCm\7.1\bin;" + $env:PATH
$env:HSA_OVERRIDE_GFX_VERSION="11.0.0"
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:PYTHONUNBUFFERED="1"
$env:TOKENIZERS_PARALLELISM="false"
python -m aether.training.pretrain --config configs/aether_200m.yaml `
    --output-dir runs/aether_200m_run2 --no-tui --resume auto
```

`--resume auto` finds the newest of `rolling.pt` / `interrupted.pt` / `final.pt` in the
output dir. Drop `--resume auto` for a fresh start.

Launch it **detached** so an aborted tool call cannot kill it:

```powershell
Start-Process -FilePath "python" -ArgumentList "-m","aether.training.pretrain",`
  "--config","configs/aether_200m.yaml","--output-dir","runs/aether_200m_run2","--no-tui",`
  "--resume","auto" -WorkingDirectory "C:\Users\haziq\Documents\niga" -WindowStyle Hidden `
  -PassThru -RedirectStandardOutput "runs\aether_200m_run2\stdout.log" `
  -RedirectStandardError "runs\aether_200m_run2\stderr.log"
```

Verify the pipeline before a long run:

```powershell
python -m pytest tests/ -q        # must be 113 passed
python scripts/benchmark.py       # must be >= 8 TFLOPS (measured 26.30)
```

## 4. Compute reality (the binding constraint)

Measured **~4,000 tok/s** at L=1024, micro_batch 2, on the RX 7600.

```
6.55 B tokens (100K steps x 65,536)  /  4,000 tok/s  =  19.4 days continuous
1,000 steps = 65.6 M tokens          =  ~4.5 hours
```

So: **100K steps is ~19 days**, not the 5–6 days the original plan assumed. Any decision
about the token budget has to start from this number. Cheap knobs if more tokens are
needed per day: raise `micro_batch` (memory permitting), and note L=2048 costs ~1.5× per
token vs L=1024.

## 5. Where quality stands / how to tell if it is working

Reference points measured on this exact data + tokenizer (Llama-3 128K):

| quantity | nats/token |
|---|---|
| random init (ln 128256) | 11.76 |
| unigram entropy | 7.71 |
| bigram conditional entropy | 4.45 |
| expected for a ~200M model at ~6.5 B tokens on FineWeb-Edu | **~3.0–3.2** |

Reproduce any of these with the scratchpad script pattern used on 13 Sep (stream
FineWeb-Edu → tokenize → count n-grams → `model(input_ids, labels=ids)["task_loss"]`).

**Do not quote `eval_step_*.json` as a benchmark.** It is an 8-sample loglikelihood probe
(`hellaswag=0.00%` at step 200 is 0/8 — pure noise at loss 9.34). Real numbers need the
trained checkpoint plus lm-eval-harness, and the peer table is already researched:

- SmolLM2-135M (2T tokens): HellaSwag 42.1, ARC-avg 43.9, PIQA 68.4, WinoGrande 51.3, MMLU 31.5
- SmolLM2-360M (4T): HellaSwag 54.5, ARC-avg 53.0, PIQA 71.7, WinoGrande 52.5, MMLU 35.8
- Pythia-160M (300B): HellaSwag 29.9, ARC-e 43.5, ARC-c 18.8, PIQA 62.7, WinoGrande 53.1
- MobileLLM-125M (1T): HellaSwag 38.9, ARC-e 43.9, PIQA 65.3
- Qwen2.5-0.5B (18T): HellaSwag 51.2, PIQA 69.9

Honest framing: 6.55 B tokens is 100–600× less than every model in that table. The
defensible claim is "matches SmolLM2-135M-class quality at a fraction of the tokens", not
"beats SmolLM2-360M".

## 6. Open items (nothing is half-finished in code)

1. **First real benchmark** once a checkpoint has enough tokens (≥ ~1 B is the earliest
   worth reporting; ~3–6 B for the intended comparison). Path: `aether/utils/export_hf.py`
   → `aether/eval/harness.py::run_lm_eval`.
2. **Edge/inference metrics** — no numbers exist yet. Targets in
   `docs/benchmark_targets.md` §Stretch: ≥50 tok/s decode on the RX 7600, <4 GB at L=4096,
   TTFT/inter-token latency. The PolyMamba state is only 16 floats/layer/order, which is
   genuinely edge-friendly; FSA's DCT + gather path is the one to measure.
3. **A real baseline.** `ablations/00_baseline.py` trains a vanilla transformer on
   **random tokens** (stability only, `--proxy`). A same-tokenizer same-data baseline run
   is still missing, and it is the only way to claim the novel components help.
4. **SFT / DPO** — code exists (`aether/post_train/`), never executed successfully.
5. Everything above is blocked on **tokens**, i.e. on wall-clock GPU time.

## 7. Gotchas that will bite again

- **Never run a blocking `Start-Sleep`/long foreground command while a background training
  job is alive** — aborting that call killed the training process tree once (13 Sep 17:07).
  Launch detached (`Start-Process`) and poll with short commands.
- Device 0 is the **integrated GPU** and hard-crashes ROCm. Always go through
  `select_device()` / `CUDA_VISIBLE_DEVICES=1`.
- Do not run GPU work (pytest, profilers, benchmarks) while training — it starves the run
  and the tokens/s collapse (600 ↔ 4,000).
- `runs/aether_200m_dead_v1/` holds the destroyed pre-fix run (its `final.pt` scores
  12.56 nats). Keep it for the post-mortem, never resume from it.
- Checkpoints are fp32 now: `rolling.pt` ≈ 2.2 GB, not 1.1 GB.
