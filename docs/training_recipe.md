# Training Recipe

Updated 31 July 2026 (PAF-5). Matches `configs/aether_200m.yaml` and the fixed trainer.

## Pretrain

### Data
- **Primary:** FineWeb-Edu `sample-10BT` (streaming).
- **Tokenizer:** Llama-3 BPE 128K vocab (`unsloth/Llama-3.2-1B-Instruct`, un-gated mirror ? PAF-4).
- **Sequence packing:** yes (docs separated by EOS, sliced into fixed L, zero/pad tail).
- **Background producer thread** streams raw doc texts; the training thread only tokenizes
  from the buffer, so the GPU is not starved.

### Curriculum

| Steps (fraction) | Seq len |
|------------------|---------|
| 0-20%            | 1024 |
| 20-100%          | 2048 |

`grad_accum` is recomputed every step from the current seq_len so the effective batch stays
constant at 65,536 tokens. (L=4096 is a stretch goal only after measuring memory at L=2048.)

### Optimizer
- **Algorithm:** `torch.optim.AdamW(fused=True)` (no bitsandbytes ? PAF-3: no ROCm Windows
  binary; bnb import crashes on this machine).
- **b1, b2:** 0.9, 0.95. **Weight decay:** 0.1 on nn.Linear/Embedding weights **only**;
  no decay on norms, biases, `A_log`, `mask_logits`, `mix_gate`, `mixer.gate`, `route_proj`,
  and embeddings.
- **Gradient clipping:** 1.0 global norm.
- **LR schedule:** 1000-step linear warmup 0 -> 1.2e-4, then cosine to 1.2e-5 (min_lr_ratio 0.1).
  Peak 1.2e-4 is deliberately below the reproducible LR-instability onset (~1.5e-4) that
  broke three full runs; see `docs/debug_report_2026-07-31.md` §B26.

### Batch
- **Micro-batch:** 2 x L (8,192 tokens at L=1024; 4,096 at L=2048).
- **Grad accumulation:** `effective_batch_tokens / (micro_batch * seq_len)` (32 at L=1024,
  16 at L=2048).
- **Effective batch:** 65,536 tokens constant.
- **Total:** 100,000 steps x 65,536 = 6.55B tokens (~5-6 days at the measured 29.48 TFLOPS).

### Precision
- bf16 params, **bf16 Adam moments** (empirically verified 9 Aug 2026 — torch creates state
  in the param dtype for `fused=True` + bf16 params, NOT fp32), **no master weights**.
  ~0.75 GB optim state. Forcing fp32 state would add ~0.75 GB and OOM at micro_batch 2.
- fp32 for: softmax, MoD routing scores, SSM scan, A construction, DCT, RMSNorm stats.

### PolyMamba-2 stability
- **dt init:** log-uniform over [0.005, 0.1] (Mamba-style range), NOT a single 0.001.
- **Radius clamp (REQUIRED):** in the forward, `dt = softplus(A_log).clamp_min(dt_floor)` with
  `dt_floor[k] = (1-max_radius)/min_eig(M[k])`, `max_radius=0.95`. This caps ρ ≤ 0.95 so the
  SSM *backward* gradient amplification `1/(1−ρ)` stays ≈ 20× (was ~1100× at ρ=0.9991, which
  permanently exploded grad norms in every full run — see B27). Do NOT let ρ drift toward 1.
- FSA provides the long-range context (cross-chunk attention over past chunk summaries),
  so the SSM does not need near-integrator memory.

### Auxiliary losses
- **L_ortho (FSA):** weight 0.01, ramped 0 -> target over first 10%.
- **L_kbal (PolyMamba-2):** weight 0.01, ramped 0 -> target over first 5%.
- **L_mbd (MoD):** weight 0.01, ramped 0 -> target over first 5%.
All three are computed inside the block forward and returned as block outputs (checkpoint-safe).

### FSA stability
- The chunk **mean is centered out before the DCT** (cross-chunk path only). This removes the
  DC bin (= sqrt(chunk_size) * chunk-mean ≈ 5.7x) whose magnitude inflated cross-attention
  logits/outputs and drove the `mask_logits` gradient explosion. No division is used, so the
  near-zero high-frequency DCT coefficients do NOT amplify gradients (L2-normalizing them did
  — reverted). The intra-chunk path keeps the raw (uncentered) k/v for strict within-chunk
  causality.

### Component knobs (wired every step)
- FSA soft top-k temperature: 1.0 -> 0.1 over first 10%.
- FSA Gumbel noise: 1.0 -> 0 over first 10% (exploration).
- MoD router temperature: constant 0.1.

### Checkpointing
- Gradient checkpointing at block level (`use_reentrant=False`), mandatory.
- Save every 5,000 steps; `final.pt` at the end; `latest.pt`/`latest.txt` pointer.
- Checkpoint metadata includes `tokens_seen` and `model_config` (so SFT/DPO/eval can rebuild
  the exact architecture).

### Eval during pretrain
- Every 500 steps: 8 HellaSwag + 8 ARC-Easy loglikelihood probes + 8 generations (cheap trend
  signal only; not a benchmark).
- Final: full lm-eval-harness via `scripts/eval.sh` (auto-exports the `.pt` to HF format).

## SFT
- **Data:** `HuggingFaceTB/smol-smoltalk`, ~1B tokens, Llama-3 chat template.
- **Optimizer:** `adamw_torch_fused`, LR 1e-5, cosine, 2 epochs.
- **Collator:** `DataCollatorForLanguageModeling` (padding -> -100 labels).
- Throughput: ~13 h on RX 7600.

## DPO
- **Data:** `HuggingFaceH4/ultrafeedback_binarized` (`train_prefs`), beta 0.1.
- **Optimizer:** `adamw_torch_fused`, LR 5e-7, 1 epoch.
- `ref_model=None` lets TRL build the reference internally.
- Throughput: ~4 h.

## Final eval

`scripts/eval.sh` runs `aether.eval.harness.run_lm_eval`, which auto-exports the `.pt`
checkpoint to a trust_remote_code HF dir (`aether/utils/export_hf.py`) and loads it with
`HFLM(trust_remote_code=True)`. Targets in `docs/benchmark_targets.md`.

## Failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Loss spikes / NaN early | PolyMamba A unstable | Check `pm_max_spectral_radius` (TUI); should stay < 1.0 |
| Loss stuck near log(vocab) | future leak regression | Run `tests/test_model_math.py::test_aether_full_model_causality` |
| Loss stuck near half log(vocab) | MoD zero-poisoning regression | Run `tests/test_model.py::test_mod_unselected_tokens_bypass_block_span` |
| MoD always picks the same positions | positional collapse | watch `mod route ratios` in the TUI; increase `mod.loss_weight` |
| OOM at L=2048 | activations | reduce `micro_batch` to 2; re-run |
