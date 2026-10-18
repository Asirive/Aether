# Debug & Fix Report ? 31 July 2026

Full-system audit of AETHER-200M (architecture, math, implementation, training pipeline,
post-training, eval, scripts, docs). Every finding was verified by reading the code and, where
possible, by direct empirical probes on the machine (RX 7600, torch 2.14.0a0+rocm).

**Pre-fix test state:** 75/77 passing (2 data-stream failures). Suite runtime 15m 25s (the 2
failing tests each blocked on a 60 s queue timeout many times).
**Post-fix test state:** 97/97 passing in 21 s. GPU smoke run: 186,615,276 params, 5 steps,
checkpoint saved, no NaN.

---

## P0 ? Training-breaking bugs (all empirically confirmed, all fixed)

### B1. FSA was not causal ? future-token leakage (ARCHITECTURE FLAW)
- **Where:** old `aether/components/fsa.py` (DCT along the full sequence axis + full bidirectional
  attention, no causal mask anywhere).
- **Why fatal:** the DCT mixes every position (past and future) into every frequency bin, so each
  output position contained future-token information. In a next-token LM the model learns to copy
  the future: training loss collapses, autoregressive eval is garbage.
- **Probe:** changing tokens 16..31 changed logits at positions 0..15 by 1.281.
- **Fix:** redesigned FSA as **Causal Chunk-Compressed FSA (CC-FSA)**. K/V are split into chunks of
  `chunk_size=32`; each chunk gets a DCT-II within the chunk (local ? no cross-position leak beyond
  the chunk); each head keeps a learned top-k set of frequency bins (`keep_bins = C/4 = 8`) as the
  chunk summary. Queries attend to (a) compressed summaries of chunks lying *fully* in the past
  (strict chunk-level causality) and (b) raw time-domain keys inside their own chunk with a
  standard causal mask. A per-head scalar gate (`mix_gate`) blends the two paths. The per-head
  rotation matrices (604 MB at L=4096) are gone; head symmetry-breaking now comes from random
  mask-logit init + orthogonality loss.
- **Side effect:** the docs claimed ~4x attention compute reduction, which was false for the old
  code (full LxL matmul with zeroed bins). CC-FSA makes it real: cross-chunk attention runs over
  nC*k_c = L/4 keys, intra-chunk over C=32 keys.
- **Regression tests:** `test_fsa_causality_no_future_leak`,
  `test_fsa_causality_mid_sequence_perturbation`, `test_aether_full_model_causality`.

### B2. MoD router zero-poisoned unselected tokens
- **Where:** old `mod_router.py` returned `x * mask`; `model.py` fed the masked x through the
  remaining blocks.
- **Why fatal:** the STE mask is binary in forward, so unselected tokens became exactly 0. Every
  op is bias-free and RMSNorm(0)=0, so they stayed 0 through all later blocks, producing zero
  logits and a permanent `log(vocab)=11.76` CE loss on those positions, with zero gradient to
  everything before the router.
- **Fix:** router returns `x` **unchanged** plus a per-token gate (B, L, 1). `model.py` applies
  the gate residually around each routed block span: `x = gate * block(x) + (1 - gate) * x`
  (span = until the next routing point). Unselected tokens get exact identity passthrough.
- **Regression tests:** `test_mod_unselected_tokens_bypass_block_span`,
  `test_mod_router_residual_gate_math`.
### B3. PolyMamba balance loss was identically zero
- **Where:** old `PolyMamba2.balance_loss()` computed the gate on a dummy **zero vector**;
  softmax(0) is uniform, so loss = relu(log K - log K - 0.3) = 0 forever.
- **Probe:** loss = 0.00000000 even with a force-collapsed gate.
- **Fix:** the loss is computed inside `forward` from the real per-token gates: entropy of the
  mean gate distribution over (batch, seq), `relu(log K - H(p) - 0.3)`, kept in the autograd
  graph (returned out of the checkpointed block as a block output ? see B6).
- **Note:** it measures *global* order usage (mean over tokens), the correct collapse notion;
  per-token routing diversity is intended behavior.

### B4. MoD balance loss was identically zero
- **Where:** old router computed a KL over `selected_count/k`; top-k selects exactly k by
  construction, so the ratio was always 1 and the loss 0.
- **Fix:** loss is now MSE between the mean **soft** selection probability per position and the
  target keep ratio (uniform positional load, gradient flows through the soft mask).

### B5. Gradient-accumulation count was frozen at init
- **Where:** `trainer.py` computed `grad_accum` once from the first curriculum phase (L=1024).
  At L=2048 the effective batch silently doubled to 131,072 tokens/step.
- **Fix:** `compute_grad_accum()` helper, recomputed every step from the current curriculum
  seq_len. Effective batch is now constant at 65,536 tokens/step across all phases.

### B6. Aux-loss annealing was display-only + checkpoint-escape risk
- **Where:** trainer computed `fsa_temperature` only for the TUI; FSA/router hardcoded T=0.1;
  `gumbel_noise` never used. Aux losses were gathered *after* forward by calling mixer loss
  methods outside the gradient-checkpointed blocks ? once those losses carry real gradients,
  tensors stored on `self` escape the checkpoint graph.
- **Fix:** `model.set_component_knobs(fsa_temperature, fsa_gumbel)` is called every step;
  `fsa_gumbel_noise()` schedule added (1.0 -> 0 over warmup). Blocks now return
  `(x, kbal, ortho)` tuples so aux losses are proper checkpoint outputs with intact graphs.

### B7. pyproject depended on bitsandbytes (contradicts PAF-3)
- bnb is installed on this machine and **crashes on import** (no ROCm Windows binary). Removed
  from dependencies; added missing `rich` (the TUI imports it but it was undeclared).

### B8/B9. post_train scripts could not run
- `sft.py`: `optim="adamw_8bit"` (requires bnb), `num_proc=4` on Windows (spawn crash), no data
  collator. Fixed: `adamw_torch_fused`, `num_proc=0` on win32, `DataCollatorForLanguageModeling`.
- `dpo.py`: missing `train_dataset` entirely (TypeError on construction), `adamw_8bit`.
  Fixed: loads `ultrafeedback_binarized` (`train_prefs`), `adamw_torch_fused`,
  `processing_class`/`tokenizer` fallback for TRL version differences.

### B10. sft.sh/dpo.sh loaded the wrong architecture from the wrong path
- They built `AetherConfig()` with defaults (vocab 32k, d=768, 12 layers) vs the real
  128256/576/24, and looked for `checkpoints/aether-200m/pretrain/final/model.pt`, which the
  trainer never writes.
- **Fix:** trainer now stores the full model config in checkpoint metadata
  (`metadata.model_config`); the scripts rebuild the model from it. Default path:
  `runs/aether_200m/final.pt`.

### B11. lm-eval could not load checkpoints
- `HFLM(pretrained=...)` needs HF format (config.json + safetensors); trainer saves raw `.pt`.
- **Fix:** new `aether/utils/export_hf.py` writes a trust_remote_code HF package
  (config/configuration/modeling + safetensors + tokenizer); `harness.run_lm_eval` auto-exports
  from any `.pt` containing `metadata.model_config`.

### B12. Missing/broken script references
- `scripts/benchmark.py` did not exist (referenced by README and env_setup_windows.ps1).
- README pointed at `scripts/pretrain.ps1` (real file: `pretrain_200m.ps1`).
- `scripts/pretrain.sh` ran the random-token ablation `04_aether_full.py`, not pretraining.
- All fixed/created.
---

## P1 ? Stability / quality fixes

- **B13 Weight decay on stability-critical params:** `A_log`, `mask_logits`, `mix_gate`,
  `mixer.gate`, `route_proj`, embeddings are now in the no-decay group (decaying A_log pushes
  dt toward softplus(0)=0.693 and changes SSM dynamics; decaying mask_logits fights the masks).
- **B14 FSA rotation memory (obsolete):** removed with CC-FSA (was 151 MB GPU copies per call
  at L=2048, x4 per FSA layer).
- **B15 Spectral radius device:** now computed in fp32 on CPU (ROCm `eigvals` on GPU/bf16 is
  unreliable; the health metric previously failed silently under try/except).
- **B16 Token budget honesty:** 200k steps x 65,536 = 13.1B tokens, not 30B. At the measured
  29.48 TFLOPS with grad-checkpoint recompute this is ~11 days ? over the 1-week budget once
  SFT/DPO/eval are included. Config now uses **100k steps = 6.55B tokens (~5-6 days)**.
  AGENTS.md updated accordingly.
- **B17 `_loglikelihood_choice` BPE boundary:** ctx and choice are now encoded separately (the
  joint encode could merge across the boundary and corrupt `choice_len`); removed the ARC
  double space.
- **B18 data.py off-by-one:** the short-chunk fallback deleted `seq_len` tokens while consuming
  `seq_len+1`, and could produce ragged batches when the stream was exhausted; now deletes
  `chunk_len` and pads with EOS.
- **B19 `aether_25m_proxy.yaml`** was missing entire config sections (eval/logging/tui/output)
  and crashed `PretrainConfig`; now complete and trainer-compatible.
- **B20 tokens_per_sec** now uses full optimizer-step wall time (was fwd+bwd only).
- **B21** `latest.txt` pointer fallback when Windows symlink creation fails.
- **B22 test_data_stream fake** predated the producer-thread design (returned empty batches
  after 60 s timeouts) ? fixed; suite 15m25s -> 21s.

## Accepted risks (documented, not bugs)

- **bf16 params without master weights (PAF-3):** updates smaller than the bf16 ulp are lost.
  Mitigation: LR floor 5e-5, cosine schedule, no fp16 anywhere.
- **FSA intra-chunk attention is dense within C=32:** by design (local exactness); the
  cross-chunk path provides global mixing at L/4 key width.
- **Resume does not restore data-stream position or RNG:** acceptable for streaming pretraining.
- **Periodic eval uses 8+8 built-in dev samples:** fast trend signal only; final eval uses
  lm-eval-harness proper.
- **Ablation scripts (00-04) train on random tokens:** they validate stability only, not
  component quality. Real component ablations must use the actual FineWeb-Edu stream at 25M
  proxy scale via `configs/aether_25m_proxy.yaml`.
## B23 — 9 Aug 2026: label double-shift turned pretraining into a 2-ahead objective

**Severity: CRITICAL. Root cause of the "loss plateau at ~7.5 + degenerate canary
'the of, the of'" collapse seen in every full-size run.**

**The bug.** Two components used different label conventions:
- `data.py` pre-shifted: `labels = c[1:]` (labels[t] = token t+1, i.e. the loader already
  built "next token" targets).
- `model.py` forward shifted again: `shift_logits = logits[..., :-1, :]`,
  `shift_labels = labels[..., 1:]`.

Combined, the training objective became P(t+2 | prefix) — the model was trained to predict
**the token 2 positions ahead**. Every unit test used the HF convention
(`labels = input_ids.clone()`), which only exercises the model's own shift, so the mismatch
(which exists only on the data-loader path in real training) silently passed the 97-test
suite. The trainer was never exercised end-to-end with the real stream until full runs.

**Empirical signature (all explained by this one bug):**
- loss drops to ~7.5-7.7 (near unigram entropy) then stalls — conditional learning cannot
  engage when alignments are wrong
- canary cycles "the of, the of, the of" — that is literally the 2-step transition matrix
  of English (P(t+2 | t) alternates function words)
- eval 0-12% — eval measures 1-ahead, the model was trained 2-ahead
- grad norms climbing into the millions — the misaligned objective is noisy/ambiguous

**The fix.** `data.py` now returns `labels = input_ids.clone()` (HF convention); the shift
lives only in `model.forward`. One-line change + a permanent regression test:
`tests/test_label_alignment.py` trains a tiny Aether on a deterministic successor ring
(i -> i+1) and requires 1-ahead accuracy > 0.9 and 2-ahead accuracy < 0.3. With the bug,
1-ahead accuracy was ~0%. With the fix: 1-ahead = 100%, 2-ahead = 0%.

**Also fixed in the same pass (evidence-based, secondary):**
- **FSA ortho-loss weight 0.001 -> 0.01.** The ortho loss (on raw mask_logits, init std
  1.0) was climbing from ~0.27 to ~28.75 as heads converged on the same high-energy
  frequency bins; at weight 0.001 the steering force was negligible (~0.029 nats).
  0.01 gives 10x stronger diversity pressure. Applied to `aether_200m.yaml` and
  `aether_25m_proxy.yaml`.
- Added `test_fineweb_stream_labels_match_hf_convention` — locks the loader convention so
  the bug cannot silently return.

**What was NOT the cause (other-practitioner claims verified against code, all false):**
- PolyMamba A-matrix explosion — A is Hurwitz-stable by construction
  (A = I - softplus(dt)·M, M normalized); monitored spectral radius 0.9999 < 1 at every step.
- MoD "winner-take-all" at layer 20 (kept=1.00) — that is the configured target ratio
  (0.50/0.70/0.85/1.00/1.00) and monitored kept ratios match targets exactly.
- Missing pre-LN — blocks are already pre-LN
  (`mix = mixer(norm1(x)); x = x + mix; x = x + ffn(norm2(x))`).
- LR too aggressive — the 3e-4 peak was kept. The spikes at LR ~2e-4+ were a symptom of
  the misaligned objective (noisy gradients), not a separate cause.

**Validation after fix:** 99/99 tests pass (~37 s). Full-200M real-data run restarted;
loss should now keep descending below the old ~7.4 floor and grad norms should stay
O(1)-O(10), not O(10^6).
## B24 — 9 Aug 2026: CUDA OOM at ~step 25 (fragmentation + 2.1 GB fp32-logits transient)

**Symptom.** Training died silently around step 20-25 with
`torch.OutOfMemoryError: CUDA out of memory` (tried to allocate 1.96 GiB, GPU had 0 bytes
free). The full 65K-token runs survived because their allocator hadn't fragmented to
0-bytes-free at the moment of the canary; a fresh `--no-tui` run fragmented the same pool
and OOM'd.

**Root cause (two compounding factors).**
1. The 200M config sits only ~1.4 GB under the 8 GB limit (peak active 6.6 GB + weights +
   bf16 Adam state + grads ≈ 8.4-8.6 GB reserved). The canary generation's transient
   allocations tip it over when the caching allocator has no free blocks.
2. `_cross_entropy_chunked` materialized the full shifted logits in fp32 per chunk
   (`chunk_logits.float()`), a ~2.1 GB transient for a 4096-token micro-batch.

**Fixes (all small, safe, evidence-based).**
- `scripts/pretrain_200m.ps1`: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the
  allocator flag recommended by the OOM traceback itself).
- `trainer.py`: `torch.cuda.empty_cache()` before canary generation and before periodic eval.
- `model.py`: `_cross_entropy_chunked` chunk_tokens 1024 -> 512 (halves the fp32 temp).
- `configs/aether_200m.yaml`: micro_batch 4 -> 2 (halves activation peak; ~same throughput,
  grad_accum auto-recomputes 16 -> 32). This is the real safety margin.

**Verification.** 120-step real-data run: no OOM, loss 12.4 -> 10.61, grad norms ~2 the
whole way, checkpoint saved. Prior to the fix the equivalent run OOM'd at step ~25.

## B25 — 9 Aug 2026: theoretical audit (probes) — all pass

Systematic math/implementation audit of every component, verified with direct probes
(see the session log in the compacted context):
- **SSM parallel scan == naive sequential recurrence** for seq_len in {1,2,3,7,32,33,64,128}
  and state_dim in {1,16}: max error 8.8e-5 (fp32 rounding). PASS.
- **Gradient accumulation is mathematically exact**: 2 micro-batches loss/2 == 1 full batch,
  max weight diff 2.2e-7. PASS.
- **Label alignment**: successor-ring probe, 1-ahead accuracy 100%, 2-ahead 0%. PASS.
- **DCT-II** orthonormal + invertible (covered by test_dct_math). PASS.
- **PolyMamba A** Hurwitz-stable: monitored spectral radius 0.9999 < 1. PASS.
- **Optimizer state dtype**: empirically bf16 (not fp32 as PAF-3 originally claimed) —
  corrected the docs. Accepted tradeoff for 8 GB budget; do not change without re-checking
  memory (fp32 state would add ~0.75 GB and OOM at micro_batch 2).

**Test suite: 99/99 pass (~44 s). Smoke test: 5 steps clean, checkpoint saved.**
All 54 Python files + 4 YAML configs parse. No lingering GPU processes.
## B26 — 9 Aug 2026: reproducible LR-instability at ~1.5e-4 (the "breaks at step ~550" bug)

**Symptom.** Even after the label fix (B23) produced healthy learning through step 500
(loss 6.57, semi-coherent canary like "a new. The 2000s, the"), every run still broke at
~step 550-600: grad norms spiked (11 -> 19 -> 35 -> 94 -> 116 -> 302), the spike-guard
skipped the worst steps, but loss still rose (6.57 -> 7.11 -> 7.23) and the canary
degenerated ("a 20110000000000000000", "and the other other and the other"). This happened
in **all three full runs regardless of LR peak (5e-4, 3e-4) or label objective** — the
onset always coincided with LR crossing ~1.5-1.8e-4.

**Root cause (two independent structural gradient amplifiers).**
1. **PolyMamba A was a near-integrator.** dt init was a single 0.001 for every channel, so
   A = I - dt*M had spectral radius 0.9999 — the state never decayed within a 1024-token
   sequence (memory ~10,000 steps). The recurrence amplified gradients ~1/(1-0.999) = 1000x.
2. **FSA head-collapse inflated magnitudes.** ortho_loss climbed to ~24.4 (heads converged
   to the same frequency bins), and the DC bin of a chunk = sqrt(32) * chunk-mean ≈ 5.7x,
   inflating cross-attention logits/outputs.

**Fixes (all verified).**
- **PolyMamba dt init:** log-uniform over [0.005, 0.1] (Mamba-style range). most channels
  now decay fast; measured spectral radius max 0.99882 < 1 (was 0.9999). FSA provides the
  global context, so the SSM doesn't need near-integrator memory.
- **FSA:** center the chunk mean out before the DCT (cross-chunk path only) — removes the
  DC bin magnitude spike without any division (L2-normalizing the compressed k/v was tried
  first but amplified near-zero high-frequency-coefficient gradients; reverted).
- **LR:** peak 3e-4 -> **1.2e-4** (below the ~1.5e-4 onset). Slower but stable.
- **Diagnostic:** spike-guard now logs top-5 parameter grads by norm when grad_norm > 20,
  so if instability recurs the exact module is identifiable.

**Validation strategy:** 900-step real-data run (`--config configs/validate_stab.yaml`)
extending past the step-550 danger zone, monitored for loss continuing to descend and grad
norms staying O(1)-O(10). Tests: 99/99 pass; component stability probe passes (spectral
radius 0.99882, FSA output bounded).
---

## B27 — PolyMamba SSM backward-gradient amplification (THE recurring-explosion root cause)

**Symptom:** every full run learned well (loss 12 → 6-7, canary improving) for a few hundred
steps, then grad_norm permanently exploded to 200-400 and the model froze. The hotspot grads
were consistently `blocks.{0,2}.mixer.in_proj` (early PolyMamba blocks) + `mod_router.route_proj`.
The TUI logged `PolyMamba max spectral radius: 0.9991`.

**Root cause (forward stable, backward unstable):** in the SSM `h_t = A h_{t-1} + u_t`, the
gradient of the loss at position `j` w.r.t. the input at position `t` is `(Aᵀ)^(j-t)`. Summed
over all future positions, the amplification is

```
dL/du_t ≈ (1 / (1 − ρ)) · ||dL/dh||      where ρ = spectral radius
```

With ρ = 0.9991 → `1/(1−0.9991) ≈ 1100×` backward amplification. The forward pass stays
numerically fine (loss finite, output not NaN — the model "looks" healthy), but the associatve
scan's backward pass amplifies ~1000× the loss gradient once the model drives `A` toward the
identity to gain long memory. The order-1 tridiagonal channel has a small min eigenvalue
(~0.18 after the `/1.10` normalization), so even `dt=0.005` gives `1 − 0.005·0.18 ≈ 0.9991`.

**Why it was permanent:** the spike-guard skipped any step with `grad_norm > threshold`. Once
the amplified gradients locked in, **every** step was skipped (`optimizer.step()` never ran) —
the model was frozen forever and could not recover. This is a *mechanism* bug in addition to
the *scale* bug.

**Fix (both):**
1. **Bound the spectral radius** (root cause). `PolyMamba2` now takes `max_radius=0.95` and
   clamps `dt = softplus(A_log).clamp_min(dt_floor)` where
   `dt_floor[k] = (1 − max_radius) / min_eig(structure_M[k])`. This makes the **largest**
   eigenvalue of `A` satisfy `≤ max_radius` by construction, so the backward amplification is
   `≤ 1/(1−0.95) = 20×` instead of ~1100×. The SSM no longer needs long memory — FSA supplies
   global context. Verified: forcing `A_log = −10` (near-integrator) still yields radius 0.95.
2. **Don't freeze on large-but-finite grads** (mechanism). The spike-guard now skips only on
   non-finite grads or loss spikes; large-but-finite grads are handled by `grad_clip` (scale,
   direction preserved) and the optimizer still steps. The model can no longer get permanently
   stuck.

**Validation:** `test_polymamba.py` + `test_polymamba_math.py` 15/15 pass; a standalone probe
confirmed radius is clamped to 0.95 even with `A_log` forced to near-integrator values.
`test_model_math.py` 11/11 pass (the earlier full-suite "crash" mid-`test_model_math` was a GPU
OOM from running the whole suite in one process — it passes alone).

**Action required:** old checkpoints (`final.pt`, `rolling.pt`, `interrupted.pt`) were trained
with `dt` already driven into the ρ≈0.999 regime. They are **dead** — do not resume them.
Retrain from scratch with the clamped `max_radius` so `dt` can never reach the unstable regime.
