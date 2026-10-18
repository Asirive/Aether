# aether/training/ — Training Pipeline Context

## Data Flow

```
HF datasets → tokenize (Llama-3 BPE) → pack into fixed L sequences → shuffle → batch
```

## Streaming vs Pre-tokenization

For 30B tokens at 200M params, **streaming** is preferred:
- No need to store 30B tokens on disk
- PyArrow handles the read overhead
- `datasets.load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)`

**Sequence packing:**
- Concatenate docs separated by EOS tokens
- Slice into fixed L sequences
- Truncate last partial sequence
- This achieves ~95%+ token utilization vs 70-80% with per-doc padding

## Curriculum Schedule

| Phase | Steps (fraction) | Seq Len | Notes |
|-------|------------------|---------|-------|
| Warmup | 0-20% | 1024 | Saves compute; CC-FSA cross path inactive (first chunk only) |
| Standard | 20-100% | 2048 | Default; CC-FSA cross path active |

L=4096 is a stretch goal only after measuring memory at L=2048. `grad_accum` is recomputed
every step so the effective batch stays 65,536 tokens across phases.

The DCT cache (`dct_matrix(chunk_size, ...)`) is keyed by `(C, device, dtype)`; C=32 is fixed,
so no rebuild is needed across curriculum phases.

## Optimizer Constraints

- **`torch.optim.AdamW(fused=True)`** with bf16 params + fp32 moments, **no master weights**
  (PAF-3: bitsandbytes has no ROCm Windows wheel and crashes on import here).
- Total optim state ~1.6 GB for 200M params; fits the 8GB RX 7600 budget.
- Weight decay 0.1 on nn.Linear/Embedding weights **only**; no decay on norms, biases, `A_log`,
  `mask_logits`, `mix_gate`, `mixer.gate`, `route_proj`, and embeddings.
- `grad_accum` is recomputed every step from the current curriculum seq_len so the effective
  batch is constant at 65,536 tokens across phases (fixed 31 Jul 2026: was frozen at init,
  doubling the effective batch at L=2048).

## LR Schedule

- Linear warmup 0 -> 3e-4 over 1000 steps.
- Cosine 3e-4 -> 3e-5 over remaining steps (min_lr_ratio 0.1).
- Global grad clip 1.0; spike-guard skips updates when loss > 1.5x EMA or grad_norm
  exceeds `optim.grad_spike_threshold` (100.0 default).

## Label Convention (CRITICAL — do not break)

The data loader returns **`labels = input_ids.clone()`** (HF convention). The *only*
shift lives in `model.forward`:
```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```
**Do NOT pre-shift labels in any data loader.** `labels = c[1:]` in the loader combined
with the model shift trains P(t+2 | prefix) — the 2-ahead objective that caused the
8 Aug 2026 "loss plateau at ~7.5 + 'the of, the of' canary" collapse (PAF-6, B23).
Guarded by `test_fineweb_stream_labels_match_hf_convention` and
`tests/test_label_alignment.py` (successor-ring probe: 1-ahead must reach 100%).

## Gradient Checkpointing

Block-level, `use_reentrant=False`. Blocks return `(x, kbal, ortho)` tuples so aux losses
are valid checkpoint outputs. ~3-4x memory savings, ~30% extra compute. Mandatory at 200M.
## Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM at L=4096 batch 8 | Activations too large | Reduce micro-batch to 4 |
| Loss spikes early | PolyMamba A unstable | Check spectral_radius metric |
| Eval loss diverges from train | Eval uses different seq_len | Verify eval matches curriculum |
| Slow token throughput | Data loader bottleneck | Check num_workers; use pinned memory |
| Loss stalls at ~7.5 + canary alternates "the of the of" | Label double-shift (2-ahead objective) | Revert loader to labels == input_ids; run test_label_alignment |
