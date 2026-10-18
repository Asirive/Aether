# aether/ops/ — Primitives Context

## DCT Module

Cached DCT-II matrix applied via matmul. NOT a custom CUDA kernel — pure PyTorch matmul with cached matrix.

**Why matmul, not FFT?**
- DCT-II is its own operation; we want full control over the basis (no FFT phase conventions to worry about)
- Cached `W ∈ R^{L×L}` is dense but applied once per forward pass; matmul is well-optimized
- IDCT is `W^T @ x`, also a single matmul

**Memory:** 4 KB per cache entry at L=2048 (`2048² × 4 bytes`). Trivial.

**Precision:** compute in fp32, cast result to bf16 for the matmul downstream.

## Soft Top-k Module

Differentiable top-k mask using perturbed softmax with straight-through gradient. Now also
returns the **soft mask** when `return_soft=True` (used by the MoD router for the load-balance
loss and by FSA for the bin weights).

**Algorithm:**
1. Optional Gumbel noise (wired every step in training via `model.set_component_knobs`;
   annealed 1.0 -> 0 over the FSA warmup). Disabled in eval.
2. `kthvalue` gives the k-th largest threshold; `hard_mask = perturbed >= kth`.
3. `soft_mask = sigmoid((logits - kth) / max(temperature, 1e-3))` — always computed (the old
   `temperature >= 1.0` early-return skipped the soft gradient path, defeating warmup).
4. STE: `ste_mask = hard_mask + (soft_mask - soft_mask.detach())` — forward value is hard,
   gradient flows through `soft_mask`.

**Why STE + Gumbel:** hard top-k is non-differentiable; STE routes gradient to near-threshold
logits; Gumbel noise explores alternative selections early in training. After warmup the
Gumbel weight -> 0 and selection is deterministic.

## Cache Invalidation

The DCT matrix cache (`lru_cache`) is keyed by `(seq_len, device, dtype)`. When the model moves
device the cache rebuilds on the new device. FSA's cross-chunk and intra-chunk masks are keyed
by `(padded_len, str(device))` / `(n_chunks, pad, str(device))` and are plain Python dicts
(not LRU) — they persist for the lifetime of the module, which is fine (small, deterministic).

## Unit tests required

- DCT/IDCT invertibility, orthonormality, device cache, gradient flow
- soft top-k: exact k ones at low temperature, soft->hard anneal, gradient flow, return_soft
