# aether/components/ ? Component-Specific Gotchas

Updated 31 July 2026 (PAF-5). Read `docs/debug_report_2026-07-31.md` for context.

## CC-FSA (Causal Chunk-Compressed Freq-Sparse Attention)

### Init Requirements

- **mask_logits** `N(0, 0.02^2)` over `(n_heads, chunk_size)` ? random tie-break gives per-head
  bin-set diversity at init (the old cold-start collapse came from identical zero logits).
- **mix_gate** init zeros so the initial blend is 0.5 cross / 0.5 intra.
- The old per-head orthogonal pre-rotation is GONE. It was 151 MB of GPU copies per forward call
  at L=2048 and did not solve causality (the leak was the DCT itself).

### Forward pass rules

1. K/V are split into chunks of `chunk_size=32`; tail zero-padded to a multiple of C.
2. Per-chunk DCT-II computed in fp32; bins cast to bf16 for matmuls.
3. Per-head learned top-k (`keep_bins = C * topk_ratio = 8`) gathers the kept bins (real compute
   reduction, not zero-masked). STE on the soft top-k gives gradient to `mask_logits`.
4. **Cross-chunk path:** strict causality ? position t attends only to bins of chunks with
   `(j+1)*C <= t` (chunk fully in the past). Masked-renorm softmax (rows with no visible
   chunks produce exactly zero). First chunk -> no cross context by design.
5. **Intra-chunk path:** standard causal attention inside the current chunk (tril mask, pad-aware
   for the last partial chunk).
6. Blend: `out = sigmoid(mix_gate) * out_cross + (1 - sigmoid(mix_gate)) * out_intra`.

### Causality is a HARD requirement

Regression test: `tests/test_fsa.py::test_fsa_causality_no_future_leak` and
`tests/test_model_math.py::test_aether_full_model_causality`. If either fails, FSA is leaking
the future ? do not train.

### Unit tests

- forward shape (incl. seq_len not a multiple of C)
- causality at multiple cut points
- first-chunk cross context is zero with mix_gate forced to +inf
- mask diversity (Jaccard overlap of per-head top-k sets < 0.75)
- orthogonality loss bounds; gradient flow to mask_logits and mix_gate
- cross-chunk mask is strictly causal; intra-chunk mask is tril + pad-aware
- chunk DCT roundtrip

## PolyMamba-2 SSM

### Init Requirements

- **A Hurwitz-stable AND radius-bounded every step**: `A_log = log(exp(0.001)-1)` at init; M
  normalized so `|eig(M)| <= 1/1.1`. In the forward, `dt = softplus(A_log).clamp_min(dt_floor)`
  where `dt_floor[k] = (1-max_radius)/min_eig(M[k])` with default `max_radius=0.95`. This caps
  the **largest** eigenvalue of A at `≤ 0.95`, bounding the SSM *backward* gradient
  amplification to `1/(1-0.95) ≈ 20×` (was ~1100× at ρ=0.999). FSA supplies long-range context,
  so the SSM must NOT be a near-integrator (see B27 in `docs/debug_report_2026-07-31.md`).
- B, C ~ `N(0, 1/d_inner)`; D zeros; gate ~ `N(0, 0.01^2)`.

### Forward pass rules

1. SSM scan in fp32 (Hillis-Steele; combine rule `(M1,v1)oplus(M2,v2)=(M2M1, v2+M2v1)`).
2. `A_log` excluded from weight decay (decaying it changes SSM dynamics).
3. spectral_radius() computed in fp32 on CPU (ROCm eigvals on GPU/bf16 is unreliable).

### Balance loss (fixed 31 Jul 2026)

- Computed INSIDE forward from the real per-token gates: `L = relu(log K - H(mean(pi)) - 0.3)`.
- Measures GLOBAL order usage (mean over batch+seq). Per-token routing diversity is intended.
- Returned as a block output tuple `(x, kbal, ortho)` so the graph is checkpoint-safe.
- Regression tests: `test_balance_loss_positive_when_gate_collapsed_on_data`,
  `test_balance_loss_zero_when_gate_uniform_on_data`.

### Unit tests

- forward shape; spectral radius < 1 at init AND after training steps; balance loss bounds;
  balance loss responds to global collapse; gradient flow through A_log and gate.

## MoD-GDR (Mixture-of-Depths router)

### Init Requirements

- `route_proj` ~ `N(0, 0.01^2)` so initial scores are near-uniform.

### Forward pass rules

1. The router returns `x` UNCHANGED plus a gate `(B, L, 1)`. It does NOT mask x.
2. The model applies the gate residually around each routed block in the span
   [routing_point, next_routing_point): `x = gate*block(x) + (1-gate)*x`.
3. Unselected tokens get exact identity passthrough (fixed 31 Jul 2026 zero-poisoning bug).
4. Keep ratios: 0.50/0.70/0.85/1.00/1.00 at blocks 4/8/12/16/20. Ratio 1.0 = select all.

### Load-balance loss

- MSE of mean soft selection prob per position vs target ratio. The hard count is exact by
  construction so a count-based loss is identically 0 (fixed 31 Jul 2026).

### Unit tests

- gate count matches ratio; router returns x unchanged; gate is binary in forward; residual
  gate math (selected->block output, unselected->identity); keep_ratio=1 selects all;
  non-routing point is identity; gradient flows to route_proj.

## Common Patterns

All three components use the same `to_fp32` helper for stability-critical ops. NEVER use bf16
for softmax, MoD routing scores, SSM state updates, A construction, or DCT.
