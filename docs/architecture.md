# AETHER-200M Architecture Specification

Updated 31 July 2026 to match the shipped implementation exactly. See
`docs/debug_report_2026-07-31.md` for the audit that produced the CC-FSA redesign.

## Overview

A 186.6M-parameter decoder-only causal language model for a single AMD RX 7600 8GB.

```
Input tokens
   v
Token Embedding (tied with LM head)        vocab=128256, d=576
   v
[24 AETHER Blocks; even = PolyMamba-2, odd = CC-FSA]
   + MoD residual gates at blocks 4, 8, 12, 16, 20
   v
Final RMSNorm
   v
LM Head (tied)
```

Causality is a hard requirement and is regression-tested at component and model level
(`tests/test_fsa.py`, `tests/test_model_math.py::test_aether_full_model_causality`).

## Components

### CC-FSA ? Causal Chunk-Compressed Freq-Sparse Attention

Replaces attention in **odd blocks** (layer_id % 2 == 1).

**Algorithm:**
1. Q, K, V projections: `d=576 -> h=9 x d_k=64`.
2. K/V are split into chunks of `chunk_size=32` positions (zero-padded tail).
3. Each chunk is transformed by a DCT-II (`W in R^{32x32}`, fp32) along the chunk axis.
4. Each head keeps a learned top-k set of frequency bins (`keep_bins = 32 * 0.25 = 8`) as the
   chunk summary. Selection uses soft top-k with a straight-through estimator; bins are
   **gathered** (real compute reduction, not zero-masking).
5. **Cross-chunk path:** query at position t attends to compressed summaries of chunks that
   lie *fully* in the past (`(j+1)*C <= t`). Rows with no visible chunks (the first chunk)
   produce exactly zero output via masked-renormalized softmax.
6. **Intra-chunk path:** standard causal attention inside the current chunk (time domain,
   blocked per chunk, `C x C`).
7. Per-head scalar blend: `out = sigmoid(g_h) * out_cross + (1 - sigmoid(g_h)) * out_intra`.
8. Output projection `d -> d`.

**Cost:** cross `O(L * (L/C * k_c) * d)` = `O(L * (L/4) * d)` + intra `O(L * C * d)` +
DCT `O(L * C * d)` ? about **4x fewer attention FLOPs than dense**, now honest (the old design
computed a full LxL matmul with zeroed bins and the 4x claim was false).

**Prior art (must acknowledge):** FNet (Lee-Thorp 2021), GFNet (Rao 2021), AFNO (Guibas 2021),
NSA (DeepSeek 2025) ? NSA in particular compresses tokens into chunk summaries with causal
attention; CC-FSA differs by using a DCT basis + per-head learned frequency-band masks instead
of an MLP compressor.

**Numerical requirements:**
- DCT in fp32, result cast to bf16 for matmuls.
- Softmax in fp32 with masked-renorm (never raw -inf rows -> NaN).
- mask_logits init `N(0, 0.02)` (random tie-break = per-head diversity at init); mix_gate init 0
  (blend = 0.5/0.5 at init).
- Soft top-k temperature annealed 1.0 -> 0.1 and Gumbel noise 1.0 -> 0 over the first 10% of
  steps (wired every step via `model.set_component_knobs`).
- Orthogonality aux loss on sigmoid masks, weight 0.001, ramped over first 10%.

### PolyMamba-2 SSM

Replaces attention in **even blocks** (layer_id % 2 == 0).

**Algorithm:**
1. Input projection `d -> 2 * d_inner`, SwiGLU gating.
2. K=2 parallel SSMs with state dim N=16, constant-A recurrence `h_t = A h_{t-1} + B x_t`,
   `y_t = C h_t + D x_t`, computed with a Hillis-Steele parallel scan in fp32.
3. A is Hurwitz-stable by construction: `A = I - softplus(A_log) * M`, `A_log` init so
   `dt ~= 0.001`; M is the polynomial structure matrix normalized so `|eig(M)| <= 1/1.1`
   (order 0: scaled identity; order 1: symmetric tridiagonal Jacobi).
4. Per-token gate `pi_t = softmax(W_gate x_t)` mixes the K order outputs.
5. Output projection `d_inner -> d`.

**Balance loss (fixed 31 Jul 2026):** computed inside `forward` from the real gates:
`L_kbal = relu(log K - H(mean_{batch,seq}(pi)) - 0.3)`. It measures *global* order usage;
per-token diversity is intended. Weight 0.01, ramped over first 5% of steps.

**Stability guards:** spectral radius computed in fp32 on CPU each logging step; gradient clip
1.0 global; `A_log` excluded from weight decay.

### MoD-GDR ? Mixture-of-Depths router (residual gate)

Euclidean top-k depth routing at blocks 4, 8, 12, 16, 20 with keep ratios
0.50 / 0.70 / 0.85 / 1.00 / 1.00.

**Algorithm:**
1. At routing point i, `s_t = W_route x_t` (fp32), top-k via soft top-k STE.
2. The router returns x **unchanged** plus a binary gate `(B, L, 1)`.
3. The model applies the gate residually around every block in the span [i, next routing
   point): `x = gate * block(x) + (1 - gate) * x`. Unselected tokens get exact identity
   passthrough (fixed 31 Jul 2026: the old `x * mask` zeroed them permanently).
4. Load-balance loss: MSE of the mean soft selection probability per position vs the target
   keep ratio (the hard count is exact by construction, so a count-based loss is identically 0).
   Weight 0.01, ramped over first 5%.

Note: gating is functional MoD ? compute is not reduced in v1 (blocks still run densely, the
gate zeroes the *update* for skipped tokens). The paper contribution is routing quality and
training stability, not wall-clock savings.

## Block assembly

```python
def forward(x):
    mix = mixer(rmsnorm1(x))          # PolyMamba-2 (even) or CC-FSA (odd)
    x = x + mix
    x = x + swiglu_ffn(rmsnorm2(x))
    return x, kbal, ortho             # tuple: checkpoint-safe aux losses
```

Blocks return aux losses as outputs so gradient checkpointing keeps their graphs intact.

## Full architecture

```
vocab: 128,256 (Llama-3 BPE, unsloth mirror)
d_model: 576
d_inner: 1536
n_layers: 24
n_heads: 9 (CC-FSA only; PolyMamba-2 is single-head)
d_k: 64
max_seq_len: 2048 (curriculum 1024 -> 2048)
RoPE: not used (CC-FSA cross path is position-free by chunk order;
      intra-chunk locality + PolyMamba recurrence carry order information)
tie_embeddings: yes
Total params: 186,615,276 (measured)
```

## Param budget (measured, bf16)

| Component | Params |
|-----------|--------|
| Token embedding (tied) | 73.8M |
| 12 x CC-FSA block (mixer + SwiGLU + norms) | ~45.3M |
| 12 x PolyMamba-2 block (mixer + SwiGLU + norms) | ~67.5M |
| MoD router | 0.001M |
| **Total** | **186.6M** |

## Numerical policy

- **bf16:** linears, FFN activations, embeddings, attention scores (with max-subtraction).
- **fp32:** softmax, MoD routing scores, SSM state updates + scan, A-matrix construction,
  DCT/IDCT, RMSNorm statistics, all aux losses.
- **No fp32 master weights** (PAF-3): bf16 params + fp32 Adam moments via fused AdamW.
- **Gradient checkpointing** at block level (`use_reentrant=False`); blocks return
  `(x, kbal, ortho)` tuples so aux losses are valid checkpoint outputs.

## Throughput estimate (RX 7600, L=2048)

- Measured peak: 29.48 TFLOPS bf16 matmul.
- Expect ~12-18 TFLOPS sustained training with recompute; ~2.5-4K tokens/s.
- 100K steps x 65,536 tokens = 6.55B tokens ~= 5-6 days.

## References

- FNet: arXiv:2105.03824; GFNet: arXiv:2101.11639; AFNO: arXiv:2109.10219
- NSA: arXiv:2502.11089
- Mamba: arXiv:2312.00752; Liquid-S4: arXiv:2209.12951; CTDG-SSM: arXiv:2606.04672
- MoD: arXiv:2404.02258
