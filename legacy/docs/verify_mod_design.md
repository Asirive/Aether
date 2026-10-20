# MoD design audit — gate reuse, shared router, residual semantics

**Angle:** the MoD design itself (`aether/model.py::AetherForCausalLM.forward`,
`aether/components/mod_router.py`, `aether/ops/topk_soft.py`).
**Date:** 15 Sep 2026, round 1. CPU-only, tiny tensors, `torch.set_num_threads(1)`.

## VERDICT: NOT A BUG that prevents learning.

None of the five flagged points is a defect that stops the model from learning. Items 1
and 4 are the *intended* MoD semantics; items 2 and 5 are real but saturate/immaterial at
scale; item 3 is a minor mis-specification of the (shared) routing loss, not a blocker.
The one genuine residual risk is the STE-path magnitude coupling (bonus below), which is
the same class as the PAF-9 collapse but empirically bounded in the post-fix run.

---

## 1. Gate on the FULL block output — NOT A BUG (it IS residual-flow + gated sublayers)

`aether/model.py:172-175`

```python
if gate is not None:
    block_in = x
    block_out, kbal, ortho = block(block_in)
    x = gate * block_out + (1.0 - gate) * block_in
```

`AetherBlock.forward` (`model.py:98-108`) folds the residual in *first*:
`block_out = block_in + mix + ffn`. Therefore

```
gate*block_out + (1-gate)*block_in  ==  block_in + gate*(block_out - block_in)
                                     ==  block_in + gate*(mix + ffn)
```

Executed (tiny model, 6 layers, routing point 2, `d_model=64`):

- `max|gate*block_out+(1-gate)*block_in - (block_in + gate*(block_out-block_in))| = 0.0`
- `max|gated -(block_in + gate*(mix+ffn))| = 2.4e-7` (fp32 round-off only)
- unselected tokens: `max|gated - block_in| = 0.0` exactly; selected: `max|gated - block_out| = 0.0`
- `gate` values are exactly `{0.0, 1.0}`

So the gate does **not** scale the residual: the algebra cancels it, leaving the canonical
Mixture-of-Depths form (Raposo et al. 2024) — residual always flows, only the sublayer
contribution `F(x)=mix+ffn` is gated. **This is defensible and is the intended semantics**;
"it gates the full block output" is a code-reading illusion, not a mathematical difference.

Consequence of the hard `{0,1}` gate + 4-block span reuse: an **unselected token is an exact
identity through 4 consecutive blocks**, so it receives *no* block contribution
(`mix`/`ffn` discarded, and the block params get zero gradient *from that token*). The block
is still computed densely for all tokens, so there is **no FLOP saving** — the "MoD" here is
a pure gating/regularisation mechanism, not a compute-saving one. That is a naming/design
criticism, not a learning blocker.

## 2. Residual-norm divergence — real but SELF-CORRECTING (saturating), not runaway

Measured with an instrumented forward that splits the pre-final-norm residual by the routing
mask at the routing point (gauge: `norm_selected / norm_unselected`).

My run (6 layers, `points=(2,)`, ratio 0.5, seq 64, batch 4, AdamW lr 1e-3, 300 steps):

| step | ratio | | step | ratio |
|---|---|---|---|---|
| 0 | 0.999 | | 140 | 1.170 |
| 20 | 1.141 | | 180 | 1.165 |
| 40 | **1.237** | | 240 | 1.170 |
| 60 | 1.212 | | 300 | **1.154** |

Reproduction of the prior script `exp3_split_axis.py` (8 layers, `points=(2,6)`,
ratios `(0.5,0.7)`, measured at block-6 entry against the point-2 mask, 400 steps):

`0.987 → 1.358@40 → 1.348@80 → 1.229@160 → 1.202@240 → 1.268@360`

Multi-span run (8 layers, `points=(1,5)`, ratios `(0.5,0.7)`): `0.982 → 1.260@30 → … → 1.157@300`;
final norm by depth: 0 spans = 1.051, 1 span = 1.230, 2 spans = 1.298 (deep/no-span = **1.235**).
Matched no-MoD baseline: ratio exactly **1.000** at every step (all tokens share one path).

**Conclusion:** the ratio rises to a peak (~1.24–1.36) then *decays and plateaus* at ~1.15–1.27.
It is **self-correcting/saturating, not runaway** — the per-block input `RMSNorm` bounds block
outputs and the final `RMSNorm` bounds the logits, so there is no feedback channel. The prior
"1.0 → 1.96 monotonic divergence" was **not reproduced by either script**; treat 1.96 as
metric-dependent/suspect (my metric is an honest selected-vs-unselected norm ratio).

## 3. keep_ratio=1.0 routing points — gate IS a no-op; the routing LOSS is not (minor mis-spec)

Verified: with `keep_ratio=1.0`, `k = seq_len` (`mod_router.py:67`), `kth_value` = the minimum,
`hard_mask` is all-ones → `gate` is exactly `{1.0}` and `x == block_out` exactly (0.0 diff)
(`mod_router.py:24-25`, `topk_soft.py`). So the **gate at layers 16 and 20 is a no-op**, and
because the gate is reused to the next routing point, **layers 16-23 are never gated at all**.
Only the spans at 4/8/12 (ratios 0.5/0.7/0.85) do anything.

But the routing *loss* at those points is not zero and is the dominant contributor
(real schedule `(0.5,0.7,0.85,1.0,1.0)`, seq 1024, `d_model=64`):

| point | ratio | bal | z (×1e-2) | combined | raw ∂/∂route_proj |
|---|---|---|---|---|---|
| L4 | 0.50 | 2.24e-4 | 6.9e-5 | **2.93e-4** | 7.24e-3 |
| L8 | 0.70 | 3.58e-2 | 6.9e-5 | **3.58e-2** | 1.69e-2 |
| L12 | 0.85 | 0.108 | 6.9e-5 | **0.108** | 2.81e-2 |
| L16 | 1.00 | 0.192 | 6.9e-5 | **0.192** | 3.70e-2 |
| L20 | 1.00 | 0.192 | 6.9e-5 | **0.192** | 3.70e-2 |

The two no-op points contribute 0.385 of the 0.529 total mod loss (**73 %**) and the *largest*
gradient into the single shared `route_proj`. Root cause: `balance_loss = mse(mean(sigmoid),
k/seq_len)` (`mod_router.py:74-76`) — `usage = soft.mean(dim=0)` saturates at ≈0.5 whenever the
score spread ≪ temperature (here std≈0.075, T=1), so it only matches `target=ratio` at ratio≈0.5.
At ratio 1.0 the target (1.0) is unreachable through a finite-weight sigmoid, giving a persistent
MSE ≈ (0.5−1)² ≈ 0.19 that pushes the **shared** router to inflate its score spread — a gradient
that has nothing to do with depth routing and interacts with the 0.5/0.85 points. Weighted by
`mod_loss_weight=0.01` the effect is small (~0.004 on the loss), so it is a smell, not a blocker.

**Minimal fix if desired:** drop routing points with `keep_ratio >= 1.0` (they cannot route), i.e.
use `routing_points=(4,8,12)`; or clamp the balance target to the achievable sigmoid mean.

## 4. `k` over the SEQUENCE axis — NOT A BUG (this IS per-token MoD / fixed capacity)

`k = max(1, int(seq_len*keep_ratio))` (`mod_router.py:67`), `kthvalue(perturbed, n-k+1, dim=-1)`
over the `(B,L)` score tensor (`topk_soft.py:24`) takes top-k over the sequence axis, per batch row.
Executed: both a flat/zero row and a random row route exactly `k`; per-*column* counts vary (0-4 in a
16-col test). This is the intended MoD *static-computation* rule: every sequence routes a fixed
fraction of positions, independent of content. The alternative — a **per-token threshold**
(`score > τ`, variable compute) — is a *different* family (content-adaptive depth), and measured it
selects wildly variable counts per row (0/23, 0/22, 40/24). Choosing fixed capacity is the MoD design
choice; it matters (fixed vs adaptive compute) but is not a bug. Caveat: "exactly k" holds only when
there are no ties (see item 5).

## 5. `hard_mask = (perturbed >= kth_value)` over-selects on ties — CONFIRMED, immaterial at scale

`topk_soft.py:25`. Constructed ties:
`[0.5,0.5,0.5,0.1], k=2 → selects 3` (+1); all-equal `(1,8), k=4 → selects 8` (+4);
`(1,16)` with one 1.0 and `k=8 → selects 16`.

At real sequence lengths (`soft_topk_mask` called in bf16 under autocast):
fp32 → extra = 0 always. bf16 → extra ≤ 4/row; at ratio 0.85, 44-56 % of rows over-select by
~0.65-1.06 tokens (≤0.4 % relative); at ratio 0.5, 0-9 % of rows, ≤0.09 tokens.
**Degenerate (all-zero / constant) hidden rows select 100 % at any ratio** (1024/1024, 2048/2048) —
this is the practically relevant failure and it breaks the docstring's claim that the hard count is
"exact by construction". At scale the effect is negligible *unless* flat rows occur.
**Minimal fix:** build the mask from the `indices` already computed (`topk_soft.py:26`) instead of `>=`.

## Bonus (residual-semantics coupling) — the one genuine latent risk

The PAF-9 fix made the router's *own* loss gradient scale-invariant (verified: 1.00× under a 64× input
scale). But the gradient that actually **trains the router to route** flows through the gated residual
(STE path): `∂/∂route_proj ∝ (block_out - block_in) · ∂L/∂x`. Measured, that grows with the sublayer
contribution magnitude — 15.5× → 248× → 3971× as `(block_out - block_in)` scales 1→4→16→64
(~∝ scale²); corroborated by the prior `exp2_driven.py`. This channel is *inherent to the gating form*
(it would be identical if you gated only the sublayers) and is **not** covered by the `rms_norm` fix.
It is the same class as the collapse mechanism, but the post-fix run was stable (grad norm 1.5–1.9 to
step 440), so it is a latent risk, not a demonstrated learning blocker.

## MINIMAL FIXES (all optional; none is required to learn)

1. `aether/components/mod_router.py` + `configs/aether_200m.yaml`: drop `keep_ratio>=1.0` routing
   points (16, 20) — they gate nothing and only inject a mis-specified balance gradient.
2. `aether/ops/topk_soft.py:25`: build `hard_mask` from the topk `indices` (exact-k by construction).
3. (If the residual-magnitude coupling is a concern) detach or normalise the STE's
   `(block_out - block_in)` factor, or bound block output magnitude.

## CONFIDENCE: 80 %

**Would falsify:**
- Item 2: a longer/at-scale run in which the selected/unselected norm ratio keeps climbing past
  ~1.4 instead of plateauing.
- Bonus: a run in which `route_proj`'s STE-path gradient dominates the global grad norm
  (would elevate the coupling from latent to a real collapse driver).
- Item 5: evidence that all-zero/constant hidden rows actually occur in the training batches.
