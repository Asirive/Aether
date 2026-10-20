# Scale-bound verification: is any activation/weight scale unbounded?

**Agent angle:** build the missing instrument (the repo logs only `grad_norm`) and measure whether
activation and weight scales are bounded — and whether, after the 15 Sep router fix, any
unbounded/uncorrected quantity remains.

**Instrument:** `scripts/verify_scale_bounds.py` (CPU, single thread, ~2 min for 400 steps).
It trains a tiny AETHER (`d_model=64`, 4 layers, 272k params, tied embeddings, `vocab=64`,
`seq=32`) on a hard-but-learnable trigram task and logs **every step**: per-layer residual RMS,
block-local forward gain, `||token_embedding.weight||_F`, logit RMS / max|logit|, every RMSNorm
`||weight||`, every mixer output RMS, `route_proj`/`route_norm` norms, FSA `mask_logits` norm,
and the pre-clip grad norm. The gated-residual reconstruction is validated against the true
pre-final-norm residual (max abs error `0.000e+00`).

Config = post-15-Sep: `embed_lr_scale=0.1`, router input RMSNorm + z-loss, two-sided `dt` clamp.

---

## 1. Trend table (400 steps, the requested run)

`×` = end/start. "still climbing" = positive slope over the last 100 steps **and** last-100 value
above the mid-run value (i.e. no saturation).

| quantity | start | mid(200) | end(400) | × | mono% | still climbing |
|---|--:|--:|--:|--:|--:|:--:|
| loss | 4.206 | 3.931 | 3.667 | 0.87 | 48.5 | no (falls) |
| `res_rms_0` (residual after blk0) | 0.0202 | 15.46 | 12.98 | **643** | 54.0 | no |
| `res_rms_3` (residual after blk3) | 0.0202 | 17.42 | 15.85 | **786** | 55.2 | no |
| `block_gain_0` (blk0 out/in RMS) | 1.00 | 703.8 | 549.8 | **550** | 54.2 | no |
| `mixer_rms_0` (PolyMamba out) | 1.7e-6 | 16.52 | 13.70 | **8.0e6** | 54.5 | no |
| `embed_weight_norm` (tied head) | 1.293 | 1.539 | 1.630 | 1.26 | 89.2 | **YES** |
| `logit_rms` | 0.231 | 0.541 | 0.652 | 2.82 | 55.5 | **YES** |
| `logit_absmax` | 1.55 | 2.02 | 2.54 | 1.64 | 51.0 | **YES** |
| `norm_final` (last RMSNorm gain) | 8.00 | 10.01 | 10.96 | 1.37 | 88.5 | **YES** |
| `route_norm` (router gain) | 8.00 | 7.89 | 8.10 | 1.01 | 66.8 | **YES** |
| `route_proj` (router projection) | 0.0755 | 0.503 | 0.822 | **10.9** | 87.8 | **YES** |
| `fsa_mask_logits` | 14.75 | 14.76 | 14.71 | 1.00 | 34.0 | no |
| grad_norm | 0.72 | 0.76 | ~1.0 | — | — | no (stable) |

### 1200-step confirmation (batch 4, same task)

| quantity | start | end(1200) | × | mono% | still climbing |
|---|--:|--:|--:|--:|:--:|
| loss | 4.225 | 3.537 | 0.84 | 49.5 | no (still falling) |
| `res_rms_0` | 0.0202 | 23.74 (peak ~64 @300) | **1174** | 50.1 | no (peaks then relaxes) |
| `mixer_rms_0` | 1.8e-6 | 10.99 | **6.0e6** | 52.2 | no |
| `embed_weight_norm` | 1.293 | 1.873 | 1.45 | 94.9 | **YES** |
| `logit_rms` | 0.233 | 0.855 | 3.68 | 51.1 | no (saturates ~step 750) |
| `norm_final` | 8.00 | 13.39 | 1.67 | 93.3 | **YES** |
| `route_norm` | 8.00 | 9.28 | 1.16 | 70.7 | **YES** |
| `route_proj` | 0.0755 | 2.955 | **39.1** | 87.5 | **YES** |
| `fsa_mask_logits` | 14.75 | 14.41 | 0.98 | 20.2 | no |
| grad_norm | 1.01 | 2.07 (peak 2.83 @900) | ~2 | — | no |

**Which grow monotonically?** Everything except `fsa_mask_logits` (~flat) and the block-1/2 gains
(~flat).
**Which are still climbing at the end (no saturation)?** `route_proj` (×10.9 / ×39.1),
`norm_final` (×1.37 / ×1.67), `embed_weight_norm` (×1.26 / ×1.45), `logit_rms`/`logit_absmax`,
`route_norm` (~×1.1). The residual and mixer norms grow much *faster* (×600–10⁷) but **saturate**:
they peak around step 300 then relax as the block readouts converge.

---

## 2. Fastest-growing quantity and its cause

* **Fastest overall = the residual stream and its per-block additive contribution** (residual
  ×643→1174; PolyMamba mixer output ×6e6 relative to its ~0 init). **Cause = residual accumulation:**
  `aether/model.py:106-107` does `x = x + mix; x = x + self.ffn(self.norm2(x))` with no bound on the
  sum. At init the block readouts are ≈0 (`out_proj` std `0.02/√d_inner`), so the residual starts at
  the embedding RMS (~0.02) and is inflated by the blocks' learned O(1) contributions. It is a
  **null-ish direction for the loss**: every sublayer reads `RMSNorm(x)` (scale-invariant,
  `model.py:21-24`), so the loss exerts no restoring force on the residual's global scale. This one
  *does saturate* here (block outputs are produced by weight-decayed weights from scale-invariant
  inputs) — but nothing **explicitly** bounds it.
* **Fastest-growing quantity that does NOT saturate = `mod_router.route_proj.weight`** (×10.9 at 400
  steps, ×39.1 at 1200 steps, monotone 87–88%). **Cause = a specific un-decayed parameter:** it is in
  the no-decay set (`optim.py:9`), and its nominal bound — the ST-MoE z-loss added on 15 Sep — is
  **effectively 1e-4**, because `z_loss_weight=1e-2` (`mod_router.py:34`) is multiplied again by
  `mod_loss_weight=1e-2` at `model.py:203-207`. A 1e-4-weight term cannot counter a unit Adam step.
  Once the top-k gate saturates, `signal`-w.r.t.-`route_proj` is ~0, so the norm drifts freely.
* **Not** the RMSNorm gains or the embedding as *rate* leaders, but they are the next two and they
  also never saturate: `norm_final` ×1.67 (mono 93%) and `token_embedding` ×1.45 (mono 95%) — both in
  the no-decay group.

---

## 3. Every tensor whose scale is NOT constrained by construction

Constraint inventory from `aether/training/optim.py` (`_NO_DECAY_SUBSTRINGS`, lines 9/26-34/40) —
confirmed by dumping the live param groups:

* **`weight_decay=0.0` group (19 params):** all `norm1`/`norm2` gains, final `norm` gain,
  `route_norm_weight`, `route_proj.weight`, `A_log`, `mixer.gate.weight`, `mask_logits`, `mix_gate`.
* **`weight_decay=0.0, lr×0.1` group:** `token_embedding.weight`.

Ranked by measured growth rate (unconstrained ones only):

| rank | tensor | constructed bound | growth (400 / 1200 steps) | still climbing? | dangerous? |
|---|---|:--:|:--:|:--:|:--:|
| 1 | residual stream `x` (activation) | none | ×643 / ×1174 | no (saturates) | no |
| 2 | mixer/block outputs (activations) | none | up to ×8e6 / ×6e6 | no | no |
| 3 | `mod_router.route_proj.weight` | z-loss, eff. **1e-4** only | ×10.9 / **×39.1** | **YES** | no (gate ∈[0,1]) |
| 4 | `norm.weight` (final RMSNorm gain) | none | ×1.37 / ×1.67 | **YES** | no |
| 5 | `token_embedding.weight` (tied head) | 0.1× LR only, no decay | ×1.26 / ×1.45 | **YES** | no (need monitor) |
| 6 | `mod_router.route_norm_weight` | none | ×1.01 / ×1.16 | YES | no |
| 7 | per-layer `norm1/norm2` gains | none | ×0.73–1.10 | mixed | no |
| 8 | `fsa.mask_logits` | ortho loss is scale-invariant ⇒ none | ×1.00 / ×0.98 | no | no |
| 9 | `poly.mixer.gate.weight`, `fsa.mix_gate` | none | — | — | no (softmax/sigmoid bounds effect) |
| 10 | `A_log` | no decay **but** two-sided `dt` clamp (`polymamba.py:96-104`) | bounded | — | no |
| — | `B,C,D`, all `*_proj`/`ffn`/`w_q/k/v/o` linears | weight_decay = 0.1 ⇒ bounded | — | — | no |

**After the 15 Sep router fix: yes, unbounded/uncorrected quantities remain** — the residual stream,
the RMSNorm gains, the tied embedding (hence logit scale), the router gains/`route_proj`, and the FSA
`mask_logits` norm. **But none is a collapse mechanism:** they are all scale-degenerate (each is
either consumed through a scale-invariant RMSNorm, a bounded softmax/sigmoid gate, or is a pure
parameter scale), and the 1200-step repro shows grad_norm bounded (peak 2.83) with loss still
falling. The one genuinely *dangerous* path pre-fix — an **unbounded activation fed into a hard
top-k gate** — is gone: `mod_router.py:69` now does `F.rms_norm(x)` before `route_proj`.

---

## 4. Minimal set of bounds (top 3)

Only the still-climbing, unconstrained tensors need bounds. Weight decay alone is **not** sufficient:
moving `route_proj` into the decay group only reduced its 400-step growth from ×10.88 to ×8.77 (Adam
decoupled decay is ~10× weaker than a unit Adam step). Use loss/architecture terms:

1. **Router score scale → `route_proj`** (rank 3).
   * Make the existing z-loss real: the aux term arrives at coefficient `1e-4`
     (`z_loss_weight` 1e-2 @ `mod_router.py:34` × `mod_loss_weight` 1e-2 @ `model.py:203-207`).
     Raise `z_loss_weight` to ~1e-1 (or drop the double-scaling) so the effective coefficient is
     ~1e-3, and/or remove `"route_proj"` from `_NO_DECAY_SUBSTRINGS` at **`optim.py:9`**.
   * Minimal hard guard: clamp `route_proj.weight` norm in `MoDRouter.forward` (**`mod_router.py:69-71`**).
2. **Logit scale → covers the embedding (rank 5), the final-norm gain (rank 4) and `logit_rms/absmax`.**
   Add a **logit soft-cap** (`logits = cap*tanh(logits/cap)`, Gemma-2 style) or a **logit z-loss**
   `1e-4·logsumexp(logits)²` inside `_compute_logits` at **`aether/model.py:276-279`**. Because
   `logit ≈ (final gain)·E`, one soft-cap transitively bounds both the gain and the embedding scale.
   Alternatively/additionally give the tied embedding weight decay by changing
   `"weight_decay": 0.0` → the base value in the embed group at **`optim.py:40`**.
3. **RMSNorm gains (rank 4/7) and `mask_logits` norm (rank 8).** The logit soft-cap (item 2) bounds
   the gains where it matters (they only affect the output through the logit scale). `mask_logits`
   has no bound at all (the orthogonality loss normalises first, `fsa.py:160-168`), but is
   self-limiting via the sigmoid in `soft_topk_mask`; if a hard bound is wanted, add an L2 penalty on
   `mask_logits` in `FreqSparseAttention.orthogonality_loss` (**`fsa.py:160`**).
4. **QK-norm / branch scale: not required.** FSA `q,k` already come from an RMSNorm'd input through
   weight-decayed `w_q/w_k`, and the MoD gate is in [0,1]; both are already bounded.
5. **Residual stream (rank 1–2): monitor, don't cap.** It is bounded in practice (saturates here).
   The missing telemetry is the fix: log `res_rms` per layer every N steps (the new instrument does
   exactly this) so any *future* unbounded rise is visible before it overflows bf16.

---

## 5. Verdict

* **Remaining dangerous unbounded quantity?** **NO** — after the router fix the collapse path is
  closed; 1200 CPU steps keep grad_norm ≤ 2.83 with loss falling. (NOT A BUG for stability.)
* **Remaining unbounded-at-all quantity?** **YES**, several (see §3) — all benign/scale-degenerate.
* **Weakest link:** the z-loss shipped on 15 Sep is effectively **1e-4** (double-scaled), so it does
  not do its stated job — `route_proj` still grew **×39** over 1200 steps. This is latent, not
  currently harmful, but it is the single cheapest thing to correct.

**Confidence:** 80%. **Falsified by:** a long enough real run (or the 200M config) in which
`res_rms` or `logit_absmax` exceeds the bf16 range / grad_norm grows without the router — which
would move the residual/logit scale from "saturating" to "dangerous".
