# `embed_lr_scale: 0.1` — does it cause the post-fix learning regression?

**Agent angle:** prove or disprove that `optim.embed_lr_scale: 0.1` (`configs/aether_200m.yaml:52`)
is responsible for the post-fix run learning ~1.2 nats slower at steps 200–400, with a
controlled CPU experiment.

**Instrument:** `scripts/verify_embed_lr_scale.py` (CPU, single thread, `d_model=64`,
4 layers, 232,200 params, tied embeddings, `vocab=512`, `seq=32`; ~50 s per 400-step arm).
Two arms share the **same init** and the **same batches**; the only differing argument is
`embed_lr_scale`.

**Verdict: CONFIRMED** — `embed_lr_scale=0.1` alone reproduces the observed gap almost
exactly (real gap 1.18 nats @200 → 2.43 nats @400; toy gap 1.38 → 2.82 over the same
steps, ratio 0.99–1.16× from step 300 on). It is also **not needed**: the stated
motivation (bounding logit-scale / head-gradient growth) is not delivered by an LR scale.

---

## 1. Mechanism verification (executed, `aether/training/optim.py`)

`_split_param_groups` (`optim.py:12-41`) puts every trainable parameter into exactly one of
three lists; the embedding branch is fired by `embed_lr_scale != 1.0 and "token_embedding"
in lowered` (`optim.py:24-25`) and the group gets `lr_scale` (`optim.py:39-40`), which
`build_optimizer` converts to a per-group LR at `optim.py:53-54`:
`group["lr"] = lr * group.pop("lr_scale", 1.0)`.

Verified by constructing the real `build_optimizer`/`build_scheduler` on the tiny model:

| `embed_lr_scale` | groups | embedding group | initial LR |
|---|---|---|---|
| `1.0` | 2 (`decay` 30 params, `no_decay` 20) | merged into `no_decay` (`_NO_DECAY_SUBSTRINGS` contains `"embedding"`, `optim.py:9`) | `1.2e-4` |
| `0.1` | 3 (`decay` 30, `no_decay` 19, **`['token_embedding.weight']`**) | its own single-parameter group | `1.2e-05` |

Partition is exact: 30 + 19 + 1 = 50 = every `requires_grad=True` parameter, no duplicates,
nothing dropped (`model.py` has 4 frozen `structure_M` params but no trainable ones are lost).

**`LambdaLR` preserves the ratio.** `build_scheduler` (`optim.py:63-76`) passes one `lr_lambda`
to `LambdaLR`, so `get_lr()` returns `initial_lr * lambda(step)` for **each** group with the
group's own `initial_lr`. Measured on the real schedule (`lr=1.2e-4`, `warmup=1000`,
`total=100000`, `min_lr_ratio=0.1`):

```
step      0: lr per group = ['0.000000e+00', '0.000000e+00', '0.000000e+00']  ratio nan
step    500: lr per group = ['6.000000e-05', '6.000000e-05', '6.000000e-06']  ratio 0.1000
step   1000: lr per group = ['1.200000e-04', '1.200000e-04', '1.200000e-05']  ratio 0.1000
step   5000: lr per group = ['1.195656e-04', '1.195656e-04', '1.195656e-05']  ratio 0.1000
```

The 10× ratio is preserved exactly through warmup *and* cosine decay. So the mechanism
works as intended — this is **not** a code bug; it is a hyper-parameter that is set wrong.

**Why this hits the head too.** `tie_embeddings=True` ⇒ `lm_head = None` (`model.py:137-140`)
⇒ `_compute_logits` does `F.linear(x, self.token_embedding.weight)` (`model.py:276-279`).
The single scaled tensor is **both** the input embedding and the output projection, so the
"head" learns 10× slower as well. In the real config that tensor is
128,256 × 576 = **73.9 M / 186.6 M = 39.6 % of all parameters**
(toy: 32,768 / 232,200 = 14.1 % — the toy is if anything *less* embedding-heavy than the real model).

Also note `trainer.py:685,740` log `lr=self.optimizer.param_groups[0]["lr"]`, i.e. only the
**base** group LR — the embedding's LR is invisible in `train.log`, which is why the
0.1× group has gone unremarked.

---

## 2. Measured curves (400 CPU steps, constant `lr=1e-4`, mean of 3 seeds)

Loss is the pure task CE (`result["task_loss"]`). Data: a deterministic cycle of 16 distinct
tokens → the successor function is exactly learnable, `CE: ln(512)=6.24 → 0`.

| step | scale=1.0 | scale=0.5 | scale=0.3 | **scale=0.1** | gap(1.0−0.1) |
|--:|--:|--:|--:|--:|--:|
| 25 | 5.8747 | 6.0046 | 6.0518 | 6.0920 | 0.217 |
| 50 | 5.4265 | 5.6776 | 5.7726 | 5.8587 | 0.432 |
| 75 | 4.9923 | 5.3124 | 5.4416 | 5.6080 | 0.616 |
| 100 | 4.5839 | 4.9787 | 5.1093 | 5.2628 | 0.679 |
| 150 | 3.9788 | 4.5799 | 4.7729 | 4.9287 | 0.950 |
| **200** | **3.4815** | 4.3324 | 4.6259 | **4.8579** | **1.376** |
| 250 | 3.0235 | 4.0931 | 4.4909 | 4.8148 | 1.791 |
| 300 | 2.5987 | 3.8490 | 4.3510 | 4.7698 | 2.171 |
| 350 | 2.2111 | 3.5991 | 4.2047 | 4.7249 | 2.514 |
| **400** | **1.8624** | 3.3531 | 4.0594 | **4.6791** | **2.817** |

### 1000-step continuation (seed 0) — the gap does not close

| step | 1.0 | 0.5 | 0.3 | 0.1 |
|--:|--:|--:|--:|--:|
| 400 | 1.8829 | 3.3653 | 4.0670 | 4.6836 |
| 700 | 0.5089 | 1.9739 | 3.1457 | 4.3891 |
| **1000** | **0.0944** | **0.9412** | **2.2330** | **4.0651** |
| `eval CE` | 0.0938 | 0.9414 | 2.2348 | 4.0668 |
| `logit_rms` | 0.9003 | 0.6621 | 0.4494 | 0.2269 |
| `emb_std` | 0.09785 | 0.05333 | 0.03591 | 0.02227 |
| `emb_grad@1000` | 0.1854 | 0.9611 | 0.9771 | **0.9933** |

The 0.1× arm is **not** merely delayed — it flattens onto a much worse floor (≈4.0 nats
here) and the gap is still *widening* at 1000 steps (+2.80 @400 → +3.97 @1000).

---

## 3. Steps of progress lost by `0.1×`

Matching losses between arms (`1.0×` curve as reference):

| horizon | `0.1×` loss | equals `1.0×` at step | steps lost | slowdown |
|---|---|---|---|---|
| 400 | 4.6854 (seed 0) | 95 | 305 / 400 | **4.22×** |
| 400 | 4.6897 (seed 1) | 92 | 308 / 400 | 4.36× |
| 400 | 4.6621 (seed 2) | 96 | 304 / 400 | 4.18× |
| 1000 | 4.0651 (seed 0) | 146 | 854 / 1000 | **6.85×** |

And the reverse direction never happens inside the horizon: the `0.1×` arm does **not**
reach the `1.0×` arm's step-400 loss (1.86) by step 1000. So ≈ **305 of 400 steps (76 %) of
early progress are destroyed**, and the handicap compounds rather than washing out.

---

## 4. Does it explain the real 1.2-nat gap?

**First, the real gap is bigger than the briefing's table says.** The briefing
(`docs/VERIFY_CONTEXT.md:39-44`) quotes run2 as `~8.8 @300, ~8.3 @400`. The actual log
(`runs/aether_200m_run2/train.log`, the segment that ran to step 1650) says `7.9087 @300`
and `7.2801 @400`. Corrected:

| step | run2 (scale 1.0) | run3 (scale 0.1) | real gap | toy gap (only scale differs) | toy/real |
|--:|--:|--:|--:|--:|--:|
| 100 | 10.6506 | 10.8739 | 0.223 | 0.679 | 3.04 |
| 150 | 10.0425 | 10.6695 | 0.627 | 0.950 | 1.52 |
| **200** | **9.3422** | **10.5254** | **1.183** | **1.376** | **1.16** |
| 250 | 8.5205 | 10.2622 | 1.742 | 1.791 | 1.03 |
| 300 | 7.9087 | 10.1110 | 2.202 | 2.171 | 0.99 |
| 350 | 7.3566 | 9.8103 | 2.454 | 2.514 | 1.02 |
| **400** | **7.2801** | **9.7059** | **2.426** | **2.817** | **1.16** |

`embed_lr_scale` is the **only** experimental variable in the toy, and it reproduces the
observed gap to within 1–17 % at every step from 200 onward, with the same growth shape
(gap ≈ 0 from step 1 to 50, then diverging). So:

- **Yes, it is a plausible explanation — and more than plausible: it is *sufficient* to
  account for the entire observed gap.** On this evidence the other three 15-Sep fixes
  (router RMSNorm+z-loss+temperature, two-sided `dt` clamp, FSA index fix) need not
  contribute anything to the loss regression; they cannot be ruled out as *contributors*
  from this experiment alone (run2↔run3 differ in four ways), but no unexplained residual is
  left over once `embed_lr_scale` is credited.
- The gap is **not** "too large for that alone". The briefing understated the real gap
  (2.43 nats @400, not 1.4), and the toy exceeds even that.
- The toy is a worst case only in the sense that its learnable structure (a deterministic
  successor cycle) lives entirely in the tied matrix. But the real early phase is precisely
  the embedding/head-dominated regime: per `AGENTS.md` §PAF-9 the loss at step 400 is still
  above the **empirical unigram entropy of 7.71 nats** (`VERIFY_CONTEXT.md:62-64`), i.e. it
  is still learning unigram/bigram statistics, and 39.6 % of the model's parameters are that
  one matrix. Warmup also makes the real case worse, not better: both arms ramp together, so
  the ratio is preserved while the embedding's *absolute* movement is tiny for the first
  1000 steps.

---

## 5. Recommendation — smallest safe value

**Set `optim.embed_lr_scale: 1.0` (delete the scaling).** If a stability hedge is mandated,
**0.5 is the largest cut that is defensible; 0.1 and 0.3 are not.**

Justification from the measurements:

1. **The motivation does not hold up.** `AGENTS.md` §PAF-9 fix 4 says the cut exists to bound
   "the logit-scale growth that drives the head gradient". An LR multiplier cannot bound a
   gradient — it only slows the parameter trajectory. Measured pre-clip
   `‖∇ token_embedding.weight‖₂` (mean over 3 seeds, 400-step run):

   | steps | 1.0 | 0.5 | 0.3 | 0.1 |
   |--:|--:|--:|--:|--:|
   | 1–100 | 0.9847 | 0.9820 | 0.9802 | 0.9781 |
   | 201–300 | 0.9798 | 0.9911 | 0.9940 | 0.9960 |
   | 301–400 | 0.9713 | 0.9874 | 0.9925 | **0.9957** |

   The embedding gradient is *identical* across scales, and by step 301–400 the **total**
   pre-clip grad norm is *larger* with the cut (2.055 @0.1× vs 1.567 @1.0×) because the
   model is further from the minimum. The one thing the cut does deliver is a smaller
   logit RMS (0.90 → 0.23 at 1000 steps) — but that is a *consequence of under-training*
   (the 0.1× run never learned the task), not a stability control.
2. **The actual PAF-9 hazard is already fixed, and independently of this knob.** The
   step-1300 collapse came from a *discrete gate* fed by an unbounded input
   (`‖x‖²/T` router gradient). Fix 1 removed it with `F.rms_norm` on the router input plus
   the z-loss. `embed_lr_scale` is a second-order hedge against a mechanism that no longer
   exists — and it is the only one of the four fixes the user can see in the loss curve.
   Corroborating: post-fix grad norm is *higher* (1.5–1.9 vs 0.8–1.0), so the knob did not
   even move the metric it was meant to protect.
3. **Cost ladder** (400-step toy, gap vs `1.0×`): `0.5` → +1.49 nats, `0.3` → +2.19,
   `0.1` → **+2.82**; at 1000 steps `0.5` → +0.85 nats, `0.3` → +2.14, `0.1` → **+3.97**.
   There is no knee: the cost is monotone in the size of the cut, and only `1.0` is free.
   `0.5` keeps the run on the same trajectory (0.94 nats at 1000 steps, still descending
   fast) at a visible but survivable cost; `0.3` and `0.1` diverge onto a different, much
   worse trajectory.

### Minimal fix

`configs/aether_200m.yaml:52`

```yaml
-  embed_lr_scale: 0.1
+  embed_lr_scale: 1.0
```

No code change: with `embed_lr_scale == 1.0` the parameter is skipped by
`optim.py:24` and `token_embedding.weight` falls back into the existing `no_decay` group
(`optim.py:26-34`, via `"embedding"` in `_NO_DECAY_SUBSTRINGS`) with weight decay 0.0 —
exactly the pre-fix behaviour, verified above (`n_groups=2`, LR `1.2e-4`).

---

## 6. Confidence and falsifiers

**Confidence that `embed_lr_scale=0.1` is a first-order cause of the regression: 85 %.**
(Confidence that it is *a* substantial cause at all: ≈96 % — its isolated effect is large,
monotone in the cut, and quantitatively the right size.)

What would falsify / weaken this:

- A real-config A/B (`embed_lr_scale` 1.0 vs 0.1, everything else identical) showing
  < 0.3 nats gap at step 400. That would mean the toy's embedding-dominated task is
  unrepresentative of FineWeb-Edu early training.
- Showing the router `temperature` 0.1→1.0 change alone costs ≥ 2 nats by step 400 — then
  `embed_lr_scale` could be a minor contributor and the four-fix commit would need
  apportioning. This experiment cannot separate them (run2↔run3 differ in four ways).
- Evidence that the real model's early loss reduction routes mostly through the body rather
  than the tied matrix (e.g. a frozen-embedding control reaching the run2 loss trajectory).

Known limitations of the evidence: the toy's 512-token vocab and 16-token cycle make the
successor function the only learnable structure; absolute nats are not transferable, only
the scale of the gap relative to the same-steps reference. The toy's embedding is 14.1 % of
parameters vs 39.6 % in the real model, which biases the toy *downwards*, not upwards.

### Exact reproduction

```powershell
$env:OMP_NUM_THREADS=1
python scripts/verify_embed_lr_scale.py --steps 400  --lr 1e-4 --scales 1.0,0.5,0.3,0.1 --seeds 0,1,2 --out curves400.json
python scripts/verify_embed_lr_scale.py --steps 1000 --lr 1e-4 --scales 1.0,0.5,0.3,0.1 --seeds 0     --out curves1000.json
```
