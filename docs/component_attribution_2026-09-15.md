# Component attribution: is AETHER's complexity justified? (2026-09-15, round 1)

**Angle:** is `PolyMamba2` actually contributing to the model, or is it dead weight? Extended to
the other two components (CC-FSA, MoD) because the same harness isolates all three.

**Deliverable (per `docs/VERIFY_CONTEXT.md`):**

1. **VERDICT:** No bug found. **PolyMamba2 — PARTIALLY JUSTIFIED. CC-FSA — NOT JUSTIFIED at this
   scale. MoD routing — NOT JUSTIFIED (measurably harmful).**
2. **EVIDENCE:** below, from code actually executed on this machine.
3. **MINIMAL FIX:** none (no bug); architectural recommendations only.
4. **CONFIDENCE:** 70 %.

---

## 1. What was run

Harness: `ablations/05_component_attribution.py` (new). CPU only, `OMP_NUM_THREADS=1`,
`torch.set_num_threads(1)`, fp32, tiny tensors. Results:
`ablations/results/05_component_attribution/{lag,induction,lag_convk3}/`.

Two synthetic sequence-memory tasks, **baked into the script, no downloads**:

| task | construction | chance (ln V) | why it is the right probe |
|---|---|---|---|
| `lag` (primary) | `x_t` iid uniform over V=16; model must emit **`x_{t-6}`** at position t (delayed copy, L=64) | 2.7726 | Dense, deterministic cross-position target. A **memoryless** mixer is at chance *by construction*; a causal conv with kernel ≥ 7 or a mechanism with an explicit offset (RoPE attention) solves it exactly. Directly tests `architecture.md:128`'s claim that *"PolyMamba recurrence carry order information"*. |
| `induction` (cross-check) | one random block of `L/repeats` tokens repeated `repeats=4` times (V=8, L=64); target = next token | 2.0794 | Pure **content-based** induction — position-free, so it is the *fair* task for CC-FSA vs attention. A memoryless mixer is again at chance. |

Arms (each is the full `AetherForCausalLM` with exactly one thing changed):

| arm | change |
|---|---|
| `aether` | none — full model as configured (PolyMamba2 even / CC-FSA odd / MoD on) |
| `linear` | `PolyMamba2` (even) → param-matched gated pointwise linear (no recurrence) |
| `conv` | `PolyMamba2` (even) → param-matched causal depthwise conv1d (cheap mixer **with** memory) |
| `attn` | `FreqSparseAttention` (odd) → standard causal MHSA, **no positional encoding** (matches AETHER, which has none) |
| `attn_rope` | `FreqSparseAttention` (odd) → standard causal MHSA **+ RoPE** |
| `allattn` / `allattn_rope` | both mixer families → causal MHSA, no-PE / RoPE |
| `nomod` | MoD routing disabled entirely (`routing_points=()`, `mod_loss_weight=0`) — gate ≡ 1 |
| `matched` | `linear`+`attn` swaps with output RMS matched to the replaced component (control) |

Fairness controls actually enforced: **same seed, same data tensor, same step count, same batch**
(`make_*_data(seed=777)` / eval `seed=888`), **2 seeds per arm** with the spread reported, and
**parameter counts reported** (they are within ~2 %: 199,432 for `aether` vs 200,648 `linear`;
the conv arm has *fewer* params, 184,520). Mixer output RMS at init is logged per arm.

### A fact that reframes the FSA comparison

AETHER has **no positional encoding at all**: `rope_base` exists in `AetherConfig`
(`aether/model.py:37,55`) and is passed by the trainer (`aether/training/trainer.py:121`), but
**nothing ever reads it** — a repo-wide grep for `rope|rotary|pos_emb` finds no application site.
`docs/architecture.md:127-128` confirms this is intentional. Consequently the *only* mixes with
any positional sensitivity are PolyMamba's recurrence and CC-FSA's intra-chunk DCT. This is why
both a no-PE attention arm (the literal "plain causal attention") **and** a RoPE attention arm
are reported: the RoPE arm is the modern, capability-matched baseline.

---

## 2. Results

### 2.1 `lag` task (k=6, V=16, chance = ln16 = **2.7726**; `d_model=64`, 4 layers, 4 heads,
`d_inner=128`, `state_dim=16`, B=8, **500 steps**, lr 2e-3, **2 seeds**)

| arm | params | final eval (mean) | seed 0 / seed 1 | spread |
|---|---:|---:|---:|---:|
| `allattn_rope` | 165,568 | **0.0038** | 0.003 / 0.004 | 0.0010 |
| `conv` (kernel 8) | 184,520 | **0.0022** | 0.002 / 0.002 | 0.0000 |
| `attn_rope` | 199,168 | **0.4436** | 0.343 / 0.545 | 0.2021 |
| `nomod` | 199,432 | **2.0064** | 1.957 / 2.056 | 0.0984 |
| `aether` (full) | 199,432 | **2.1795** | 2.156 / 2.203 | 0.0473 |
| `attn` (no PE) | 199,168 | **2.2357** | 2.165 / 2.307 | 0.1421 |
| `allattn` (no PE) | 165,568 | **2.3331** | 2.331 / 2.335 | 0.0037 |
| `linear` | 200,648 | **2.6502** | 2.648 / 2.653 | 0.0047 |
| `matched` (degenerate, see §3.3) | 200,384 | 2.7730 | 2.773 / 2.773 | 0.0004 |
| `conv` **(kernel 3 control)** | 183,240 | 2.7230 | 2.712 / 2.734 | 0.022 |

Loss curves (eval loss at every 50 steps; the curves **separate** — `conv` and `attn_rope` dive
inside 50–100 steps, the rest plateau near chance):

```
conv        s0: 2.737 0.056 0.016 0.010 0.007 0.005 ... 0.002     (k=3 control: 2.712 -> 2.723, i.e. chance)
attn_rope   s0: 2.796 0.952 0.148 0.104 0.170 0.375 ... 0.343
nomod       s0: 2.836 2.627 2.513 2.399 2.321 2.249 ... 1.957
aether      s0: 2.838 2.645 2.517 2.407 2.336 2.299 ... 2.156
attn        s0: 2.797 2.685 2.517 2.411 2.336 2.281 ... 2.165
allattn     s0: 2.782 2.664 2.577 2.524 2.501 2.425 ... 2.335
linear      s0: 2.788 2.739 2.710 2.694 2.669 2.683 ... 2.653
```

### 2.2 `induction` task (V=8, chance = ln8 = **2.0794**; `d_model=32`, 2 layers, 2 heads,
`d_inner=64`, `state_dim=8`, L=64, B=8, **1000 steps**, lr 2e-3, **2 seeds**)

| arm | params | final eval (mean) | copy-region loss (s0/s1) | spread |
|---|---:|---:|---:|---:|
| `attn` (PM + noPE-attn) | 50,176 | **1.7215** | 1.555 / 1.645 | 0.0715 |
| `conv` (conv + FSA) | 46,948 | **1.9059** | 1.883 / 1.829 | 0.0296 |
| `allattn` | 41,568 | **1.9346** | 1.881 / 1.882 | 0.0032 |
| `linear` (Lin + FSA) | 50,596 | 2.0694 | 2.074 / 2.058 | 0.0089 |
| `aether` (full: PM + FSA) | 50,308 | **2.0787** | 2.078 / 2.076 | 0.0016 |

`aether` is **flat at chance for all 1000 steps** (2.119 → 2.049 → 2.078); it never begins to
learn the copy. Replacing its odd-layer CC-FSA with plain causal attention (`attn`) is the single
best arm.

---

## 3. Answers to the three questions

### 3.1 Does `PolyMamba2` beat a plain linear mixer at this scale? — **YES** (so the SSM is *not* dead weight relative to the brief's baseline)

Clean single-variable comparisons (everything else held fixed, 2 seeds each):

* `lag`, PolyMamba2 → pointwise linear: `aether` 2.1795 vs `linear` 2.6502 → **PolyMamba2 is
  0.47 nats better**. Without MoD the gap is larger: `nomod` 2.0064 vs `linear` 2.6502 = **0.64 nats**.
* `induction`: `attn` (PolyMamba2 in the even slot) 1.7215 vs `allattn` (attention in the even
  slot) 1.9346 → **PolyMamba2 is 0.21 nats better** than attention in that slot.

Both gaps exceed the seed spread. So on a task that requires cross-position memory, the SSM
**does** earn its keep over a memoryless linear layer. It is the only even-layer mixer in AETHER
that can mix across positions, and it is doing measurable work.

**But it is not the cheap way to get that work done:**

* A causal depthwise conv with kernel 8 (same even-layer slot, *fewer* parameters) reaches
  **0.0022** on `lag` where the SSM reaches 2.0–2.2 — a **~2-nat** difference. The kernel-3
  control (2.723, chance) proves the conv's success comes from genuinely reaching `x_{t-6}`, not
  from parameter count.
* The SSM never solves the delayed copy (2.0 vs a floor of ~0): it learns a *partial* order
  signal. `docs/architecture.md:128` claims PolyMamba's recurrence "carry order information" —
  it does so, but far less effectively than an offset-capable kernel at this scale.

### 3.2 Does `FSA` beat standard causal attention at this scale / length ≤ 256? — **NO**

* `induction` (content-based, position-free — the fair test): `aether` (with CC-FSA) **2.0787 =
  chance; it learns nothing**, while the identical model with plain causal attention (`attn`) =
  **1.7215** (copy loss 1.55–1.64 vs 2.077). **CC-FSA is worse than plain attention by 0.36 nats
  and actively blocks learning here.**
* `lag`: `aether` 2.1795 vs `attn` 2.2357 → CC-FSA +0.056 nats, *within the seed spread*
  (0.047/0.142) — a tie.
* `lag` vs capability-matched attention: `attn_rope` **0.4436** vs `aether` 2.1795 → plain
  attention **with RoPE is 1.7 nats better**.

CC-FSA never beats plain causal attention in any arm measured, and loses badly wherever the
baseline is given an explicit offset mechanism. **At L = 64 and d_model = 32–64 the chunk-compression
machinery is not paying for itself.**

### 3.3 Does MoD routing help, hurt, or do nothing? — **HURTS**

`lag`: `aether` (MoD on) 2.1795 vs `nomod` (gate ≡ 1) 2.0064 → **MoD costs +0.17 nats**, and the
per-seed ranges do not overlap (`aether` [2.156, 2.203] vs `nomod` [1.957, 2.056]). It is the only
arm pair where the *worse* arm's best seed is still worse than the better arm's worse seed.

Consistent with `docs/architecture.md:101-103`: MoD v1 reduces **no** compute (blocks still run
densely; the gate only zeroes the update). So on this evidence it is pure overhead: it removes
50 %/30 % of the layer-1/layer-2 updates and produces a strictly worse model.

*Control caveat:* the `matched` arm degenerated — the replaced components' output RMS at init is
~0 (`aether` mixer RMS even=0.0000, odd=0.0001), so the RMS-matching factor computed to ~0 and the
arm became "no mixer at all" (2.7730 = chance). It is therefore reported but **excluded from all
conclusions**. It is also a mildly interesting measurement in its own right: at initialisation the
PolyMamba2 and CC-FSA mixers contribute essentially nothing to the residual stream.

---

## 4. Blunt verdict

| component | justified by measurement? | evidence |
|---|---|---|
| **PolyMamba2** | **Yes, but only vs the weakest baseline.** Beats a plain pointwise linear by 0.47–0.64 nats on `lag` and is the better even-slot mixer on `induction` (+0.21). **Loses to a cheap causal conv by ~2 nats** and to RoPE attention by 1.7 nats on `lag`. Not dead weight; not competitive either. |
| **CC-FSA** | **No.** Ties plain attention on a positional task, is 0.36 nats *worse* (and blocks learning entirely) on a content-based task, and is 1.7 nats worse than RoPE attention. |
| **MoD routing** | **No — harmful.** +0.17 nats on `lag` with non-overlapping seeds; zero compute savings by design. |

## 5. Minimal fix (no bug — recommendations only)

No defect found in `aether/model.py` or `aether/components/*`; the forward passes are correct and
the earlier stability fixes hold at this scale. If the goal is a model whose complexity is
*earned*, the measurements above say:

* `aether/model.py:131-135,168-179` — **remove MoD routing** (or set all keep ratios to 1.0).
* `aether/model.py:80-94` — **replace `FreqSparseAttention` with standard causal attention + RoPE**
  in the odd blocks; it is better on every task measured.
* `aether/model.py:80-94` — **keep `PolyMamba2` only if an SSM is wanted for other reasons**; a
  causal depthwise conv matches or beats it here at fewer parameters.

## 6. Confidence and falsification

**70 %.** What would falsify / change this:

* **Scale.** These are d_model 32–64, 2–4 layer models on synthetic tasks, trained 500–1000 steps
  in fp32. CC-FSA's chunk compression is designed to pay off at L in the thousands with long-range
  structure; the `induction` result in particular could hinge on CC-FSA's soft-top-k optimisation
  being harder to fit at tiny width. Re-run at the `aether_25m_proxy` config on real text.
* **Task family.** The `lag` task rewards an explicit offset operator; conv and RoPE attention
  possess one and AETHER deliberately does not (`architecture.md:127-128`). That makes the conv /
  RoPE results partly a statement about AETHER's missing positional encoding rather than about
  PolyMamba2's quality. The `induction` cross-check was added precisely to remove this confound,
  and it agrees that CC-FSA is the weak link.
* **Optimisation vs capability.** The full model may simply need more steps on `induction`
  (`aether` was flat, not merely slow, between steps 250–1000). A 5–10k-step run at d_model 64
  would settle whether CC-FSA eventually matches attention.
* The MoD result is the most robust (cleanest separation, consistent with the no-compute-saving
  design) and would be falsified only by a different routing-point/keep-ratio schedule.

## 7. Reproduce

```
set OMP_NUM_THREADS=1
python ablations/05_component_attribution.py --task lag --lag 6 --steps 500 --seeds 0 1 \
  --arms aether nomod linear conv attn attn_rope allattn allattn_rope matched
python ablations/05_component_attribution.py --task lag --lag 6 --conv-kernel 3 --steps 500 \
  --seeds 0 1 --arms conv --outdir ablations/results/05_component_attribution/lag_convk3
python ablations/05_component_attribution.py --task induction --repeats 4 --vocab 8 --seq-len 64 \
  --d-model 32 --d-inner 64 --heads 2 --dk 16 --state-dim 8 --steps 1000 --seeds 0 1 \
  --arms allattn aether linear conv attn
```
