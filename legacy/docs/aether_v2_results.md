# Aether component study and the v2 architecture (CPU lab results)

Every number here comes from `ablations/lab.py`: ~6.9M-parameter 8-layer models, 800 steps
x 16 x 256 bytes of Python source, bf16, same data order for every arm, held-out bits per
byte (lower is better). Seed-to-seed SD within a cell is about 0.007-0.01 bpb. Raw results
are in `ablations/results/` (06 = v1 factorial, 07 = v2 factorial, 08 = layer patterns,
10 = LFM3-style comparison, 11 = after the SSM init fix, 12 = copy probe).

These are small, short runs. They rank architectures reliably at this scale; they do not
prove anything at 1B parameters. That is what the GPU rehearsal (`docs/vast_rehearsal.md`) is for.

## 1. Original Aether (v1) components, 2^3 factorial

Factors: F = CC-FSA instead of attention in odd layers, P = PolyMamba-2 instead of attention
in even layers, M = MoD-GDR router. Metric: strictly causal (prefix-only) bpb.

| cell | bpb | vs transformer |
|---|---|---|
| transformer | 2.161 | |
| +P (PolyMamba-2) | 2.123 | -0.038 (but 22% more parameters) |
| +F (CC-FSA) | 2.227 | +0.066 |
| +M (MoD-GDR) | 2.401 | +0.240 |
| F+P+M (= Aether v1) | 2.413 | +0.252 |

* No v1 component helped by more than noise, and F and M hurt. F and P interfere
  (adding F on top of P: +0.31).
* **MoD-GDR is not causal**: its sequence-level top-k lets future tokens change past
  logits by up to 3.2 (measured). Its training loss is optimistic; generation is worse.
* Root causes found in code: PolyMamba-2 is a time-invariant SSM with a 16-dim state shared
  by all channels and a 0.95 decay cap (memory ~20 tokens); CC-FSA zeroes the DC bin by
  mean-centring before the DCT, uses no positional encoding and a fixed gate between paths.

## 2. Aether v2 components (each fixes a v1 defect), 2^3 factorial

F2 = chunk-compressed attention (`fsa2`), P2 = selective SSM (Mamba-2 / SSD form, `pm2`),
M2 = causal per-token depth router (`mod2`). All three verified causal (leak = 0).

| effect | bpb |
|---|---|
| P2 main effect | **-0.477** |
| F2 main effect | +0.061 |
| M2 main effect | +0.090 |
| add F2 on top of P2 | -0.020 (synergy: F2 hurts alone, helps with P2) |
| add M2 on top of P2 | -0.012 (neutral) |

The selective SSM is the component that matters. The router gives nothing and needs
data-dependent control flow, which NPUs handle badly, so v2 drops it.

## 3. Layer patterns (after the SSM init fix), 2 seeds each

| 8-layer pattern | bpb (mean) | CPU tok/s |
|---|---|---|
| transformer (all attention) | 2.038 | 3,250 |
| LFM3-style: conv, conv, SWA, attn (x2) | 1.820 | 3,600 |
| all SSM | 1.753 (1 seed, pre-fix) | 2,060 |
| SSM, SSM, SSM, fsa2 (x2) | 1.678 | 2,150 |
| SSM, SSM, SSM, attn (x2) | 1.660 | 2,190 |
| conv, SSM, conv, fsa2 (x2) | 1.680 | 2,660-3,050 |
| **conv, SSM, conv, attn (x2)** | **1.677** | **3,050** |
| LFM3 layout with our SSM instead of conv | 1.683 (1 seed) | 2,470 |

* Any pattern with SSMs plus one attention-type layer in four beats the transformer by
  ~0.36 bpb (18%) and LFM3-style by ~0.14 bpb (8%), far outside seed noise.
* Half the SSM layers can be cheap gated short convs with no quality loss and ~1.4x speed.
* Same-slot comparison: SSM beats short conv by 0.10 bpb (attn/pm2 1.73 vs attn/conv 1.83).
  What beats LFM3 is the SSM, not the layout.
* Compressed vs full attention in the 1-in-4 slot: within noise on bpb (-0.018 and -0.003
  for full), but full attention wins decisively on exact retrieval (section 5) and is the
  better-supported op on an NPU.
* All-SSM is worse than SSM + some attention.

## 4. Fixes made along the way

* **SSM step-0 gradient spike (88 vs 3 for attention)**: the shared init gave the SSM input
  projection std 0.02/sqrt(d), so the output gate was ~0 and the gated RMSNorm amplified its
  gradient. Mamba-style init (1/sqrt(3 d_in)) cut it to 18 and improved bpb by ~0.02-0.07.
* **Strict-metric "gap"**: the prefix-only metric sampled different positions from the full
  metric. Now both are reported at the same positions; for v2 they match exactly.
* **Lost results**: parallel jobs overwrote a shared JSON. Writes are locked now and the
  analyzer also reads every RESULT line from the logs.

## 5. In-context retrieval (copy task)

The language-model runs are too short for any model, attention included, to learn copying
(copy gain -0.14 to -0.16 everywhere), and MQAR does not train within the CPU budget. So
retrieval was tested directly: `ablations/recall.py --task copy` trains 4-layer models to
reproduce a span of 128 random tokens after seeing it once (1000 steps; results/14_copy).

| 4-layer design | exact-copy accuracy |
|---|---|
| attention x4 | 100% |
| **conv, SSM, conv, attention (v2)** | **100% / 100% (seed 0 / 1)** |
| LFM3-style: conv, conv, SWA, attention | 3.6% / 15.7% (seed 0 / 1) |
| conv, SSM, conv, compressed attention | 3.4% / 1.7% (seed 0 / 1) |
| SSM, SSM, SSM, compressed attention | 0.8% |
| SSM x4 | 0.4% |
| conv x4 (control) | 0.4% |

* The v2 design keeps attention's exact retrieval: one full-attention layer in four suffices.
* Compressed attention cannot copy exactly from beyond its raw window (summaries are lossy),
  and SSMs alone cannot store 128 random tokens. This is why v2 uses full attention in the
  1-in-4 slot, not fsa2, even though the two tie on bits per byte.
* The LFM3-style stack also has one full-attention layer but did not learn the task in 1000
  steps (3.6% and 15.7% over two seeds).

## 6. Scale-up check (2.2x parameters, 4x steps), 3 seeds

d_model 384, SwiGLU 1152, 8 layers (~15.4M params), 3200 steps (results/13_scale).
bpb on all positions of 96 held-out windows (~24k predictions per number):

| 8-layer design | seed 0 | seed 1 | seed 2 | mean | copy gain (s0/s1/s2) |
|---|---|---|---|---|---|
| transformer | 1.601 | 1.365 | 1.430 | 1.466 | -0.20 / +0.48 / +0.22 |
| LFM3-style | 1.477 | 1.445 | 1.430 | 1.451 | -0.12 / -0.05 / -0.01 |
| **conv, SSM, conv, attn (v2)** | **1.305** | **1.333** | **1.339** | **1.325** | +0.19 / +0.05 / +0.16 |

* v2 beats the transformer on all three seeds, by 0.30, 0.03 and 0.09 bpb (mean -0.14,
  -9.6%), and beats LFM3-style on all three (mean -0.13).
* The margin over the transformer depends on when the transformer forms induction heads
  (in-context copying): a sudden drop in its curve, at step ~1600 on seed 1, ~2400 on seed 2,
  not by 3200 on seed 0. v2 learns copying smoothly and early; its three seeds lie within
  0.034 bpb, the transformer's within 0.24. With 3 paired seeds the direction is consistent
  but the size of the gap is uncertain (paired t ~ 1.7).
* The strict prefix-only metric (1.505/1.442/1.448 for v2 vs 1.627/1.453/1.449) samples only
  576 positions per run, so it is too noisy to rank these runs; it exists to detect leaks,
  and the leak is exactly 0 for every model here.
* Reading: v2 reaches a given quality sooner and more reliably; whether a transformer
  catches up with much longer training is what the 6400-step runs (queued) and the 0.54B
  rehearsal test. At 0.54B and 1e8+ tokens both will have induction heads early, so the
  rehearsal compares them after that transition.
* Max gradient norm: transformer 83 / 16 / 20, v2 17 / 18 / 19, LFM3 13 / 14 / 13.

**Longer training (results/15_long): 6400 steps, seed 0, same 15M models.**

| step | 1600 | 3200 | 4000 | 4800 | 5600 | 6400 |
|---|---|---|---|---|---|---|
| transformer | 1.798 | 1.602 | 1.482 | 1.390 | 1.354 | **1.337** |
| v2 | 1.527 | 1.305 | 1.257 | 1.223 | 1.220 | **1.211** |
| gap | -0.27 | -0.30 | -0.23 | -0.17 | -0.13 | **-0.13** |

* With twice the training, the transformer forms induction heads (copy gain +0.25) and the
  gap narrows, but it levels off at -0.13 bpb (-9.4%) over the last 1600 steps rather than
  closing. v2 still copies better (copy gain +0.38) and also leads on the strict metric
  (1.379 vs 1.452).
* Single seed. Together with the three 3200-step seeds, v2 led in every run (4 of 4).
* **Seed 1 reverses it.** 6400 steps, seed 1:

  | step | 1600 | 3200 | 4000 | 4800 | 5600 | 6400 |
  |---|---|---|---|---|---|---|
  | transformer | 1.597 | 1.380 | 1.302 | 1.261 | 1.245 | **1.233** |
  | v2 | 1.597 | 1.367 | 1.309 | 1.296 | 1.266 | 1.273 |

  The transformer formed induction heads early on this seed, passed v2 at step 4000 and
  finished 0.04 bpb ahead, with a much stronger copy gain (+0.64 vs +0.22) and a better strict
  metric (1.403 vs 1.435). v2's val_bpb rose over the last 400 steps (1.263 -> 1.273) while its
  train_bpb kept falling (0.81 vs transformer 0.95): it is overfitting the ~9 MB stdlib corpus,
  which 6400 steps covers ~3 times.
* Long-run score: 1 win, 1 loss. The 3200-step wins (3 of 3) are real but are an early-training
  advantage; they do not show v2 is better at convergence.
* Two explanations, tested in results/20_bigdata: (1) data repetition, via `--data big`
  (~109 MB deduplicated, ~0.24 epochs at 6400 steps); (2) too little attention (2 of 8 layers)
  for retrieval once the transformer's induction heads form, via two variants with 3 and 4
  attention layers.

**Larger model (results/19_d512): d_model 512, d_inner 1536, ~27M params, 3200 steps, seed 0.**

| design | val_bpb | strict causal | copy gain | max grad norm |
|---|---|---|---|---|
| transformer | 1.621 | 1.673 | -0.19 | 33.5 |
| v2 | **1.327** | **1.515** | +0.09 | 21.7 |

* -0.29 bpb (-18%). At this budget the larger transformer has not formed induction heads yet
  (negative copy gain), so this is mostly the same early-training advantage as at 15M, not
  proof that the gap holds at convergence. Single seed.
* Caveat seen in every long run: v2's train/val gap is larger (here train 0.98 vs val 1.33;
  transformer 1.35 vs 1.62). Validation is held-out files, so v2 fits the training files
  faster and some of that does not transfer. The corpus is ~21.6 MB and 6400 steps is ~1.2
  epochs, so with real pretraining-scale data this matters less, but the rehearsal must
  compare held-out loss, not train loss.

**No data repetition (results/20_bigdata): `--data big`, ~109 MB deduplicated Python
(stdlib + installed packages), 6400 steps = ~0.24 epochs, d384, seed 0.**

| design | attn layers | 1600 | 3200 | 4800 | 6400 | strict causal | copy gain |
|---|---|---|---|---|---|---|---|
| transformer | 8 | 1.763 | 1.404 | 1.264 | 1.209 | 1.123 | +0.80 |
| v2 (conv-pm2-conv-attn x2) | 2 | **1.539** | **1.283** | **1.161** | **1.099** | 1.075 | +0.69 |
| v2, last two attn | 3 | 1.524 | 1.286 | 1.163 | 1.109 | **1.062** | +0.66 |
| v2, alternating attn | 4 | 1.552 | 1.292 | 1.162 | 1.106 | 1.072 | +0.65 |

* v2 beats the transformer by 0.110 bpb (-9.1%) at 6400 steps, and the gap is flat over the
  second half (-0.12, -0.10, -0.11). This time the transformer has formed induction heads
  (copy gain +0.80, its strongest of any run) and still loses on both metrics.
* Train and val stay within 0.03 for every design: the overfitting seen on the 9 MB corpus is
  gone. So the seed-1 reversal above is best explained by data repetition, which v2 fits
  faster; real pretraining does not repeat data.
* More attention does not help: 3 or 4 attention layers are within 0.01 of v2's 2. Keep
  2 of 8, which also keeps the KV cache at 1/4 of the transformer's.
* **Seed 1 (the seed that reversed on the small corpus) confirms it:**

  | seed 1 | 1600 | 3200 | 4800 | 6400 | strict causal | copy gain |
  |---|---|---|---|---|---|---|
  | transformer | 1.743 | 1.425 | 1.297 | 1.233 | 1.095 | +0.84 |
  | v2 | **1.520** | **1.263** | **1.160** | **1.103** | **1.003** | +0.80 |

  -0.130 bpb (-10.5%), with near-equal copy gain. Over both seeds on non-repeating data, v2
  wins 2 of 2 at 6400 steps by 0.11-0.13 bpb, at 1/4 of the transformer's KV cache.

**Component ablations in the trustworthy setting (results/21_components_big): big corpus,
6400 steps, d384, seed 0, ~15.4M params each. Each row swaps one component of v2.**

| design (x2) | change vs v2 | val_bpb | vs v2 | strict causal | copy gain | train tok/s |
|---|---|---|---|---|---|---|
| ssm-ssm-ssm-attn | conv -> SSM | **1.096** | -0.002 | **1.054** | +0.74 | 1472 |
| **v2: conv-ssm-conv-attn** | - | 1.099 | 0 | 1.075 | +0.69 | 2111 |
| attn-ssm-attn-attn | conv -> attn | 1.114 | +0.015 | 1.107 | +0.63 | 1833 |
| conv-conv-conv-attn | SSM -> conv | 1.151 | +0.052 | 1.093 | +0.62 | 2401 |
| transformer | all attn | 1.209 | +0.110 | 1.123 | +0.80 | 2225 |
| conv-ssm-conv-ssm | attn -> SSM | 1.220 | +0.121 | 1.192 | +0.14 | 1749 |

What each component contributes, and how they interact:
* **Attention is essential, but 2 of 8 layers is enough.** Removing it is the worst change
  (+0.121, copy gain collapses to +0.14): recurrent layers alone cannot retrieve. Adding more
  (3, 4 or 6 of 8, above) never helps.
* **The SSM is the main source of the gain over a transformer.** Replacing it with conv costs
  +0.052; conv + attention alone already beats the transformer by 0.06, and adding the SSM
  takes it to 0.11.
* **Conv and SSM are complementary with attention, and conv is the cheap filler.** In the
  conv slots, SSM ties (-0.002, within seed noise of ~0.01) and attention is worse (+0.015).
  Conv gives the same quality as the SSM there at 1.4x the training throughput, a smaller
  state and the simplest NPU kernel, so v2 keeps conv.
* ssm3_attn is the quality-equivalent alternative (slightly better strict metric and copy
  gain); `scripts/export_onnx.py --pattern ssm3_attn` already exports it. Choose it only if
  the Vast rehearsal shows a gap at 0.54B.

**Second seed for the two closest alternatives (results/21_components_big).**

| design (x2) | seed 0 | seed 1 | mean | vs v2 mean | strict (s0/s1) | copy gain (s0/s1) | train tok/s |
|---|---|---|---|---|---|---|---|
| ssm-ssm-ssm-attn | 1.096 | 1.093 | **1.094** | -0.006 | 1.054 / 1.038 | +0.74 / +0.73 | ~1490 |
| conv-ssm-ssm-attn | 1.102 | 1.096 | 1.099 | -0.002 | 1.058 / 1.014 | +0.71 / +0.73 | ~1715 |
| v2: conv-ssm-conv-attn | 1.099 | 1.103 | 1.101 | 0 | 1.075 / 1.003 | +0.69 / +0.80 | ~2110 |
| transformer | 1.209 | 1.233 | 1.221 | +0.120 | 1.123 / 1.095 | +0.80 / +0.84 | ~2120 |

* All three hybrids are within 0.01 of each other, against a 0.12 gap to the transformer.
  ssm3_attn is ahead on both seeds but by less than seed-to-seed noise (0.002 and 0.010),
  the strict metric splits 1-1 with v2, and it trains ~30% slower here.
* Decision rule set before these runs: switch the default only for >0.01 on both seeds. Not
  met, so v2 stays the default. ssm3_attn is the optional third arm of the Vast rehearsal
  (`docs/vast_rehearsal.md`), where 0.54B decides it.

## 7. Deployment check at the real size (0.54B, static-shape ONNX)

`scripts/export_onnx.py` exports one decode step with every shape fixed, as the Snapdragon
NPU runtime (QNN through ONNX Runtime) requires: conv/SSM layers carry their constant-size
state, attention layers a 2048-slot KV buffer written by mask. Untrained weights, ONNX
Runtime on 1 x86 core with other jobs running, so only the ratio matters.

| 0.54B student | ONNX matches PyTorch | dynamic dims | decode state (fp32, 2k ctx) | ms per token |
|---|---|---|---|---|
| vanilla transformer (28 attn) | 1e-5 | 0 | 470 MB | 729 |
| **v2 (14 conv, 7 SSM, 7 attn)** | 5e-6 | 0 | **120 MB** | **254** |
| ssm3_attn backup (21 SSM, 7 attn) | 7e-6 | 0 | 125 MB | 226 |

* ssm3_attn (the optional Vast arm) costs the same on device: same 7 attention layers, so
  the same KV cache; SSM state adds 5 MB over v2's conv state; latency within run-to-run
  noise of v2 (measured later, idle machine). Switching to it is free for deployment if the
  0.54B run favours it.

* Both export cleanly with only standard ops (MatMul, Mul/Add, Softmax, Where, Sigmoid,
  Exp, Softplus, ReduceMean/Sum); no Einsum, loops or dynamic shapes.
* The 2.9x latency gap here is mostly the cost of moving the KV state in and out each step;
  on compute alone (per-token multiply-adds) v2 saves 13% at 2k context, 35% at 8k and 58%
  at 32k, because only 7 of 28 layers read a KV cache.
* KV memory per token of context: 229 KB (transformer) vs 57 KB (v2), fp32. At 32k context
  that is 7.5 GB vs 1.9 GB (halve both for fp16), which decides what fits next to the
  vision and audio models on a 16-32 GB board.
* Not tested here: INT8 accuracy on trained weights and the QNN backend itself (needs the
  Q8B and Qualcomm's SDK).

## 8. Quantization sensitivity (results/16_quant)

Lab models (7M, 800 steps, seed 0), quantized after training, fp32 eval. Increase in bpb:

| design | fp16 (incl. SSM state) | int8 weights | int4 weights, group 32 | int8 weights + int8 activations |
|---|---|---|---|---|
| transformer (2.056) | +0.000 | +0.000 | +0.004 | +0.006 |
| **v2 (1.686)** | +0.000 | +0.000 | +0.005 | +0.006 |
| LFM3-style (1.817) | +0.000 | +0.000 | +0.010 | +0.005 |

* v2 is as robust to quantization as the transformer. Its quality lead survives every
  format, including the NPU setting (static per-tensor int8 activations) and 4-bit weights.
* Not covered: the SSM's internal recurrence (decay, state update) in int8/int16 rather than
  fp16; that needs the QNN toolchain on the board.

## 9. Length generalization (results/17_longctx)

Lab models (7M, 800 steps; seeds 0 and 1, LFM3 seed 0 only) trained on 256-byte windows only, then scored on
2048-byte held-out windows. bpb by position:

| design | 0-256 (trained) | 256-512 | 512-1024 | 1024-2048 |
|---|---|---|---|---|
| transformer, seed 0 / 1 | 2.015 / 1.966 | 2.774 / 2.684 | 3.375 / 3.448 | 3.845 / 3.847 |
| **v2, seed 0 / 1** | **1.577 / 1.605** | **1.553 / 1.596** | **1.488 / 1.516** | **1.567 / 1.580** |
| LFM3-style | 1.726 | 1.684 | 1.596 | 1.682 |

* The transformer breaks immediately past its training length (bpb nearly doubles). v2
  stays at or below its in-length bpb out to 8x the training length; so does LFM3-style.
* Reason: only 1 layer in 4 uses RoPE attention; the conv/SSM layers carry position and
  recent context without positional encodings, so unseen RoPE angles do less damage.
* Caveat: transformers are usually fixed for this with RoPE scaling (YaRN, position
  interpolation) or long-context fine-tuning, so this is a robustness advantage, not proof
  that v2 uses long context better. The slight rise at 1024-2048 suggests v2's attention
  layers also degrade somewhat; long-context fine-tuning is still advisable.
* Exact retrieval does **not** extrapolate, for either design (results/18_copy_len): trained
  on the 128-token copy task (both 100%), then tested on 256 and 512-token spans, the
  transformer scores 0.5% / 0.4% and v2 0.5% / 0.4% (chance is 0.4%). v2's robustness is in
  average prediction quality, not in exact recall at unseen lengths. The deployed model
  must be trained (or fine-tuned) at the context length it will use.

## 10. What the CPU lab cannot answer
* **Scale.** 15M parameters and 3200 steps at most. Effects can shrink or flip at 0.5-1B.
* **Real speed.** The SSM scan is plain PyTorch; the NPU/GPU numbers need the target hardware.

## Recommended v2 for the rehearsal and the Snapdragon target

`conv, SSM, conv, attn` repeated (28 layers at 0.54B), no router: equal-best quality,
exact retrieval as good as a transformer, fastest of the SSM designs, constant-memory decode in 21 of 28 layers, and only standard
ops (matmul, depthwise conv, elementwise, softmax attention). Run in `scripts/distill.py`
with `--pattern hybrid_attn`.

## 11. Spatial grounding on CPU (results/22_spatial): no architecture signal yet

Synthetic scenes (64 px, encoder trained from scratch, CE only, no teacher), d256 x 8 layers,
3000 steps x 8 examples, lr 3e-4, loss on the first answer token (`--answer-tokens 1`).

| task (chance) | student | probe acc | final train CE |
|---|---|---|---|
| one square, left/right (0.50) | vanilla | **1.00** (800 steps) | 0.0002 |
| three shapes, 4 relations (0.25) | aether_v2 hybrid_attn | 0.25 | 1.388 |
| three shapes, 4 relations (0.25) | vanilla | 0.25 | 1.387 |
| three shapes, 4 relations (0.25) | aether_v2 ssm3_attn | 0.25 | 1.387 |

* The pipeline works (one square: 100%). Two earlier bugs made every run look like chance:
  the loss covered the whole answer, so the single deciding token was swamped (runs in
  `old_full_answer_loss/`), and the probe scored untrained tokens and answers outside the task.
* The three-shape task stayed at chance for **all three** designs: train CE = ln 4, i.e. each
  learned only the answer prior. Finding the two named objects and comparing their positions,
  with a vision encoder learned from scratch on 24k examples, is beyond this CPU budget. That
  is a limit of the test, not evidence for or against any language core.
* The comparison has to be made on Vast, where the encoder is pretrained (SigLIP 2) and the
  teacher's full distribution gives a signal at every token (`docs/vision_stage.md`).
