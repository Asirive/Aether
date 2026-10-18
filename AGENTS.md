# AETHER-200M — Agent Memory

This file is the single source of truth for agents (human or AI) working on this project. Read it first. Trust it. If reality contradicts it, update it — but document why.

**Picking the project back up? Read `docs/RESUME_NOTES.md` first** — it records the live run,
the exact resume command, the compute budget and the open items. Session-level findings are
in `docs/debug_report_2026-09-13.md` and AGENTS.md §PAF-8.

---

## 1. Mission

Train a **200M-parameter** language model on **AMD RX 7600 (8GB)** that is **competitive with SmolLM2-360M-Instruct** at 55% the parameters. Then post-train it (SFT + DPO). Release the code and a short paper.

**Timeline:** 1 week total (environment + pretrain + SFT + DPO + eval).

---

## 2. Non-Negotiable Decisions (do NOT re-litigate)

| Decision | Value | Reason |
|----------|-------|--------|
| Model size | **~187M params (d_model=576, d_inner=1536, n_layers=24, n_heads=9)** — measured 186,615,276 | Original 1B plan was infeasible on RX 7600 (5,000–15,000× compute gap vs SOTA). 200M is honest. Actual count derives from `vocab_size=128256` (Llama-3 has 128K tokens, not 32K as previously claimed). See PAF-4. |
| Components | **CC-FSA + PolyMamba-2 + MoD-GDR only (3)** | Original 6-component design was over-engineered. HPE was duplicate of Wiola/SRPE. TT-Mem was duplicate of Tensor Memory. AR-FFN overclaimed against Minima/LoRA-MoE. FSA was redesigned 31 Jul 2026 as Causal Chunk-Compressed FSA after a future-leak audit finding (see PAF-5). |
| GDR geometry | **Euclidean MoD top-k, residual gate** | Hyperbolic routing was overkill, "Möbius addressing" was a false claim (it was Euclidean softmax). Use Raposo et al. 2024 Mixture-of-Depths directly. Router returns a gate; caller applies `x = gate*block(x) + (1-gate)*x` over the block span (fixed 31 Jul 2026 zero-poisoning bug, PAF-5). |
| Backend | **ROCm HIP SDK 7.1.1 on Windows 11 native** | DirectML torch-directml is a dead project (last release Sep 2024, bf16 permanently rejected). WSL2 ROCm has no `/dev/kfd` for gfx1102. **HIP SDK 7.1.1 has official native Windows support for RX 7600 (gfx1102)** — bypasses WSL2 entirely. See `scripts/env_setup_windows.ps1`. |
| Tokenizer | **Llama-3 BPE 128K vocab (`unsloth/Llama-3.2-1B-Instruct` mirror)** | Standard, well-supported, mature. The original `meta-llama/Llama-3.2-1B-Instruct` is gated and not auto-downloadable; the unsloth mirror is byte-identical. See PAF-4. |
| PolyMamba-2 orders | **K=2 (not 4)** | K=4 at 200M blew the budget. K=2 still demonstrates the "poly" novelty (K=1 = Mamba baseline). |
| Sequence length | **L=1024 → L=2048 curriculum, L=2048 cap** | Curriculum avoids cold-start overhead. L=2048 cap for 8GB VRAM safety with the new FSA; L=4096 is a stretch goal only after memory is measured at L=2048. |
| Optimizer | **`AdamW(fused=True)` with fp32 master weights + bf16 autocast compute** (updated 13 Sep 2026, see PAF-8) | bnb ships CUDA/CPU wheels on Windows but **never** ROCm Windows wheels (confirmed from bnb release notes: ROCm builds are Linux x86-64 only since v0.48; bnb import crashes on this machine). See PAF-3. The original bf16-parameter/no-master-weight choice was measured to freeze most parameters; fp32 master weights cost ~8% throughput once the LM head is chunked. |
| Training compute | **100K steps × 65,536 tokens = 6.55B tokens** | 200K steps (13.1B) ≈ 11 days at the measured 29.48 TFLOPS — blows the 1-week budget once SFT/DPO/eval are included. 6.55B ≈ 5–6 days. Stretch: resume to 200K (13.1B) only if ahead of schedule. Effective batch is held constant across curriculum phases (grad_accum recomputed per step). |

---

## 3. Dropped Components (do NOT reintroduce)

- **HPE (Helical Position Encoding)** — duplicate of Wiola/SRPE (arXiv:2607.01394, July 2026). Use RoPE.
- **TT-Mem (Tensor-Train Memory)** — duplicate of Tensor Memory (arXiv:2605.27686, May 2026); math wrong (8192 cores ≠ scalar TT); "Möbius" claim was false. Use standard FFN.
- **AR-FFN (Adaptive-Rank FFN)** — overclaimed; Minima (Feb 2026) and LoRA-MoE in the prior 6 months cover the slot. Use SwiGLU.
- **Hyperbolic geometry anywhere** — no published benefit at 200M scale; adds engineering for no measurable gain.
- **MoE** — too complex for 1-week timeline and benefits don't kick in until >1B params.

---

## 4. Audit Findings (frozen)

The architecture went through a 10-agent audit. Critical findings to remember:

- **AR-FFN arithmetic was wrong** (rank options inconsistent with 4 GFLOPs claim). Don't trust any FLOP math without re-deriving from scratch.
- **Standard transformer baseline was 5–10× wrong** in the original plan. Always state units explicitly (per-seq vs per-token).
- **MoD exit decision is non-differentiable.** Must use straight-through estimator or soft top-k. Hard thresholding fails.
- **FSA cold-start collapses all heads to the same mask.** Per-head orthogonal pre-rotation at init is mandatory.
- **PolyMamba-2 A matrices can have spectral radius ≥ 1.** Must use Hurwitz-stable init: `A = I - softplus(dt)·M` with **dt log-uniform over [0.005, 0.1]** (updated 9 Aug 2026 — a single `dt=0.001` for all channels made A a near-integrator with radius 0.9999, amplifying gradients ~1000× and destabilizing training at LR ~1.5e-4; see B26). **Updated again (B27):** even with log-uniform dt, the order-1 tridiagonal channel's small min-eigenvalue let `dt` drift so ρ reached 0.9991 — the *backward* scan gradient amplifies `1/(1−ρ) ≈ 1100×`, permanently exploding grad norms in every full run. The forward now clamps `dt ≥ (1−max_radius)/min_eig(M)` with `max_radius=0.95`, capping ρ at 0.95 (backward amplification ≈ 20×). **Do NOT set `max_radius` above ~0.97** without re-checking gradient stability.
- **TT-Mem was not actually a scalar TT** (boundary ranks wrong). If anyone tries to resurrect it, the math must use `r_0 = r_K = 1` or it's a Tensor Ring.
- **TT-Mem bf16 numerical collapse** (K=8 sequential matmuls compound errors). Avoid even attempting on bf16.
- **GDR with hyperbolic distance has vanishing gradients near boundary.** Project anchors to `||p|| < 1 - 1e-3` and clamp arccosh argument.
- **DirectML bf16 permanently rejected** (Microsoft closed Issue #688 as "Not planned" in Jul 2025). Do not waste time on DirectML.

Full audit: `docs/math_audit.md`.

---

## 5. Active Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| HIP SDK install on Windows requires admin + reboot | High | Bundled AMD driver replaces existing Adrenalin. Verify `hipInfo` works after reboot and a bf16 matmul benchmark hits ≥8 TFLOPS before any code is written |
| MoD top-k routing collapse (all tokens take same path) | High | L_mbd with KL-to-uniform target, ramped over first 5% of steps |
| PolyMamba-2 SSM blow-up | High | Hurwitz-stable init, unit test that spectral radius < 1 at every step |
| FSA cold-start collapse | Medium | Per-head orthogonal pre-rotation at init; soft top-k during first 10% of steps |
| L=4096 OOM at long context | Low | 200M has huge headroom (~6 GB free); but gradient checkpointing + recompute attention scores in backward is mandatory |
| 30B tokens insufficient to match SmolLM2-360M | Medium | Honest stretch goal: 50B if time permits. Realistic outcome: close to but possibly under SmolLM2-360M. |
| Week-1 ROCm bugs eat days | High | Phase 0 is environment-only. If it doesn't work in 8 hours, switch to ZLUDA-on-Windows backup. |

---

## PAF — Platform Adjustment Factors

Adjustments forced by the actual shipped platform. Each is a deviation from the project's "ideal" plan, with the reason and the substitute.

### PAF-1 — DirectML closed, WSL2 ROCm has no KFD for gfx1102

- **Original plan:** DirectML as a "fallback" Python torch-directml backend.
- **Reality (MS Issue #688):** DirectML torch-directml last release Sep 2024; bf16 permanently rejected as "Not planned" Jul 2025.
- **Reality (WSL2 ROCm):** WSL2 Linux kernel does not expose `/dev/kfd` or `/dev/dri` for the gfx1102 (RDNA 3) GPU. WSL2 ROCm support is CDNA-only.
- **Substitute:** Native Windows install of **AMD ROCm HIP SDK 7.1.1** (installs as a regular Windows driver package, no WSL2, no Linux kernel). PyTorch for HIP is pulled from the TheRock unified wheel index at `https://rocm.nightlies.amd.com/whl-multi-arch/` with extra `device-gfx1102` — produces `torch 2.14.0a0+rocm7.15.0a20260722`. RX 7600 = `gfx1102`.
- **Required env vars (set by `scripts/benchmark.ps1`):**
  - `HSA_OVERRIDE_GFX_VERSION=11.0.0` — benign for gfx1102
  - `TORCH_DISABLE_CUDA_PROFILING=1`
  - `HIP_ENABLE_TRACING=0`
  - `HSA_ENABLE_PROFILING=0`
  - `CUDA_VISIBLE_DEVICES=1` — to skip integrated GPU (gfx1036)

### PAF-2 — HIP SDK 7.1.1 path validated, ≥29 TFLOPS bf16

- HIP SDK 7.1.1 has official native Windows support for gfx1102.
- Confirmed via `scripts/benchmark.ps1`: `29.48 TFLOPS` sustained bf16 matmul, well above the 8 TFLOPS exit criterion.

### PAF-4 — Llama-3 has 128K vocab (not 32K); Meta's tokenizer is gated

- **Original plan:** Llama-3 BPE 32K vocab from `meta-llama/Llama-3.2-1B-Instruct`.
- **Reality A (vocab):** Llama-3 BPE has `vocab_size=128256` (128K), not 32K. Confirmed by `AutoTokenizer.from_pretrained(...).vocab_size`. Old claim was wrong.
- **Reality B (gating):** `meta-llama/Llama-3.2-1B-Instruct` returns `401 GatedRepoError` for unauthenticated requests (verified Feb–Jul 2026).
- **Substitutes:**
  - Tokenizer: `unsloth/Llama-3.2-1B-Instruct` — un-gated mirror, byte-identical weights + tokenizer. Verified via `https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json` → 200.
  - Fallback: `Xenova/llama-3-tokenizer` — also un-gated, 128K vocab, BPE identical.
- **Architecture impact:** 128K embedding × d_model = 73M at d_model=576, 82M at d_model=640. This shrinks the available transformer budget from ~200M to ~110M. We picked **`d_model=576, d_inner=1536, n_layers=24, n_heads=9, mod_routing_points=(4,8,12,16,20)`** → 186.7M total (73M embedding, 113M transformer). See `configs/aether_200m.yaml`.
- **EOS handling:** Llama-3 EOS is `<|eot_id|>` id 128009 (or `<|end_of_text|>` id 128001). Old code hardcoded `EOS_TOKEN_ID = 2` which is GPT-2/Llama-1. Fixed in `aether/training/data.py` to read `tokenizer.eos_token_id` by default; explicit override still possible.

### PAF-3 — `bitsandbytes` ROCm Windows wheel does not exist; fall back to fused AdamW

- **Research method:** Read bnb `Release notes` from v0.46 to v0.49.2 (current 0.49.2, Feb 16 2026).
- **Findings from upstream release notes:**
  - v0.48.0 (30 Sep 2025): "ROCm: Add 6.4 and 7.0 builds" — **Linux x86-64 only**.
  - v0.49.0 (11 Dec 2025): "ROCm: Add build for ROCm 7.1" — **Linux x86-64 only**.
  - v0.49.2 (16 Feb 2026): "ROCm 7.2 build is now included" — **Linux x86-64 only**.
  - The Windows wheel (`win_amd64`) shipped in 0.49.2 contains: `libbitsandbytes_cpu.dll`, `libbitsandbytes_cuda{118,120,...130}.dll`, `libbitsandbytes_xpu.dll`. **No `libbitsandbytes_rocm*.dll`** for any ROCm version.
- **Implication:** bnb 8-bit AdamW is impossible on Windows + ROCm HIP SDK 7.1.1 + TheRock torch.
- **Substitute:** `torch.optim.AdamW(fused=True)` with **fp32 master weights** (superseded 13 Sep 2026 — see PAF-8; the earlier bf16-parameter/no-master-weight variant was measured to freeze `A_log`, RMSNorm, `mask_logits` and part of the embedding). Adam state (m, v) is created in the **param dtype**, so fp32 params give fp32 moments. Total optim state ≈ 2.2 GB (fp32 params + fp32 m+v); this fits only because the LM head is now chunked. **Do NOT "fix" this without re-checking the memory budget.**
- **Do NOT re-attempt** to install bnb ROCm Windows wheel. None will exist before AETHER-200M finishes.

### PAF-5 — 31 July 2026 full-system debug: FSA redesigned, MoD semantics fixed, dead losses revived

A full architecture/math/implementation audit (see `docs/debug_report_2026-07-31.md`) found and fixed four training-breaking bugs:

1. **FSA leaked future tokens.** DCT along the full sequence axis mixed future positions into every frequency bin; probe showed past logits changing by 1.281 when future tokens changed. **FSA is now Causal Chunk-Compressed FSA (CC-FSA):** chunk size 32, per-chunk DCT-II, per-head learned top-k=8 frequency bins as chunk summaries, strict chunk-level causal mask for the cross-chunk path, standard causal mask inside the current chunk, per-head scalar blend gate. Per-head pre-rotation matrices removed (random mask-logit init + ortho loss now handle symmetry breaking).
2. **MoD zero-poisoned unselected tokens.** Router multiplied `x * mask`; unselected tokens became 0 and stayed 0 (bias-free net, RMSNorm(0)=0), producing permanent `log(V)` loss and no gradient. **Router now returns x unchanged + a gate; the model applies `x = gate*block(x) + (1-gate)*x` over the block span.**
3. **PolyMamba `balance_loss` was identically 0** (computed on a dummy zero vector). Now computed from real per-token gates inside `forward` (entropy of the mean gate distribution), kept in the autograd graph via block tuple outputs.
4. **MoD balance loss was identically 0** (top-k count is exact by construction). Now MSE of mean soft usage per position vs target ratio.

Also fixed: grad_accum frozen at init (effective batch doubled at L=2048), aux annealing never wired to modules, bitsandbytes in pyproject (crashes on import here), `sft.py`/`dpo.sh` breakage (adamw_8bit, missing train_dataset, wrong config/paths), lm-eval couldn't load raw `.pt` (new `aether/utils/export_hf.py`), missing `scripts/benchmark.py`, `pretrain.sh` running an ablation, weight decay on `A_log`/`mask_logits`/`route_proj`/gate/embedding, spectral radius on GPU (now CPU fp32), token budget honesty (100K steps = 6.55B tokens). Tests: 97/97 in ~21 s (was 75/77 in 15 min).

### PAF-6 — 9 Aug 2026: label double-shift (the "loss plateau at ~7.5 + the-of-the-of canary" root cause)

`data.py` pre-shifted labels (`labels = c[1:]`) while `model.forward` shifted them again
(`logits[:-1]` vs `labels[1:]`). Training silently learned **P(t+2 | prefix)**, not
next-token — unit tests only exercised the HF convention (`labels = input_ids.clone()`) so
the 97-test suite could not catch it. Every full-size run showed the same signature: loss
dropping to unigram entropy (~7.5) then stalling, canary degenerating to "the of, the of"
(the 2-step transition matrix of English), eval stuck at 0–12%, grad norms exploding as LR
ramps.

**Fix:** `data.py` now returns `labels = input_ids.clone()`. Verified by the deterministic
successor-ring probe in `tests/test_label_alignment.py` (1-ahead accuracy 100%, 2-ahead 0%
after fix; ~0% 1-ahead before). **FSA ortho-loss weight raised 0.001 → 0.01** (heads were
converging on the same frequency bins; loss climbing 0.27 → 28.75). Full log in
`docs/debug_report_2026-07-31.md` §B23. Tests: 99/99 (~37 s).

**Do NOT re-introduce a pre-shift in any data loader.** The convention is: loader returns
`labels = input_ids`; the *only* shift lives in `model.forward` (`logits[:-1]` vs
`labels[1:]`). Guarded by `test_fineweb_stream_labels_match_hf_convention`.

### PAF-7 — 9 Aug 2026: reproducible LR-instability at ~1.5e-4 (the "breaks at step ~550" bug)

Even after the label fix (PAF-6), every full run broke at ~step 550-600 when LR crossed
~1.5-1.8e-4: grad norms spiked (up to 300), loss rose after the spike-guard, canary
degenerated. This occurred at LR peak 5e-4 AND 3e-4, with both objectives — a structural
LR ceiling, not a data/label problem.

**Two root causes (both fixed):**
1. PolyMamba `A` was a near-integrator (dt=0.001 everywhere -> radius 0.9999, memory
   ~10,000 steps) amplifying recurrence gradients ~1000×. **Fix:** log-uniform dt over
   [0.005, 0.1] (Mamba-style range); radius now 0.99882. FSA provides long-range context.
2. FSA heads collapsed to the same frequency bins (ortho_loss ~24.4) and the DC bin
   (= sqrt(32)*chunk-mean) inflated attention magnitudes. **Fix:** L2-normalize the
   compressed cross-chunk k/v along d_k.

**LR:** peak lowered to **1.2e-4** (below the ~1.5e-4 onset). Slower but stable.
**Diagnostic:** spike-guard logs top-5 param grads by norm (threshold 20) to pinpoint any
future instability. Full log in `docs/debug_report_2026-07-31.md` §B26.

### PAF-8 — 13 Sep 2026: dead grad-norm guard, guard death-spiral, frozen parameters, CPU-bound scan

Full audit in `docs/debug_report_2026-09-13.md`. Four independent defects, each measured:

1. **`grad_spike_threshold` was dead code** (assigned, never read). A 19,523-norm gradient
   was clipped to 1.0 and applied at step ~1700, destroying the model (loss 5.73 → 12.57,
   worse than random init). The guard now measures the pre-clip norm via
   `clip_grad_norm_(params, inf)` and skips above the threshold.
2. **Guard death-spiral.** The loss EMA only updated on applied steps, so after one bad
   update every later step compared against a stale EMA and was skipped forever (335
   consecutive skips, `grad_norm=0.000` in the log). The EMA now tracks slowly during skips,
   re-seeds after `max_consecutive_skips`, and arms a forced clipped update; after
   `wedged_skip_limit` skips the run stops loudly instead of spinning.
3. **bf16 optimizer state rounded updates to zero.** Adam's first step has `|dw| = lr`, so it
   probes bf16 rounding directly: with `lr=1.2e-4`, `A_log` changed 0.0 % of elements,
   RMSNorm 0.0 %, `mask_logits` 2.3 %, embedding 87.5 %. The learned SSM timescales and the
   learned FSA frequency bins never trained at all. **Fix: fp32 master weights + bf16
   autocast compute** (the SSM scan is wrapped in `autocast(enabled=False)` to stay fp32).
4. **The scan was CPU-launch-bound.** Hillis–Steele materialised `(bsz, seq_len, N, N)` and
   `torch.where` over it 10×: ~14,000 `as_strided` calls per micro-batch. Replaced with a
   vector-only doubling scan over the `(N, N)` powers only.
5. **The LM head materialised `(B, L, 128256)` logits** (~2 GB with the shift copy and its
   gradient), which made fp32 master weights spill into shared memory (735 tok/s).
   `_chunked_causal_loss` projects one chunk at a time — measured −0.92 GB, restoring
   **3,900–3,976 tok/s** in the real trainer.

**Budget correction:** 6.55 B tokens at 3,900 tok/s ≈ **19.4 days**, not 5–6. Plan accordingly.

### PAF-9 — 15 Sep 2026: THE root cause of the step-1300 collapse (found by 10 parallel investigators)

All four full runs died the same way: healthy to ~1200 steps (loss ~5.7, grad norm ~1.0),
then a **slow** grad-norm climb with no schedule change (1.0 → 9.5 @1300 → 23 → 46 @1400),
settling at ~13–14 with every step skipped forever. The guard was a patch, not a fix.

**Root cause: the MoD router read the un-normalised residual stream.**

`MoDRouter.forward` did `scores = route_proj(x)` on the **raw** residual — the only `Linear`
in the model that did not read an RMSNorm'd input — and fed it to a hard top-k with a
straight-through estimator at `temperature=0.1` (a near-step function). Consequences,
all measured:

- The router's parameter gradient scales as **‖x‖²/T** with no bound: 461 → 4879 as ‖x‖ grew
  8 → 511. Nothing bounded it (no input norm, no z-loss, no weight decay, one shared
  `route_proj` across all 5 routing points with contradictory keep ratios 0.5/0.7/0.85/1.0).
- 31–58 % of tokens flipped their hard route **every step** → a permanently non-stationary
  objective and a gradient that never settles.
- The residual norm grows during training (nothing bounds it either), so the router gradient
  grows with it — hence a *slow* climb and onset ~250–700 steps after the LR peak.
- `mod_router.route_proj.weight` is in the top-5 hot gradients in **100 %** of 822
  diagnostic lines, and was **64 % of the entire pre-clip grad norm** in one run.
- A CPU mini-reproduction reproduced the collapse exactly: raw router → grad_norm
  0.96 → 10.14; with `RMSNorm(x)` → the router gradient peaks then **decays**, norm stable.

**Explicitly disproved as causes** (measured, so do not re-litigate): PolyMamba SSM
divergence (ρ(A) = 0.950000 in all 12 layers, zero inf/NaN in all 861 tensors, `A_log`
gradient *identically zero* on clamped channels); mixed precision (the bf16-parameter run
collapses at the same step with the same hotspots); the LR schedule (C¹-flat, 0.011 % change
across the whole onset window, and Adam's step is provably ≤ 1.14·lr); the data pipeline
(0 splices, 0 degenerate chunks, correct EOS).

**Fixes (all in this commit):**
1. `mod_router.py` — `F.rms_norm` on the router input with a learned gain (scale-invariance),
   `temperature` 0.1 → **1.0**, and an ST-MoE router z-loss `1e-2 · mean(score²)` folded into
   the returned routing loss. Verified: gradient growth over a 64× input-scale sweep is
   10.6× → **0.91×** (flat).
2. `polymamba.py` — `dt` now clamped **two-sided** (`_compute_dt_bounds`). The old clamp was
   `clamp_min` only, so ρ(A) was free to exceed 1 once `dt > 2/λ_max` (A_log ≈ 2.08).
   Verified: ρ = 0.950000 for `A_log` ∈ [-10, +20].
3. `fsa.py` / `topk_soft.py` — `soft_topk_mask` now returns the indices it actually selected,
   and the FSA gathers both the DCT coefficients and their weights with those. Previously the
   weights came from the **Gumbel-perturbed** mask while the indices came from the clean
   logits, silently dropping ~46 % of the 8 compressed bins every step at gumbel≈0.87.
   Verified: `bin_weights` ≡ 1.0000 at gumbel 0.0/0.5/1.0.
4. `optim.py` — `embed_lr_scale` (config `optim.embed_lr_scale: 0.1`) puts the tied embedding
   in its own param group at 10 % LR, bounding the logit-scale growth that drives the
   head gradient.

Regression tests for all four: `test_router_gradient_is_invariant_to_residual_scale`,
`test_polymamba_spectral_radius_bounded_for_extreme_alogs`,
`test_parallel_scan_matches_sequential_reference`, plus the existing suites. 116 tests.

**Consequence for checkpoints:** fix 1 adds a new parameter (`mod_router.route_norm_weight`),
so checkpoints saved before 15 Sep will NOT load with `strict=True`. Start fresh, or
`load_state_dict(..., strict=False)`.

**Do NOT re-introduce a raw (unnormalised) input into any router/gate.** Any discrete gate
fed by an unbounded activation will reproduce this collapse.

### PAF-10 — 15 Sep 2026: PAF-9 was fixable but incomplete (10-agent verification round)

PAF-9's router fix removed the grad-norm collapse (run3: **zero** `[spike-guard]` events
through step 440, vs a collapse at 1274 before) — but it **introduced a learning regression**
and left the router's aux loss broken. Both were found by a 10-agent audit.

**Regression: `embed_lr_scale: 0.1` (reverted to 1.0).** A controlled A/B (only that argument
changed) reproduces the real gap: at matched steps the 0.1× arm is **+1.38 nats @200, +2.82
@400, +3.97 @1000**, and it flattens on a worse floor. The justification was also wrong — an
LR multiplier cannot bound a gradient: measured pre-clip `‖∇embedding‖` is *identical* across
scales, and the total grad norm is *larger* with the cut. The tied embedding **is** the LM
head (39.6 % of all parameters) and early training lives there.

**The consensus finding — the router's aux loss was still broken, three ways:**
1. **`keep_ratio = 1.0` routing points (layers 16, 20) were structurally unsatisfiable.** With
   `k = seq_len`, `soft = sigmoid(s − min(s)) ≥ 0.5` can never reach the target 1.0, so those
   two points contributed **64 % of the entire mod aux loss** and the *largest* gradient into
   the shared `route_proj` — a permanent, unbounded score-**inflation** force. They are also
   functional no-ops (gate ≡ 1, so `x = block(x)`).
2. **The z-loss was decorative.** `z_loss_weight=1e-2` nested inside `mod_loss_weight=1e-2`
   gives an effective **1e-4** — 100× below ST-MoE's recipe. Measured: it changed the router
   score RMS by 1–17 %; 100× the coefficient actually bounds it. Now `z_loss_weight = 1.0`
   (effective 1e-2 at full ramp).
3. **`route_proj` has no weight decay** and nothing else bounds it.

Together: the same saturation that killed the runs was still reachable *through the weights*
instead of through `‖x‖`.

**Other confirmed defects fixed in this round:**
- **PolyMamba `A_log` was 88 % frozen** — `dt` was initialised log-uniform in `[0.005, 0.1]`
  while the clamp floor is `[0.055, 0.460]`, and `clamp_min` has **zero gradient**. Measured:
  only **3 of 32** channels received gradient, and 1500 AdamW steps moved the clamped entries
  by *exactly* 0.0. The entire order-1 tridiagonal channel — the only thing making this
  "Poly"-Mamba — had a compile-time-constant transition matrix. `dt` is now initialised
  *inside* `[floor, ceil]`: **32/32 channels receive gradient**, ρ(A) = 0.938/0.923.
- **`soft_topk_mask` selected more than k on ties** (`perturbed >= kth`): with all-equal
  scores it selected **64 of 64** instead of 32, and both balance and z-loss gradients were
  exactly zero — a saddle the aux loss actively pushed toward. The mask is now built from the
  returned indices.
- **`mask_logits` init was `N(0, 1.0)`** while three docs specify `N(0, 0.02²)`; at std 1.0 the
  anti-collapse ortho loss is 50× too weak to steer (2500× weaker relative to parameter
  scale). Now 0.02. The ortho loss also used `.sum()` over `n_heads(n_heads−1) = 72` pairs —
  now `.mean()`.
- **Residual-output projections** (`out_proj`, `w_down`, `w_o`) now use the standard
  `0.02/√(2·n_layers)` scaling like GPT-2/Llama.

**Instrumentation added:** the periodic log line now carries `route_rms` (router score RMS)
and `pm_rho` (PolyMamba spectral radius). Both agents that found the router problems asked
for this — the quantities were computed but never logged, so saturation was invisible.

**Disproved / not-a-bug in this round** (measured, do not re-litigate): the chunked CE is
exact (6.1e-07 vs full logits; the residual is the deliberate fp32 upcast); the task loss is a
true mean (ratio 1.0000003 vs a single big batch); FSA causality is bit-exact (0.0 change at
all positions < t for t = 31…1023); the residual-stream growth at init (~1.0×) is *normal* for
pre-norm; `_apply_aux_ramping` is correct.

**Still open / flagged, not fixed:** `rope_base` is configured but **never applied** — the
model has no explicit positional encoding (order arrives only via the SSM recurrence and
FSA's causal chunking). Component attribution on tiny models showed MoD routing *hurting*
(+0.17 nats) and CC-FSA not beating plain attention at L ≤ 256. A same-recipe vanilla
baseline on real data is still missing and is the only way to justify the hybrid.

---

---

## 6. Code Conventions

- **No comments in code** unless explicitly asked. Code is self-documenting via names.
- **Type hints everywhere.** Strict types in `pyproject.toml`.
- **fp32 master weights + bf16 autocast compute** (PAF-8). bf16 only for the compute inside matmuls; parameters, gradients and optimizer state are fp32. fp32 for stability-critical ops (Softmax, MoD load-balancing, MoE-style routing, the SSM scan, DCT, RMSNorm stats) and A-matrix initialization.
- **Gradient checkpointing mandatory** at 200M for memory headroom.
- **bf16 reduction inattention scores**: standard trick (subtract max before exp).
- **Weights init:** standard small init for linears; Hurwitz for SSM A; small W_gate for PolyMamba-2 gate; uniform logits for FSA mask.
- **Tests required** for every component before integration into `model.py`.

---

## 7. Verification Commands (run after every change)

```powershell
# Environment (after HIP SDK 7.1.1 install + reboot)
hipInfo                          # must list gfx1102
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # must be True

# Throughput benchmark (Phase 0 exit criterion)
python scripts/benchmark.py      # must show >= 8 TFLOPS sustained bf16 (measured 26.3)

# Unit tests (113 tests, ~28 s)
pytest tests/ -v                 # must pass before integration

# Smoke test (Phase 1 exit criterion, ~20 s on GPU)
python scripts/smoke_pretrain.py --steps 5 --seq-len 256 --micro-batch 2  # must finish + save checkpoint

# Proxy ablation sanity (random tokens, stability only)
python ablations/04_aether_full.py --proxy --steps 1000  # loss must decrease
```

**Device note:** this machine has two ROCm devices — index 0 is the integrated GPU
(`AMD Radeon(TM) Graphics`, 6.63 GB) and **hard-crashes with a 0xC0000005 access violation**
on kernels it does not support; index 1 is the RX 7600 (8.57 GB). `aether.utils.device.select_device()`
picks the discrete GPU and calls `torch.cuda.set_device`, so plain `"cuda"` resolves correctly.
Tests get this via `tests/conftest.py`. Never hardcode `device=0`.

---

## 8. File Layout

```
aether/              # Core package
  components/        # FSA, PolyMamba-2, MoD-GDR
  ops/               # DCT, soft top-k
  training/          # Data, optim, curriculum, losses
  post_train/        # SFT, DPO
  eval/              # lm-eval-harness runner
  utils/             # Logging, checkpoint, numerics
configs/             # YAML configs
ablations/           # 25M proxy ablations
tests/               # Unit tests
scripts/             # Shell entry points
docs/                # Architecture, audit, recipes
```

Each subdirectory has its own AGENTS.md for context-specific guidance.

---

## 9. "Do NOT" List

- Do NOT add hyperbolic geometry anywhere. It's gone.
- Do NOT reintroduce TT-Mem, HPE, or AR-FFN. They are dead.
- Do NOT use DirectML. Do NOT use WSL2 ROCm (KFD missing for gfx1102). Use **ROCm HIP SDK 7.1.1 on Windows 11 native** — that's the only path that works for RX 7600.
- Do NOT train >200M params without re-checking memory budget.
- Do NOT skip gradient checkpointing.
- Do NOT use fp16 (overflow risk in attention logits). Use bf16.
- Do NOT put parameters, gradients or optimizer state in bf16 (PAF-8: the update is below one bf16 ULP for |w| ≳ 0.015, so `A_log`/RMSNorm/`mask_logits` never train). Master weights are fp32; bf16 is compute-only, via autocast.
- Do NOT materialise the full `(B, L, vocab)` logits in the training path. Use `_chunked_causal_loss`; the logging path only computes logits when `labels is None` or `return_logits=True`.
- Do NOT flatten batch and sequence together when chunking the loss — that pairs the last token of one document with the first token of the next.
- Do NOT guard only the loss: keep the absolute pre-clip `grad_spike_threshold` skip, and keep the anti-freeze (EMA re-seed + forced clipped update + `wedged_skip_limit`) working for every skip reason.
- Do NOT add comments to code unless asked.
- Do NOT commit checkpoints, data, or .env files.
- Do NOT claim novelty for FSA or PolyMamba-2 without acknowledging prior art (FNet/AFNO/NSA and Liquid-S4/CTDG-SSM respectively).

---

## 10. Project Owner

**Haziq** (GitHub: @IRSPlays). Pain-first work ethic. Brutal honesty preferred over glazing.

If something is wrong, say so. If something works, say so concisely.
