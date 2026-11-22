# Vision stage (Asirive Cortex): image encoder + projector + Aether

Runs after the text stage (`docs/vast_rehearsal.md`). Goal: a language core that describes
scenes and answers questions for a blind user, with spatial understanding as the priority.

```
camera -> vision encoder (SigLIP 2, frozen) -> projector -> Aether (language) -> speech
```

Safety-critical facts (obstacles, steps, distance) should come from dedicated detectors and
a depth model running on the NPU; Cortex turns them plus the image into guidance and answers
questions. Do not let the language model alone decide that a path is clear.

## What is distilled

For every image + text example the teacher (Qwen3-VL-4B by default) gives a probability
over its whole vocabulary for each next text token. The student, which sees the encoder's
image tokens followed by the same text tokens, is trained to match that distribution
(KL at temperature T, weight `--kd-weight`) plus a little ordinary next-token loss. Only text
positions are scored; the image tokens are context. The student's vocabulary is the
teacher's, so `--init` (a text-stage checkpoint) only works if both stages use teachers with
the same tokenizer.

## 1. Check the teachers and the encoder (about 10 minutes)

```bash
python3 scripts/check_vl_teacher.py --teacher Qwen/Qwen3-VL-4B-Instruct --teacher Qwen/Qwen3.5-2B
```

Prints PASS/FAIL for loading, vocabulary, memory, image/text alignment and each teacher's
accuracy on the synthetic spatial probe (chance 0.25), then the encoder's token count.
Send the JSON (`runs/vl_teacher_check.json`). Choose the teacher with the better probe
accuracy unless it fails a check. If the vocabulary differs from the text-stage teacher
(Qwen3-0.6B: 151,936), the text stage must be redone with a teacher sharing it, or the
vision stage started without `--init`.

## 2. Short run (one GPU-hour)

```bash
python3 scripts/distill_vl.py --student aether_v2 --teacher Qwen/Qwen3-VL-4B-Instruct \
  --encoder google/siglip2-base-patch16-224 --init runs/kd_aether/ckpt.pt \
  --dataset <dataset id> --image-col image --prompt-col question --answer-col answers \
  --steps 500 --out runs/vl_aether
```

* `--dataset` is any Hugging Face image dataset with an image column and a text column.
  Candidates (verify ids and column names on the instance with `datasets`): VizWiz VQA
  (photos taken by blind users, `answers` is a list), Visual Genome relationships or region
  descriptions, RefCOCO/RefCOCOg. `--prompt-col ""` uses `--prompt` for every image
  (captioning data).
* `--synthetic-frac 0.1` mixes in synthetic spatial scenes, so the probe is in-distribution.
* Spatial probe accuracy is logged at every eval; the student starts at chance and should
  climb above it. Restartable from `runs/vl_aether/ckpt.pt`.

## 3. Then

Compare `--student vanilla` on the same data if the text stage left that question open.
Add a held-out eval on real data (VizWiz validation) before a long run.

## CPU result (results/22_spatial)

See `docs/aether_v2_results.md` section on spatial grounding.
