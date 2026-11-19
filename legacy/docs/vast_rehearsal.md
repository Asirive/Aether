# GPU rehearsal: distil Qwen3-0.6B into Aether v2 vs a vanilla transformer

Goal: at 0.54B parameters and real data, does the Aether v2 student (conv / SSM / conv /
attention, see docs/aether_v2_results.md) beat a parameter-matched vanilla attention student distilled from
the same teacher on the same tokens? And does it keep in-context retrieval (copy probe)?

| student | params | layers |
|---|---|---|
| `aether_v2` (pattern `hybrid_attn`) | ~538M | 28 = 7 x (conv, ssm, conv, attn) |
| `vanilla` | 537.3M | 28 x attention |

Both: d_model 1024, SwiGLU 3072, the teacher's 151,936-token vocabulary, tied embeddings.

## 1. Rent

Vast.ai search filters: **Secure Cloud only**, 1x RTX 5090 (32 GB), max CUDA >= 12.8,
reliability >= 99%, disk >= 100 GB, PyTorch template. A host outside mainland China
(Hugging Face is blocked there). ~$0.50/hr at the time of writing.

## 2. Get the code onto the box

The repo is private. Either clone with a fine-grained GitHub token that has read access to
this repo only, or copy a tarball:

```bash
# on your machine
git archive --format=tar.gz -o aether.tgz ccr-6777893e-1czf2h
scp -P <port> aether.tgz root@<host>:~
# on the instance
mkdir Aether && tar xzf aether.tgz -C Aether && cd Aether/legacy
```

## 3. Setup + smoke test (~15 min, ~$0.15)

```bash
bash scripts/vast_setup.sh
```

Checks the GPU has a matching PyTorch kernel (installs a CUDA 12.8 build if not; RTX 50xx
needs it), installs packages, runs the unit tests, checks Hugging Face is reachable, then
runs 30 training steps per student and prints tokens/s, peak memory and the estimated hours
for 1B tokens. **Stop here and send the printed table** before committing to the full runs.

If a smoke run runs out of memory, add `--act-ckpt` (recompute blocks in backward).

## 4. Full runs (run one after the other in tmux)

```bash
tmux new -s kd
TOK=5e8   # tokens per student; pick from the smoke-test hours estimate
python3 scripts/distill.py --student aether_v2 --tokens $TOK --out runs/kd_aether  2>&1 | tee -a runs/kd_aether.out
python3 scripts/distill.py --student vanilla   --tokens $TOK --out runs/kd_vanilla 2>&1 | tee -a runs/kd_vanilla.out
```

Both runs see the same documents in the same order and the same held-out set. If the
instance is interrupted, rerun the same command: it resumes from `runs/<name>/ckpt.pt`.

Monitor: `python3 scripts/compare_kd.py runs/kd_aether runs/kd_vanilla`

## 5. What to bring back

`runs/kd_*/log.jsonl` and `meta.json` (small). Checkpoints only if we continue from them.
Then **destroy** the instance (a stopped instance still bills for disk).

## Reading the result

* `student_bpb` on the held-out set: lower is better; the gap to `teacher_bpb` is what
  distillation has not yet transferred.
* `copy_gain` = loss on a span minus loss on its second repeat. A model that retrieves from
  context scores well above 0. If `aether_v2` is clearly below `vanilla` here, its
  retrieval is weaker, which matters for a chat/voice model even if bpb is better.
* `tok_per_s`: the SSM scan is plain PyTorch (no fused kernel), so this understates
  Aether's speed; it is not the on-device speed.
