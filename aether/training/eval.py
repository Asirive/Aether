from __future__ import annotations

import json
import math
import re
import string
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


HELLASWAG_DEV_SAMPLES: list[dict] = [
    {
        "ctx": "A man is sitting on a roof. he",
        "endings": [
            "is holding a rope.",
            "is pulling a hose.",
            "is cutting the grass.",
            "is laying on his back and relaxing.",
        ],
        "label": 3,
    },
    {
        "ctx": "The woman is a doctor. she",
        "endings": [
            "is cooking dinner for her family.",
            "takes care of patients every day.",
            "is at a football game.",
            "is studying for an exam at university.",
        ],
        "label": 1,
    },
    {
        "ctx": "A child is playing with a toy car on the floor. The car",
        "endings": [
            "is flying through the sky like an airplane.",
            "rolls forward and hits a wall.",
            "turns into a real vehicle.",
            "starts barking like a dog.",
        ],
        "label": 1,
    },
    {
        "ctx": "It is raining heavily outside, so",
        "endings": [
            "people are going to the beach to swim.",
            "we will stay indoors and read books.",
            "the sun is shining brightly in the sky.",
            "everyone is sunbathing on the grass.",
        ],
        "label": 1,
    },
    {
        "ctx": "She opened the refrigerator door and",
        "endings": [
            "started flying like a bird.",
            "took out some vegetables to cook for dinner.",
            "climbed a tall tree.",
            "began painting the walls of the kitchen.",
        ],
        "label": 1,
    },
    {
        "ctx": "The teacher walked into the classroom and",
        "endings": [
            "started swimming across the floor.",
            "greeted the students with a smile.",
            "flew out of the open window.",
            "turned into a robot.",
        ],
        "label": 1,
    },
    {
        "ctx": "He poured coffee into the cup and",
        "endings": [
            "threw the cup into the ocean.",
            "took a sip to wake himself up.",
            "planted the cup in the garden.",
            "used the cup as a hat.",
        ],
        "label": 1,
    },
    {
        "ctx": "The cat climbed up the tree and",
        "endings": [
            "started reading a book.",
            "got stuck on a high branch.",
            "barked at a passing car.",
            "dove underwater to catch fish.",
        ],
        "label": 1,
    },
]


ARC_EASY_DEV_SAMPLES: list[dict] = [
    {
        "question": "What is the main source of energy for most ecosystems on Earth?",
        "choices": ["Wind", "Sunlight", "Coal", "Oil"],
        "answer": 1,
    },
    {
        "question": "Which of the following is a mammal?",
        "choices": ["Snake", "Eagle", "Dolphin", "Frog"],
        "answer": 2,
    },
    {
        "question": "What do plants need to perform photosynthesis?",
        "choices": ["Oxygen and water", "Carbon dioxide and sunlight", "Nitrogen and salt", "Helium and sand"],
        "answer": 1,
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
        "answer": 1,
    },
    {
        "question": "How many legs does an insect have?",
        "choices": ["Four", "Six", "Eight", "Ten"],
        "answer": 1,
    },
    {
        "question": "What is the boiling point of water at sea level in Celsius?",
        "choices": ["50", "75", "100", "150"],
        "answer": 2,
    },
    {
        "question": "Which of these is a solid?",
        "choices": ["Steam", "Oxygen", "Ice", "Juice"],
        "answer": 2,
    },
    {
        "question": "Which animal lays eggs?",
        "choices": ["Cat", "Dog", "Chicken", "Whale"],
        "answer": 2,
    },
]


GENERATION_PROMPTS: list[str] = [
    "The capital of France is",
    "Once upon a time, in a small village by the sea,",
    "The three branches of government are",
    "To make a cake, you need flour, sugar, eggs, and",
    "Albert Einstein is famous for",
    "The largest planet in our solar system is",
    "If you mix red and blue, you get",
    "The water cycle begins when",
]


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _loglikelihood_choice(
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    tokenizer,
    ctx: str,
    choice: str,
    device: torch.device,
    max_ctx_tokens: int = 1024,
) -> float:
    ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
    choice_ids = tokenizer.encode(" " + choice.lstrip(), add_special_tokens=False)
    full_ids = ctx_ids + choice_ids
    if len(full_ids) > max_ctx_tokens:
        full_ids = full_ids[-max_ctx_tokens:]
        if len(full_ids) <= len(choice_ids):
            return float("-inf")
    choice_len = len(choice_ids)
    if choice_len <= 0:
        return float("-inf")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model_forward(input_ids)["logits"]
    log_probs = F.log_softmax(logits[0, -choice_len - 1: -1].float(), dim=-1)
    target = torch.tensor(full_ids[-choice_len:], device=device)
    return float(log_probs.gather(-1, target.unsqueeze(-1)).sum().item())


def evaluate_choices(
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    tokenizer,
    samples: list[dict],
    device: torch.device,
    task: str,
) -> dict:
    correct = 0
    results: list[dict] = []
    for idx, sample in enumerate(samples):
        if task == "hellaswag":
            ctx = sample["ctx"]
            endings = sample["endings"]
            label = sample["label"]
            scores = [
                _loglikelihood_choice(model_forward, tokenizer, ctx, e, device)
                for e in endings
            ]
            pred = int(max(range(len(scores)), key=lambda i: scores[i]))
            ok = (pred == label)
            results.append({
                "idx": idx, "ctx": ctx, "choices": endings,
                "scores": scores, "pred": pred, "label": label, "ok": ok,
            })
        elif task == "arc_easy":
            question = sample["question"]
            choices = sample["choices"]
            label = sample["answer"]
            ctx = f"Question: {question}\nAnswer:"
            scores = [
                _loglikelihood_choice(model_forward, tokenizer, ctx, " " + c, device)
                for c in choices
            ]
            pred = int(max(range(len(scores)), key=lambda i: scores[i]))
            ok = (pred == label)
            results.append({
                "idx": idx, "ctx": question, "choices": choices,
                "scores": scores, "pred": pred, "label": label, "ok": ok,
            })
        if ok:
            correct += 1
    return {
        "task": task,
        "n": len(samples),
        "correct": correct,
        "accuracy": correct / max(1, len(samples)),
        "details": results,
    }


@torch.no_grad()
def generate_samples(
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    tokenizer,
    device: torch.device,
    prompts: list[str],
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 50,
) -> list[dict]:
    samples: list[dict] = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        cur = input_ids
        for _ in range(max_new_tokens):
            logits = model_forward(cur)["logits"][:, -1, :].float()
            logits = logits / max(temperature, 1e-3)
            if top_k and top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            cur = torch.cat([cur, nxt], dim=-1)
            if nxt.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(cur[0].tolist(), skip_special_tokens=True)
        samples.append({"prompt": prompt, "continuation": text[len(prompt):]})
    return samples


def save_eval_log(eval_results: dict, samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"eval": eval_results, "generations": samples}
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


CANARY_PROMPTS: list[str] = [
    "The capital of France is",
    "Once upon a time, in a small village by the sea,",
    "The largest planet in our solar system is",
    "If you mix red and blue paint, you get",
    "def fibonacci(n):",
    "Five plus three equals",
]


@torch.no_grad()
def generate_canary_samples(
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    tokenizer,
    device: torch.device,
    prompts: list[str] | None = None,
    max_new_tokens: int = 20,
) -> list[tuple[str, str]]:
    if prompts is None:
        prompts = CANARY_PROMPTS
    results: list[tuple[str, str]] = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        cur = input_ids
        prompt_len = input_ids.shape[1]
        for _ in range(max_new_tokens):
            logits = model_forward(cur)["logits"][:, -1, :].float()
            nxt = logits.argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=-1)
            if nxt.item() == tokenizer.eos_token_id:
                break
        continuation = tokenizer.decode(cur[0, prompt_len:].tolist(), skip_special_tokens=True)
        results.append((prompt, continuation))
    return results