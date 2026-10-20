from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

_DEFAULT_TOKENIZER = "unsloth/Llama-3.2-1B-Instruct"


def build_tokenizer(name: str = _DEFAULT_TOKENIZER) -> AutoTokenizer:
    cache_dir = Path.home() / ".cache" / "huggingface" / "tokenizers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer