from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset


def run_sft(
    model,
    tokenizer,
    output_dir: str | Path,
    dataset_name: str = "HuggingFaceTB/smol-smoltalk",
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-5,
    max_seq_length: int = 2048,
) -> None:
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split="train")

    def format_and_tokenize(example):
        messages = example.get("messages", [])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return tokenizer(text, truncation=True, max_length=max_seq_length, padding=False)

    num_proc = 0 if os.name == "nt" else 4
    tokenized = dataset.map(
        format_and_tokenize, remove_columns=dataset.column_names, num_proc=num_proc or None
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        warmup_ratio=0.0,
        lr_scheduler_type="cosine",
        report_to="wandb",
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=0 if os.name == "nt" else 2,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
