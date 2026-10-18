from __future__ import annotations

from pathlib import Path

from datasets import load_dataset


def run_dpo(
    model,
    tokenizer,
    output_dir: str | Path,
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-7,
    beta: float = 0.1,
    max_length: int = 1024,
    max_prompt_length: int = 512,
) -> None:
    from trl import DPOConfig, DPOTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split="train_prefs")

    config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="wandb",
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
    )

    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=dataset,
    )
    try:
        trainer = DPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
