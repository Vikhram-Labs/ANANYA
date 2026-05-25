"""Continued pretraining on legal corpus — causal LM with packing."""

from __future__ import annotations

from pathlib import Path

from ananya.training.qlora import QLoRAConfig, load_model_and_tokenizer
from ananya.utils.logging import get_logger

logger = get_logger(__name__)


def run_cpt(
    cfg: QLoRAConfig,
    train_jsonl: str,
    max_steps: int = 3000,
) -> Path:
    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    model, tokenizer, _ = load_model_and_tokenizer(cfg)
    ds = load_dataset("json", data_files={"train": train_jsonl}, split="train")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate * 0.25,
        max_steps=max_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        bf16=True,
        optim=cfg.optim,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    out = Path(cfg.output_dir) / "final"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    return out
