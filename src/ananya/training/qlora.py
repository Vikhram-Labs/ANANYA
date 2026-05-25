"""QLoRA training — Unsloth-first with HF+PEFT fallback for Colab T4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ananya.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QLoRAConfig:
    model_name: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    output_dir: str = "outputs/checkpoints/sft"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    optim: str = "paged_adamw_8bit"
    use_unsloth: bool = True


def load_model_and_tokenizer(cfg: QLoRAConfig):
    if cfg.use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg.model_name,
                max_seq_length=cfg.max_seq_length,
                load_in_4bit=cfg.load_in_4bit,
                dtype=None,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.target_modules
                or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth",
            )
            return model, tokenizer, "unsloth"
        except ImportError:
            logger.warning("Unsloth not installed — falling back to transformers+peft")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules
        or ["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    return model, tokenizer, "peft"


def format_chat(example: dict[str, Any], tokenizer) -> dict[str, str]:
    messages = example.get("messages", [])
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        text = "\n".join(parts)
    return {"text": text}


def run_sft(cfg: QLoRAConfig, dataset_path: str, resume_from: str | None = None) -> Path:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer, backend = load_model_and_tokenizer(cfg)
    if resume_from:
        from peft import PeftModel

        if backend == "peft":
            model = PeftModel.from_pretrained(model, resume_from)

    data = load_dataset("json", data_files={"train": f"{dataset_path}/train.jsonl"}, split="train")
    if "messages" in data.column_names:
        data = data.map(lambda x: format_chat(x, tokenizer), remove_columns=data.column_names)

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        optim=cfg.optim,
        fp16=False,
        bf16=True,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=data, args=training_args)
    trainer.train(resume_from_checkpoint=resume_from is not None)
    out = Path(cfg.output_dir) / "final"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    return out


def merge_and_save(adapter_dir: str, base_model: str, output_dir: str) -> None:
    """Merge LoRA into base for HF push — run on CPU if VRAM low."""
    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, device_map="cpu")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.save_pretrained(output_dir)
