#!/usr/bin/env python3
"""Colab T4 SFT entrypoint — mount Drive, set HF_TOKEN, run."""

from __future__ import annotations

import argparse
import os

from ananya.training.qlora import QLoRAConfig, run_sft
from ananya.utils.config import load_config
from ananya.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/sft_colab_t4.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg_yaml = load_config(args.config)
    set_seed(cfg_yaml.get("project", {}).get("seed", 42))

    model_cfg = cfg_yaml["model"]
    hp = cfg_yaml["hyperparameters"]
    mem = cfg_yaml.get("memory", {})
    lora = cfg_yaml.get("training", {}).get("lora", cfg_yaml.get("lora", {}))

    qlora = QLoRAConfig(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        load_in_4bit=mem.get("load_in_4bit", True),
        lora_r=lora.get("r", 16),
        lora_alpha=lora.get("lora_alpha", 32),
        output_dir=cfg_yaml.get("paths", {}).get("checkpoints", "outputs/checkpoints/sft"),
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp.get("num_train_epochs", 2),
        use_unsloth=mem.get("use_unsloth", True),
    )

    dataset_path = cfg_yaml["dataset"]["path"]
    out = run_sft(qlora, dataset_path, resume_from=args.resume)
    print(f"Saved adapter to {out}")

    if os.environ.get("HF_TOKEN"):
        from ananya.hub.push import push_model

        push_model(str(out), repo_id=os.environ.get("HF_MODEL_REPO", ""))


if __name__ == "__main__":
    main()
