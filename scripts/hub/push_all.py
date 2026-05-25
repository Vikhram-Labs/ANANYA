#!/usr/bin/env python3
import argparse
import os

from ananya.hub.push import push_dataset, push_model
from ananya.utils.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--dataset-only", action="store_true")
    p.add_argument("--model-only", action="store_true")
    p.add_argument("--version", default="v0.1.0")
    args = p.parse_args()
    cfg = load_config(args.config)
    hf = cfg["huggingface"]

    if not args.model_only:
        push_dataset(
            cfg["paths"]["splits"],
            hf["dataset_repo"],
            private=hf.get("private", False),
            version_tag=args.version,
        )
    if not args.dataset_only:
        push_model(
            f"{cfg['paths']['checkpoints']}/sft/final",
            hf["model_repo"],
            base_model=cfg["training"]["default_base"],
            private=hf.get("private", False),
            version_tag=args.version,
        )


if __name__ == "__main__":
    main()
