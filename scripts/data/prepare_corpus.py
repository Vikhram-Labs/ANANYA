#!/usr/bin/env python3
"""End-to-end corpus preparation."""

import argparse
from pathlib import Path

from ananya.data.cleaning import clean_jsonl
from ananya.data.constitution_ingest import ingest_constitution
from ananya.data.instruct_builder import build_instruct_from_articles
from ananya.data.splits import write_splits
from ananya.utils.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--constitution-json", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    raw = Path(cfg["paths"]["raw"])
    processed = Path(cfg["paths"]["processed"])
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    const_raw = raw / "constitution_en.jsonl"
    ingest_constitution(const_raw, Path(args.constitution_json) if args.constitution_json else None)

    clean_path = processed / "constitution_en_clean.jsonl"
    clean_jsonl(str(const_raw), str(clean_path))

    split_dir = Path(cfg["paths"]["splits"]) / "pretrain"
    write_splits(clean_path, split_dir)

    instruct_path = processed / "instruct_en.jsonl"
    build_instruct_from_articles(str(clean_path), str(instruct_path))
    write_splits(instruct_path, Path(cfg["paths"]["splits"]) / "instruct", group_key="article")

    print("Done:", split_dir)


if __name__ == "__main__":
    main()
