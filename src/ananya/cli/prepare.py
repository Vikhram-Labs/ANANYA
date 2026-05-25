"""CLI: data preparation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ananya.data.corpus_gather import gather_all
from ananya.data.instruct_builder import build_instruct_from_articles
from ananya.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(prog="ananya-prepare")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--constitution-json", default=None)
    p.add_argument("--build-instruct", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    raw = Path(cfg["paths"]["raw"])
    processed = Path(cfg["paths"]["processed"])
    counts = gather_all(raw, processed, Path(args.constitution_json) if args.constitution_json else None)
    print("Gather counts:", counts)
    if args.build_instruct:
        articles = raw / "constitution_en.jsonl"
        out = Path(cfg["paths"]["processed"]) / "instruct_en.jsonl"
        n = build_instruct_from_articles(str(articles), str(out))
        print(f"Built {n} instruction records")


if __name__ == "__main__":
    main()
