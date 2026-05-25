#!/usr/bin/env python3
import argparse
from pathlib import Path

from ananya.rag.indexer import VectorIndexer
from ananya.utils.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    inp = Path(args.input or f"{cfg['paths']['processed']}/constitution_en_clean.jsonl")
    out = Path(args.output or cfg["paths"]["indices"])
    idx = VectorIndexer(
        embedding_model=cfg["rag"]["embedding_model"],
        store_type=cfg["rag"]["vector_store"],
    )
    n = idx.build_from_jsonl(inp, out / "constitution", chunk_size=cfg["rag"]["chunk_size"])
    print(f"Indexed {n} chunks -> {out}")


if __name__ == "__main__":
    main()
