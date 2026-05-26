#!/usr/bin/env python3
"""Colab-ready IndicTrans2 batch translation CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from ananya.translation.pipeline import TranslationPipeline, make_indictrans2_translator
from ananya.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch translate legal corpus via IndicTrans2")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--input", required=True, help="Input JSONL with text field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--langs", nargs="+", default=["hi", "ta", "bn", "te", "mr"])
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--model", default="ai4bharat/indictrans2-en-indic-1B")
    args = parser.parse_args()

    print(f"DEBUG: Input args parsed. Output path: {args.output}")
    cfg = load_config(args.config)
    glossary = cfg["translation"]["glossary_path"]

    # In Colab: replace stub with real IndicTrans2 inference
    try:
        print("DEBUG: Calling make_indictrans2_translator...")
        translator = make_indictrans2_translator(args.model)
        print("DEBUG: Successfully created translator!")
    except Exception as e:
        print(f"DEBUG: Exception in make_indictrans2_translator: {e}")
        translator = None

    print("DEBUG: Initializing TranslationPipeline...")
    pipe = TranslationPipeline(
        glossary_path=glossary,
        cache_dir=cfg["translation"]["cache_dir"],
        max_retries=cfg["translation"]["max_retries"],
        batch_size=cfg["translation"]["batch_size"],
        translator=translator,
    )
    
    print(f"DEBUG: Starting build_parallel_corpus on {args.input}...")
    try:
        count = pipe.build_parallel_corpus(
            Path(args.input),
            Path(args.output),
            target_langs=args.langs,
            text_field=args.text_field,
        )
        print(f"Wrote {count} alignment records to {args.output}")
    except Exception as e:
        print(f"DEBUG: Exception in build_parallel_corpus: {e}")
        raise

if __name__ == "__main__":
    print("DEBUG: Script started!")
    main()
    print("DEBUG: Script finished perfectly!")
