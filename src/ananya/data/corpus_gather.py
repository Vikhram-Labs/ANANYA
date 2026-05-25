"""Orchestrate multi-source legal corpus gathering into unified JSONL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ananya.data.constitution_ingest import ingest_constitution
from ananya.data.cleaning import clean_jsonl
from ananya.data.splits import write_splits
from ananya.utils.logging import get_logger

logger = get_logger(__name__)

SOURCE_REGISTRY = {
    "constitution": "constitution_en.jsonl",
    "amendments": "amendments_en.jsonl",
    "parliament_debates": "debates_en.jsonl",
    "sc_summaries": "sc_summaries_en.jsonl",
    "schemes": "schemes_en.jsonl",
    "legal_qa": "legal_qa_en.jsonl",
}


def gather_all(
    raw_dir: Path,
    processed_dir: Path,
    constitution_json: Path | None = None,
) -> dict[str, int]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    const_path = raw_dir / SOURCE_REGISTRY["constitution"]
    counts["constitution"] = ingest_constitution(const_path, constitution_json)

    for name, filename in SOURCE_REGISTRY.items():
        if name == "constitution":
            continue
        src = raw_dir / filename
        if not src.exists():
            logger.warning("Optional source missing: %s", src)
            continue
        out = processed_dir / f"clean_{filename}"
        counts[name] = clean_jsonl(str(src), str(out))

    merged_in = processed_dir / "clean_constitution_en.jsonl"
    if not merged_in.exists():
        clean_jsonl(str(raw_dir / SOURCE_REGISTRY["constitution"]), str(merged_in))

    split_dir = processed_dir.parent / "splits" / "pretrain"
    counts["splits"] = sum(
        write_splits(merged_in, split_dir, group_key="article").values()
    )
    return counts
