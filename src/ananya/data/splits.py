"""Train / validation / test splits with stratification by article & language."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from indiclegalslm.utils.io import read_jsonl, write_jsonl


def stratified_split(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    group_key: str = "article",
) -> dict[str, list[dict[str, Any]]]:
    random.seed(seed)
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        g = r.get(group_key) or r.get("id", "unknown")
        groups[g].append(r)

    keys = list(groups.keys())
    random.shuffle(keys)
    n = len(keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for k, recs in groups.items():
        if k in train_keys:
            splits["train"].extend(recs)
        elif k in val_keys:
            splits["validation"].extend(recs)
        else:
            splits["test"].extend(recs)
    return splits


def write_splits(
    input_jsonl: Path,
    output_dir: Path,
    group_key: str = "article",
) -> dict[str, int]:
    records = list(read_jsonl(input_jsonl))
    splits = stratified_split(records, group_key=group_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, recs in splits.items():
        path = output_dir / f"{name}.jsonl"
        counts[name] = write_jsonl(path, recs)
    return counts


def to_hf_dataset(split_dir: Path) -> DatasetDict:
    data: dict[str, Dataset] = {}
    for split in ("train", "validation", "test"):
        path = split_dir / f"{split}.jsonl"
        if path.exists():
            records = list(read_jsonl(path))
            data[split] = Dataset.from_list(records)
    return DatasetDict(data)
