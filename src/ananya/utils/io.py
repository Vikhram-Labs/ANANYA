"""JSONL I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any] | BaseModel],
    append: bool = False,
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            if isinstance(rec, BaseModel):
                rec = rec.model_dump(mode="json")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count
