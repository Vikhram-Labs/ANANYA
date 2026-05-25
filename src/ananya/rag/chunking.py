"""Article-aware chunking for constitutional RAG."""

from __future__ import annotations

import hashlib
from typing import Any, Iterator

from ananya.schemas.datasets import LanguageCode, RetrievalRecord, SourceMetadata, SourceType


def chunk_records(
    records: list[dict[str, Any]],
    chunk_size: int = 400,
    overlap: int = 50,
    text_key: str = "text",
    language: LanguageCode = LanguageCode.EN,
) -> Iterator[RetrievalRecord]:
    source = SourceMetadata(
        source_id="rag_index",
        source_type=SourceType.CONSTITUTION,
        language=language,
        license="public_domain_government",
    )
    for rec in records:
        text = rec.get(text_key, "")
        article = rec.get("article")
        title = rec.get("title")
        if len(text) <= chunk_size:
            yield RetrievalRecord(
                chunk_id=_cid(text, article),
                text=text,
                language=language,
                article=article,
                title=title,
                source=source,
                metadata={"chunk_index": 0},
            )
            continue
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            yield RetrievalRecord(
                chunk_id=_cid(chunk, article, idx),
                text=chunk,
                language=language,
                article=article,
                title=title,
                source=source,
                metadata={"chunk_index": idx, "start": start, "end": end},
            )
            start = end - overlap
            idx += 1
            if end >= len(text):
                break


def _cid(text: str, article: int | None, idx: int = 0) -> str:
    h = hashlib.sha256(f"{article}:{idx}:{text[:80]}".encode()).hexdigest()[:12]
    return f"chunk-{article or 'x'}-{idx}-{h}"
