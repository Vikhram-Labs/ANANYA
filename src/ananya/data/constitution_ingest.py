"""Ingest articles from indianconstitution package or local JSON."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterator

from indiclegalslm.schemas.datasets import LanguageCode, PretrainRecord, SourceMetadata, SourceType
from indiclegalslm.utils.io import write_jsonl


def load_constitution_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "articles" in data:
        return data["articles"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized constitution JSON structure: {path}")


def load_from_indianconstitution_package() -> list[dict[str, Any]]:
    """Load via installed indianconstitution package if available."""
    try:
        import indianconstitution as ic  # type: ignore

        if hasattr(ic, "get_all_articles"):
            return ic.get_all_articles()
        if hasattr(ic, "articles"):
            return list(ic.articles)
    except ImportError:
        pass
    raise ImportError(
        "Install indianconstitution or provide --constitution-json path. "
        "pip install indianconstitution"
    )


def articles_to_pretrain_records(
    articles: list[dict[str, Any]],
    language: LanguageCode = LanguageCode.EN,
) -> Iterator[PretrainRecord]:
    source = SourceMetadata(
        source_id="constitution_of_india",
        source_type=SourceType.CONSTITUTION,
        url="https://www.india.gov.in/my-government/constitution-india",
        license="public_domain_government",
        language=language,
    )
    for art in articles:
        article_num = int(art.get("article", art.get("number", 0)))
        title = art.get("title", "")
        desc = art.get("description", art.get("content", ""))
        text = f"Article {article_num}: {title}\n\n{desc}".strip()
        yield PretrainRecord(
            id=f"const-en-{article_num:03d}",
            text=text,
            language=language,
            article=article_num,
            title=title,
            source=source,
            char_count=len(text),
        )


def ingest_constitution(
    output_path: Path,
    constitution_json: Path | None = None,
) -> int:
    if constitution_json:
        articles = load_constitution_json(constitution_json)
    else:
        articles = load_from_indianconstitution_package()
    records = list(articles_to_pretrain_records(articles))
    return write_jsonl(output_path, records)


def article_dict_to_chunk_text(article: dict[str, Any], max_chars: int = 2000) -> list[str]:
    """Split long articles for RAG chunking."""
    base = f"Article {article['article']}: {article['title']}\n\n{article['description']}"
    if len(base) <= max_chars:
        return [base]
    paragraphs = base.split("\n\n")
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}".strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks
