"""Instruction-tuning dataset generation from constitution + QA templates."""

from __future__ import annotations

import uuid
from typing import Any, Iterator

from indiclegalslm.schemas.datasets import InstructionRecord, LanguageCode, SourceMetadata, SourceType
from indiclegalslm.utils.io import read_jsonl, write_jsonl

SYSTEM_PROMPT = (
    "You are IndicLegalSLM, a sovereign Indian constitutional assistant. "
    "Answer using only constitutionally grounded reasoning. Cite article numbers. "
    "If uncertain, say you do not know."
)


def _article_fields(article: dict[str, Any]) -> tuple[int, str, str]:
    num = int(article["article"])
    title = article.get("title", "")
    desc = article.get("description") or article.get("text", "")
    if desc.startswith(f"Article {num}"):
        parts = desc.split("\n\n", 1)
        desc = parts[1] if len(parts) > 1 else desc
    return num, title, desc


def constitutional_qa_pair(article: dict[str, Any], lang: LanguageCode = LanguageCode.EN) -> InstructionRecord:
    num, title, desc = _article_fields(article)
    source = SourceMetadata(
        source_id=f"const-art-{num}",
        source_type=SourceType.CONSTITUTION,
        language=lang,
        license="public_domain_government",
    )
    question = f"What does Article {num} of the Indian Constitution ({title}) state?"
    return InstructionRecord(
        id=str(uuid.uuid4()),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Article {num} ({title}): {desc}"},
        ],
        task="constitutional_qa",
        language=lang,
        article=num,
        source=source,
        references=[f"Article {num}"],
    )


def summarization_pair(article: dict[str, Any], lang: LanguageCode = LanguageCode.EN) -> InstructionRecord:
    num, title, desc = _article_fields(article)
    source = SourceMetadata(
        source_id=f"const-art-{num}",
        source_type=SourceType.CONSTITUTION,
        language=lang,
        license="public_domain_government",
    )
    summary = desc[:500] + ("..." if len(desc) > 500 else "")
    return InstructionRecord(
        id=str(uuid.uuid4()),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Summarize Article {num} ({title}) in plain language for a citizen.",
            },
            {"role": "assistant", "content": summary},
        ],
        task="summarization",
        language=lang,
        article=num,
        source=source,
    )


def build_instruct_from_articles(
    articles_jsonl: str,
    output_path: str,
    tasks: list[str] | None = None,
) -> int:
    tasks = tasks or ["constitutional_qa", "summarization"]
    records: list[InstructionRecord] = []
    for row in read_jsonl(articles_jsonl):
        art = row
        if "article" not in art:
            continue
        if "constitutional_qa" in tasks:
            records.append(constitutional_qa_pair(art))
        if "summarization" in tasks:
            records.append(summarization_pair(art))
    return write_jsonl(output_path, records)
