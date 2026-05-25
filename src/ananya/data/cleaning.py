"""Dataset cleaning — dedup, length filter, PII scrub, quality gates."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Callable

from ananya.utils.io import read_jsonl, write_jsonl


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


PII_PATTERNS = [
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"\b\d{10}\b"),  # phone-like
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),  # aadhaar-like — redact in legal corpus
]


def scrub_pii(text: str) -> str:
    for pat in PII_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def clean_record(
    record: dict[str, Any],
    text_key: str = "text",
    min_chars: int = 20,
    max_chars: int = 50000,
) -> dict[str, Any] | None:
    text = record.get(text_key) or record.get("description", "")
    if not text:
        msgs = record.get("messages", [])
        if msgs:
            text = msgs[-1].get("content", "")
    text = normalize_unicode(scrub_pii(collapse_whitespace(str(text))))
    if len(text) < min_chars or len(text) > max_chars:
        return None
    record = dict(record)
    if text_key in record:
        record[text_key] = text
    elif "description" in record:
        record["description"] = text
    record["_content_hash"] = content_hash(text)
    return record


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in records:
        h = r.get("_content_hash") or content_hash(str(r))
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


def clean_jsonl(
    input_path: str,
    output_path: str,
    text_key: str = "text",
    extra_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> int:
    cleaned: list[dict[str, Any]] = []
    for rec in read_jsonl(input_path):
        c = clean_record(rec, text_key=text_key)
        if c is None:
            continue
        if extra_filter and not extra_filter(c):
            continue
        cleaned.append(c)
    cleaned = deduplicate(cleaned)
    return write_jsonl(output_path, cleaned)
