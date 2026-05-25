"""JSONL / HuggingFace dataset schemas with metadata contracts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class LanguageCode(str, Enum):
    EN = "en"
    HI = "hi"
    TA = "ta"
    BN = "bn"
    TE = "te"
    MR = "mr"


class SourceType(str, Enum):
    CONSTITUTION = "constitution"
    AMENDMENT = "amendment"
    PARLIAMENT_DEBATE = "parliament_debate"
    SC_SUMMARY = "supreme_court_summary"
    SCHEME = "governance_scheme"
    LEGAL_QA = "legal_qa"
    SYNTHETIC = "synthetic"


class SourceMetadata(BaseModel):
    source_id: str
    source_type: SourceType
    url: str | None = None
    license: str = "unknown"
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    language: LanguageCode
    version: str = "1.0"


class PretrainRecord(BaseModel):
    """Continued pretraining — plain text with rich metadata."""

    id: str
    text: str
    language: LanguageCode
    article: str | None = None
    part: str | None = None
    schedule: str | None = None
    title: str | None = None
    source: SourceMetadata
    char_count: int | None = None
    token_estimate: int | None = None

    @field_validator("text")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must be non-empty")
        return v


class InstructionRecord(BaseModel):
    """SFT / chat tuning — ChatML-compatible messages."""

    id: str
    messages: list[dict[str, str]]
    task: str  # constitutional_qa | summarization | reasoning | comparison
    language: LanguageCode
    article: str | None = None
    source: SourceMetadata
    difficulty: str | None = None
    references: list[str] = Field(default_factory=list)

    @field_validator("messages")
    @classmethod
    def valid_roles(cls, v: list[dict[str, str]]) -> list[dict[str, str]]:
        allowed = {"system", "user", "assistant"}
        for m in v:
            if m.get("role") not in allowed:
                raise ValueError(f"invalid role in {m}")
            if "content" not in m:
                raise ValueError("message must have content")
        return v


class RetrievalRecord(BaseModel):
    """RAG index chunks with citation anchors."""

    chunk_id: str
    text: str
    language: LanguageCode
    article: str | None
    title: str | None
    part: str | None
    embedding_model: str | None = None
    source: SourceMetadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    """Held-out benchmark items."""

    id: str
    task: str
    question: str
    reference_answer: str
    language: LanguageCode
    article: str | None = None
    choices: list[str] | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    rubric: dict[str, Any] | None = None
    source: SourceMetadata


class AlignmentRecord(BaseModel):
    """Parallel multilingual alignment for cross-lingual eval."""

    align_id: str
    en_text: str
    translations: dict[LanguageCode, str]
    article: str | None
    glossary_locked: bool = True
    source: SourceMetadata
