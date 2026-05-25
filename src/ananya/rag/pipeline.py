"""Citation-aware multilingual RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ananya.rag.indexer import VectorIndexer
from ananya.utils.logging import get_logger

logger = get_logger(__name__)

CITATION_TEMPLATE = "[Art. {article}] {title}"


class RAGPipeline:
    def __init__(
        self,
        index_dir: str | Path,
        embedding_model: str | None = None,
        top_k: int = 5,
        generate_fn: Callable[[str], str] | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.top_k = top_k
        self.indexer = VectorIndexer(embedding_model or self._read_embedder())
        self.indexer.load(self.index_dir)
        self.generate_fn = generate_fn

    def _read_embedder(self) -> str:
        return (self.index_dir / "embedder.txt").read_text().strip()

    def retrieve(self, query: str, language: str | None = None) -> list[dict[str, Any]]:
        hits = self.indexer.search(query, top_k=self.top_k)
        if language:
            lang_hits = [h for h in hits if h.get("language") == language]
            if lang_hits:
                hits = lang_hits + [h for h in hits if h not in lang_hits]
        return hits

    def format_context(self, hits: list[dict[str, Any]]) -> str:
        blocks = []
        for h in hits:
            cite = CITATION_TEMPLATE.format(
                article=h.get("article", "?"),
                title=h.get("title", ""),
            )
            blocks.append(f"{cite}\n{h.get('text', '')}")
        return "\n\n---\n\n".join(blocks)

    def build_prompt(self, query: str, hits: list[dict[str, Any]]) -> str:
        context = self.format_context(hits)
        return (
            "Use ONLY the following constitutional excerpts. Cite article numbers.\n\n"
            f"{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def answer(self, query: str, language: str | None = None) -> dict[str, Any]:
        hits = self.retrieve(query, language=language)
        prompt = self.build_prompt(query, hits)
        if self.generate_fn is None:
            return {"answer": None, "prompt": prompt, "citations": hits}
        answer = self.generate_fn(prompt)
        return {
            "answer": answer,
            "citations": [
                {"article": h.get("article"), "title": h.get("title"), "score": h.get("score")}
                for h in hits
            ],
            "prompt": prompt,
        }
