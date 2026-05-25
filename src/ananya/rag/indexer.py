"""FAISS / Chroma vector indexing."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ananya.rag.chunking import chunk_records
from ananya.utils.io import read_jsonl
from ananya.utils.logging import get_logger

logger = get_logger(__name__)


class VectorIndexer:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        store_type: str = "faiss",
    ):
        self.embedding_model_name = embedding_model
        self.store_type = store_type
        self._model = None
        self._index = None
        self._meta: list[dict[str, Any]] = []

    def _load_embedder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model_name)

    def build_from_jsonl(
        self,
        jsonl_path: Path,
        output_dir: Path,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ) -> int:
        records = list(read_jsonl(jsonl_path))
        chunks = list(chunk_records(records, chunk_size=chunk_size, overlap=chunk_overlap))
        texts = [c.text for c in chunks]
        self._meta = [c.model_dump(mode="json") for c in chunks]

        self._load_embedder()
        embeddings = self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.json").write_text(json.dumps(self._meta, ensure_ascii=False, indent=2))

        if self.store_type == "faiss":
            import faiss

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            faiss.write_index(index, str(output_dir / "index.faiss"))
            self._index = index
        else:
            import chromadb

            client = chromadb.PersistentClient(path=str(output_dir / "chroma"))
            coll = client.get_or_create_collection("indic_legal")
            coll.add(
                ids=[m["chunk_id"] for m in self._meta],
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[{k: str(v) for k, v in m.items() if k != "text"} for m in self._meta],
            )

        with (output_dir / "embedder.txt").open("w") as f:
            f.write(self.embedding_model_name)
        return len(chunks)

    def load(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        self.embedding_model_name = (index_dir / "embedder.txt").read_text().strip()
        self._meta = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        if self.store_type == "faiss":
            import faiss

            self._index = faiss.read_index(str(index_dir / "index.faiss"))
        self._load_embedder()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self._load_embedder()
        q = self._model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)

        if self.store_type == "faiss":
            scores, indices = self._index.search(q, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                m = dict(self._meta[idx])
                m["score"] = float(score)
                results.append(m)
            return results
        raise NotImplementedError("Chroma load+search: use pipeline with chroma client")
