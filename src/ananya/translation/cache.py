"""Translation memory cache — SQLite-backed for Colab persistence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class TranslationMemory:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tm (
                cache_key TEXT PRIMARY KEY,
                source_lang TEXT,
                target_lang TEXT,
                source_text TEXT,
                translated_text TEXT
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def key(text: str, src: str, tgt: str) -> str:
        raw = f"{src}|{tgt}|{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, text: str, src: str, tgt: str) -> str | None:
        k = self.key(text, src, tgt)
        row = self._conn.execute(
            "SELECT translated_text FROM tm WHERE cache_key = ?", (k,)
        ).fetchone()
        return row[0] if row else None

    def set(self, text: str, src: str, tgt: str, translation: str) -> None:
        k = self.key(text, src, tgt)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO tm (cache_key, source_lang, target_lang, source_text, translated_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (k, src, tgt, text, translation),
        )
        self._conn.commit()

    def export_jsonl(self, path: Path) -> int:
        rows = self._conn.execute("SELECT source_lang, target_lang, source_text, translated_text FROM tm").fetchall()
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as f:
            for src, tgt, st, tt in rows:
                f.write(json.dumps({"src": src, "tgt": tgt, "source": st, "translation": tt}, ensure_ascii=False) + "\n")
                count += 1
        return count
