"""IndicTrans2-compatible batch translation with retry + terminology."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ananya.schemas.datasets import AlignmentRecord, LanguageCode, SourceMetadata, SourceType
from ananya.translation.cache import TranslationMemory
from ananya.translation.terminology import TerminologyEngine
from ananya.utils.io import read_jsonl, write_jsonl
from ananya.utils.logging import get_logger

logger = get_logger(__name__)

LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "mr": "mar_Deva",
}


class TranslationPipeline:
    def __init__(
        self,
        glossary_path: str,
        cache_dir: str = "data/cache/translation_memory",
        max_retries: int = 3,
        batch_size: int = 8,
        translator: Callable[[list[str], str, str], list[str]] | None = None,
    ):
        self.terminology = TerminologyEngine(glossary_path)
        self.tm = TranslationMemory(Path(cache_dir) / "tm.db")
        self.max_retries = max_retries
        self.batch_size = batch_size
        self._translator = translator or self._default_translator

    def _default_translator(self, texts: list[str], src: str, tgt: str) -> list[str]:
        """Fallback: identity (plug IndicTrans2 in Colab)."""
        logger.warning("Using stub translator — install indic-trans2 for production.")
        return texts

    def translate_batch(self, texts: list[str], src: str, tgt: str) -> list[str]:
        results: list[str | None] = [None] * len(texts)
        to_translate: list[tuple[int, str, dict[str, str]]] = []

        for i, text in enumerate(texts):
            cached = self.tm.get(text, src, tgt)
            if cached:
                results[i] = cached
                continue
            masked, ph = self.terminology.mask(text, tgt)
            to_translate.append((i, masked, ph))

        for start in range(0, len(to_translate), self.batch_size):
            batch = to_translate[start : start + self.batch_size]
            indices = [b[0] for b in batch]
            masked_texts = [b[1] for b in batch]
            placeholders = [b[2] for b in batch]

            for attempt in range(self.max_retries):
                try:
                    translated = self._translator(masked_texts, src, tgt)
                    break
                except Exception as e:
                    logger.error("Translation attempt %d failed: %s", attempt + 1, e)
                    if attempt == self.max_retries - 1:
                        translated = masked_texts
                    time.sleep(2**attempt)

            for idx, tr, ph, orig_idx in zip(indices, translated, placeholders, indices):
                restored = self.terminology.restore(tr, ph)
                orig_text = texts[orig_idx]
                self.tm.set(orig_text, src, tgt, restored)
                results[orig_idx] = restored

        return [r if r is not None else texts[i] for i, r in enumerate(results)]

    def build_parallel_corpus(
        self,
        input_jsonl: Path,
        output_jsonl: Path,
        target_langs: list[str],
        text_field: str = "text",
    ) -> int:
        alignments: list[AlignmentRecord] = []
        source_meta = SourceMetadata(
            source_id="parallel_corpus",
            source_type=SourceType.CONSTITUTION,
            language=LanguageCode.EN,
            license="public_domain_government",
        )

        for row in read_jsonl(input_jsonl):
            en_text = row.get(text_field, "")
            if not en_text:
                continue
            translations: dict[LanguageCode, str] = {LanguageCode.EN: en_text}
            for lang in target_langs:
                if lang == "en":
                    continue
                tr = self.translate_batch([en_text], "en", lang)[0]
                translations[LanguageCode(lang)] = tr

            alignments.append(
                AlignmentRecord(
                    align_id=str(uuid.uuid4()),
                    en_text=en_text,
                    translations=translations,
                    article=row.get("article"),
                    source=source_meta,
                )
            )

        return write_jsonl(output_jsonl, alignments)


def make_indictrans2_translator(model_name: str = "ai4bharat/indictrans2-indic-en-1B"):
    """Factory for IndicTrans2 — call from Colab after installing indic-trans2."""

    def _translate(texts: list[str], src: str, tgt: str) -> list[str]:
        # Placeholder hook — users wire AI4Bharat inference here
        # from indictrans import IndicTransToolkit
        raise NotImplementedError(
            f"Wire IndicTrans2 inference for {model_name}. See scripts/translate/run_indictrans2.py"
        )

    return _translate
