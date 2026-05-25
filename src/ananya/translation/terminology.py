"""Legal terminology consistency — mask, translate, restore."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class TerminologyEngine:
    def __init__(self, glossary_path: str | Path):
        with Path(glossary_path).open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.terms: list[dict[str, str]] = data.get("terms", [])
        self.patterns = [
            re.compile(p, re.I) for p in data.get("protected_patterns", [])
        ]
        self._placeholder_map: dict[str, str] = {}

    def mask(self, text: str, target_lang: str) -> tuple[str, dict[str, str]]:
        placeholders: dict[str, str] = {}
        masked = text
        idx = 0

        for term in self.terms:
            en = term.get("en", "")
            if not en:
                continue
            tgt = term.get(target_lang, en)
            if en in masked:
                key = f"__LEGALTERM_{idx}__"
                masked = masked.replace(en, key)
                placeholders[key] = tgt
                idx += 1

        for pat in self.patterns:
            for m in pat.finditer(masked):
                key = f"__LEGALPAT_{idx}__"
                placeholders[key] = m.group(0)
                masked = masked[: m.start()] + key + masked[m.end() :]
                idx += 1

        return masked, placeholders

    def restore(self, text: str, placeholders: dict[str, str]) -> str:
        for key, value in placeholders.items():
            text = text.replace(key, value)
        return text

    def validate_glossary_coverage(self, text: str, lang: str) -> list[str]:
        missing = []
        for term in self.terms:
            expected = term.get(lang)
            if expected and expected not in text:
                en = term.get("en", "")
                if en and en.lower() in text.lower():
                    missing.append(en)
        return missing
