"""YAML config loader with env substitution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def _substitute_env(value: Any) -> Any:
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(0))

        return pattern.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "extends" in raw:
        base_path = (path.parent / raw.pop("extends")).resolve()
        base = load_config(base_path)
        base.update(raw)
        raw = base
    return _substitute_env(raw)
