"""Hugging Face Hub automation — datasets, models, semantic versioning."""

from __future__ import annotations

import os
import re
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from ananya.hub.cards import render_dataset_card, render_model_card
from ananya.utils.logging import get_logger

logger = get_logger(__name__)

# Naming: {org}/indic-legal-{component}-{size}-{stage}-v{major}
REPO_PATTERN = r"^[\w-]+/indic-legal-[\w-]+$"


def bump_version(tag: str, part: str = "patch") -> str:
    m = re.match(r"v?(\d+)\.(\d+)\.(\d+)", tag.lstrip("v"))
    if not m:
        return "v0.1.0"
    major, minor, patch = map(int, m.groups())
    if part == "major":
        return f"v{major + 1}.0.0"
    if part == "minor":
        return f"v{major}.{minor + 1}.0"
    return f"v{major}.{minor}.{patch + 1}"


def push_dataset(
    data_dir: str,
    repo_id: str,
    private: bool = False,
    version_tag: str = "v0.1.0",
) -> str:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    card = render_dataset_card(repo_id, version_tag)
    (Path(data_dir) / "README.md").write_text(card, encoding="utf-8")
    api.upload_folder(
        folder_path=data_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Release {version_tag}",
    )
    api.create_tag(repo_id, tag=version_tag, repo_type="dataset")
    logger.info("Pushed dataset %s @ %s", repo_id, version_tag)
    return repo_id


def push_model(
    model_dir: str,
    repo_id: str,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    private: bool = False,
    version_tag: str = "v0.1.0",
) -> str:
    if not repo_id:
        raise ValueError("HF_MODEL_REPO required")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    card = render_model_card(repo_id, base_model, version_tag)
    out = Path(model_dir)
    (out / "README.md").write_text(card, encoding="utf-8")
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Release {version_tag}",
    )
    api.create_tag(repo_id, tag=version_tag, repo_type="model")
    logger.info("Pushed model %s @ %s", repo_id, version_tag)
    return repo_id
