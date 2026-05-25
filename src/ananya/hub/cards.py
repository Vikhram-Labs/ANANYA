"""Model card & dataset card templates (HF Hub)."""

from __future__ import annotations


def render_model_card(repo_id: str, base_model: str, version: str) -> str:
    return f"""---
language:
- en
- hi
- ta
- bn
- te
- mr
license: apache-2.0
base_model: {base_model}
tags:
- legal
- constitution
- indic
- qlora
- sovereign-ai
library_name: transformers
pipeline_tag: text-generation
---

# {repo_id} ({version})

**ANANYA** — sovereign Indian constitutional multilingual SLM (QLoRA on `{base_model}`).

## Intended use

- Constitutional Q&A with article citations
- Multilingual legal reasoning (hi, ta, bn, te, mr)
- RAG-backed answers over Indian Constitution

## Limitations

Not legal advice. May hallucinate; verify against official Gazette text.

## Training

- QLoRA / Unsloth on Colab T4
- Continued pretraining + instruction tuning
- See [GitHub](https://github.com/vikhram-labs/ANANYA)

## Evaluation

| Task | EM | F1 | ROUGE-L |
|------|-----|-----|---------|
| constitutional_reasoning | — | — | — |

## Citation

```bibtex
@misc{{ananya2025,
  title={{ANANYA: A Multilingual Indian Constitutional Legal SLM}},
  author={{ANANYA Contributors}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```
"""


def render_dataset_card(repo_id: str, version: str) -> str:
    return f"""---
language:
- en
- hi
- ta
- bn
- te
- mr
license: apache-2.0
task_categories:
- text-generation
- question-answering
tags:
- legal
- constitution
- multilingual
pretty_name: Indic Legal Constitution Multilingual Corpus
size_categories:
- 1K<n<10K
---

# {repo_id} ({version})

Multilingual Indian legal / constitutional corpus for CPT, SFT, and RAG.

## Schema

- `pretrain`: text + article metadata
- `instruct`: ChatML messages
- `retrieval`: chunked articles
- `evaluation`: benchmark items
- `alignment`: parallel translations

## Sources

Constitution (english), amendments, SC summaries, synthetic QA.

## Provenance

Each record includes `source` metadata (type, license, language).
"""
