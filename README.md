# ANANYA

**Adaptive Neural Architecture for Native-language Yielded AI** — QLoRA fine-tuning, RAG, and Hugging Face-native deployment optimized for **free Colab T4 GPUs**.

Built on structured constitution data (via the [`indianconstitution`](https://pypi.org/project/indianconstitution) package) and extended into a full legal-multilingual research stack.

[![CI](https://img.shields.io/badge/CI-github_actions-blue)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![HF](https://img.shields.io/badge/🤗-Datasets%20%26%20Models-yellow)](https://huggingface.co/)

## Features

- **Parameter-efficient training**: QLoRA / PEFT / Unsloth on Qwen2.5-1.5B (T4-safe)
- **Multilingual pipeline**: Hindi, Tamil, Bengali, Telugu, Marathi (+ English source)
- **Legal terminology engine**: glossary mask-and-restore for IndicTrans2
- **RAG**: FAISS + multilingual embeddings, citation-aware answers
- **AnanyaBench**: constitutional reasoning, hallucination, cross-lingual retrieval
- **Deployment**: Gradio demo, FastAPI, Docker, GGUF export path

## Quick Start (local)

```bash
git clone https://github.com/vikhram-labs/ANANYA.git
cd ANANYA
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env  # set HF_TOKEN

# Prepare sample corpus
python scripts/data/prepare_corpus.py --constitution-json data/raw/sample_constitution.json

# Build RAG index
python scripts/rag/build_index.py

# Gradio (model optional if RAG-only)
python apps/gradio_demo.py
```

## Colab T4 Training

Open [`notebooks/01_colab_setup_and_train.ipynb`](notebooks/01_colab_setup_and_train.ipynb) or run:

```bash
pip install -r requirements-colab.txt
python scripts/train/run_sft_colab.py --config configs/training/sft_colab_t4.yaml
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for memory tuning and full pipeline order.

## Repository Layout

```
configs/          YAML configuration & legal glossary
src/ananya/  Core library (data, train, rag, eval, hub)
scripts/          CLI entrypoints for each stage
benchmarks/       AnanyaBench tasks
apps/             Gradio + FastAPI
notebooks/        Colab workflows
docs/             Architecture, research, branding
```

## Hugging Face Repos (convention)

| Artifact | Example ID |
|----------|------------|
| Dataset | `{org}/ananya-constitution-multilingual` |
| Model | `{org}/ananya-1.5b-instruct-v0.1.0` |

```bash
export HF_TOKEN=...
python scripts/hub/push_all.py --version v0.1.0
```

## API

```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is Article 14?", "use_rag": true}'
```

## Docker

```bash
docker compose -f docker/docker-compose.yml up api
```

## Documentation

- [Architecture & implementation plan](docs/ARCHITECTURE.md)
- [Research & paper outline](docs/RESEARCH.md)
- [Branding & launch](docs/BRANDING.md)

## Citation

```bibtex
@misc{ananya2025,
  title={ANANYA: Multilingual Indian Constitutional Legal SLM},
  author={ANANYA Contributors},
  year={2025},
  howpublished={\url{https://github.com/vikhram-labs/ANANYA}}
}
```

## Disclaimer

This project provides **informational** constitutional NLP assistance, not legal advice. Verify all outputs against official government publications.

## License

Apache-2.0 — see [LICENSE](LICENSE).
