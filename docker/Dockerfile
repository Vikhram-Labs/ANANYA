FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY apps ./apps
COPY configs ./configs

ENV MODEL_PATH=/models/adapter
ENV RAG_INDEX=/data/indices/constitution
ENV BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

EXPOSE 8000 7860

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
