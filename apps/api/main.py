"""FastAPI backend — quantized inference + RAG."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ananya.inference.generate import chat_generate
from ananya.rag.pipeline import RAGPipeline

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    base = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter = os.environ.get("MODEL_PATH")
    index_dir = os.environ.get("RAG_INDEX", "data/indices/constitution")

    if os.path.isdir(index_dir):
        _state["rag"] = RAGPipeline(index_dir=index_dir)

    try:
        from ananya.inference.quantize import load_4bit_for_inference

        _state["model"], _state["tokenizer"] = load_4bit_for_inference(base, adapter)
    except Exception:
        _state["model"] = _state["tokenizer"] = None

    yield
    _state.clear()


app = FastAPI(title="ANANYA API", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    language: str | None = None
    use_rag: bool = True
    max_new_tokens: int = Field(default=512, le=2048)


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state.get("model") is not None,
        "rag_loaded": _state.get("rag") is not None,
    }


@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest):
    rag: RAGPipeline | None = _state.get("rag")
    model, tokenizer = _state.get("model"), _state.get("tokenizer")

    if req.use_rag and rag:
        if model and tokenizer:

            def gen(prompt: str) -> str:
                return chat_generate(
                    model,
                    tokenizer,
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=req.max_new_tokens,
                )

            rag.generate_fn = gen
        result = rag.answer(req.question, language=req.language)
        return QueryResponse(
            answer=result.get("answer") or "",
            citations=result.get("citations", []),
        )

    if model and tokenizer:
        ans = chat_generate(
            model,
            tokenizer,
            [
                {"role": "system", "content": "ANANYA — cite articles."},
                {"role": "user", "content": req.question},
            ],
            max_new_tokens=req.max_new_tokens,
        )
        return QueryResponse(answer=ans)

    return QueryResponse(answer="Model not loaded.", citations=[])
