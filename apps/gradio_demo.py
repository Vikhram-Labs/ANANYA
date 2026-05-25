"""Gradio demo — constitutional Q&A with optional RAG."""

from __future__ import annotations

import os

import gradio as gr

from ananya.inference.generate import chat_generate
from ananya.rag.pipeline import RAGPipeline


def build_demo():
    model = tokenizer = None
    rag: RAGPipeline | None = None

    index_dir = os.environ.get("RAG_INDEX", "data/indices/constitution")
    model_path = os.environ.get("MODEL_PATH", "outputs/checkpoints/sft/final")

    if os.path.isdir(index_dir):
        rag = RAGPipeline(index_dir=index_dir)

    try:
        from ananya.inference.quantize import load_4bit_for_inference

        model, tokenizer = load_4bit_for_inference(
            os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            adapter_path=model_path if os.path.isdir(model_path) else None,
        )
    except Exception as e:
        print(f"Model load skipped: {e}")

    def respond(query: str, language: str, use_rag: bool):
        if use_rag and rag:
            gen_fn = None
            if model and tokenizer:

                def gen_fn(prompt: str) -> str:
                    return chat_generate(
                        model,
                        tokenizer,
                        [{"role": "user", "content": prompt}],
                        max_new_tokens=512,
                    )

            rag.generate_fn = gen_fn
            result = rag.answer(query, language=language if language != "auto" else None)
            cites = "\n".join(
                f"- Art. {c['article']}: {c['title']} (score={c.get('score', 0):.2f})"
                for c in result.get("citations", [])
            )
            return result.get("answer") or result.get("prompt", ""), cites

        if model and tokenizer:
            ans = chat_generate(
                model,
                tokenizer,
                [
                    {
                        "role": "system",
                        "content": "You are ANANYA. Cite articles. Not legal advice.",
                    },
                    {"role": "user", "content": query},
                ],
            )
            return ans, ""
        return "Load MODEL_PATH or enable RAG index.", ""

    with gr.Blocks(title="ANANYA") as demo:
        gr.Markdown("# ANANYA — Indian Constitutional Assistant")
        with gr.Row():
            query = gr.Textbox(label="Question", placeholder="What are Fundamental Rights?")
            lang = gr.Dropdown(
                ["auto", "en", "hi", "ta", "bn", "te", "mr"],
                value="auto",
                label="Language",
            )
            use_rag = gr.Checkbox(value=True, label="Use RAG")
        btn = gr.Button("Ask")
        answer = gr.Textbox(label="Answer", lines=8)
        citations = gr.Textbox(label="Citations", lines=4)
        btn.click(respond, [query, lang, use_rag], [answer, citations])
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", share=False)
