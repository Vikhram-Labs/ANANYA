#!/usr/bin/env python3
import argparse
import os

from ananya.evaluation.runner import EvaluationRunner
from ananya.inference.generate import chat_generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks", default="benchmarks")
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = p.parse_args()

    model = tokenizer = None
    if args.model_path:
        from ananya.inference.quantize import load_4bit_for_inference

        model, tokenizer = load_4bit_for_inference(args.base_model, args.model_path)

    def predict_fn(q: str) -> str:
        if model is None:
            return ""
        return chat_generate(
            model,
            tokenizer,
            [{"role": "user", "content": q}],
            max_new_tokens=256,
        )

    runner = EvaluationRunner(args.benchmarks)
    lb = runner.run_all(predict_fn)
    print(lb)


if __name__ == "__main__":
    main()
