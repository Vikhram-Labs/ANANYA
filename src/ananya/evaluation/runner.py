"""Benchmark evaluation runner with leaderboard JSON export."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ananya.evaluation.metrics import compute_metrics
from ananya.utils.io import read_jsonl
from ananya.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationRunner:
    def __init__(
        self,
        benchmark_dir: str | Path,
        output_dir: str | Path = "outputs/reports",
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_task(self, task_name: str) -> list[dict[str, Any]]:
        path = self.benchmark_dir / f"{task_name}.jsonl"
        return list(read_jsonl(path))

    def run_task(
        self,
        task_name: str,
        predict_fn: Callable[[str], str],
        task_type: str = "qa",
    ) -> dict[str, float]:
        items = self.load_task(task_name)
        preds, refs = [], []
        for item in items:
            q = item.get("question", item.get("prompt", ""))
            preds.append(predict_fn(q))
            refs.append(item.get("reference_answer", item.get("answer", "")))

        metrics = compute_metrics(preds, refs, task=task_type)
        report = {
            "task": task_name,
            "task_type": task_type,
            "n": len(items),
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }
        out_path = self.output_dir / f"{task_name}_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out_path)
        return metrics

    def run_all(
        self,
        predict_fn: Callable[[str], str],
        tasks: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        tasks = tasks or [
            "constitutional_reasoning",
            "multilingual_qa",
            "legal_summarization",
            "hallucination_detection",
            "cross_lingual_retrieval",
            "article_comparison",
            "amendment_understanding",
        ]
        leaderboard: dict[str, dict[str, float]] = {}
        for task in tasks:
            path = self.benchmark_dir / f"{task}.jsonl"
            if not path.exists():
                logger.warning("Skipping missing benchmark: %s", path)
                continue
            task_type = "retrieval" if "retrieval" in task else "qa"
            if "hallucination" in task:
                task_type = "hallucination"
            leaderboard[task] = self.run_task(task, predict_fn, task_type=task_type)

        lb_path = self.output_dir / "leaderboard.json"
        lb_path.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
        return leaderboard
