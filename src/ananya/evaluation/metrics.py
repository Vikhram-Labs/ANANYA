"""Evaluation metrics for legal / constitutional NLP."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(ref))


def token_f1(pred: str, ref: str) -> float:
    p_tok = normalize_answer(pred).split()
    r_tok = normalize_answer(ref).split()
    if not p_tok and not r_tok:
        return 1.0
    if not p_tok or not r_tok:
        return 0.0
    common = Counter(p_tok) & Counter(r_tok)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_tok)
    rec = num_same / len(r_tok)
    return 2 * prec * rec / (prec + rec)


def retrieval_recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    hits = sum(1 for g in gold_ids if g in top)
    return hits / len(gold_ids)


def article_citation_accuracy(pred: str, gold_articles: list[int]) -> float:
    """Fraction of cited articles in prediction that match gold."""
    found = {int(m) for m in re.findall(r"Article\s+(\d+)", pred, re.I)}
    if not gold_articles:
        return 1.0 if not found else 0.0
    if not found:
        return 0.0
    return len(found & set(gold_articles)) / len(set(gold_articles))


def hallucination_rate(
    pred: str,
    evidence_texts: list[str],
    threshold: float = 0.3,
) -> float:
    """Heuristic: fraction of prediction tokens not supported by evidence n-grams."""
    pred_tokens = set(normalize_answer(pred).split())
    if not pred_tokens:
        return 0.0
    evidence = " ".join(evidence_texts)
    ev_tokens = set(normalize_answer(evidence).split())
    unsupported = pred_tokens - ev_tokens
    rate = len(unsupported) / len(pred_tokens)
    return float(rate > threshold)


def compute_metrics(
    predictions: list[str],
    references: list[str],
    task: str = "qa",
    **kwargs: Any,
) -> dict[str, float]:
    scores: dict[str, list[float]] = {
        "exact_match": [],
        "f1": [],
    }
    for pred, ref in zip(predictions, references):
        scores["exact_match"].append(exact_match(pred, ref))
        scores["f1"].append(token_f1(pred, ref))

    out = {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}

    if task == "retrieval":
        k = kwargs.get("k", 5)
        recalls = []
        for ret, gold in zip(kwargs.get("retrieved", []), kwargs.get("gold_ids", [])):
            recalls.append(retrieval_recall_at_k(ret, gold, k))
        out[f"recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0

    if task == "hallucination":
        rates = []
        for pred, ev in zip(predictions, kwargs.get("evidence", [])):
            rates.append(hallucination_rate(pred, ev if isinstance(ev, list) else [ev]))
        out["hallucination_rate"] = sum(rates) / len(rates) if rates else 0.0

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_vals = []
        for pred, ref in zip(predictions, references):
            rouge_vals.append(scorer.score(ref, pred)["rougeL"].fmeasure)
        out["rouge_l"] = sum(rouge_vals) / len(rouge_vals)
    except ImportError:
        pass

    return out
