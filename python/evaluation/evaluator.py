"""Evaluation pipeline for multilingual misinformation detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.logger import log_evaluation_records

_VALID_LABELS = {"true", "false", "uncertain"}
_FALLBACK_DATASET = [
    {
        "claim": "Water boils at 100 degrees Celsius",
        "label": "true",
        "language": "en",
    },
    {
        "claim": "Vaccines contain microchips",
        "label": "false",
        "language": "en",
    },
]


def _normalize_label(label: str) -> str:
    normalized = (label or "").strip().lower()
    if normalized not in _VALID_LABELS:
        return "uncertain"
    return normalized


def _precision_recall_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for actual, predicted in zip(y_true, y_pred):
        if predicted == actual:
            tp += 1
        else:
            fp += 1
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _retrieval_quality_score(claim: str, evidence: list[str]) -> float:
    claim_tokens = {token for token in claim.lower().split() if token}
    if not claim_tokens:
        return 0.0
    for item in evidence:
        evidence_tokens = {token for token in item.lower().split() if token}
        if claim_tokens & evidence_tokens:
            return 1.0
    return 0.0


def _explanation_quality_score(explanation: str, evidence: list[str]) -> float:
    length_ok = 1.0 if len((explanation or "").split()) >= 12 else 0.0
    citation_markers = ["source", "evidence", "[", "]", "http", "chunk"]
    has_citation = any(marker in (explanation or "").lower() for marker in citation_markers)
    has_evidence = len(evidence) > 0
    return (length_ok + (1.0 if has_citation or has_evidence else 0.0)) / 2.0


def _load_samples(dataset_file: Path) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    if not dataset_file.exists():
        return _FALLBACK_DATASET, {
            "status": "failed",
            "error": "Dataset missing",
            "hint": "Create data/evaluation_dataset.json or run dataset generator",
        }

    samples = json.loads(dataset_file.read_text(encoding="utf-8"))
    return samples, None


def run_evaluation(dataset_path: str | Path) -> dict[str, Any]:
    from rag.pipeline import run_pipeline

    dataset_file = Path(dataset_path)
    samples, missing_info = _load_samples(dataset_file)

    y_true: list[str] = []
    y_pred: list[str] = []
    retrieval_scores: list[float] = []
    explanation_scores: list[float] = []
    records: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        claim = str(sample.get("claim", "")).strip()
        true_label = _normalize_label(str(sample.get("label", "uncertain")))

        result = run_pipeline(claim)
        pred_label = _normalize_label(str(result.get("verdict", "uncertain")))
        evidence = [str(item) for item in result.get("evidence", [])]
        explanation = str(result.get("explanation", ""))

        retrieval_score = _retrieval_quality_score(claim, evidence)
        explanation_score = _explanation_quality_score(explanation, evidence)

        y_true.append(true_label)
        y_pred.append(pred_label)
        retrieval_scores.append(retrieval_score)
        explanation_scores.append(explanation_score)

        records.append(
            {
                "index": idx,
                "claim": claim,
                "language": sample.get("language", "en"),
                "ground_truth": true_label,
                "predicted": pred_label,
                "is_correct": pred_label == true_label,
                "explanation": explanation,
                "retrieved_evidence": evidence,
                "retrieval_score": retrieval_score,
                "explanation_score": explanation_score,
            }
        )

    total = len(y_true)
    accuracy = sum(int(a == p) for a, p in zip(y_true, y_pred)) / total if total else 0.0
    precision, recall, f1 = _precision_recall_f1(y_true, y_pred)
    retrieval_score = sum(retrieval_scores) / total if total else 0.0
    explanation_score = sum(explanation_scores) / total if total else 0.0

    log_evaluation_records(records)

    response: dict[str, Any] = {
        "status": "success" if missing_info is None else "failed",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "retrieval_score": round(retrieval_score, 4),
        "explanation_quality": round(explanation_score, 4),
        "total_samples": total,
        "fallback_dataset_used": missing_info is not None,
    }
    if missing_info is not None:
        response.update(missing_info)
    return response
