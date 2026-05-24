from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def operation_accuracy(predictions: list[str], targets: list[str]) -> float:
    if not targets:
        return 0.0
    return sum(p == t for p, t in zip(predictions, targets)) / len(targets)


def confusion_matrix(predictions: list[str], targets: list[str]) -> tuple[list[str], list[list[int]]]:
    labels = sorted(set(predictions) | set(targets))
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for pred, target in zip(predictions, targets):
        matrix[idx[target]][idx[pred]] += 1
    return labels, matrix


def precision_recall_f1(predictions: list[str], targets: list[str]) -> dict[str, dict[str, float]]:
    labels = sorted(set(predictions) | set(targets))
    result: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(p == label and t == label for p, t in zip(predictions, targets))
        fp = sum(p == label and t != label for p, t in zip(predictions, targets))
        fn = sum(p != label and t == label for p, t in zip(predictions, targets))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {"precision": precision, "recall": recall, "f1": f1, "support": float(tp + fn)}
    return result


def top_k_accuracy(topk_predictions: list[list[str]], targets: list[str], k: int = 5) -> float | None:
    if not topk_predictions:
        return None
    if not targets:
        return 0.0
    hits = 0
    for preds, target in zip(topk_predictions, targets):
        hits += target in preds[:k]
    return hits / len(targets)


def summarize_text_metrics(
    predictions: list[str],
    targets: list[str],
    topk_predictions: list[list[str]] | None = None,
) -> dict[str, Any]:
    labels, matrix = confusion_matrix(predictions, targets)
    per_class = precision_recall_f1(predictions, targets)
    metrics: dict[str, Any] = {
        "operation_type_accuracy": operation_accuracy(predictions, targets),
        "labels": labels,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "stop_metrics": per_class.get("<STOP>", {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}),
        "target_histogram": dict(Counter(targets)),
        "prediction_histogram": dict(Counter(predictions)),
    }
    if topk_predictions is not None:
        metrics["top_5_accuracy"] = top_k_accuracy(topk_predictions, targets, k=5)
    return metrics

