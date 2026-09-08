"""Set-based metrics for citation-level retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Sequence, Set


def _f1(predicted: Set[str], gold: Set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted)
    recall = true_positive / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def macro_f1(
    predictions: Sequence[Iterable[str]],
    gold_labels: Sequence[Iterable[str]],
) -> float:
    """Average per-query F1 across retrieval examples."""

    if len(predictions) != len(gold_labels):
        raise ValueError("predictions and gold_labels must have the same length")
    if not predictions:
        return 0.0
    return sum(_f1(set(predicted), set(gold)) for predicted, gold in zip(predictions, gold_labels)) / len(predictions)


def micro_f1(
    predictions: Sequence[Iterable[str]],
    gold_labels: Sequence[Iterable[str]],
) -> float:
    """Compute F1 after pooling citation counts across all queries."""

    if len(predictions) != len(gold_labels):
        raise ValueError("predictions and gold_labels must have the same length")
    true_positive = false_positive = false_negative = 0
    for predicted_values, gold_values in zip(predictions, gold_labels):
        predicted = set(predicted_values)
        gold = set(gold_values)
        true_positive += len(predicted & gold)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)

    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator
