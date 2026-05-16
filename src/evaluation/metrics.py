from __future__ import annotations

from collections.abc import Iterable

from src.validation.diff_analyzer import DiffAnalyzer
from src.validation.edit_classifier import coarse_error_type


def compute_metrics(rows: Iterable[dict]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return _zero_metrics()

    exact = sum(row["prediction"] == row["target"] for row in rows) / len(rows)
    dirty = [row for row in rows if not row.get("is_clean", False)]
    clean = [row for row in rows if row.get("is_clean", False)]

    dirty_improved = _safe_rate(sum(row["prediction"] == row["target"] and row["source"] != row["target"] for row in dirty), len(dirty))
    dirty_worse = _safe_rate(sum(row["prediction"] not in {row["source"], row["target"]} for row in dirty), len(dirty))
    clean_over = _safe_rate(sum(row["prediction"] != row["target"] for row in clean), len(clean))

    edit_scores = _edit_scores(rows)
    metrics = {
        "exact_match": exact,
        "dirty_improved_rate": dirty_improved,
        "dirty_worse_rate": dirty_worse,
        "clean_overcorrection_rate": clean_over,
        **edit_scores,
    }
    return metrics


def combined_score(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = weights or {
        "exact_match": 1.0,
        "edit_f1": 1.0,
        "spelling_f1": 1.0,
        "punctuation_f1": 1.0,
        "dirty_improved_rate": 1.0,
        "dirty_worse_rate": -2.0,
        "clean_overcorrection_rate": -3.0,
    }
    return sum(metrics.get(name, 0.0) * weight for name, weight in weights.items())


def _edit_scores(rows: list[dict]) -> dict[str, float]:
    analyzer = DiffAnalyzer()
    predicted: list[str] = []
    gold: list[str] = []
    predicted_spelling: list[str] = []
    gold_spelling: list[str] = []
    predicted_punct: list[str] = []
    gold_punct: list[str] = []

    for row in rows:
        pred_edits = analyzer.analyze(row["source"], row["prediction"])
        gold_edits = analyzer.analyze(row["source"], row["target"])
        predicted.extend(_keys(pred_edits))
        gold.extend(_keys(gold_edits))
        predicted_spelling.extend(_keys([edit for edit in pred_edits if coarse_error_type(edit.edit_type) in {"spelling", "split_join", "hyphen", "case"}]))
        gold_spelling.extend(_keys([edit for edit in gold_edits if coarse_error_type(edit.edit_type) in {"spelling", "split_join", "hyphen", "case"}]))
        predicted_punct.extend(_keys([edit for edit in pred_edits if coarse_error_type(edit.edit_type) in {"punctuation", "final_punctuation"}]))
        gold_punct.extend(_keys([edit for edit in gold_edits if coarse_error_type(edit.edit_type) in {"punctuation", "final_punctuation"}]))

    edit_p, edit_r, edit_f1 = _precision_recall_f1(predicted, gold)
    spelling_p, spelling_r, spelling_f1 = _precision_recall_f1(predicted_spelling, gold_spelling)
    punctuation_p, punctuation_r, punctuation_f1 = _precision_recall_f1(predicted_punct, gold_punct)

    return {
        "edit_precision": edit_p,
        "edit_recall": edit_r,
        "edit_f1": edit_f1,
        "spelling_precision": spelling_p,
        "spelling_recall": spelling_r,
        "spelling_f1": spelling_f1,
        "punctuation_precision": punctuation_p,
        "punctuation_recall": punctuation_r,
        "punctuation_f1": punctuation_f1,
    }


def _keys(edits: list) -> list[str]:
    return [f"{edit.edit_type}:{edit.source}->{edit.replacement}" for edit in edits]


def _precision_recall_f1(predicted: list[str], gold: list[str]) -> tuple[float, float, float]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    true_positive = len(predicted_set & gold_set)
    precision = _safe_rate(true_positive, len(predicted_set))
    recall = _safe_rate(true_positive, len(gold_set))
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _zero_metrics() -> dict[str, float]:
    names = [
        "exact_match",
        "dirty_improved_rate",
        "dirty_worse_rate",
        "clean_overcorrection_rate",
        "spelling_precision",
        "spelling_recall",
        "spelling_f1",
        "punctuation_precision",
        "punctuation_recall",
        "punctuation_f1",
        "edit_precision",
        "edit_recall",
        "edit_f1",
    ]
    return dict.fromkeys(names, 0.0)
