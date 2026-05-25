from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.schema.edit_types import coarse_error_type


DEFAULT_COMBINED_SCORE_WEIGHTS = {
    "exact_match": 1.0,
    "edit_f1": 1.0,
    "spelling_f1": 1.0,
    "punctuation_f1": 1.0,
    "dirty_improved_rate": 1.0,
    "dirty_worse_rate": -2.0,
    "clean_overcorrection_rate": -3.0,
}


@dataclass(frozen=True)
class _RowMetricAnalysis:
    row: dict[str, Any]
    predicted_edits: list[Edit]
    gold_edits: list[Edit]
    true_positive: int
    false_positive: int
    false_negative: int


def compute_metrics(rows: Iterable[dict], weights: dict[str, float] | None = None) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        metrics = _zero_metrics()
        metrics["combined_score"] = combined_score(metrics, weights)
        metrics.update(_scoped_metrics_from_analyses([], weights))
        return metrics

    analyses = _analyze_rows(rows)
    metrics = _base_metrics_from_analyses(analyses)
    metrics["combined_score"] = combined_score(metrics, weights)
    metrics.update(_scoped_metrics_from_analyses(analyses, weights))
    return metrics


def compute_metrics_from_edits(
    rows: Iterable[dict],
    predicted_edits_by_row: list[list[Edit]],
    gold_edits_by_row: list[list[Edit]],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    rows = list(rows)
    if len(rows) != len(predicted_edits_by_row) or len(rows) != len(gold_edits_by_row):
        raise ValueError("rows and edit lists must have the same length")
    if not rows:
        metrics = _zero_metrics()
        metrics["combined_score"] = combined_score(metrics, weights)
        metrics.update(_scoped_metrics_from_analyses([], weights))
        return metrics

    analyses: list[_RowMetricAnalysis] = []
    for row, predicted_edits, gold_edits in zip(rows, predicted_edits_by_row, gold_edits_by_row, strict=False):
        counts = _edit_counts(predicted_edits, gold_edits)
        analyses.append(
            _RowMetricAnalysis(
                row=row,
                predicted_edits=predicted_edits,
                gold_edits=gold_edits,
                true_positive=counts["true_positive"],
                false_positive=counts["false_positive"],
                false_negative=counts["false_negative"],
            )
        )
    metrics = _base_metrics_from_analyses(analyses)
    metrics["combined_score"] = combined_score(metrics, weights)
    metrics.update(_scoped_metrics_from_analyses(analyses, weights))
    return metrics


def _analyze_rows(rows: list[dict[str, Any]]) -> list[_RowMetricAnalysis]:
    analyzer = DiffAnalyzer()
    analyses: list[_RowMetricAnalysis] = []
    for row in rows:
        predicted_edits = analyzer.analyze(row["source"], row["prediction"])
        gold_edits = analyzer.analyze(row["source"], row["target"])
        counts = _edit_counts(predicted_edits, gold_edits)
        analyses.append(
            _RowMetricAnalysis(
                row=row,
                predicted_edits=predicted_edits,
                gold_edits=gold_edits,
                true_positive=counts["true_positive"],
                false_positive=counts["false_positive"],
                false_negative=counts["false_negative"],
            )
        )
    return analyses


def _base_metrics_from_analyses(analyses: list[_RowMetricAnalysis]) -> dict[str, float]:
    if not analyses:
        return _zero_metrics()

    exact = sum(_is_exact(analysis.row) for analysis in analyses) / len(analyses)
    dirty = [analysis for analysis in analyses if not analysis.row.get("is_clean", False)]
    clean = [analysis for analysis in analyses if analysis.row.get("is_clean", False)]

    dirty_improved = _safe_rate(sum(_is_dirty_improved_analysis(analysis) for analysis in dirty), len(dirty))
    dirty_worse = _safe_rate(sum(_is_dirty_worse_analysis(analysis) for analysis in dirty), len(dirty))
    clean_over = _safe_rate(sum(not _is_exact(analysis.row) for analysis in clean), len(clean))

    edit_scores = _edit_scores_from_analyses(analyses)
    return {
        "exact_match": exact,
        "dirty_improved_rate": dirty_improved,
        "dirty_worse_rate": dirty_worse,
        "clean_overcorrection_rate": clean_over,
        **edit_scores,
    }


def combined_score(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_COMBINED_SCORE_WEIGHTS
    return sum(metrics.get(name, 0.0) * weight for name, weight in weights.items())


def _scoped_metrics_from_analyses(
    analyses: list[_RowMetricAnalysis],
    weights: dict[str, float] | None,
) -> dict[str, float]:
    scopes = {
        "real": [analysis for analysis in analyses if _is_real(analysis.row)],
        "synthetic": [analysis for analysis in analyses if _is_synthetic(analysis.row)],
        "clean": [analysis for analysis in analyses if analysis.row.get("is_clean", False)],
    }
    scoped: dict[str, float] = {}
    for scope_name, scope_analyses in scopes.items():
        scope_metrics = _base_metrics_from_analyses(scope_analyses)
        scope_metrics["combined_score"] = combined_score(scope_metrics, weights)
        scoped.update({f"{scope_name}_{name}": value for name, value in scope_metrics.items()})
    return scoped


def _edit_scores_from_analyses(analyses: list[_RowMetricAnalysis]) -> dict[str, float]:
    predicted: Counter[tuple[Any, ...]] = Counter()
    gold: Counter[tuple[Any, ...]] = Counter()
    predicted_spelling: Counter[tuple[Any, ...]] = Counter()
    gold_spelling: Counter[tuple[Any, ...]] = Counter()
    predicted_punct: Counter[tuple[Any, ...]] = Counter()
    gold_punct: Counter[tuple[Any, ...]] = Counter()

    for row_id, analysis in enumerate(analyses):
        pred_edits = analysis.predicted_edits
        gold_edits = analysis.gold_edits
        predicted.update(_edit_keys(pred_edits, row_id=row_id))
        gold.update(_edit_keys(gold_edits, row_id=row_id))
        predicted_spelling.update(
            _edit_keys(
                _filter_edits_by_coarse_type(pred_edits, {"spelling", "split_join", "hyphen", "case"}),
                row_id=row_id,
            )
        )
        gold_spelling.update(
            _edit_keys(
                _filter_edits_by_coarse_type(gold_edits, {"spelling", "split_join", "hyphen", "case"}),
                row_id=row_id,
            )
        )
        predicted_punct.update(
            _edit_keys(
                _filter_edits_by_coarse_type(pred_edits, {"punctuation", "final_punctuation"}),
                row_id=row_id,
            )
        )
        gold_punct.update(
            _edit_keys(
                _filter_edits_by_coarse_type(gold_edits, {"punctuation", "final_punctuation"}),
                row_id=row_id,
            )
        )

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


def edit_key(edit: Edit, row_id: int | None = None) -> tuple[Any, ...]:
    key = (edit.edit_type, edit.start, edit.end, edit.source, edit.replacement)
    if row_id is None:
        return key
    return (row_id, *key)


def mark_correct_edits(predicted_edits: list[Edit], gold_edits: list[Edit]) -> list[bool]:
    remaining_gold = Counter(edit_key(edit) for edit in gold_edits)
    flags: list[bool] = []
    for edit in predicted_edits:
        key = edit_key(edit)
        is_correct = remaining_gold[key] > 0
        flags.append(is_correct)
        if is_correct:
            remaining_gold[key] -= 1
    return flags


def is_dirty_improved_row(row: dict[str, Any]) -> bool:
    if row.get("is_clean", False) or row["source"] == row["target"]:
        return False
    return _row_edit_counts(row)["true_positive"] > 0


def is_dirty_worse_row(row: dict[str, Any]) -> bool:
    if row.get("is_clean", False) or row["source"] == row["target"]:
        return False
    return _row_edit_counts(row)["false_positive"] > 0


def _is_dirty_improved_analysis(analysis: _RowMetricAnalysis) -> bool:
    row = analysis.row
    if row.get("is_clean", False) or row["source"] == row["target"]:
        return False
    return analysis.true_positive > 0


def _is_dirty_worse_analysis(analysis: _RowMetricAnalysis) -> bool:
    row = analysis.row
    if row.get("is_clean", False) or row["source"] == row["target"]:
        return False
    return analysis.false_positive > 0


def _row_edit_counts(row: dict[str, Any]) -> dict[str, int]:
    analyzer = DiffAnalyzer()
    predicted_edits = analyzer.analyze(row["source"], row["prediction"])
    gold_edits = analyzer.analyze(row["source"], row["target"])
    return _edit_counts(predicted_edits, gold_edits)


def _edit_counts(predicted_edits: list[Edit], gold_edits: list[Edit]) -> dict[str, int]:
    predicted = Counter(edit_key(edit) for edit in predicted_edits)
    gold = Counter(edit_key(edit) for edit in gold_edits)
    true_positive = sum((predicted & gold).values())
    predicted_total = sum(predicted.values())
    gold_total = sum(gold.values())
    return {
        "true_positive": true_positive,
        "false_positive": max(0, predicted_total - true_positive),
        "false_negative": max(0, gold_total - true_positive),
    }


def _edit_keys(edits: list[Edit], row_id: int) -> list[tuple[Any, ...]]:
    return [edit_key(edit, row_id=row_id) for edit in edits]


def _filter_edits_by_coarse_type(edits: list[Edit], coarse_types: set[str]) -> list[Edit]:
    return [edit for edit in edits if coarse_error_type(edit.edit_type) in coarse_types]


def _precision_recall_f1(predicted: Counter[tuple[Any, ...]], gold: Counter[tuple[Any, ...]]) -> tuple[float, float, float]:
    true_positive = sum((predicted & gold).values())
    predicted_total = sum(predicted.values())
    gold_total = sum(gold.values())
    precision = _safe_rate(true_positive, predicted_total)
    recall = _safe_rate(true_positive, gold_total)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _is_exact(row: dict[str, Any]) -> bool:
    return row["prediction"] == row["target"]


def _is_real(row: dict[str, Any]) -> bool:
    return not row.get("is_clean", False) and not _as_bool(row.get("is_synthetic", False))


def _is_synthetic(row: dict[str, Any]) -> bool:
    return not row.get("is_clean", False) and _as_bool(row.get("is_synthetic", False))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


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
