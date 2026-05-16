from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.metrics import compute_metrics
from src.evaluation.reports import write_edit_logs, write_required_evaluation_reports
from src.inference.corrector import Corrector


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    edit_scores: list[dict[str, Any]]
    accepted_edits: list[dict[str, Any]]
    rejected_edits: list[dict[str, Any]]


def evaluate_rows(
    rows: Iterable[dict],
    corrector: Corrector | None = None,
    output_dir: str | Path | None = None,
    *,
    metric_weights: dict[str, float] | None = None,
    show_progress: bool = False,
) -> dict[str, float]:
    return evaluate_rows_detailed(
        rows,
        corrector=corrector,
        output_dir=output_dir,
        metric_weights=metric_weights,
        show_progress=show_progress,
    ).metrics


def evaluate_rows_detailed(
    rows: Iterable[dict],
    corrector: Corrector | None = None,
    output_dir: str | Path | None = None,
    *,
    metric_weights: dict[str, float] | None = None,
    show_progress: bool = False,
) -> EvaluationResult:
    corrector = corrector or Corrector()
    row_list = list(rows)
    evaluated = []
    accepted: list[dict] = []
    rejected: list[dict] = []
    edit_scores: list[dict[str, Any]] = []
    for row_id, row in enumerate(_with_progress(row_list, enabled=show_progress, description="Evaluating")):
        result = corrector.correct(row["source"])
        prediction = result.corrected_text
        evaluated_row = {**row, "prediction": prediction}
        evaluated.append(evaluated_row)
        row_is_correct = prediction == row["target"]
        for edit in result.edits:
            serialized = {
                "row_id": row_id,
                "source": edit.source,
                "replacement": edit.replacement,
                "edit_type": edit.edit_type,
                "status": edit.status,
                "reason": edit.reason,
                "confidence": edit.confidence,
            }
            if edit.status == "accepted":
                accepted.append(serialized)
            else:
                rejected.append(serialized)
            edit_scores.append(
                {
                    "confidence": edit.confidence,
                    "is_correct": row_is_correct and edit.status == "accepted",
                }
            )
    metrics = compute_metrics(evaluated, weights=metric_weights)
    if output_dir is not None:
        write_required_evaluation_reports(evaluated, metrics, output_dir)
        write_edit_logs(accepted, rejected, output_dir)
    return EvaluationResult(
        metrics=metrics,
        edit_scores=edit_scores or [{"confidence": 0.0, "is_correct": False}],
        accepted_edits=accepted,
        rejected_edits=rejected,
    )


def _with_progress(rows: list[dict], *, enabled: bool, description: str):
    if not enabled:
        return rows
    try:
        from tqdm.auto import tqdm
    except Exception:
        return rows
    return tqdm(rows, total=len(rows), desc=description, dynamic_ncols=True)
