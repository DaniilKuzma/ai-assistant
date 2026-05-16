from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.evaluation.metrics import compute_metrics
from src.evaluation.reports import write_edit_logs, write_required_evaluation_reports
from src.inference.corrector import Corrector


def evaluate_rows(
    rows: Iterable[dict],
    corrector: Corrector | None = None,
    output_dir: str | Path | None = None,
    *,
    metric_weights: dict[str, float] | None = None,
    show_progress: bool = False,
) -> dict[str, float]:
    corrector = corrector or Corrector()
    row_list = list(rows)
    evaluated = []
    accepted: list[dict] = []
    rejected: list[dict] = []
    for row_id, row in enumerate(_with_progress(row_list, enabled=show_progress, description="Evaluating")):
        result = corrector.correct(row["source"])
        prediction = result.corrected_text
        evaluated.append({**row, "prediction": prediction})
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
    metrics = compute_metrics(evaluated, weights=metric_weights)
    if output_dir is not None:
        write_required_evaluation_reports(evaluated, metrics, output_dir)
        write_edit_logs(accepted, rejected, output_dir)
    return metrics


def _with_progress(rows: list[dict], *, enabled: bool, description: str):
    if not enabled:
        return rows
    try:
        from tqdm.auto import tqdm
    except Exception:
        return rows
    return tqdm(rows, total=len(rows), desc=description, dynamic_ncols=True)
