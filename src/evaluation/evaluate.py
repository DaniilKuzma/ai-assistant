from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.candidates.candidate_generator import CandidateGenerator
from src.evaluation.candidate_recall import write_candidate_recall_reports
from src.evaluation.metrics import compute_metrics, mark_correct_edits
from src.evaluation.reports import write_edit_logs, write_required_evaluation_reports
from src.evaluation.rule_metrics import write_rule_reports
from src.evaluation.score_distribution import write_candidate_score_distribution_by_rule
from src.inference.corrector import Corrector
from src.validation.diff_analyzer import DiffAnalyzer


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
    report_metadata: dict[str, Any] | None = None,
    candidate_generator: CandidateGenerator | None = None,
    candidate_recall_max_candidates: int | None = None,
) -> dict[str, float]:
    return evaluate_rows_detailed(
        rows,
        corrector=corrector,
        output_dir=output_dir,
        metric_weights=metric_weights,
        show_progress=show_progress,
        report_metadata=report_metadata,
        candidate_generator=candidate_generator,
        candidate_recall_max_candidates=candidate_recall_max_candidates,
    ).metrics


def evaluate_rows_detailed(
    rows: Iterable[dict],
    corrector: Corrector | None = None,
    output_dir: str | Path | None = None,
    *,
    metric_weights: dict[str, float] | None = None,
    show_progress: bool = False,
    report_metadata: dict[str, Any] | None = None,
    candidate_generator: CandidateGenerator | None = None,
    candidate_recall_max_candidates: int | None = None,
) -> EvaluationResult:
    corrector = corrector or Corrector()
    row_list = list(rows)
    evaluated = []
    accepted: list[dict] = []
    rejected: list[dict] = []
    edit_scores: list[dict[str, Any]] = []
    candidate_decisions: list[dict[str, Any]] = []
    analyzer = DiffAnalyzer()
    total_gold_edits = 0
    for row_id, row in enumerate(_with_progress(row_list, enabled=show_progress, description="Evaluating")):
        result = corrector.correct(row["source"])
        for decision in getattr(corrector, "last_candidate_decisions", []) or []:
            candidate_decisions.append({**decision, "row_id": row_id})
        prediction = result.corrected_text
        evaluated_row = {**row, "prediction": prediction}
        evaluated.append(evaluated_row)
        gold_edits = analyzer.analyze(row["source"], row["target"])
        total_gold_edits += len(gold_edits)
        accepted_for_scoring = [edit for edit in result.edits if edit.status == "accepted"]
        correct_flags = mark_correct_edits(accepted_for_scoring, gold_edits)
        correctness_by_identity = {
            id(edit): is_correct
            for edit, is_correct in zip(accepted_for_scoring, correct_flags, strict=False)
        }
        for edit in result.edits:
            serialized = {
                "row_id": row_id,
                "source": edit.source,
                "replacement": edit.replacement,
                "edit_type": edit.edit_type,
                "start": edit.start,
                "end": edit.end,
                "rule_id": edit.rule_id,
                "status": edit.status,
                "reason": edit.reason,
                "confidence": edit.confidence,
            }
            if edit.status == "accepted":
                accepted.append(serialized)
            else:
                rejected.append(serialized)
            if edit.status == "accepted":
                edit_scores.append(
                    {
                        "confidence": edit.confidence,
                        "is_correct": correctness_by_identity.get(id(edit), False),
                        "rule_id": edit.rule_id,
                    }
                )
    for score in edit_scores:
        score["total_gold_edits"] = total_gold_edits
    metrics = compute_metrics(evaluated, weights=metric_weights)
    if output_dir is not None:
        write_required_evaluation_reports(evaluated, metrics, output_dir, metadata=report_metadata)
        write_edit_logs(accepted, rejected, output_dir)
        write_rule_reports(evaluated, accepted, rejected, output_dir)
        write_candidate_score_distribution_by_rule(evaluated, candidate_decisions, accepted, rejected, output_dir)
        write_candidate_recall_reports(
            evaluated,
            output_dir,
            candidate_generator=candidate_generator,
            max_candidates=candidate_recall_max_candidates,
        )
    return EvaluationResult(
        metrics=metrics,
        edit_scores=edit_scores
        or [{"confidence": 0.0, "is_correct": False, "rule_id": "", "total_gold_edits": total_gold_edits}],
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
