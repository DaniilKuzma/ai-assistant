from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Any

from src.candidates.candidate_generator import Candidate
from src.evaluation.evaluate import EvaluationReportOptions, EvaluationResult, write_evaluation_outputs
from src.evaluation.metrics import compute_metrics_from_edits, mark_correct_edits
from src.inference.model_corrector import (
    BatchedFeatureModelScores,
    ModelCandidatePrediction,
    ModelPunctuationPrediction,
    TrainedModelCorrector,
    _annotate_decisions_with_validation,
    _apply_candidates,
    _apply_punctuation_predictions,
    _select_candidates_with_trace,
)
from src.inference.postprocess import normalize_spacing
from src.validation.diff_analyzer import DiffAnalyzer


CORE_TRAINING_METRICS = [
    "combined_score",
    "clean_overcorrection_rate",
    "dirty_worse_rate",
    "edit_precision",
    "edit_recall",
    "spelling_f1",
    "punctuation_f1",
]

PROFILE_COLUMNS = [
    "total_eval_time_sec",
    "feature_load_time_sec",
    "model_forward_time_sec",
    "thresholding_time_sec",
    "validation_time_sec",
    "edit_realization_time_sec",
    "metrics_time_sec",
    "detailed_reports_time_sec",
    "rows_per_sec",
    "eval_batch_size",
    "gpu_available",
    "cuda_device",
    "feature_cache_hit",
    "number_of_eval_examples",
    "bottleneck",
]


class EvaluationMode(str, Enum):
    FAST_DURING_TRAINING = "fast_during_training"
    FULL_AFTER_TRAINING = "full_after_training"


@dataclass(frozen=True)
class EvaluationRuntimeConfig:
    batch_size: int
    fast_during_training: bool
    max_eval_examples_per_epoch: int
    full_eval_after_training: bool
    full_eval_examples: int
    write_detailed_reports_during_training: bool
    write_accepted_rejected_during_training: bool
    write_score_distribution_during_training: bool
    write_rule_worse_examples_during_training: bool


@dataclass(frozen=True)
class EvaluationProfile:
    total_eval_time_sec: float
    feature_load_time_sec: float
    model_forward_time_sec: float
    thresholding_time_sec: float
    validation_time_sec: float
    edit_realization_time_sec: float
    metrics_time_sec: float
    detailed_reports_time_sec: float
    rows_per_sec: float
    eval_batch_size: int
    gpu_available: bool
    cuda_device: str
    feature_cache_hit: bool
    number_of_eval_examples: int

    @property
    def bottleneck(self) -> str:
        timings = {
            "feature_load_time_sec": self.feature_load_time_sec,
            "model_forward_time_sec": self.model_forward_time_sec,
            "thresholding_time_sec": self.thresholding_time_sec,
            "validation_time_sec": self.validation_time_sec,
            "edit_realization_time_sec": self.edit_realization_time_sec,
            "metrics_time_sec": self.metrics_time_sec,
            "detailed_reports_time_sec": self.detailed_reports_time_sec,
        }
        name, value = max(timings.items(), key=lambda item: item[1])
        if self.total_eval_time_sec > 300.0 and self.number_of_eval_examples >= 5000:
            return f"SLOW_EVAL_BOTTLENECK: {name} dominates ({value:.3f}s)"
        return f"{name} dominates ({value:.3f}s)"

    def as_row(self) -> dict[str, Any]:
        return {
            "total_eval_time_sec": round(self.total_eval_time_sec, 3),
            "feature_load_time_sec": round(self.feature_load_time_sec, 3),
            "model_forward_time_sec": round(self.model_forward_time_sec, 3),
            "thresholding_time_sec": round(self.thresholding_time_sec, 3),
            "validation_time_sec": round(self.validation_time_sec, 3),
            "edit_realization_time_sec": round(self.edit_realization_time_sec, 3),
            "metrics_time_sec": round(self.metrics_time_sec, 3),
            "detailed_reports_time_sec": round(self.detailed_reports_time_sec, 3),
            "rows_per_sec": round(self.rows_per_sec, 3),
            "eval_batch_size": int(self.eval_batch_size),
            "gpu_available": bool(self.gpu_available),
            "cuda_device": self.cuda_device,
            "feature_cache_hit": bool(self.feature_cache_hit),
            "number_of_eval_examples": int(self.number_of_eval_examples),
            "bottleneck": self.bottleneck,
        }


@dataclass(frozen=True)
class ProfiledEvaluationResult:
    evaluation: EvaluationResult
    profile: EvaluationProfile


def evaluation_batch_size(config: dict[str, Any]) -> int:
    evaluation_config = config.get("evaluation", {}) or {}
    training_config = config.get("training", {}) or {}
    return max(1, int(evaluation_config.get("batch_size", training_config.get("batch_size", 1)) or 1))


def evaluation_runtime_config(config: dict[str, Any]) -> EvaluationRuntimeConfig:
    evaluation_config = config.get("evaluation", {}) or {}
    return EvaluationRuntimeConfig(
        batch_size=evaluation_batch_size(config),
        fast_during_training=bool(evaluation_config.get("fast_during_training", True)),
        max_eval_examples_per_epoch=int(evaluation_config.get("max_eval_examples_per_epoch", 1000)),
        full_eval_after_training=bool(evaluation_config.get("full_eval_after_training", True)),
        full_eval_examples=int(evaluation_config.get("full_eval_examples", 5000)),
        write_detailed_reports_during_training=bool(
            evaluation_config.get("write_detailed_reports_during_training", False)
        ),
        write_accepted_rejected_during_training=bool(
            evaluation_config.get("write_accepted_rejected_during_training", False)
        ),
        write_score_distribution_during_training=bool(
            evaluation_config.get("write_score_distribution_during_training", False)
        ),
        write_rule_worse_examples_during_training=bool(
            evaluation_config.get("write_rule_worse_examples_during_training", False)
        ),
    )


def limit_rows_for_evaluation(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    mode: EvaluationMode,
) -> list[dict[str, Any]]:
    settings = evaluation_runtime_config(config)
    if mode == EvaluationMode.FAST_DURING_TRAINING:
        limit = settings.max_eval_examples_per_epoch
    elif settings.full_eval_after_training:
        limit = settings.full_eval_examples
    else:
        limit = len(rows)
    return rows[: max(0, min(int(limit), len(rows)))]


def training_fast_report_options(config: dict[str, Any]) -> EvaluationReportOptions:
    settings = evaluation_runtime_config(config)
    if settings.write_detailed_reports_during_training:
        return EvaluationReportOptions(
            write_required_reports=True,
            write_edit_logs=settings.write_accepted_rejected_during_training,
            write_rule_reports=settings.write_rule_worse_examples_during_training,
            write_score_distribution=settings.write_score_distribution_during_training,
            write_candidate_recall=settings.write_rule_worse_examples_during_training,
        )
    return EvaluationReportOptions.for_training_fast_eval()


def evaluate_features_detailed(
    rows: list[dict[str, Any]],
    features: list[Any],
    *,
    corrector: TrainedModelCorrector,
    output_dir: str | Path | None = None,
    metric_weights: dict[str, float] | None = None,
    report_metadata: dict[str, Any] | None = None,
    report_options: EvaluationReportOptions | None = None,
    batch_size: int,
    mixed_precision: bool,
    feature_load_time_sec: float = 0.0,
    feature_cache_hit: bool = False,
    show_progress: bool = False,
) -> ProfiledEvaluationResult:
    if len(rows) != len(features):
        raise ValueError(f"rows/features length mismatch: {len(rows)} rows vs {len(features)} features")
    if not hasattr(corrector.backend, "score_features_batched"):
        raise TypeError("corrector backend does not support batched feature scoring")

    total_started = time.perf_counter()
    scores, model_forward_time = corrector.backend.score_features_batched(
        features,
        batch_size=batch_size,
        mixed_precision=mixed_precision,
    )
    analyzer = DiffAnalyzer()
    evaluated: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    edit_scores: list[dict[str, Any]] = []
    candidate_decisions: list[dict[str, Any]] = []
    collect_detailed_outputs = output_dir is not None
    predicted_edits_by_row: list[list[Any]] = []
    gold_edits_by_row: list[list[Any]] = []
    total_gold_edits = 0
    thresholding_time = 0.0
    validation_time = 0.0
    realization_time = 0.0
    metrics_time = 0.0

    iterable = _with_progress(list(enumerate(zip(rows, features, scores, strict=False))), enabled=show_progress)
    for row_id, (row, feature, feature_scores) in iterable:
        threshold_started = time.perf_counter()
        predictions = _candidate_predictions_from_feature(feature, feature_scores)
        selected, decisions = _select_candidates_with_trace(predictions, corrector.thresholds)
        thresholding_time += time.perf_counter() - threshold_started

        realization_started = time.perf_counter()
        proposed = _apply_candidates(str(row["source"]), selected)
        punctuation_predictions = _punctuation_predictions_from_feature(corrector, feature, feature_scores)
        proposed, punctuation_edits = _apply_punctuation_predictions(
            proposed,
            punctuation_predictions,
            corrector.thresholds,
            punctuation_candidates=_punctuation_candidates_from_feature(feature),
        )
        proposed = normalize_spacing(proposed)
        trusted_edits = [*selected, *punctuation_edits]
        realization_time += time.perf_counter() - realization_started

        validation_started = time.perf_counter()
        validation = corrector.validator.validate(str(row["source"]), proposed, trusted_edits=trusted_edits)
        _annotate_decisions_with_validation(decisions, validation.edits)
        corrected = validation.apply_accepted()
        validation_time += time.perf_counter() - validation_started

        metrics_started = time.perf_counter()
        if collect_detailed_outputs:
            for decision in decisions:
                candidate_decisions.append({**decision, "row_id": row_id})
        evaluated_row = {**row, "prediction": corrected}
        evaluated.append(evaluated_row)
        gold_edits = analyzer.analyze(row["source"], row["target"])
        gold_edits_by_row.append(gold_edits)
        total_gold_edits += len(gold_edits)
        predicted_edits_by_row.append(analyzer.analyze(row["source"], corrected))
        if collect_detailed_outputs:
            accepted_for_scoring = [edit for edit in validation.edits if edit.status == "accepted"]
            correct_flags = mark_correct_edits(accepted_for_scoring, gold_edits)
            correctness_by_identity = {
                id(edit): is_correct for edit, is_correct in zip(accepted_for_scoring, correct_flags, strict=False)
            }
            for edit in validation.edits:
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
                    edit_scores.append(
                        {
                            "confidence": edit.confidence,
                            "is_correct": correctness_by_identity.get(id(edit), False),
                            "rule_id": edit.rule_id,
                        }
                    )
                else:
                    rejected.append(serialized)
        metrics_time += time.perf_counter() - metrics_started

    for score in edit_scores:
        score["total_gold_edits"] = total_gold_edits

    metrics_started = time.perf_counter()
    metrics = compute_metrics_from_edits(evaluated, predicted_edits_by_row, gold_edits_by_row, weights=metric_weights)
    metrics_time += time.perf_counter() - metrics_started

    reports_started = time.perf_counter()
    if output_dir is not None:
        write_evaluation_outputs(
            evaluated,
            metrics,
            output_dir,
            accepted=accepted,
            rejected=rejected,
            candidate_decisions=candidate_decisions,
            metadata=report_metadata,
            candidate_generator=getattr(corrector, "candidates", None),
            candidate_recall_max_candidates=getattr(corrector.backend, "max_candidates", None),
            report_options=report_options,
        )
    detailed_reports_time = time.perf_counter() - reports_started

    total_time = time.perf_counter() - total_started + feature_load_time_sec
    profile = EvaluationProfile(
        total_eval_time_sec=total_time,
        feature_load_time_sec=feature_load_time_sec,
        model_forward_time_sec=model_forward_time,
        thresholding_time_sec=thresholding_time,
        validation_time_sec=validation_time,
        edit_realization_time_sec=realization_time,
        metrics_time_sec=metrics_time,
        detailed_reports_time_sec=detailed_reports_time,
        rows_per_sec=len(rows) / total_time if total_time > 0 else 0.0,
        eval_batch_size=batch_size,
        gpu_available=_gpu_available(),
        cuda_device=_cuda_device(),
        feature_cache_hit=feature_cache_hit,
        number_of_eval_examples=len(rows),
    )
    return ProfiledEvaluationResult(
        evaluation=EvaluationResult(
            metrics=metrics,
            edit_scores=edit_scores
            or [{"confidence": 0.0, "is_correct": False, "rule_id": "", "total_gold_edits": total_gold_edits}],
            accepted_edits=accepted,
            rejected_edits=rejected,
        ),
        profile=profile,
    )


def write_evaluation_profile_reports(profile: EvaluationProfile, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    row = profile.as_row()
    with (output / "evaluation_profile.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    (output / "evaluation_profile_summary.md").write_text(_profile_summary(profile), encoding="utf-8")


def evaluation_profile_metadata(profile: EvaluationProfile) -> dict[str, Any]:
    return profile.as_row()


def core_training_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0)) for key in CORE_TRAINING_METRICS}


def _candidate_predictions_from_feature(
    feature: Any,
    scores: BatchedFeatureModelScores,
) -> list[ModelCandidatePrediction]:
    predictions: list[ModelCandidatePrediction] = []
    for index, candidate in enumerate(_candidates_from_feature(feature)):
        predictions.append(
            ModelCandidatePrediction(
                candidate=candidate,
                score=float(scores.candidate_scores[index]),
                confidence=float(scores.candidate_confidences[index]),
                rule_id=candidate.rule_id,
            )
        )
    return predictions


def _candidates_from_feature(feature: Any) -> list[Candidate]:
    candidates: list[Candidate] = []
    masks = list(getattr(feature, "candidate_mask", []))
    for index, active in enumerate(masks):
        if not active:
            continue
        start, end = _candidate_char_span(feature, index)
        gap_index = _candidate_gap_index(feature, index)
        candidates.append(
            Candidate(
                source=str(feature.candidate_sources[index]),
                replacement=str(feature.candidate_replacements[index]),
                edit_type=str(feature.candidate_edit_types[index]),
                start=start,
                end=end,
                confidence=1.0,
                requires_model=bool(feature.candidate_requires_model[index]),
                rule_id=str(feature.candidate_rule_ids[index]),
                mode=feature.candidate_modes[index],
                action=_value_at(feature, "candidate_actions", index, ""),
                label=_value_at(feature, "candidate_punctuation_labels", index, ""),
                gap_index=gap_index,
                requires=tuple(_value_at(feature, "candidate_requires", index, tuple())),
                group=str(_value_at(feature, "candidate_groups", index, "")),
            )
        )
    return candidates


def _punctuation_candidates_from_feature(feature: Any) -> list[Candidate]:
    return [
        candidate
        for candidate in _candidates_from_feature(feature)
        if candidate.edit_type
        in {
            "punctuation_insert",
            "punctuation_delete",
            "punctuation_replace",
            "final_punctuation",
        }
        and candidate.gap_index is not None
        and candidate.action
        and candidate.label
    ]


def _punctuation_predictions_from_feature(
    corrector: TrainedModelCorrector,
    feature: Any,
    scores: BatchedFeatureModelScores,
) -> list[ModelPunctuationPrediction]:
    backend = corrector.backend
    predictions: list[ModelPunctuationPrediction] = []
    for gap_index, active in enumerate(feature.punctuation_gap_mask):
        if not active or gap_index >= len(scores.punctuation_label_ids):
            continue
        label = backend.punctuation_by_id.get(int(scores.punctuation_label_ids[gap_index]), "NONE")
        action = backend.punctuation_action_by_id.get(int(scores.punctuation_action_ids[gap_index]), "KEEP_NONE")
        predictions.append(
            ModelPunctuationPrediction(
                gap_index=gap_index,
                label=label,
                confidence=float(scores.punctuation_confidences[gap_index]),
                action=action,
            )
        )
    return predictions


def _candidate_char_span(feature: Any, index: int) -> tuple[int, int]:
    spans = getattr(feature, "candidate_char_spans", None)
    if spans is not None and index < len(spans):
        start, end = spans[index]
        return int(start), int(end)
    return 0, 0


def _candidate_gap_index(feature: Any, index: int) -> int | None:
    value = _value_at(feature, "candidate_gap_indexes", index, -1)
    value = int(value)
    return value if value >= 0 else None


def _value_at(feature: Any, name: str, index: int, default: Any) -> Any:
    values = getattr(feature, name, None)
    if values is None or index >= len(values):
        return default
    return values[index]


def _with_progress(values: list[Any], *, enabled: bool) -> Any:
    if not enabled:
        return values
    try:
        from tqdm.auto import tqdm
    except Exception:
        return values
    return tqdm(values, total=len(values), desc="Evaluating cached features", dynamic_ncols=True)


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_device() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:
        return ""


def _profile_summary(profile: EvaluationProfile) -> str:
    row = profile.as_row()
    lines = [
        "# Evaluation Profile Summary",
        "",
        f"- total_eval_time_sec: {row['total_eval_time_sec']}",
        f"- feature_load_time_sec: {row['feature_load_time_sec']}",
        f"- model_forward_time_sec: {row['model_forward_time_sec']}",
        f"- thresholding_time_sec: {row['thresholding_time_sec']}",
        f"- validation_time_sec: {row['validation_time_sec']}",
        f"- edit_realization_time_sec: {row['edit_realization_time_sec']}",
        f"- metrics_time_sec: {row['metrics_time_sec']}",
        f"- detailed_reports_time_sec: {row['detailed_reports_time_sec']}",
        f"- rows_per_sec: {row['rows_per_sec']}",
        f"- eval_batch_size: {row['eval_batch_size']}",
        f"- gpu_available: {row['gpu_available']}",
        f"- cuda_device: {row['cuda_device']}",
        f"- feature_cache_hit: {row['feature_cache_hit']}",
        f"- number_of_eval_examples: {row['number_of_eval_examples']}",
        f"- bottleneck: {row['bottleneck']}",
    ]
    if profile.total_eval_time_sec > 300.0 and profile.number_of_eval_examples >= 5000:
        lines.extend(
            [
                "",
                "## Bottleneck",
                "",
                f"Evaluation exceeded 5 minutes on {profile.number_of_eval_examples} examples. {profile.bottleneck}",
            ]
        )
    return "\n".join(lines) + "\n"
