from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import yaml

from src.candidates.matching import candidate_matches_edit
from src.config.load_config import load_config
from src.evaluation.fast_eval import (
    _candidate_predictions_from_feature,
    _punctuation_candidates_from_feature,
    _punctuation_predictions_from_feature,
    evaluation_batch_size,
)
from src.evaluation.metrics import compute_metrics_from_edits, mark_correct_edits
from src.inference.model_corrector import (
    TrainedModelCorrector,
    _annotate_decisions_with_validation,
    _apply_candidates,
    _apply_punctuation_predictions,
    _select_candidates_with_trace,
)
from src.inference.postprocess import normalize_spacing
from src.training.train import _build_features_with_metadata, _load_rows_for_split
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.strict_validator import StrictValidator


EXPLICIT_SWEEP_THRESHOLDS: dict[str, list[float]] = {
    "subject_predicate_dash": [0.990, 0.992, 0.994, 0.996],
    "direct_speech_colon": [0.900, 0.910, 0.920, 0.940],
    "hyphen_particles": [0.50, 0.53, 0.56, 0.60],
    "hyphen_koe_koy": [0.45, 0.48, 0.50, 0.55],
    "context_to_zhe": [0.80, 0.82, 0.84, 0.86],
    "ne_adverb": [0.74, 0.76, 0.78, 0.82],
    "ne_participle": [0.78, 0.80, 0.82, 0.86],
    "ne_adjective": [0.70, 0.72, 0.74, 0.78],
    "dictionary_fuzzy": [0.75, 0.78, 0.80, 0.84],
    "cy_exception": [0.58, 0.60, 0.62, 0.64],
    "swapped_letters_candidate": [0.76, 0.78, 0.80, 0.84],
}
STRICT_WATCH_RULES = {
    "hyphen_po_adverbs",
    "context_za_to",
    "double_consonant_candidate",
    "n_nn_deverbal_adjective",
    "direct_speech_quotes",
    "enumeration_colon",
    "explanation_colon",
}
SWEEP_RULES = tuple(EXPLICIT_SWEEP_THRESHOLDS) + tuple(sorted(STRICT_WATCH_RULES))


@dataclass(frozen=True)
class SafetyCaps:
    clean_overcorrection_rate: float = 0.005
    dirty_worse_rate: float = 0.003
    real_dirty_worse_rate: float = 0.05
    edit_precision: float = 0.90
    punctuation_f1: float = 0.70


@dataclass(frozen=True)
class ReplayEvaluation:
    row_ids: list[int]
    metrics: dict[str, float]
    evaluated_rows: list[dict[str, Any]]
    accepted_edits: list[dict[str, Any]]
    rejected_edits: list[dict[str, Any]]
    candidate_decisions: list[dict[str, Any]]
    predicted_edits_by_row: list[list[Edit]]
    gold_edits_by_row: list[list[Edit]]


def default_output_dir(config: dict[str, Any], timestamp: str | None = None) -> Path:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    return reports_dir / "threshold_calibration" / timestamp


def passes_safety_caps(metrics: dict[str, float], caps: SafetyCaps = SafetyCaps()) -> bool:
    return (
        float(metrics.get("clean_overcorrection_rate", 0.0)) <= caps.clean_overcorrection_rate
        and float(metrics.get("dirty_worse_rate", 0.0)) <= caps.dirty_worse_rate
        and float(metrics.get("real_dirty_worse_rate", 0.0)) <= caps.real_dirty_worse_rate
        and float(metrics.get("edit_precision", 0.0)) >= caps.edit_precision
        and float(metrics.get("punctuation_f1", 0.0)) >= caps.punctuation_f1
    )


def select_best_safe_threshold(
    *,
    current_threshold: float,
    current_metrics: dict[str, float],
    candidate_rows: list[dict[str, Any]],
    caps: SafetyCaps = SafetyCaps(),
) -> dict[str, Any]:
    current_row = {"candidate_threshold": float(current_threshold), **current_metrics}
    safe_rows = [row for row in candidate_rows if passes_safety_caps(_metrics_from_row(row), caps)]
    if passes_safety_caps(current_metrics, caps):
        safe_rows.append(current_row)
    if not safe_rows:
        return current_row
    return max(safe_rows, key=_threshold_objective)


def write_calibration_reports(
    *,
    output_dir: Path,
    latest_dir: Path,
    sweep_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    summary: str,
    recommended_thresholds: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(sweep_rows).to_csv(output_dir / "threshold_sweep_by_rule.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(output_dir / "threshold_calibration_decision_by_rule.csv", index=False)
    (output_dir / "threshold_sweep_summary.md").write_text(summary, encoding="utf-8")
    (output_dir / "recommended_thresholds.yaml").write_text(
        yaml.safe_dump(recommended_thresholds, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (latest_dir / "latest_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (latest_dir / "latest_recommended_thresholds.yaml").write_text(
        yaml.safe_dump(recommended_thresholds, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def run_calibration(
    *,
    config_path: str | Path,
    split: str,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    baseline_mode: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    if baseline_mode:
        config = _config_with_threshold_mode(config, baseline_mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) if output_dir is not None else default_output_dir(config, timestamp=timestamp)
    latest_dir = run_dir.parent

    rows = _load_rows_for_split(_config_with_limit(config, split, limit), split=split, fallback_rows=[])
    if limit is not None:
        rows = rows[: int(limit)]
    corrector = TrainedModelCorrector.from_config(config)
    feature_config = _model_eval_feature_config(_config_with_limit(config, split, limit))
    feature_run = _build_features_with_metadata(
        feature_config,
        rows,
        split=split,
        limit=limit,
        tokenizer=getattr(corrector.backend, "tokenizer", None),
    )
    eval_batch_size = int(batch_size) if batch_size is not None else evaluation_batch_size(config)
    print(f"Scoring {len(rows)} rows with batch_size={eval_batch_size}...", flush=True)
    scores, model_forward_time = corrector.backend.score_features_batched(
        feature_run.features,
        batch_size=eval_batch_size,
        mixed_precision=bool(config.get("training", {}).get("mixed_precision", True)),
    )
    print(f"Model scoring finished in {model_forward_time:.3f}s; replaying threshold sweeps...", flush=True)

    baseline_thresholds = _active_thresholds(config)
    analyzer = DiffAnalyzer()
    gold_edits_by_row = [
        analyzer.analyze(str(row.get("source", "")), str(row.get("target", "")))
        for row in rows
    ]
    baseline = _evaluate_with_cached_scores(
        rows,
        feature_run.features,
        scores,
        corrector,
        baseline_thresholds,
        precomputed_gold_edits_by_row=gold_edits_by_row,
        collect_decisions=True,
    )
    score_stats = _score_stats_by_rule(rows, baseline.candidate_decisions)
    pre_guard_metrics = _read_existing_summary_metrics(config)
    baseline_rule_thresholds = {
        rule_id: _current_threshold_for_rule(rule_id, baseline.candidate_decisions, baseline_thresholds)
        for rule_id in SWEEP_RULES
    }

    accepted_changes: dict[str, float] = {}
    current_thresholds = dict(baseline_thresholds)
    current_evaluation = baseline
    sweep_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for pass_index, rules in enumerate((SWEEP_RULES, tuple(accepted_changes)), start=1):
        for rule_id in rules:
            threshold_key = f"{rule_id}_threshold"
            current_threshold = float(
                current_thresholds.get(
                    threshold_key,
                    baseline_rule_thresholds.get(rule_id, current_thresholds.get("default_threshold", 0.85)),
                )
            )
            candidate_thresholds = _candidate_thresholds_for_rule(rule_id, current_threshold, score_stats.get(rule_id, {}))
            print(
                f"Sweeping pass={pass_index} rule={rule_id} thresholds={','.join(str(value) for value in candidate_thresholds)}",
                flush=True,
            )
            candidate_rows: list[dict[str, Any]] = []
            evaluations_by_threshold: dict[float, ReplayEvaluation] = {}
            for candidate_threshold in candidate_thresholds:
                if abs(float(candidate_threshold) - float(current_threshold)) < 1e-9:
                    trial = current_evaluation
                else:
                    trial_thresholds = dict(current_thresholds)
                    trial_thresholds[threshold_key] = float(candidate_threshold)
                    affected_row_indexes = _affected_row_indexes(
                        rule_id,
                        baseline.candidate_decisions,
                        current_threshold,
                        float(candidate_threshold),
                    )
                    patch = _evaluate_with_cached_scores(
                        rows,
                        feature_run.features,
                        scores,
                        corrector,
                        trial_thresholds,
                        precomputed_gold_edits_by_row=gold_edits_by_row,
                        row_indexes=affected_row_indexes,
                        collect_decisions=False,
                    )
                    trial = _combine_evaluation(current_evaluation, patch)
                evaluations_by_threshold[float(candidate_threshold)] = trial
                row = {
                    "pass_index": pass_index,
                    "rule_id": rule_id,
                    "threshold_key": threshold_key,
                    "current_threshold": current_threshold,
                    "candidate_threshold": float(candidate_threshold),
                    **_metrics_for_sweep_row(trial.metrics),
                    **_rule_impact_counts(rule_id, rows, trial),
                }
                row["passes_safety_caps"] = passes_safety_caps(trial.metrics)
                candidate_rows.append(row)
                sweep_rows.append(row)

            selected = select_best_safe_threshold(
                current_threshold=current_threshold,
                current_metrics=current_evaluation.metrics,
                candidate_rows=candidate_rows,
            )
            selected_threshold = float(selected.get("candidate_threshold", current_threshold))
            selected_evaluation = evaluations_by_threshold.get(selected_threshold, current_evaluation)
            previous_metrics = dict(current_evaluation.metrics)
            recommendation, reason = _recommendation_for_selection(
                rule_id,
                current_threshold,
                selected_threshold,
                current_evaluation,
                selected_evaluation,
                selected,
            )
            if recommendation in {"lower", "raise"}:
                current_thresholds[threshold_key] = selected_threshold
                current_evaluation = selected_evaluation
                accepted_changes[threshold_key] = selected_threshold

            stats = score_stats.get(rule_id, {})
            decision_rows.append(
                {
                    "rule_id": rule_id,
                    "threshold_key": threshold_key,
                    "current_threshold": current_threshold,
                    "recommended_threshold": selected_threshold if recommendation in {"lower", "raise"} else current_threshold,
                    "positive_score_p10": stats.get("positive_score_p10", 0.0),
                    "positive_score_p50": stats.get("positive_score_p50", 0.0),
                    "positive_score_p90": stats.get("positive_score_p90", 0.0),
                    "negative_score_mean": stats.get("negative_score_mean", 0.0),
                    "negative_score_p90": stats.get("negative_score_p90", 0.0),
                    "clean_false_positives_at_threshold": int(selected.get("clean_false_positives", 0)),
                    "synthetic_true_positives_at_threshold": int(selected.get("synthetic_true_positives", 0)),
                    "real_true_positives_at_threshold": int(selected.get("real_true_positives", 0)),
                    "dirty_worse_impact": float(selected.get("dirty_worse_rate", 0.0))
                    - float(previous_metrics.get("dirty_worse_rate", 0.0)),
                    "recommendation": recommendation,
                    "reason": reason,
                }
            )

    final_evaluation = current_evaluation
    verdict = _verdict(baseline.metrics, final_evaluation.metrics, accepted_changes)
    recommended_thresholds = _recommended_thresholds_payload(config, current_thresholds, accepted_changes, baseline_mode)
    manifest = {
        "timestamp": timestamp,
        "config_path": str(config_path),
        "adapter_output_dir": str(config.get("paths", {}).get("adapter_output_dir", "models/adapters/latest")),
        "heads_output_dir": str(config.get("paths", {}).get("heads_output_dir", "models/heads/latest")),
        "dataset_path": str(config.get("data", {}).get("processed_train_path", "")),
        "evaluation_split": split,
        "baseline_thresholds_mode": _threshold_mode(config),
        "feature_cache_enabled": bool(feature_run.cache_result.enabled),
        "feature_cache_hit": bool(feature_run.cache_result.hit),
        "feature_cache_path": str(feature_run.cache_result.path),
        "model_forward_time_sec": round(float(model_forward_time), 3),
        "accepted_threshold_changes": accepted_changes,
        "rejected_threshold_rules": [
            row["rule_id"] for row in decision_rows if row.get("recommendation") not in {"lower", "raise"}
        ],
        "final_recommended_thresholds_path": str(run_dir / "recommended_thresholds.yaml"),
        "verdict": verdict,
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    summary = _summary_markdown(
        verdict=verdict,
        pre_guard_metrics=pre_guard_metrics,
        baseline_metrics=baseline.metrics,
        final_metrics=final_evaluation.metrics,
        accepted_changes=accepted_changes,
        output_dir=run_dir,
    )
    write_calibration_reports(
        output_dir=run_dir,
        latest_dir=latest_dir,
        sweep_rows=sweep_rows,
        decision_rows=decision_rows,
        summary=summary,
        recommended_thresholds=recommended_thresholds,
        manifest=manifest,
    )
    return {
        "output_dir": str(run_dir),
        "recommended_thresholds": recommended_thresholds,
        "baseline_metrics": baseline.metrics,
        "final_metrics": final_evaluation.metrics,
        "manifest": manifest,
        "verdict": verdict,
    }


def _evaluate_with_cached_scores(
    rows: list[dict[str, Any]],
    features: list[Any],
    scores: list[Any],
    corrector: TrainedModelCorrector,
    thresholds: dict[str, float],
    *,
    precomputed_gold_edits_by_row: list[list[Edit]] | None = None,
    row_indexes: list[int] | None = None,
    collect_decisions: bool = False,
) -> ReplayEvaluation:
    analyzer = DiffAnalyzer()
    validator = StrictValidator(
        context_pair_threshold=float(thresholds.get("context_pair_threshold", 0.98)),
        tsya_threshold=float(thresholds.get("tsya_threshold", thresholds.get("context_pair_threshold", 0.98))),
    )
    evaluated_rows: list[dict[str, Any]] = []
    accepted_edits: list[dict[str, Any]] = []
    rejected_edits: list[dict[str, Any]] = []
    candidate_decisions: list[dict[str, Any]] = []
    predicted_edits_by_row: list[list[Edit]] = []
    evaluated_gold_edits_by_row: list[list[Edit]] = []

    row_ids = row_indexes if row_indexes is not None else list(range(min(len(rows), len(features), len(scores))))
    for row_id in row_ids:
        row = rows[row_id]
        feature = features[row_id]
        feature_scores = scores[row_id]
        predictions = _candidate_predictions_from_feature(feature, feature_scores)
        selected, decisions = _select_candidates_with_trace(predictions, thresholds)
        proposed = _apply_candidates(str(row["source"]), selected)
        proposed, punctuation_edits = _apply_punctuation_predictions(
            proposed,
            _punctuation_predictions_from_feature(corrector, feature, feature_scores),
            thresholds,
            punctuation_candidates=_punctuation_candidates_from_feature(feature),
        )
        proposed = normalize_spacing(proposed)
        validation = validator.validate(str(row["source"]), proposed, trusted_edits=[*selected, *punctuation_edits])
        if collect_decisions:
            _annotate_decisions_with_validation(decisions, validation.edits)
        corrected = validation.apply_accepted()
        evaluated_row = {**row, "prediction": corrected}
        evaluated_rows.append(evaluated_row)
        if precomputed_gold_edits_by_row is not None and row_id < len(precomputed_gold_edits_by_row):
            gold_edits = precomputed_gold_edits_by_row[row_id]
        else:
            gold_edits = analyzer.analyze(row["source"], row["target"])
        predicted_edits = analyzer.analyze(row["source"], corrected)
        evaluated_gold_edits_by_row.append(gold_edits)
        predicted_edits_by_row.append(predicted_edits)

        if collect_decisions:
            for decision in decisions:
                candidate_decisions.append({**decision, "row_id": row_id})
        accepted_for_scoring = [edit for edit in validation.edits if edit.status == "accepted"]
        correct_flags = mark_correct_edits(accepted_for_scoring, gold_edits)
        correctness_by_identity = {
            id(edit): is_correct
            for edit, is_correct in zip(accepted_for_scoring, correct_flags, strict=False)
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
                "is_correct": correctness_by_identity.get(id(edit), False),
            }
            if edit.status == "accepted":
                accepted_edits.append(serialized)
            else:
                rejected_edits.append(serialized)

    metrics = compute_metrics_from_edits(evaluated_rows, predicted_edits_by_row, evaluated_gold_edits_by_row)
    return ReplayEvaluation(
        row_ids=list(row_ids),
        metrics=metrics,
        evaluated_rows=evaluated_rows,
        accepted_edits=accepted_edits,
        rejected_edits=rejected_edits,
        candidate_decisions=candidate_decisions,
        predicted_edits_by_row=predicted_edits_by_row,
        gold_edits_by_row=evaluated_gold_edits_by_row,
    )


def _affected_row_indexes(
    rule_id: str,
    candidate_decisions: list[dict[str, Any]],
    current_threshold: float,
    candidate_threshold: float,
) -> list[int]:
    lower = min(float(current_threshold), float(candidate_threshold))
    upper = max(float(current_threshold), float(candidate_threshold))
    affected = {
        int(record.get("row_id", -1))
        for record in candidate_decisions
        if str(record.get("rule_id") or "") == rule_id
        and lower <= float(record.get("score", 0.0)) < upper
        and int(record.get("row_id", -1)) >= 0
    }
    return sorted(affected)


def _combine_evaluation(base: ReplayEvaluation, patch: ReplayEvaluation) -> ReplayEvaluation:
    if not patch.row_ids:
        return base
    affected = set(patch.row_ids)
    evaluated_rows = list(base.evaluated_rows)
    predicted_edits_by_row = list(base.predicted_edits_by_row)
    gold_edits_by_row = list(base.gold_edits_by_row)
    for row_id, evaluated_row, predicted_edits, gold_edits in zip(
        patch.row_ids,
        patch.evaluated_rows,
        patch.predicted_edits_by_row,
        patch.gold_edits_by_row,
        strict=False,
    ):
        evaluated_rows[row_id] = evaluated_row
        predicted_edits_by_row[row_id] = predicted_edits
        gold_edits_by_row[row_id] = gold_edits
    accepted_edits = [
        edit for edit in base.accepted_edits if int(edit.get("row_id", -1)) not in affected
    ] + patch.accepted_edits
    rejected_edits = [
        edit for edit in base.rejected_edits if int(edit.get("row_id", -1)) not in affected
    ] + patch.rejected_edits
    metrics = compute_metrics_from_edits(evaluated_rows, predicted_edits_by_row, gold_edits_by_row)
    return ReplayEvaluation(
        row_ids=list(base.row_ids),
        metrics=metrics,
        evaluated_rows=evaluated_rows,
        accepted_edits=accepted_edits,
        rejected_edits=rejected_edits,
        candidate_decisions=base.candidate_decisions,
        predicted_edits_by_row=predicted_edits_by_row,
        gold_edits_by_row=gold_edits_by_row,
    )


def _score_stats_by_rule(rows: list[dict[str, Any]], candidate_decisions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    analyzer = DiffAnalyzer()
    candidates_by_row: dict[int, list[Any]] = {}
    for record in candidate_decisions:
        candidates_by_row.setdefault(int(record.get("row_id", -1)), []).append(record.get("candidate"))
    gold_by_row = {
        row_id: analyzer.analyze(str(row.get("source", "")), str(row.get("target", "")), candidates=candidates_by_row.get(row_id, []))
        for row_id, row in enumerate(rows)
    }
    buckets: dict[str, dict[str, list[float]]] = {}
    for record in candidate_decisions:
        candidate = record.get("candidate")
        rule_id = str(record.get("rule_id") or getattr(candidate, "rule_id", "") or "")
        if not rule_id or candidate is None:
            continue
        bucket = buckets.setdefault(rule_id, {"positive": [], "negative": []})
        row_id = int(record.get("row_id", -1))
        score = float(record.get("score", 0.0))
        if any(candidate_matches_edit(candidate, edit) for edit in gold_by_row.get(row_id, [])):
            bucket["positive"].append(score)
        else:
            bucket["negative"].append(score)
    return {
        rule_id: {
            "positive_score_p10": _quantile(values["positive"], 0.10),
            "positive_score_p50": _quantile(values["positive"], 0.50),
            "positive_score_p90": _quantile(values["positive"], 0.90),
            "negative_score_mean": _mean(values["negative"]),
            "negative_score_p90": _quantile(values["negative"], 0.90),
        }
        for rule_id, values in buckets.items()
    }


def _rule_impact_counts(rule_id: str, rows: list[dict[str, Any]], evaluation: ReplayEvaluation) -> dict[str, int]:
    clean_false_positives = 0
    synthetic_true_positives = 0
    real_true_positives = 0
    for edit in evaluation.accepted_edits:
        if str(edit.get("rule_id") or "") != rule_id:
            continue
        row = rows[int(edit.get("row_id", -1))]
        if row.get("is_clean", False) and not bool(edit.get("is_correct", False)):
            clean_false_positives += 1
        if bool(edit.get("is_correct", False)):
            if _is_synthetic_row(row):
                synthetic_true_positives += 1
            elif _is_real_row(row):
                real_true_positives += 1
    return {
        "clean_false_positives": clean_false_positives,
        "synthetic_true_positives": synthetic_true_positives,
        "real_true_positives": real_true_positives,
    }


def _candidate_thresholds_for_rule(rule_id: str, current_threshold: float, stats: dict[str, float]) -> list[float]:
    values = list(EXPLICIT_SWEEP_THRESHOLDS.get(rule_id, []))
    if not values:
        values = [
            float(stats.get("positive_score_p50", 0.0)),
            float(stats.get("positive_score_p90", 0.0)),
        ]
    values.append(float(current_threshold))
    return sorted({round(value, 6) for value in values if 0.0 < float(value) <= 1.0})


def _recommendation_for_selection(
    rule_id: str,
    current_threshold: float,
    selected_threshold: float,
    current_evaluation: ReplayEvaluation,
    selected_evaluation: ReplayEvaluation,
    selected_row: dict[str, Any],
) -> tuple[str, str]:
    if not passes_safety_caps(selected_evaluation.metrics):
        return "keep", "candidate violates safety caps"
    if (
        int(selected_row.get("real_true_positives", 0)) == 0
        and float(selected_evaluation.metrics.get("real_dirty_worse_rate", 0.0))
        > float(current_evaluation.metrics.get("real_dirty_worse_rate", 0.0))
    ):
        return "keep", "synthetic-only gain increases real dirty-worse risk"
    if _threshold_objective(selected_row) <= _threshold_objective(current_evaluation.metrics) + 1e-9:
        if rule_id in STRICT_WATCH_RULES:
            return "needs_more_training", "no safety-safe threshold improves validation objective"
        return "keep", "no safety-safe threshold improves validation objective"
    if selected_threshold < current_threshold:
        return "lower", "safe recall gain"
    if selected_threshold > current_threshold:
        return "raise", "safer threshold improves validation objective"
    return "keep", "current threshold remains best"


def _verdict(
    baseline_metrics: dict[str, float],
    final_metrics: dict[str, float],
    accepted_changes: dict[str, float],
) -> str:
    if not passes_safety_caps(final_metrics):
        return "BLOCKED"
    spelling_gain = float(final_metrics.get("spelling_f1", 0.0)) - float(baseline_metrics.get("spelling_f1", 0.0))
    if accepted_changes and spelling_gain > 0.003:
        return "READY_FOR_THRESHOLD_UPDATE"
    return "NEEDS_MORE_TRAINING"


def _recommended_thresholds_payload(
    config: dict[str, Any],
    thresholds: dict[str, float],
    accepted_changes: dict[str, float],
    baseline_mode: str | None,
) -> dict[str, Any]:
    return {
        "mode": "calibrated_val_guarded",
        "baseline_mode": baseline_mode or _threshold_mode(config),
        "calibrated_val_guarded": thresholds,
        "accepted_changes": accepted_changes,
    }


def _summary_markdown(
    *,
    verdict: str,
    pre_guard_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    final_metrics: dict[str, float],
    accepted_changes: dict[str, float],
    output_dir: Path,
) -> str:
    lines = [
        "# Threshold Sweep Summary",
        "",
        f"- verdict: {verdict}",
        f"- output_dir: {output_dir}",
        f"- accepted_threshold_changes: {len(accepted_changes)}",
        "",
        "## Existing Report Metrics",
        "",
        *_metric_lines(pre_guard_metrics, prefix="pre_guard"),
        "",
        "## Guard-Fixed Baseline Metrics",
        "",
        *_metric_lines(baseline_metrics, prefix="baseline"),
        "",
        "## Final Recommended Metrics",
        "",
        *_metric_lines(final_metrics, prefix="final"),
        "",
        "## Accepted Changes",
        "",
    ]
    if accepted_changes:
        lines.extend(f"- {key}: {value}" for key, value in sorted(accepted_changes.items()))
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _metric_lines(metrics: dict[str, float], *, prefix: str) -> list[str]:
    base_names = [
        "exact_match",
        "dirty_improved_rate",
        "dirty_worse_rate",
        "clean_overcorrection_rate",
        "edit_precision",
        "edit_recall",
        "edit_f1",
        "spelling_precision",
        "spelling_recall",
        "spelling_f1",
        "punctuation_precision",
        "punctuation_recall",
        "punctuation_f1",
    ]
    names = list(base_names)
    for scope in ("clean", "synthetic", "real"):
        names.extend(f"{scope}_{name}" for name in base_names)
    return [f"- {prefix}_{name}: {float(metrics.get(name, 0.0)):.6f}" for name in names if name in metrics]


def _metrics_for_sweep_row(metrics: dict[str, float]) -> dict[str, float]:
    base_keys = [
        "exact_match",
        "dirty_improved_rate",
        "dirty_worse_rate",
        "clean_overcorrection_rate",
        "edit_precision",
        "edit_recall",
        "edit_f1",
        "spelling_precision",
        "spelling_recall",
        "spelling_f1",
        "punctuation_precision",
        "punctuation_recall",
        "punctuation_f1",
    ]
    keys = list(base_keys)
    for scope in ("clean", "synthetic", "real"):
        keys.extend(f"{scope}_{key}" for key in base_keys)
    return {key: float(metrics.get(key, 0.0)) for key in keys}


def _metrics_from_row(row: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in row.items() if isinstance(value, int | float)}


def _threshold_objective(row: dict[str, Any]) -> float:
    return (
        float(row.get("spelling_f1", 0.0)) * 3.0
        + float(row.get("edit_recall", 0.0))
        + float(row.get("edit_f1", 0.0))
        + float(row.get("punctuation_f1", 0.0)) * 0.25
        - float(row.get("clean_overcorrection_rate", 0.0)) * 5.0
        - float(row.get("dirty_worse_rate", 0.0)) * 3.0
        - float(row.get("real_dirty_worse_rate", 0.0)) * 2.0
    )


def _current_threshold_for_rule(
    rule_id: str,
    candidate_decisions: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> float:
    for record in candidate_decisions:
        if str(record.get("rule_id") or "") == rule_id:
            return float(record.get("threshold", 0.0))
    return float(thresholds.get(f"{rule_id}_threshold", thresholds.get("default_threshold", 0.85)))


def _read_existing_summary_metrics(config: dict[str, Any]) -> dict[str, float]:
    path = Path(config.get("paths", {}).get("reports_dir", "reports")) / "evaluation_summary.csv"
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty:
        return {}
    row = frame.iloc[0].to_dict()
    return {str(key): float(value) for key, value in row.items() if isinstance(value, int | float)}


def _active_thresholds(config: dict[str, Any]) -> dict[str, float]:
    threshold_config = config.get("thresholds", {}) or {}
    mode = _threshold_mode(config)
    return {str(key): float(value) for key, value in dict(threshold_config.get(mode, {})).items()}


def _threshold_mode(config: dict[str, Any]) -> str:
    return str(config.get("thresholds", {}).get("mode", "balanced"))


def _config_with_threshold_mode(config: dict[str, Any], mode: str) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    cloned.setdefault("thresholds", {})["mode"] = mode
    return cloned


def _config_with_limit(config: dict[str, Any], split: str, limit: int | None) -> dict[str, Any]:
    if limit is None:
        return config
    cloned = copy.deepcopy(config)
    cloned.setdefault("training", {})[_limit_key_for_split(split)] = int(limit)
    return cloned


def _model_eval_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    cloned.setdefault("evaluation", {})["feature_tokenizer"] = "model"
    return cloned


def _limit_key_for_split(split: str) -> str:
    return {
        "train": "max_train_examples",
        "val": "max_val_examples",
        "test": "max_test_examples",
    }.get(split, f"max_{split}_examples")


def _is_synthetic_row(row: dict[str, Any]) -> bool:
    return not row.get("is_clean", False) and _as_bool(row.get("is_synthetic", False))


def _is_real_row(row: dict[str, Any]) -> bool:
    return not row.get("is_clean", False) and not _as_bool(row.get("is_synthetic", False))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 6)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction), 6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate rule-specific thresholds on an existing checkpoint.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--split", default="val", help="Evaluation split. Use val for calibration.")
    parser.add_argument("--output-dir", help="Optional explicit calibration output directory.")
    parser.add_argument("--limit", type=int, help="Optional max examples for calibration.")
    parser.add_argument("--baseline-mode", help="Optional threshold profile to use as baseline.")
    parser.add_argument("--batch-size", type=int, help="Optional calibration-only evaluation batch size override.")
    args = parser.parse_args()
    result = run_calibration(
        config_path=args.config,
        split=args.split,
        output_dir=args.output_dir,
        limit=args.limit,
        baseline_mode=args.baseline_mode,
        batch_size=args.batch_size,
    )
    print(f"output_dir={result['output_dir']}")
    print(f"verdict={result['verdict']}")
    print(f"recommended_thresholds={Path(result['output_dir']) / 'recommended_thresholds.yaml'}")
    return 0


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(status)
