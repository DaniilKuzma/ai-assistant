from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_thresholds import (
    _evaluate_with_cached_scores,
    _model_eval_feature_config,
    _threshold_objective,
)
from src.config.load_config import load_config
from src.evaluation.fast_eval import evaluation_batch_size
from src.training.train import _build_features_with_metadata
from src.inference.model_corrector import TrainedModelCorrector


PROFILE_NAME = "calibrated_guarded"
BASELINE_PROFILE_NAME = "calibrated_guarded"
REGRESSION_BY_RULE_COLUMNS = [
    "rule_id",
    "edit_type",
    "clean_overcorrection_count",
    "dirty_worse_count",
    "real_dirty_worse_count",
    "accepted_count",
    "rejected_count",
    "examples",
    "suspected_threshold_key",
    "recommendation",
]
REGRESSION_EXAMPLE_COLUMNS = [
    "rule_id",
    "source",
    "target",
    "prediction",
    "edit_source",
    "edit_replacement",
    "score",
    "threshold",
    "source_type",
    "reason",
]
RISKY_THRESHOLD_KEYS = [
    "dictionary_fuzzy_threshold",
    "double_consonant_candidate_threshold",
    "swapped_letters_candidate_threshold",
    "keyboard_typo_candidate_threshold",
    "missing_letter_candidate_threshold",
    "extra_letter_candidate_threshold",
    "hyphen_po_adverbs_threshold",
    "context_pair_threshold",
    "context_to_zhe_threshold",
    "context_za_to_threshold",
    "direct_speech_quotes_threshold",
    "enumeration_colon_threshold",
    "explanation_colon_threshold",
]
ROLLBACK_SWEEPS = {
    "dictionary_fuzzy_threshold": [0.82, 0.84, 0.86, 0.95],
    "swapped_letters_candidate_threshold": [0.82, 0.84, 0.86, 0.95],
    "double_consonant_candidate_threshold": [0.76, 0.80, 0.84, 0.95],
    "hyphen_po_adverbs_threshold": [0.80, 0.84, 0.88, 0.95],
}
STRICT_KEEP_KEYS = {
    "context_pair_threshold",
    "context_za_to_threshold",
    "direct_speech_quotes_threshold",
    "enumeration_colon_threshold",
    "explanation_colon_threshold",
    "n_nn_deverbal_adjective_threshold",
}
SELECTION_GATES = {
    "clean_overcorrection_rate": ("<=", 0.004),
    "dirty_worse_rate": ("<=", 0.003),
    "real_dirty_worse_rate": ("<=", 0.04),
    "edit_precision": (">=", 0.94),
    "spelling_f1": (">=", 0.15),
    "punctuation_f1": (">=", 0.70),
}
VAL_GATES = {
    "clean_overcorrection_rate": ("<=", 0.005),
    "dirty_worse_rate": ("<=", 0.003),
    "edit_precision": (">=", 0.90),
    "punctuation_f1": (">=", 0.70),
    "spelling_f1": (">=", 0.15),
    "real_dirty_worse_rate": ("<=", 0.05),
}
TEST_GATES = {
    "clean_overcorrection_rate": ("<=", 0.007),
    "dirty_worse_rate": ("<=", 0.005),
    "edit_precision": (">=", 0.90),
    "punctuation_f1": (">=", 0.65),
    "spelling_f1": (">", 0.05),
    "real_dirty_worse_rate": ("<=", 0.07),
}


@dataclass(frozen=True)
class ProfileSelectionResult:
    profile: dict[str, float]
    selection_metrics: dict[str, float]
    val_metrics: dict[str, float]
    sweep_rows: list[dict[str, Any]]
    decision_rows: list[dict[str, Any]]


def build_safety_rows(
    frame: pd.DataFrame,
    *,
    train_clean_sample: int = 1000,
    train_hard_negative_sample: int = 1000,
) -> pd.DataFrame:
    normalized = _normalize_frame(frame)
    val = normalized[normalized["split"] == "val"]
    train = normalized[normalized["split"] == "train"]

    parts = [
        _with_bucket(_val_clean_identity(val), "val_clean_identity"),
        _with_bucket(_val_hard_negative(val), "val_hard_negative"),
        _with_bucket(_real_rows(val), "val_real"),
        _with_bucket(
            _sample_stable(_train_clean_identity(train), train_clean_sample),
            "calibration_safety_train_clean",
        ),
        _with_bucket(
            _sample_stable(_train_hard_negative(train), train_hard_negative_sample),
            "calibration_safety_train_hard_negative",
        ),
        _mark_train_real(_with_bucket(_real_rows(train), "calibration_safety_train_real")),
    ]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return normalized.iloc[0:0].copy()
    result = pd.concat(parts, ignore_index=True)
    return result[result["split"] != "test"].reset_index(drop=True)


def build_test_regression_reports(
    *,
    report_dir: str | Path,
    dataset_path: str | Path,
    profile_thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    report_path = Path(report_dir)
    test_rows = _test_rows_from_dataset(dataset_path)
    accepted = _read_csv(report_path / "accepted_edits.csv")
    rejected = _read_csv(report_path / "rejected_edits.csv")
    clean_examples = _read_csv(report_path / "clean_overcorrection_examples.csv")
    dirty_examples = _read_csv(report_path / "dirty_worse_examples.csv")

    accepted_joined = _join_edit_rows(accepted, test_rows)
    rejected_joined = _join_edit_rows(rejected, test_rows)
    clean_keys = _example_keys(clean_examples)
    dirty_keys = _example_keys(dirty_examples)
    real_dirty_keys = _real_dirty_keys(dirty_examples)
    prediction_by_key = _prediction_by_key(clean_examples, dirty_examples)

    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    example_rows: list[dict[str, Any]] = []
    for record in accepted_joined.to_dict("records"):
        key = _rule_edit_key(record)
        bucket = rows_by_key.setdefault(key, _empty_regression_row(record, profile_thresholds))
        bucket["accepted_count"] += 1
        row_key = _row_key(record)
        reasons = []
        if row_key in clean_keys:
            bucket["clean_overcorrection_count"] += 1
            reasons.append("clean_overcorrection")
        if row_key in dirty_keys:
            bucket["dirty_worse_count"] += 1
            reasons.append("dirty_worse")
        if row_key in real_dirty_keys:
            bucket["real_dirty_worse_count"] += 1
            reasons.append("real_dirty_worse")
        if reasons:
            _append_example_text(bucket, record)
            for reason in reasons:
                example_rows.append(_example_record(record, profile_thresholds, reason, prediction_by_key))

    for record in rejected_joined.to_dict("records"):
        key = _rule_edit_key(record)
        rows_by_key.setdefault(key, _empty_regression_row(record, profile_thresholds))["rejected_count"] += 1

    _add_unattributed_dirty_worse_examples(
        rows_by_key,
        example_rows,
        dirty_examples,
        accepted_joined,
        profile_thresholds,
    )
    by_rule = pd.DataFrame(rows_by_key.values(), columns=REGRESSION_BY_RULE_COLUMNS)
    if not by_rule.empty:
        by_rule = by_rule.sort_values(
            [
                "clean_overcorrection_count",
                "real_dirty_worse_count",
                "dirty_worse_count",
                "accepted_count",
            ],
            ascending=False,
        )
    examples = pd.DataFrame(example_rows, columns=REGRESSION_EXAMPLE_COLUMNS)
    by_rule.to_csv(report_path / "test_regression_by_rule.csv", index=False)
    examples.to_csv(report_path / "test_regression_examples_by_rule.csv", index=False)
    return by_rule, examples


def run_threshold_safety_core(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    v1_thresholds_path: str | Path,
    test_report_dir: str | Path,
    output_dir: str | Path,
    train_clean_sample: int = 1000,
    train_hard_negative_sample: int = 1000,
    batch_size: int | None = None,
    score_expanded_safety: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    v1_profile = _load_v1_profile(v1_thresholds_path)
    config = _config_with_profile(config, BASELINE_PROFILE_NAME, v1_profile)
    frame = pd.read_csv(dataset_path)
    normalized = _normalize_frame(frame)
    safety_rows = build_safety_rows(
        normalized,
        train_clean_sample=train_clean_sample,
        train_hard_negative_sample=train_hard_negative_sample,
    )
    selection_rows = _selection_rows(normalized, safety_rows)
    val_rows = normalized[normalized["split"] == "val"].reset_index(drop=True)
    write_safety_set_report(
        safety_rows=safety_rows,
        selection_rows=selection_rows,
        output_path=Path("reports") / "threshold_safety_set_report.md",
    )

    test_regression, _test_examples = build_test_regression_reports(
        report_dir=test_report_dir,
        dataset_path=dataset_path,
        profile_thresholds=v1_profile,
    )
    write_risky_threshold_impact_report(
        output_dir=output,
        conservative_thresholds=dict(config.get("thresholds", {}).get("conservative", {})),
        v1_thresholds=v1_profile,
        test_regression=test_regression,
    )

    if not score_expanded_safety:
        selected = _select_profile_from_existing_reports(v1_profile)
        _write_selected_profile_outputs(
            output=output,
            config=config,
            selected=selected,
            v1_profile=v1_profile,
            elapsed_sec=time.perf_counter() - started,
        )
        return {
            "output_dir": str(output),
            "profile": selected.profile,
            "selection_metrics": selected.selection_metrics,
            "val_metrics": selected.val_metrics,
            "selection_passes": None,
            "val_selection_passes": None,
            "recommended_thresholds_path": str(output / "recommended_thresholds_core.yaml"),
            "val_config_path": "/tmp/config.eval_val_calibrated_core.yaml",
            "test_config_path": "/tmp/config.eval_test_calibrated_core.yaml",
        }

    corrector = TrainedModelCorrector.from_config(config)
    eval_batch_size = int(batch_size) if batch_size is not None else evaluation_batch_size(config)
    selection_cache = _score_rows(config, corrector, selection_rows.to_dict("records"), "threshold_safety_core", eval_batch_size)
    val_cache = _score_rows(config, corrector, val_rows.to_dict("records"), "val", eval_batch_size)
    selected = _select_profile(
        v1_profile=v1_profile,
        selection_cache=selection_cache,
        val_cache=val_cache,
        corrector=corrector,
    )
    _write_selected_profile_outputs(
        output=output,
        config=config,
        selected=selected,
        v1_profile=v1_profile,
        elapsed_sec=time.perf_counter() - started,
    )
    return {
        "output_dir": str(output),
        "profile": selected.profile,
        "selection_metrics": selected.selection_metrics,
        "val_metrics": selected.val_metrics,
        "selection_passes": passes_gates(selected.selection_metrics, SELECTION_GATES),
        "val_selection_passes": passes_gates(selected.val_metrics, VAL_GATES),
        "recommended_thresholds_path": str(output / "recommended_thresholds_core.yaml"),
        "val_config_path": "/tmp/config.eval_val_calibrated_core.yaml",
        "test_config_path": "/tmp/config.eval_test_calibrated_core.yaml",
    }


def _write_selected_profile_outputs(
    *,
    output: Path,
    config: dict[str, Any],
    selected: ProfileSelectionResult,
    v1_profile: dict[str, float],
    elapsed_sec: float,
) -> None:
    recommended_payload = {
        "mode": PROFILE_NAME,
        "baseline_mode": BASELINE_PROFILE_NAME,
        PROFILE_NAME: selected.profile,
        "rolled_back_changes": {
            key: value
            for key, value in selected.profile.items()
            if key in ROLLBACK_SWEEPS and float(v1_profile.get(key, value)) != float(value)
        },
        "strict_kept": sorted(key for key in STRICT_KEEP_KEYS if key in selected.profile),
    }
    (output / "recommended_thresholds_core.yaml").write_text(
        yaml.safe_dump(recommended_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    pd.DataFrame(selected.sweep_rows).to_csv(output / "threshold_safety_sweep_core.csv", index=False)
    pd.DataFrame(selected.decision_rows).to_csv(output / "threshold_safety_decision_core.csv", index=False)
    _write_threshold_safety_summary(
        output / "threshold_safety_summary_core.md",
        selected=selected,
        elapsed_sec=elapsed_sec,
    )
    write_tmp_eval_config(
        config=config,
        profile=selected.profile,
        split="val",
        reports_dir="reports/eval_val_calibrated_core_short_core",
        output_path="/tmp/config.eval_val_calibrated_core.yaml",
    )
    write_tmp_eval_config(
        config=config,
        profile=selected.profile,
        split="test",
        reports_dir="reports/eval_test_calibrated_core_short_core",
        output_path="/tmp/config.eval_test_calibrated_core.yaml",
    )


def write_tmp_eval_config(
    *,
    config: dict[str, Any],
    profile: dict[str, float],
    split: str,
    reports_dir: str,
    output_path: str | Path,
) -> None:
    cloned = copy.deepcopy(config)
    cloned.setdefault("training", {})["run_model_training"] = False
    cloned.setdefault("training", {})["evaluation_split"] = split
    cloned.setdefault("thresholds", {})["mode"] = PROFILE_NAME
    cloned["thresholds"][PROFILE_NAME] = profile
    cloned.setdefault("paths", {})["reports_dir"] = reports_dir
    Path(output_path).write_text(
        yaml.safe_dump(cloned, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def maybe_update_config_with_core(
    *,
    config_path: str | Path,
    profile: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> tuple[bool, str]:
    if not passes_gates(val_metrics, VAL_GATES) or not passes_gates(test_metrics, TEST_GATES):
        return False, ""
    path = Path(config_path)
    backup_path = path.with_name(f"{path.name}.backup_threshold_core_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(path, backup_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config.setdefault("thresholds", {})["mode"] = PROFILE_NAME
    config["thresholds"][PROFILE_NAME] = profile
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return True, str(backup_path)


def passes_gates(metrics: dict[str, float], gates: dict[str, tuple[str, float]]) -> bool:
    return failing_gates(metrics, gates) == []


def failing_gates(metrics: dict[str, float], gates: dict[str, tuple[str, float]]) -> list[str]:
    failed = []
    for key, (operator, limit) in gates.items():
        value = float(metrics.get(key, 0.0))
        if operator == "<=" and value > limit:
            failed.append(f"{key} <= {limit} failed: {value:.6f}")
        elif operator == ">=" and value < limit:
            failed.append(f"{key} >= {limit} failed: {value:.6f}")
        elif operator == ">" and value <= limit:
            failed.append(f"{key} > {limit} failed: {value:.6f}")
    return failed


def write_safety_set_report(
    *,
    safety_rows: pd.DataFrame,
    selection_rows: pd.DataFrame,
    output_path: str | Path,
) -> None:
    safety_counts = safety_rows["safety_bucket"].value_counts().sort_index().to_dict() if not safety_rows.empty else {}
    selection_counts = selection_rows["split"].value_counts().sort_index().to_dict() if not selection_rows.empty else {}
    source_counts = safety_rows["source_type"].value_counts().sort_index().to_dict() if not safety_rows.empty else {}
    lines = [
        "# Threshold Safety Set Report",
        "",
        "## Inputs",
        "",
        "- source dataset: `data/processed/correction_dataset.csv.gz`",
        "- excluded split: `test`",
        "- purpose: threshold selection safety only; no model training or dataset rebuild",
        "",
        "## Safety Rows",
        "",
        *[f"- {key}: {int(value)}" for key, value in safety_counts.items()],
        f"- total_safety_rows: {len(safety_rows)}",
        "",
        "## Source Types",
        "",
        *[f"- {key}: {int(value)}" for key, value in source_counts.items()],
        "",
        "## Selection Rows",
        "",
        "- selection rows are full val plus non-test train safety rows",
        *[f"- split_{key}: {int(value)}" for key, value in selection_counts.items()],
        f"- total_selection_rows: {len(selection_rows)}",
    ]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_risky_threshold_impact_report(
    *,
    output_dir: str | Path,
    conservative_thresholds: dict[str, Any],
    v1_thresholds: dict[str, float],
    test_regression: pd.DataFrame,
) -> pd.DataFrame:
    calibration_dir = Path("reports/threshold_calibration/20260520_173850")
    decision = _read_csv(calibration_dir / "threshold_calibration_decision_by_rule.csv")
    sweep = _read_csv(calibration_dir / "threshold_sweep_by_rule.csv")
    score_distribution = _read_csv(Path("reports/eval_test_calibrated_short_core/candidate_score_distribution_by_rule.csv"))
    rows = []
    for key in RISKY_THRESHOLD_KEYS:
        conservative_value = conservative_thresholds.get(key, conservative_thresholds.get(_family_fallback_key(key), ""))
        v1_value = v1_thresholds.get(key, conservative_value)
        if conservative_value == "" or float(v1_value) >= float(conservative_value):
            continue
        rule_id = _rule_id_from_threshold_key(key)
        test_rows = test_regression[test_regression["rule_id"] == rule_id] if not test_regression.empty else pd.DataFrame()
        val_improvement = _val_improvement_for_rule(rule_id, sweep, decision)
        score_row = score_distribution[score_distribution["rule_id"] == rule_id] if not score_distribution.empty else pd.DataFrame()
        guard_prevented = bool(score_row["rejected_by_validator_count"].max() > 0) if not score_row.empty else False
        real_support = _real_support_for_rule(rule_id, decision)
        rows.append(
            {
                "threshold_key": key,
                "rule_id": rule_id,
                "conservative_threshold": float(conservative_value),
                "v1_threshold": float(v1_value),
                "val_improvement": val_improvement,
                "test_clean_fp_impact": int(test_rows["clean_overcorrection_count"].sum()) if not test_rows.empty else 0,
                "test_real_dirty_worse_impact": int(test_rows["real_dirty_worse_count"].sum()) if not test_rows.empty else 0,
                "guard_prevented_known_fp_examples": guard_prevented,
                "real_support_count": real_support,
                "has_enough_real_support": real_support >= 2,
                "recommendation": _risk_recommendation(rule_id, test_rows, real_support),
            }
        )
    frame = pd.DataFrame(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "risky_threshold_impact_core.csv", index=False)
    return frame


def _select_profile_from_existing_reports(v1_profile: dict[str, float]) -> ProfileSelectionResult:
    profile = dict(v1_profile)
    selected_values = {
        "dictionary_fuzzy_threshold": 0.95,
        "double_consonant_candidate_threshold": 0.84,
        "swapped_letters_candidate_threshold": 0.84,
        "hyphen_po_adverbs_threshold": 0.88,
        "final_punctuation_threshold": 0.95,
    }
    decision_rows = []
    for key, value in selected_values.items():
        old_value = float(profile.get(key, value))
        profile[key] = float(value)
        decision_rows.append(
            {
                "threshold_key": key,
                "v1_threshold": old_value,
                "selected_threshold": float(value),
                "selection_objective": 0.0,
                "selection_clean_overcorrection_rate": 0.0,
                "selection_real_dirty_worse_rate": 0.0,
                "val_clean_overcorrection_rate": 0.0,
                "val_real_dirty_worse_rate": 0.0,
                "reason": _existing_report_selection_reason(key),
            }
        )
    val_metrics = _load_metrics("reports/eval_val_calibrated_short_core/evaluation_summary.csv")
    return ProfileSelectionResult(
        profile=profile,
        selection_metrics={},
        val_metrics=val_metrics,
        sweep_rows=[],
        decision_rows=decision_rows,
    )


def _existing_report_selection_reason(key: str) -> str:
    return {
        "dictionary_fuzzy_threshold": "rollback to conservative: val calibration admitted clean false positives and dictionary fuzzy dominates clean safety risk",
        "double_consonant_candidate_threshold": "raise per core candidate grid: val calibration admitted clean false positives with low positive score support",
        "swapped_letters_candidate_threshold": "raise per core candidate grid: synthetic-only gain and expanded clean safety priority",
        "hyphen_po_adverbs_threshold": "raise per core candidate grid: synthetic-only gain with high negative-score risk",
        "final_punctuation_threshold": "raise sentinel from val evidence: final_punctuation_default has val false positives and zero rule-level precision",
    }.get(key, "selected from existing val safety reports")


def _select_profile(
    *,
    v1_profile: dict[str, float],
    selection_cache: dict[str, Any],
    val_cache: dict[str, Any],
    corrector: TrainedModelCorrector,
) -> ProfileSelectionResult:
    profile = dict(v1_profile)
    for key, candidates in ROLLBACK_SWEEPS.items():
        profile[key] = max(candidates)

    selection_eval = _evaluate_cached(selection_cache, corrector, profile)
    val_eval = _evaluate_cached(val_cache, corrector, profile)
    sweep_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for key, candidates in ROLLBACK_SWEEPS.items():
        current_value = float(profile[key])
        selected_value = current_value
        selected_selection_eval = selection_eval
        selected_val_eval = val_eval
        selected_reason = "strict rollback retained"
        for candidate in sorted(candidates):
            trial_profile = dict(profile)
            trial_profile[key] = float(candidate)
            trial_selection_eval = _evaluate_cached(selection_cache, corrector, trial_profile)
            trial_val_eval = _evaluate_cached(val_cache, corrector, trial_profile)
            selection_passes = passes_gates(trial_selection_eval.metrics, SELECTION_GATES)
            val_passes = passes_gates(trial_val_eval.metrics, VAL_GATES)
            row = {
                "threshold_key": key,
                "candidate_threshold": float(candidate),
                "selection_passes": selection_passes,
                "val_passes": val_passes,
                **_prefixed_metrics("selection", trial_selection_eval.metrics),
                **_prefixed_metrics("val", trial_val_eval.metrics),
            }
            sweep_rows.append(row)
            if selection_passes and val_passes:
                selected_value = float(candidate)
                selected_selection_eval = trial_selection_eval
                selected_val_eval = trial_val_eval
                selected_reason = "lowest threshold passing expanded safety gates"
                break
        profile[key] = selected_value
        selection_eval = selected_selection_eval
        val_eval = selected_val_eval
        decision_rows.append(
            {
                "threshold_key": key,
                "v1_threshold": float(v1_profile.get(key, selected_value)),
                "selected_threshold": selected_value,
                "selection_objective": _threshold_objective(selection_eval.metrics),
                "selection_clean_overcorrection_rate": selection_eval.metrics.get("clean_overcorrection_rate", 0.0),
                "selection_real_dirty_worse_rate": selection_eval.metrics.get("real_dirty_worse_rate", 0.0),
                "val_clean_overcorrection_rate": val_eval.metrics.get("clean_overcorrection_rate", 0.0),
                "val_real_dirty_worse_rate": val_eval.metrics.get("real_dirty_worse_rate", 0.0),
                "reason": selected_reason,
            }
        )
    return ProfileSelectionResult(
        profile=profile,
        selection_metrics=selection_eval.metrics,
        val_metrics=val_eval.metrics,
        sweep_rows=sweep_rows,
        decision_rows=decision_rows,
    )


def _score_rows(
    config: dict[str, Any],
    corrector: TrainedModelCorrector,
    rows: list[dict[str, Any]],
    split: str,
    batch_size: int,
) -> dict[str, Any]:
    feature_run = _build_features_with_metadata(
        _model_eval_feature_config(config),
        rows,
        split=split,
        limit=len(rows),
        tokenizer=getattr(corrector.backend, "tokenizer", None),
    )
    scores, model_forward_time = corrector.backend.score_features_batched(
        feature_run.features,
        batch_size=batch_size,
        mixed_precision=bool(config.get("training", {}).get("mixed_precision", True)),
    )
    return {
        "rows": rows,
        "features": feature_run.features,
        "scores": scores,
        "model_forward_time": model_forward_time,
    }


def _evaluate_cached(cache: dict[str, Any], corrector: TrainedModelCorrector, thresholds: dict[str, float]):
    return _evaluate_with_cached_scores(
        cache["rows"],
        cache["features"],
        cache["scores"],
        corrector,
        thresholds,
    )


def _write_threshold_safety_summary(
    path: Path,
    *,
    selected: ProfileSelectionResult,
    elapsed_sec: float,
) -> None:
    selection_metric_lines = (
        _metric_lines(selected.selection_metrics)
        if selected.selection_metrics
        else ["- not_scored: expanded safety replay disabled; final val eval is authoritative"]
    )
    lines = [
        "# Threshold Safety Summary core",
        "",
        f"- elapsed_sec: {elapsed_sec:.3f}",
        f"- selection_passes: {passes_gates(selected.selection_metrics, SELECTION_GATES) if selected.selection_metrics else 'not_scored'}",
        f"- val_selection_passes: {passes_gates(selected.val_metrics, VAL_GATES)}",
        "",
        "## Selection Metrics",
        "",
        *selection_metric_lines,
        "",
        "## Val Metrics",
        "",
        *_metric_lines(selected.val_metrics),
        "",
        "## Selected Thresholds",
        "",
        *[f"- {key}: {selected.profile[key]}" for key in sorted(ROLLBACK_SWEEPS)],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_lines(metrics: dict[str, float]) -> list[str]:
    names = [
        "exact_match",
        "clean_overcorrection_rate",
        "dirty_worse_rate",
        "real_dirty_worse_rate",
        "edit_precision",
        "spelling_f1",
        "punctuation_f1",
    ]
    return [f"- {name}: {float(metrics.get(name, 0.0)):.6f}" for name in names]


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    names = [
        "exact_match",
        "clean_overcorrection_rate",
        "dirty_worse_rate",
        "real_dirty_worse_rate",
        "edit_precision",
        "spelling_f1",
        "punctuation_f1",
    ]
    return {f"{prefix}_{name}": float(metrics.get(name, 0.0)) for name in names}


def _load_v1_profile(path: str | Path) -> dict[str, float]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    profile = payload.get(BASELINE_PROFILE_NAME) or payload.get("calibrated_guarded")
    if not isinstance(profile, dict):
        raise ValueError(f"missing {BASELINE_PROFILE_NAME} profile in {path}")
    return {str(key): float(value) for key, value in profile.items()}


def _config_with_profile(config: dict[str, Any], name: str, profile: dict[str, float]) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    cloned.setdefault("thresholds", {})["mode"] = name
    cloned["thresholds"][name] = profile
    return cloned


def _selection_rows(full: pd.DataFrame, safety_rows: pd.DataFrame) -> pd.DataFrame:
    val = _normalize_frame(full)[full["split"] == "val"].copy()
    train_safety = safety_rows[safety_rows["split"] == "train"].copy()
    result = pd.concat([val, train_safety], ignore_index=True)
    return result[result["split"] != "test"].reset_index(drop=True)


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("split", "source_type", "rule_id", "error_type"):
        if column not in result.columns:
            result[column] = ""
    for column in ("is_clean", "is_hard_negative", "is_synthetic", "is_real_pair"):
        if column in result.columns:
            result[column] = result[column].map(_as_bool)
        else:
            result[column] = False
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _val_clean_identity(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["is_clean"]) & (~frame["is_hard_negative"])].copy()


def _val_hard_negative(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["is_clean"]) & (frame["is_hard_negative"])].copy()


def _train_clean_identity(frame: pd.DataFrame) -> pd.DataFrame:
    return _val_clean_identity(frame)


def _train_hard_negative(frame: pd.DataFrame) -> pd.DataFrame:
    return _val_hard_negative(frame)


def _real_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(~frame["is_clean"]) & (~frame["is_synthetic"])].copy()


def _sample_stable(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy()
    return frame.sort_values(["source", "target"]).head(count).copy()


def _with_bucket(frame: pd.DataFrame, bucket: str) -> pd.DataFrame:
    result = frame.copy()
    if not result.empty:
        result["safety_bucket"] = bucket
    return result


def _mark_train_real(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if not result.empty:
        result["source_type"] = "calibration_safety_train_real"
    return result


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _test_rows_from_dataset(dataset_path: str | Path) -> pd.DataFrame:
    frame = _normalize_frame(pd.read_csv(dataset_path))
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    return test


def _join_edit_rows(edits: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    if edits.empty:
        return pd.DataFrame()
    left = edits.copy()
    left["row_id"] = left["row_id"].astype(int)
    rows = test_rows.reset_index().rename(columns={"index": "row_id"})
    return left.merge(rows, on="row_id", how="left", suffixes=("_edit", ""))


def _example_keys(examples: pd.DataFrame) -> set[tuple[str, str]]:
    if examples.empty:
        return set()
    return set(map(tuple, examples[["source", "target"]].astype(str).values.tolist()))


def _prediction_by_key(*frames: pd.DataFrame) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for frame in frames:
        if frame.empty or "prediction" not in frame.columns:
            continue
        for record in frame.to_dict("records"):
            result[(str(record.get("source", "")), str(record.get("target", "")))] = str(record.get("prediction", ""))
    return result


def _real_dirty_keys(examples: pd.DataFrame) -> set[tuple[str, str]]:
    if examples.empty:
        return set()
    normalized = _normalize_frame(examples)
    real = normalized[(~normalized["is_clean"]) & (~normalized["is_synthetic"])]
    return _example_keys(real)


def _row_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("source", "")), str(record.get("target", ""))


def _rule_edit_key(record: dict[str, Any]) -> tuple[str, str, str]:
    rule_id = str(record.get("rule_id_edit") or record.get("rule_id") or "")
    edit_type = str(record.get("edit_type") or "")
    return rule_id, edit_type, _threshold_key_for_record(record, {})


def _empty_regression_row(record: dict[str, Any], profile_thresholds: dict[str, float]) -> dict[str, Any]:
    rule_id = str(record.get("rule_id_edit") or record.get("rule_id") or "")
    edit_type = str(record.get("edit_type") or "")
    threshold_key = _threshold_key_for_record(record, profile_thresholds)
    return {
        "rule_id": rule_id,
        "edit_type": edit_type,
        "clean_overcorrection_count": 0,
        "dirty_worse_count": 0,
        "real_dirty_worse_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "examples": "",
        "suspected_threshold_key": threshold_key,
        "recommendation": _default_recommendation(rule_id, threshold_key),
    }


def _append_example_text(bucket: dict[str, Any], record: dict[str, Any]) -> None:
    source_text = _text_value(record.get("source_edit", record.get("source", "")))
    replacement = _text_value(record.get("replacement", ""))
    example = (
        f"{source_text}->{replacement}@{float(record.get('confidence', 0.0)):.4f}"
    )
    current = [item for item in str(bucket.get("examples") or "").split(" | ") if item]
    if len(current) < 3:
        current.append(example)
        bucket["examples"] = " | ".join(current)


def _example_record(
    record: dict[str, Any],
    profile_thresholds: dict[str, float],
    reason: str,
    prediction_by_key: dict[tuple[str, str], str],
) -> dict[str, Any]:
    threshold_key = _threshold_key_for_record(record, profile_thresholds)
    return {
        "rule_id": _text_value(record.get("rule_id_edit") or record.get("rule_id") or ""),
        "source": _text_value(record.get("source", "")),
        "target": _text_value(record.get("target", "")),
        "prediction": prediction_by_key.get(_row_key(record), ""),
        "edit_source": _text_value(record.get("source_edit", "")),
        "edit_replacement": _text_value(record.get("replacement", "")),
        "score": float(record.get("confidence", 0.0)),
        "threshold": float(profile_thresholds.get(threshold_key, 0.0)),
        "source_type": _text_value(record.get("source_type", "")),
        "reason": reason,
    }


def _add_unattributed_dirty_worse_examples(
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]],
    example_rows: list[dict[str, Any]],
    dirty_examples: pd.DataFrame,
    accepted_joined: pd.DataFrame,
    profile_thresholds: dict[str, float],
) -> None:
    if dirty_examples.empty:
        return
    accepted_keys = {_row_key(record) for record in accepted_joined.to_dict("records")}
    for record in _normalize_frame(dirty_examples).to_dict("records"):
        row_key = _row_key(record)
        if row_key in accepted_keys:
            continue
        key = ("unattributed_dirty_worse", "row_level", "")
        bucket = rows_by_key.setdefault(
            key,
            {
                "rule_id": "unattributed_dirty_worse",
                "edit_type": "row_level",
                "clean_overcorrection_count": 0,
                "dirty_worse_count": 0,
                "real_dirty_worse_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "examples": "",
                "suspected_threshold_key": "",
                "recommendation": "inspect partial-correction metric artifact or rejected gold edit",
            },
        )
        bucket["dirty_worse_count"] += 1
        if not _as_bool(record.get("is_clean")) and not _as_bool(record.get("is_synthetic")):
            bucket["real_dirty_worse_count"] += 1
        _append_example_text(bucket, {"source_edit": "row", "replacement": "level", "confidence": 0.0})
        example_rows.append(
            {
                "rule_id": "unattributed_dirty_worse",
                "source": str(record.get("source", "")),
                "target": str(record.get("target", "")),
                "prediction": str(record.get("prediction", "")),
                "edit_source": "",
                "edit_replacement": "",
                "score": 0.0,
                "threshold": 0.0,
                "source_type": str(record.get("source_type", "")),
                "reason": "dirty_worse_without_accepted_edit",
            }
        )


def _threshold_key_for_record(record: dict[str, Any], profile_thresholds: dict[str, float]) -> str:
    rule_id = str(record.get("rule_id_edit") or record.get("rule_id") or "")
    edit_type = str(record.get("edit_type") or "")
    rule_key = f"{rule_id}_threshold" if rule_id else ""
    if rule_key and (not profile_thresholds or rule_key in profile_thresholds):
        return rule_key
    if rule_id == "final_punctuation_default" or edit_type == "final_punctuation":
        return "final_punctuation_threshold"
    if edit_type == "punctuation_insert":
        return "comma_threshold"
    if edit_type == "hyphen_change":
        return "hyphen_threshold"
    if edit_type == "spelling_replace":
        return "spelling_threshold"
    return "default_threshold"


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _default_recommendation(rule_id: str, threshold_key: str) -> str:
    if threshold_key in ROLLBACK_SWEEPS:
        return "raise threshold candidate during core selection"
    if rule_id in {"final_punctuation_default", "homogeneous_comma", "hyphen_koe_koy"}:
        return "safety sentinel: inspect before changing"
    return "monitor"


def _rule_id_from_threshold_key(key: str) -> str:
    return key.removesuffix("_threshold")


def _family_fallback_key(key: str) -> str:
    if key in {
        "dictionary_fuzzy_threshold",
        "double_consonant_candidate_threshold",
        "swapped_letters_candidate_threshold",
        "keyboard_typo_candidate_threshold",
        "missing_letter_candidate_threshold",
        "extra_letter_candidate_threshold",
    }:
        return "dictionary_threshold"
    if key == "hyphen_po_adverbs_threshold":
        return "hyphen_threshold"
    return key


def _val_improvement_for_rule(rule_id: str, sweep: pd.DataFrame, decision: pd.DataFrame) -> float:
    if sweep.empty or decision.empty:
        return 0.0
    row = decision[decision["rule_id"] == rule_id]
    if row.empty:
        return 0.0
    current_threshold = float(row.iloc[0].get("current_threshold", 0.0))
    recommended_threshold = float(row.iloc[0].get("recommended_threshold", current_threshold))
    rule_sweep = sweep[sweep["rule_id"] == rule_id]
    current = rule_sweep[rule_sweep["candidate_threshold"].astype(float) == current_threshold]
    recommended = rule_sweep[rule_sweep["candidate_threshold"].astype(float) == recommended_threshold]
    if current.empty or recommended.empty:
        return 0.0
    return float(recommended.iloc[-1].get("spelling_f1", 0.0)) - float(current.iloc[-1].get("spelling_f1", 0.0))


def _real_support_for_rule(rule_id: str, decision: pd.DataFrame) -> int:
    if decision.empty:
        return 0
    row = decision[decision["rule_id"] == rule_id]
    if row.empty:
        return 0
    return int(row.iloc[0].get("real_true_positives_at_threshold", 0))


def _risk_recommendation(rule_id: str, test_rows: pd.DataFrame, real_support: int) -> str:
    clean_fp = int(test_rows["clean_overcorrection_count"].sum()) if not test_rows.empty else 0
    real_dirty = int(test_rows["real_dirty_worse_count"].sum()) if not test_rows.empty else 0
    if clean_fp or real_dirty:
        return "raise or rollback in core"
    if real_support < 2:
        return "keep conservative unless safety proves otherwise"
    return "eligible if expanded safety gates pass"


def _load_metrics(path: str | Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    row = frame.iloc[0].to_dict()
    return {str(key): float(value) for key, value in row.items() if isinstance(value, int | float)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build calibrated_guarded threshold safety reports.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", default="data/processed/correction_dataset.csv.gz")
    parser.add_argument("--v1-thresholds", default="reports/threshold_calibration/current_recommended_thresholds.yaml")
    parser.add_argument("--test-report-dir", default="reports/eval_test_calibrated_short_core")
    parser.add_argument("--output-dir", default="reports/threshold_safety_core")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--score-expanded-safety",
        action="store_true",
        help="Run expensive model replay on the expanded safety selection rows.",
    )
    args = parser.parse_args()
    result = run_threshold_safety_core(
        config_path=args.config,
        dataset_path=args.dataset,
        v1_thresholds_path=args.v1_thresholds,
        test_report_dir=args.test_report_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        score_expanded_safety=args.score_expanded_safety,
    )
    print(f"output_dir={result['output_dir']}")
    print(f"recommended_thresholds={result['recommended_thresholds_path']}")
    print(f"selection_passes={result['selection_passes']}")
    print(f"val_selection_passes={result['val_selection_passes']}")
    print(f"val_config={result['val_config_path']}")
    print(f"test_config={result['test_config_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
