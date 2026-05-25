from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.candidates.candidate_generator import CandidateGenerator
from src.config.candidate_dataset_config import (
    candidate_dataset_audit,
    candidate_dataset_rule_activation,
    candidate_dataset_rule_quota,
    candidate_dataset_value,
)
from src.config.load_config import load_config
from src.data import rule_data_compiler
from src.data.dataset_quality import clean_or_hard_quality_reasons, clean_or_hard_quality_pass, positive_target_quality_reasons
from src.evaluation.candidate_recall import build_candidate_recall_reports
from src.rules.capabilities import (
    active_rule_ids_for_training,
    activation_policy_from_config,
    activation_stage_for_rule,
    load_rule_capabilities,
    production_ready_flag_for_rule,
)
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id
from src.data.training_quality_audit import contains_artificial_marker_text


READY_STATUS = "READY_FOR_FULL_BUILD_PROBE_PASSED"
NOT_READY_STATUS = "NOT_READY_FOR_FULL_BUILD"
PROBE_SOURCE_PRIORITY = [
    rule_data_compiler.SOURCE_CORPUS_MINED,
    rule_data_compiler.SOURCE_SYNTAX_MINED,
    rule_data_compiler.SOURCE_MORPHOLOGY_MINED,
    rule_data_compiler.SOURCE_REAL_PATTERN_REPLAY,
    rule_data_compiler.SOURCE_RULE_LAB,
]
PROBE_REPORT_NAMES = [
    "coverage_probe_summary.json",
    "coverage_probe_report.csv",
    "coverage_probe_rejection_report.csv",
    "coverage_probe_underfilled_rules.csv",
    "coverage_probe_ready_rules.csv",
]
FORBIDDEN_ARTIFACT_NAMES = (
    "correction_dataset.csv.gz",
    "train.csv",
    "val.csv",
    "test.csv",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fast no-write RuleDataCompiler coverage probe.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports/dataset_build")
    parser.add_argument("--no-full-build", action="store_true")
    parser.add_argument("--max-clean-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.no_full_build:
        print(NOT_READY_STATUS)
        print("- missing_required_flag:--no-full-build")
        return 2

    processed_dir = Path(args.processed_dir)
    reports_dir = Path(args.reports_dir)
    before = _forbidden_artifact_snapshot(processed_dir, config_path=None)
    summary = run_probe(
        config_path=Path(args.config),
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        max_clean_rows=args.max_clean_rows,
        seed=args.seed,
    )
    after = _forbidden_artifact_snapshot(processed_dir, config_path=Path(args.config))
    forbidden_blockers = _forbidden_artifact_blockers(before, after)
    if forbidden_blockers:
        summary["blockers"] = _dedupe([*summary.get("blockers", []), *forbidden_blockers])
        summary["status"] = NOT_READY_STATUS
        _write_summary(reports_dir, summary)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(str(summary["status"]))
    if summary["status"] != READY_STATUS:
        for blocker in summary.get("blockers", []):
            print(f"- {blocker}")
        return 1
    return 0


def run_probe(
    *,
    config_path: Path,
    processed_dir: Path,
    reports_dir: Path,
    max_clean_rows: int | None,
    seed: int | None,
) -> dict[str, Any]:
    config = _probe_config(load_config(config_path), processed_dir=processed_dir, reports_dir=reports_dir, seed=seed)
    thresholds = _thresholds(config)
    clean_pool_path = Path(str(candidate_dataset_value(config, "paths.clean_pool_path", processed_dir / "clean_sentence_pool.csv.gz")))
    resolved_max_clean_rows = _resolved_max_clean_rows(config, max_clean_rows)
    clean_rows = _read_strict_clean_rows(clean_pool_path, config, max_rows=resolved_max_clean_rows)

    capabilities = _load_capabilities(config)
    policy = activation_policy_from_config(config)
    candidate_rule_ids = _candidate_rule_ids(config, capabilities, policy)
    activation = _activation_metadata(capabilities, policy)
    candidate_generator = ProbeCandidateGenerator(CandidateGenerator.from_config(config))
    generation_targets = _probe_generation_targets(config, thresholds)

    compiler_result = rule_data_compiler.compile_rule_data(
        clean_rows,
        candidate_rule_ids,
        candidate_generator,
        config,
        target_counts={
            "atomic_positive": int(generation_targets["atomic_positive"]),
            "atomic_hard_negative": int(generation_targets["atomic_hard_negative"]),
        },
    )
    recall_reports = build_candidate_recall_reports(
        compiler_result.atomic_positive_rows,
        candidate_generator=candidate_generator,
        max_candidates=int((config.get("model", {}) or {}).get("max_candidates", 16)),
        rules_config_path="configs/rules.yaml",
        trust_candidate_backed_metadata=False,
    )
    recall_by_rule = _candidate_recall_values(recall_reports.get("candidate_recall_by_rule", pd.DataFrame()))
    gold_counts_by_rule = _candidate_gold_counts(recall_reports.get("candidate_recall_by_rule", pd.DataFrame()))
    source_stats_by_rule = {
        row.rule_id: asdict(row)
        for row in compiler_result.source_stats_rows
    }
    diversity_by_rule = {
        row.rule_id: asdict(row)
        for row in compiler_result.diversity_stats_rows
    }
    top_rejection_by_rule = _top_rejection_by_rule(compiler_result.rejection_rows)
    raw_rejection_counts = _raw_rejection_rows(compiler_result.rejection_rows)
    rule_ids = sorted(set(candidate_rule_ids) | set(source_stats_by_rule) | set(recall_by_rule) | set(diversity_by_rule))

    report_rows = [
        _coverage_row(
            rule_id,
            source_stats_by_rule.get(rule_id, {}),
            diversity_by_rule.get(rule_id, {}),
            activation.get(rule_id, {}),
            top_rejection_by_rule.get(rule_id, ""),
            candidate_recall=float(recall_by_rule.get(rule_id, 0.0) or 0.0),
            gold_count=int(gold_counts_by_rule.get(rule_id, 0) or 0),
            thresholds=thresholds,
        )
        for rule_id in rule_ids
    ]
    ready_rows = [row for row in report_rows if bool(row["ready_for_full_build"])]
    underfilled_rows = [_underfilled_row(row, thresholds) for row in report_rows if not bool(row["ready_for_full_build"])]
    metrics = _metrics(report_rows, thresholds)
    quality_blockers = _accepted_quality_blockers(compiler_result.hard_negative_rows, compiler_result.atomic_positive_rows)
    destructive_drop = _destructive_atomic_positive_drop_detected(compiler_result.atomic_positive_rows)

    blockers: list[str] = []
    expected = int(thresholds["expected_min_final_active_rule_count"])
    if len(ready_rows) < expected:
        blockers.append(f"ready_rule_count_below_expected:{len(ready_rows)}<{expected}")
    if destructive_drop:
        blockers.append("destructive_atomic_positive_drop_detected")
    if quality_blockers:
        blockers.append(f"clean_hard_quality_blockers:{len(quality_blockers)}")
    if not clean_pool_path.exists():
        blockers.append(f"clean_pool_missing:{clean_pool_path.as_posix()}")
    if not candidate_rule_ids:
        blockers.append("candidate_rule_ids_empty")

    status = READY_STATUS if not blockers else NOT_READY_STATUS
    summary = {
        "status": status,
        "config_path": str(config_path),
        "processed_dir": str(processed_dir),
        "reports_dir": str(reports_dir),
        "clean_pool_path": str(clean_pool_path),
        "clean_rows_probed": int(len(clean_rows)),
        "max_clean_rows": int(resolved_max_clean_rows) if resolved_max_clean_rows is not None else None,
        "source_priority": list(PROBE_SOURCE_PRIORITY),
        "candidate_rule_count": int(len(candidate_rule_ids)),
        "ready_rule_count": int(len(ready_rows)),
        "expected_min_final_active_rule_count": expected,
        "thresholds": thresholds,
        "probe_generation_targets": generation_targets,
        "metrics": metrics,
        "destructive_atomic_positive_drop_detected": destructive_drop,
        "clean_hard_quality_blocker_count": int(len(quality_blockers)),
        "quality_blockers": quality_blockers[:50],
        "top_blockers": _top_blockers(underfilled_rows, blockers),
        "blockers": _dedupe(blockers),
        "warnings": list(compiler_result.warnings),
        "audit_errors": list(compiler_result.audit_errors),
        "reports_written": [str(reports_dir / name) for name in PROBE_REPORT_NAMES],
    }
    _write_reports(
        reports_dir,
        summary=summary,
        coverage_rows=report_rows,
        rejection_rows=raw_rejection_counts,
        underfilled_rows=underfilled_rows,
        ready_rows=ready_rows,
    )
    return summary


def recommended_next_action(reason: str) -> str:
    normalized = str(reason or "").split(":", 1)[0]
    mapping = {
        "candidate_missing": "fix_candidate_generator_or_rule_mapping",
        "strict_validator_rejected": "fix_strict_validator_support",
        "non_atomic_edit_count": "fix_operator_or_miner_to_generate_atomic_edits",
        "target_quality_failed": "fix_template_or_context_quality",
        "unsupported_edit_type": "extend_diff_or_candidate_matching",
        "opportunities_seen_0": "add_rule_specific_miner_or_templates",
        "hard_negative_count_under_min": "add_hard_negative_miner_or_templates",
        "recall_under_min": "inspect_candidate_generator_recall",
        "low_structural_diversity": "add_corpus_contexts_or_templates",
    }
    return mapping.get(normalized, "inspect_rule_data_compiler_rejections")


class ProbeCandidateGenerator:
    def __init__(self, base_generator: Any) -> None:
        self.base_generator = base_generator
        if hasattr(self.base_generator, "syntax_provider"):
            self.base_generator.syntax_provider = lambda _text: []
        self._cache: dict[str, tuple[Any, ...]] = {}

    def generate(self, text: str, **kwargs: Any) -> list[Any]:
        key = str(text)
        if key not in self._cache:
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault("dictionary_policy", "unknown_only")
            try:
                generated = self.base_generator.generate(key, **call_kwargs)
            except TypeError:
                generated = self.base_generator.generate(key)
            self._cache[key] = tuple(generated)
        return list(self._cache[key])


def _probe_config(raw_config: dict[str, Any], *, processed_dir: Path, reports_dir: Path, seed: int | None) -> dict[str, Any]:
    config = deepcopy(raw_config)
    candidate = config.setdefault("data", {}).setdefault("candidate_opportunity", {})
    paths = candidate.setdefault("paths", {})
    paths["processed_dir"] = str(processed_dir)
    paths["reports_dir"] = str(reports_dir)
    paths["clean_pool_path"] = str(processed_dir / "clean_sentence_pool.csv.gz")
    paths.setdefault("correction_dataset_path", str(processed_dir / "correction_dataset.csv.gz"))
    paths.setdefault("manifest_path", str(processed_dir / "dataset_manifest.json"))
    paths["real_error_pairs_atomic_path"] = str(processed_dir / "real_error_pairs_atomic.csv.gz")
    paths["real_error_pairs_stress_path"] = str(processed_dir / "real_error_pairs_stress.csv.gz")
    paths["real_error_pairs_validated_path"] = str(processed_dir / "real_error_pairs_validated.csv.gz")
    if seed is not None:
        candidate["synthetic_seed"] = int(seed)
    compiler_config = candidate.setdefault("rule_data_compiler", {})
    compiler_config["enabled"] = True
    compiler_config["source_priority"] = list(PROBE_SOURCE_PRIORITY)
    compiler_config.setdefault("max_probe_clean_rows", 5_000)
    configured_real_paths = [Path(str(path)) for path in compiler_config.get("real_pattern_paths", []) or []]
    processed_real_paths = [
        processed_dir / "real_error_pairs_mining.csv.gz",
        processed_dir / "real_error_pairs_rejected.csv.gz",
        processed_dir / "real_error_pairs_atomic.csv.gz",
        processed_dir / "real_error_pairs_validated.csv.gz",
    ]
    real_paths = _dedupe_paths([*configured_real_paths, *processed_real_paths])
    compiler_config["real_pattern_paths"] = [str(path) for path in real_paths if path.exists()]
    root_paths = config.setdefault("paths", {})
    root_paths["reports_dir"] = str(reports_dir)
    return config


def _thresholds(config: Mapping[str, Any]) -> dict[str, Any]:
    quota = candidate_dataset_rule_quota(config)
    activation = candidate_dataset_rule_activation(config)
    audit = candidate_dataset_audit(config)
    min_atomic = int(quota.get("min_atomic_positives_per_active_rule", 500) or 0)
    preferred_atomic = int(quota.get("preferred_atomic_positives_per_active_rule", 1500) or min_atomic)
    min_hard = int(quota.get("min_hard_negatives_per_active_rule", 200) or 0)
    min_recall = float(audit.get("min_candidate_recall_for_active_rule", 0.95) or 0.95)
    expected = int(activation.get("expected_min_final_active_rule_count", 25) or 0)
    return {
        "min_atomic": min_atomic,
        "preferred_atomic": preferred_atomic,
        "min_hard": min_hard,
        "min_recall": min_recall,
        "expected_min_final_active_rule_count": expected,
    }


def _resolved_max_clean_rows(config: Mapping[str, Any], explicit: int | None) -> int | None:
    if explicit is not None:
        return None if explicit <= 0 else int(explicit)
    compiler_config = dict(candidate_dataset_value(config, "rule_data_compiler", {}) or {})
    for key in ("max_probe_clean_rows", "max_scan_rows_per_rule"):
        value = compiler_config.get(key)
        if value is not None:
            return max(1, int(value or 1))
    return 50_000


def _probe_generation_targets(config: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, int]:
    compiler_config = dict(candidate_dataset_value(config, "rule_data_compiler", {}) or {})
    positive_cap = int(compiler_config.get("max_probe_generated_per_rule", 50) or 50)
    hard_cap = int(compiler_config.get("max_probe_hard_negatives_per_rule", positive_cap) or positive_cap)
    return {
        "atomic_positive": max(0, min(int(thresholds["preferred_atomic"]), positive_cap)),
        "atomic_hard_negative": max(0, min(int(thresholds["min_hard"]), hard_cap)),
    }


def _read_strict_clean_rows(path: Path, config: Mapping[str, Any], *, max_rows: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    usecols = {
        "text",
        "source_name",
        "source_subcorpus",
        "domain",
        "style",
        "license_status",
        "source_doc_id",
        "sentence_id",
        "hash",
    }
    chunksize = max(1, int(candidate_dataset_value(config, "clean_pool_chunksize", 10_000) or 10_000))
    rows: list[dict[str, Any]] = []
    reader = pd.read_csv(path, usecols=lambda column: column in usecols, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        for row in chunk.fillna("").to_dict("records"):
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            if contains_artificial_marker_text(text, text) or not clean_or_hard_quality_pass(text):
                continue
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def _load_capabilities(config: Mapping[str, Any]) -> list[Any]:
    try:
        return load_rule_capabilities("configs/rules.yaml", config=config)
    except TypeError:
        return load_rule_capabilities("configs/rules.yaml")


def _candidate_rule_ids(config: Mapping[str, Any], capabilities: list[Any], policy: Any) -> list[str]:
    ids = [normalize_rule_id(rule_id) for rule_id in active_rule_ids_for_training(capabilities, policy=policy)]
    configured = [normalize_rule_id(rule_id) for rule_id in candidate_dataset_rule_quota(config).get("rule_ids", []) or []]
    if configured:
        configured_set = set(configured)
        ids = [rule_id for rule_id in ids if rule_id in configured_set]
    return sorted(rule_id for rule_id in dict.fromkeys(ids) if rule_id and rule_id != UNKNOWN_RULE_ID)


def _activation_metadata(capabilities: list[Any], policy: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for capability in capabilities:
        for raw_rule_id in getattr(capability, "project_rule_ids", []) or []:
            rule_id = normalize_rule_id(str(raw_rule_id))
            if not rule_id or rule_id == UNKNOWN_RULE_ID:
                continue
            result.setdefault(
                rule_id,
                {
                    "activation_stage": activation_stage_for_rule(capability, policy),
                    "production_ready": production_ready_flag_for_rule(capability),
                },
            )
    return result


def _coverage_row(
    rule_id: str,
    source_stats: Mapping[str, Any],
    diversity: Mapping[str, Any],
    activation: Mapping[str, Any],
    top_rejection_reason: str,
    *,
    candidate_recall: float,
    gold_count: int,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    atomic_count = int(source_stats.get("total_atomic_positive_count", 0) or 0)
    hard_count = int(source_stats.get("total_hard_negative_count", 0) or 0)
    corpus_count = int(source_stats.get("corpus_mined_positive_count", 0) or 0)
    rule_lab_count = int(source_stats.get("rule_lab_positive_count", 0) or 0)
    corpus_share = _rate(corpus_count, atomic_count)
    rule_lab_share = _rate(rule_lab_count, atomic_count)
    main_blocker = _main_blocker(
        atomic_count=atomic_count,
        hard_count=hard_count,
        candidate_recall=candidate_recall,
        top_rejection_reason=top_rejection_reason,
        diversity_status=str(diversity.get("status", "")),
        thresholds=thresholds,
    )
    ready = main_blocker == ""
    return {
        "rule_id": rule_id,
        "activation_stage": str(activation.get("activation_stage", "")),
        "production_ready": bool(activation.get("production_ready", False)),
        "atomic_positive_count": atomic_count,
        "hard_negative_count": hard_count,
        "candidate_recall": candidate_recall,
        "gold_count": int(gold_count),
        "ready_for_full_build": bool(ready),
        "main_blocker": main_blocker,
        "top_rejection_reason": top_rejection_reason,
        "recommended_next_action": "" if ready else recommended_next_action(main_blocker or top_rejection_reason),
        "corpus_mined_positive_count": corpus_count,
        "syntax_mined_positive_count": int(source_stats.get("syntax_mined_positive_count", 0) or 0),
        "morphology_mined_positive_count": int(source_stats.get("morphology_mined_positive_count", 0) or 0),
        "real_pattern_replay_positive_count": int(source_stats.get("real_pattern_replay_positive_count", 0) or 0),
        "rule_lab_positive_count": rule_lab_count,
        "corpus_mined_share": corpus_share,
        "rule_lab_share": rule_lab_share,
        "corpus_mined_hard_negative_count": int(source_stats.get("corpus_mined_hard_negative_count", 0) or 0),
        "rule_lab_hard_negative_count": int(source_stats.get("rule_lab_hard_negative_count", 0) or 0),
        "unique_source_count": int(diversity.get("unique_source_count", 0) or 0),
        "unique_target_count": int(diversity.get("unique_target_count", 0) or 0),
        "unique_left_context_count": int(diversity.get("unique_left_context_count", 0) or 0),
        "unique_right_context_count": int(diversity.get("unique_right_context_count", 0) or 0),
        "unique_sentence_pattern_count": int(diversity.get("unique_sentence_pattern_count", 0) or 0),
        "dominant_template_share": float(diversity.get("dominant_template_share", 0.0) or 0.0),
        "dominant_source_type_share": float(diversity.get("dominant_source_type_share", 0.0) or 0.0),
        "near_duplicate_count": int(diversity.get("near_duplicate_count", 0) or 0),
        "structural_diversity_status": str(diversity.get("status", "empty") or "empty"),
        "structural_diversity_reason": str(diversity.get("reason", "") or ""),
    }


def _main_blocker(
    *,
    atomic_count: int,
    hard_count: int,
    candidate_recall: float,
    top_rejection_reason: str,
    diversity_status: str,
    thresholds: Mapping[str, Any],
) -> str:
    if atomic_count < int(thresholds["min_atomic"]):
        return top_rejection_reason or "opportunities_seen_0"
    if hard_count < int(thresholds["min_hard"]):
        return "hard_negative_count_under_min"
    if candidate_recall < float(thresholds["min_recall"]):
        return "recall_under_min"
    if diversity_status in {"blocked", "fail"}:
        return "low_structural_diversity"
    return ""


def _underfilled_row(row: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    main_blocker = str(row.get("main_blocker") or row.get("top_rejection_reason") or "opportunities_seen_0")
    return {
        "rule_id": str(row.get("rule_id", "")),
        "activation_stage": str(row.get("activation_stage", "")),
        "atomic_positive_count": int(row.get("atomic_positive_count", 0) or 0),
        "hard_negative_count": int(row.get("hard_negative_count", 0) or 0),
        "candidate_recall": float(row.get("candidate_recall", 0.0) or 0.0),
        "min_atomic_required": int(thresholds["min_atomic"]),
        "preferred_atomic": int(thresholds["preferred_atomic"]),
        "min_hard_required": int(thresholds["min_hard"]),
        "corpus_mined_positive_count": int(row.get("corpus_mined_positive_count", 0) or 0),
        "syntax_mined_positive_count": int(row.get("syntax_mined_positive_count", 0) or 0),
        "morphology_mined_positive_count": int(row.get("morphology_mined_positive_count", 0) or 0),
        "real_pattern_replay_positive_count": int(row.get("real_pattern_replay_positive_count", 0) or 0),
        "rule_lab_positive_count": int(row.get("rule_lab_positive_count", 0) or 0),
        "rule_lab_share": float(row.get("rule_lab_share", 0.0) or 0.0),
        "main_blocker": main_blocker,
        "top_rejection_reason": str(row.get("top_rejection_reason") or main_blocker),
        "recommended_next_action": recommended_next_action(main_blocker),
    }


def _metrics(rows: list[dict[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    recall_threshold = float(thresholds["min_recall"])
    metrics = {
        "rules_with_atomic_ge_100": _count(rows, lambda row: int(row["atomic_positive_count"]) >= 100),
        "rules_with_atomic_ge_500": _count(rows, lambda row: int(row["atomic_positive_count"]) >= 500),
        "rules_with_atomic_ge_1000": _count(rows, lambda row: int(row["atomic_positive_count"]) >= 1000),
        "rules_with_atomic_ge_1500": _count(rows, lambda row: int(row["atomic_positive_count"]) >= 1500),
        "rules_with_atomic_ge_500_and_hard_ge_100": _count(
            rows,
            lambda row: int(row["atomic_positive_count"]) >= 500 and int(row["hard_negative_count"]) >= 100,
        ),
        "rules_with_atomic_ge_500_and_hard_ge_200": _count(
            rows,
            lambda row: int(row["atomic_positive_count"]) >= 500 and int(row["hard_negative_count"]) >= 200,
        ),
        "rules_with_atomic_ge_1500_and_hard_ge_200": _count(
            rows,
            lambda row: int(row["atomic_positive_count"]) >= 1500 and int(row["hard_negative_count"]) >= 200,
        ),
        "rules_with_atomic_ge_500_and_hard_ge_200_and_recall_ge_095": _count(
            rows,
            lambda row: int(row["atomic_positive_count"]) >= 500
            and int(row["hard_negative_count"]) >= 200
            and float(row["candidate_recall"]) >= recall_threshold,
        ),
        "rules_with_atomic_ge_1500_and_hard_ge_200_and_recall_ge_095": _count(
            rows,
            lambda row: int(row["atomic_positive_count"]) >= 1500
            and int(row["hard_negative_count"]) >= 200
            and float(row["candidate_recall"]) >= recall_threshold,
        ),
        "median_corpus_mined_share": _median([float(row["corpus_mined_share"]) for row in rows if int(row["atomic_positive_count"]) > 0]),
        "median_rule_lab_share": _median([float(row["rule_lab_share"]) for row in rows if int(row["atomic_positive_count"]) > 0]),
        "rules_with_low_structural_diversity": _count(
            rows,
            lambda row: int(row["atomic_positive_count"]) > 0
            and str(row.get("structural_diversity_status") or "") not in {"ok", ""},
        ),
    }
    return metrics


def _accepted_quality_blockers(hard_rows: Iterable[Mapping[str, Any]], positive_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for index, row in enumerate(hard_rows):
        source = str(row.get("source", ""))
        for reason in clean_or_hard_quality_reasons(source):
            blockers.append(
                {
                    "row_index": index,
                    "dataset_layer": "atomic_hard_negative",
                    "rule_id": str(row.get("target_rule_id", "")),
                    "reason": reason,
                    "source": source[:300],
                }
            )
    for index, row in enumerate(positive_rows):
        target = str(row.get("target", ""))
        for reason in positive_target_quality_reasons(target):
            blockers.append(
                {
                    "row_index": index,
                    "dataset_layer": "atomic_positive",
                    "rule_id": str(row.get("rule_id", "")),
                    "reason": reason,
                    "source": str(row.get("source", ""))[:300],
                }
            )
    return blockers


def _destructive_atomic_positive_drop_detected(positive_rows: Iterable[Mapping[str, Any]]) -> bool:
    rows = list(positive_rows)
    if not rows:
        return False
    kept = [row for row in rows if not positive_target_quality_reasons(str(row.get("target", "")))]
    return len(kept) == 0


def _write_reports(
    reports_dir: Path,
    *,
    summary: Mapping[str, Any],
    coverage_rows: list[dict[str, Any]],
    rejection_rows: list[dict[str, Any]],
    underfilled_rows: list[dict[str, Any]],
    ready_rows: list[dict[str, Any]],
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(reports_dir, summary)
    pd.DataFrame(coverage_rows, columns=_coverage_columns()).to_csv(reports_dir / "coverage_probe_report.csv", index=False)
    pd.DataFrame(rejection_rows, columns=_rejection_columns()).to_csv(reports_dir / "coverage_probe_rejection_report.csv", index=False)
    pd.DataFrame(underfilled_rows, columns=_underfilled_columns()).to_csv(reports_dir / "coverage_probe_underfilled_rules.csv", index=False)
    pd.DataFrame(ready_rows, columns=_coverage_columns()).to_csv(reports_dir / "coverage_probe_ready_rules.csv", index=False)


def _write_summary(reports_dir: Path, summary: Mapping[str, Any]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "coverage_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _coverage_columns() -> list[str]:
    return [
        "rule_id",
        "activation_stage",
        "production_ready",
        "atomic_positive_count",
        "hard_negative_count",
        "candidate_recall",
        "gold_count",
        "ready_for_full_build",
        "main_blocker",
        "top_rejection_reason",
        "recommended_next_action",
        "corpus_mined_positive_count",
        "syntax_mined_positive_count",
        "morphology_mined_positive_count",
        "real_pattern_replay_positive_count",
        "rule_lab_positive_count",
        "corpus_mined_share",
        "rule_lab_share",
        "corpus_mined_hard_negative_count",
        "rule_lab_hard_negative_count",
        "unique_source_count",
        "unique_target_count",
        "unique_left_context_count",
        "unique_right_context_count",
        "unique_sentence_pattern_count",
        "dominant_template_share",
        "dominant_source_type_share",
        "near_duplicate_count",
        "structural_diversity_status",
        "structural_diversity_reason",
    ]


def _underfilled_columns() -> list[str]:
    return [
        "rule_id",
        "activation_stage",
        "atomic_positive_count",
        "hard_negative_count",
        "candidate_recall",
        "min_atomic_required",
        "preferred_atomic",
        "min_hard_required",
        "corpus_mined_positive_count",
        "syntax_mined_positive_count",
        "morphology_mined_positive_count",
        "real_pattern_replay_positive_count",
        "rule_lab_positive_count",
        "rule_lab_share",
        "main_blocker",
        "top_rejection_reason",
        "recommended_next_action",
    ]


def _rejection_columns() -> list[str]:
    return ["rule_id", "miner_name", "reason", "stage", "count", "example_source", "example_target"]


def _raw_rejection_rows(rejections: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for rejection in rejections:
        key = (
            str(rejection.get("rule_id", "")),
            str(rejection.get("miner_name", "")),
            str(rejection.get("reason", "")),
            str(rejection.get("stage", "")),
        )
        if key not in grouped:
            grouped[key] = {
                "rule_id": key[0],
                "miner_name": key[1],
                "reason": key[2],
                "stage": key[3],
                "count": 0,
                "example_source": str(rejection.get("source", ""))[:500],
                "example_target": str(rejection.get("target", ""))[:500],
            }
        grouped[key]["count"] += 1
    return sorted(grouped.values(), key=lambda row: (row["rule_id"], row["miner_name"], row["reason"], row["stage"]))


def _top_rejection_by_rule(rejections: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    counters: dict[str, Counter[str]] = {}
    for rejection in rejections:
        rule_id = str(rejection.get("rule_id", ""))
        reason = str(rejection.get("reason", ""))
        if not rule_id or not reason:
            continue
        counters.setdefault(rule_id, Counter())[reason] += 1
    return {rule_id: counter.most_common(1)[0][0] for rule_id, counter in counters.items() if counter}


def _candidate_recall_values(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "rule_id" not in frame:
        return {}
    return {
        normalize_rule_id(str(row.get("rule_id", ""))): float(row.get("candidate_recall", 0.0) or 0.0)
        for row in frame.to_dict("records")
        if normalize_rule_id(str(row.get("rule_id", ""))) != UNKNOWN_RULE_ID
    }


def _candidate_gold_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "rule_id" not in frame:
        return {}
    return {
        normalize_rule_id(str(row.get("rule_id", ""))): int(float(row.get("gold_count", 0) or 0))
        for row in frame.to_dict("records")
        if normalize_rule_id(str(row.get("rule_id", ""))) != UNKNOWN_RULE_ID
    }


def _top_blockers(underfilled_rows: Iterable[Mapping[str, Any]], blockers: Iterable[str]) -> list[dict[str, Any]]:
    counter = Counter(str(row.get("main_blocker", "")) for row in underfilled_rows if str(row.get("main_blocker", "")))
    for blocker in blockers:
        counter[str(blocker).split(":", 1)[0]] += 1
    return [{"blocker": key, "count": int(value)} for key, value in counter.most_common(20)]


def _forbidden_artifact_snapshot(processed_dir: Path, *, config_path: Path | None) -> dict[str, tuple[bool, int, int]]:
    paths = [processed_dir / name for name in FORBIDDEN_ARTIFACT_NAMES]
    if config_path is not None and config_path.exists():
        try:
            config = load_config(config_path)
            configured = candidate_dataset_value(config, "paths.correction_dataset_path", "")
            if configured:
                paths.append(Path(str(configured)))
        except Exception:
            pass
    snapshot: dict[str, tuple[bool, int, int]] = {}
    for path in _dedupe_paths(paths):
        if path.exists():
            stat = path.stat()
            snapshot[str(path.resolve())] = (True, int(stat.st_size), int(stat.st_mtime_ns))
        else:
            snapshot[str(path.resolve())] = (False, 0, 0)
    return snapshot


def _forbidden_artifact_blockers(
    before: Mapping[str, tuple[bool, int, int]],
    after: Mapping[str, tuple[bool, int, int]],
) -> list[str]:
    blockers: list[str] = []
    for path, after_state in after.items():
        before_state = before.get(path, (False, 0, 0))
        if before_state != after_state:
            blockers.append(f"forbidden_artifact_modified:{path}")
    return blockers


def _count(rows: Iterable[Mapping[str, Any]], predicate: Any) -> int:
    return sum(1 for row in rows if predicate(row))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
