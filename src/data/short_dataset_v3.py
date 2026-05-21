from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.candidates.candidate_generator import CandidateGenerator
from src.data.short_dataset_v2 import build_short_dataset_v2_from_config
from src.evaluation.candidate_recall import build_candidate_recall_reports
from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage


V3_VERDICT_READY = "READY_FOR_SHORT_TRAINING_DATASET_V3"
V3_VERDICT_SMOKE_ONLY = "READY_FOR_SMOKE_ONLY"
V2_VERDICT_READY = "READY_FOR_SHORT_TRAINING_DATASET_V2"

SYNTHETIC_OPEN_CLEAN = "synthetic_augmented_from_open_clean"
REAL_ERROR_PAIR = "real_error_pair"
CLEAN_IDENTITY_OPEN = "clean_identity_from_open_clean"
HARD_NEGATIVE_OPEN = "hard_negative_from_open_clean"

WAVE1_RULE_IDS = frozenset(
    {
        "address_comma",
        "comma_conjunction",
        "comma_subordinate",
        "comparative_turnover_comma",
        "detached_adverbial_comma",
        "direct_speech_dash",
        "homogeneous_comma",
        "hyphen_koe_koy",
        "hyphen_particles",
        "introductory_comma",
        "subject_predicate_dash",
    }
)

RISKY_LEXICAL_RULE_IDS = frozenset(
    {
        "dictionary_fuzzy",
        "swapped_letters_candidate",
        "double_consonant_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
        "keyboard_typo_candidate",
    }
)

V3_ACTIVE_TARGET_COLUMNS = [
    "rule_id",
    "tier",
    "include_in_dataset_v3",
    "reason",
    "quota_min",
    "quota_preferred",
    "quota_final",
    "candidate_recall",
    "safety_status",
    "source",
]


def build_short_dataset_v3_from_config(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Build canonical short_dataset_v3 without promoting any 60k fallback to training-ready."""

    prepared = _as_v2_compatible_config(config)
    result = build_short_dataset_v2_from_config(prepared, force=force)
    return _finalize_v3_result(prepared, result)


def resolve_v3_active_target_rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve active, tiered v3 targets from stable v2 and verified Wave 1 artifacts."""

    v3_config = config.get("data", {}).get("short_dataset_v3", {}) or {}
    rows: dict[str, dict[str, Any]] = {}
    stable_v2 = _stable_v2_rules(v3_config)
    wave1_rows = _verified_wave1_rows(v3_config)

    for rule_id, recall in stable_v2.items():
        rows[rule_id] = _target_row(
            rule_id=rule_id,
            tier="stable_v2_core",
            include=True,
            reason="stable_v2_candidate_recall_ge_0_85",
            candidate_recall=recall,
            safety_status="stable_v2_ready",
            source="v2",
        )

    for wave_row in wave1_rows:
        rule_id = str(wave_row.get("rule_id") or "")
        if not rule_id:
            continue
        include = _is_wave1_include(wave_row)
        if include:
            rows[rule_id] = _target_row(
                rule_id=rule_id,
                tier="wave1",
                include=True,
                reason="verified_wave1_candidate_backed",
                candidate_recall=_float(wave_row.get("candidate_recall"), default=1.0),
                safety_status=str(wave_row.get("status") or "READY_NEXT_DATASET"),
                source=_merge_source(rows.get(rule_id, {}).get("source"), "wave1"),
            )
        elif rule_id not in rows:
            rows[rule_id] = _target_row(
                rule_id=rule_id,
                tier="excluded",
                include=False,
                reason=_exclusion_reason(wave_row),
                candidate_recall=_float(wave_row.get("candidate_recall"), default=0.0),
                safety_status=str(wave_row.get("status") or ""),
                source="wave1",
            )

    for rule_id in _configured_bounded_phase3_rules(v3_config):
        if rule_id in rows and rows[rule_id]["include_in_dataset_v3"]:
            continue
        rows[rule_id] = _target_row(
            rule_id=rule_id,
            tier="bounded_phase3",
            include=True,
            reason="configured_bounded_phase3_candidate_backed",
            candidate_recall=1.0,
            safety_status="configured_safe",
            source="matrix_phase3",
        )

    for row in _phase3_excluded_rows(v3_config):
        rule_id = str(row.get("rule_id") or "")
        if not rule_id or rule_id in rows:
            continue
        rows[rule_id] = _target_row(
            rule_id=rule_id,
            tier="excluded",
            include=False,
            reason=_phase3_exclusion_reason(row),
            candidate_recall=_float(row.get("candidate_recall"), default=0.0),
            safety_status=str(row.get("phase3_decision") or row.get("wave1_decision") or ""),
            source="matrix_phase3",
        )

    return [rows[rule_id] for rule_id in sorted(rows)]


def active_metric_summary(
    frame: pd.DataFrame,
    *,
    count_column: str,
    metric_column: str,
    active_rule_ids: Iterable[str],
) -> dict[str, Any]:
    active_set = {str(rule_id) for rule_id in active_rule_ids}
    if frame.empty or count_column not in frame or metric_column not in frame:
        return _empty_metric_summary()
    working = frame.copy()
    working[count_column] = pd.to_numeric(working[count_column], errors="coerce").fillna(0)
    working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce").fillna(0.0)
    non_unknown = working[(working["rule_id"] != "unknown") & (working[count_column] > 0)]
    active = non_unknown[non_unknown["rule_id"].astype(str).isin(active_set)]
    return {
        "rules_with_gold": int(len(non_unknown)),
        "active_rules_with_gold": int(len(active)),
        "min_excluding_unknown": _safe_min(non_unknown, metric_column),
        "mean_excluding_unknown": _safe_mean(non_unknown, metric_column),
        "active_min_excluding_unknown": _safe_min(active, metric_column),
        "active_mean_excluding_unknown": _safe_mean(active, metric_column),
    }


def underfilled_active_rule_ids(active_rows: list[dict[str, Any]], rule_counts: dict[str, Any]) -> list[str]:
    underfilled: list[str] = []
    for row in active_rows:
        if not bool(row.get("include_in_dataset_v3")):
            continue
        rule_id = str(row.get("rule_id") or "")
        quota_min = int(row.get("quota_min") or 0)
        if int(rule_counts.get(rule_id, 0) or 0) < quota_min:
            underfilled.append(rule_id)
    return sorted(underfilled)


def _as_v2_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    data = cloned.setdefault("data", {})
    v3 = copy.deepcopy(data.get("short_dataset_v3", {}) or {})
    active_rows = resolve_v3_active_target_rules(cloned)
    included_rows = [row for row in active_rows if row["include_in_dataset_v3"]]
    rule_quotas = {
        row["rule_id"]: {
            "min_total": int(row["quota_min"]),
            "preferred_total": int(row["quota_preferred"]),
            "max_total": int(row["quota_final"]),
        }
        for row in included_rows
    }
    v2 = copy.deepcopy(v3)
    v2["enabled"] = True
    v2.setdefault("expected_total", int(v3.get("expected_total") or data.get("target_total_examples") or 100000))
    quota = dict(v2.get("active_rule_quota", {}) or {})
    quota.update(
        {
            "enabled": True,
            "rule_ids": [row["rule_id"] for row in included_rows],
            "rule_quotas": rule_quotas,
            "min_total_per_active_rule": min((int(row["quota_min"]) for row in included_rows), default=1),
            "preferred_total_per_active_rule": max((int(row["quota_preferred"]) for row in included_rows), default=1),
            "split_minimums": {},
        }
    )
    v2["active_rule_quota"] = quota
    rule_caps = dict(v2.get("rule_caps", {}) or {})
    rule_caps["rule_max_totals"] = {rule_id: quota["max_total"] for rule_id, quota in rule_quotas.items()}
    rule_caps["max_rule_share_train"] = 0.10
    v2["rule_caps"] = rule_caps
    data["short_dataset_v2"] = v2
    data["config_path"] = data.get("config_path") or "configs/config.short_dataset_v3.yaml"
    return cloned


def _finalize_v3_result(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    manifest_path_value = result.get("manifest_path") or config.get("data", {}).get("manifest_path")
    if not manifest_path_value:
        return _upgrade_result(result, None)
    manifest_path = Path(str(manifest_path_value))
    if not manifest_path.exists():
        return _upgrade_result(result, None)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _upgrade_result(result, None)

    reports_dir = Path(config.get("paths", {}).get("reports_dir") or manifest_path.parent)
    reports_dir.mkdir(parents=True, exist_ok=True)
    active_rows = resolve_v3_active_target_rules(config)
    rule_counts = {str(key): int(value) for key, value in dict(manifest.get("rule_id_counts", {}) or {}).items()}
    _attach_final_counts(active_rows, rule_counts)
    _write_active_target_rules(active_rows, reports_dir / "v3_active_target_rules.csv")

    active_rule_ids = [row["rule_id"] for row in active_rows if row["include_in_dataset_v3"]]
    recall_reports = _active_target_recall_reports(config, active_rule_ids, reports_dir=reports_dir)
    candidate_report = recall_reports["candidate_recall_by_rule"]
    gap_report = recall_reports["gap_label_coverage_by_rule"]
    candidate_report.to_csv(reports_dir / "candidate_recall_by_rule.csv", index=False)
    gap_report.to_csv(reports_dir / "gap_label_coverage_by_rule.csv", index=False)
    candidate_summary = active_metric_summary(
        candidate_report,
        count_column="gold_count",
        metric_column="candidate_recall",
        active_rule_ids=active_rule_ids,
    )
    punctuation_active_rule_ids = set(active_rule_ids) & _punctuation_rule_ids()
    gap_summary = active_metric_summary(
        gap_report,
        count_column="gold_gap_count",
        metric_column="gap_candidate_recall",
        active_rule_ids=punctuation_active_rule_ids,
    )
    underfilled = underfilled_active_rule_ids(active_rows, rule_counts)
    _write_candidate_gate_report(
        active_rows,
        candidate_report,
        gap_report,
        reports_dir / "candidate_recall_gate_report.csv",
        config=config,
        gap_active_rule_ids=punctuation_active_rule_ids,
    )
    _write_under_quota_report(active_rows, rule_counts, reports_dir / "under_quota_v3_report.csv")
    _write_wave1_coverage_report(active_rows, rule_counts, reports_dir / "wave1_rule_coverage_report.csv")

    manifest = _upgrade_manifest_to_v3(
        config=config,
        result=result,
        manifest=manifest,
        active_rows=active_rows,
        candidate_summary=candidate_summary,
        gap_summary=gap_summary,
        underfilled=underfilled,
    )
    manifest["audit_errors"] = _dedupe_errors(_v3_audit_errors(manifest, config=config))
    is_smoke = _is_smoke_config(config)
    if manifest["audit_errors"]:
        manifest["verdict"] = "BLOCKED"
    elif is_smoke:
        manifest["verdict"] = V3_VERDICT_SMOKE_ONLY
    else:
        manifest["verdict"] = V3_VERDICT_READY
    manifest["final_verdict"] = manifest["verdict"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_generation_report(reports_dir / "dataset_generation_report.md", manifest)
    return _upgrade_result(result, manifest)


def _upgrade_manifest_to_v3(
    *,
    config: dict[str, Any],
    result: dict[str, Any],
    manifest: dict[str, Any],
    active_rows: list[dict[str, Any]],
    candidate_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    underfilled: list[str],
) -> dict[str, Any]:
    data = config.get("data", {})
    v3 = data.get("short_dataset_v3", {}) or {}
    included_rows = [row for row in active_rows if row["include_in_dataset_v3"]]
    excluded_rows = [row for row in active_rows if not row["include_in_dataset_v3"]]
    active_rule_ids = sorted(row["rule_id"] for row in included_rows)
    stable_v2_ids = sorted(row["rule_id"] for row in active_rows if "v2" in str(row.get("source", "")).split("/") and row["include_in_dataset_v3"])
    wave1_ids = sorted(row["rule_id"] for row in included_rows if row["tier"] == "wave1")
    bounded_ids = sorted(row["rule_id"] for row in included_rows if row["tier"] == "bounded_phase3")

    manifest.update(
        {
            "dataset_version": "short_dataset_v3",
            "requested_total": int(data.get("target_total_examples", v3.get("requested_total", manifest.get("requested_total", 0))) or 0),
            "actual_total": int(result.get("total", manifest.get("total", 0)) or 0),
            "requested_split_sizes": dict(data.get("exact_split_sizes", {}) or {}),
            "actual_split_sizes": manifest.get("split_sizes", result.get("splits", {})),
            "fallback_used": False,
            "fallback_reason": "",
            "smoke_mode": _is_smoke_config(config),
            "smoke_verdict": V3_VERDICT_SMOKE_ONLY,
            "source_constraints": _source_constraints(config, manifest),
            "candidate_recall_summary": candidate_summary,
            "gap_label_coverage_summary": gap_summary,
            "active_rule_ids": active_rule_ids,
            "active_target_rule_ids": active_rule_ids,
            "stable_v2_core_rule_ids": stable_v2_ids,
            "wave1_rule_ids": wave1_ids,
            "bounded_phase3_rule_ids": bounded_ids,
            "excluded_rule_ids": sorted(row["rule_id"] for row in excluded_rows),
            "excluded_active_rule_ids": sorted(row["rule_id"] for row in excluded_rows),
            "active_target_rule_count": len(active_rule_ids),
            "stable_v2_core_rule_count": len(stable_v2_ids),
            "wave1_rule_count": len(wave1_ids),
            "bounded_phase3_rule_count": len(bounded_ids),
            "excluded_rule_count": len(excluded_rows),
            "low_count_active_rule_ids": underfilled,
            "active_rule_quota_summary": {
                "active_rule_count": len(active_rule_ids),
                "stable_v2_core_rule_count": len(stable_v2_ids),
                "wave1_rule_count": len(wave1_ids),
                "bounded_phase3_rule_count": len(bounded_ids),
                "excluded_count": len(excluded_rows),
                "underfilled_count": len(underfilled),
                "strategy": "tiered_v3",
            },
        }
    )
    return manifest


def _v3_audit_errors(manifest: dict[str, Any], *, config: dict[str, Any]) -> list[str]:
    data = config.get("data", {})
    v3 = data.get("short_dataset_v3", {}) or {}
    audit = dict(v3.get("audit", {}) or {})
    errors = [
        str(error)
        for error in manifest.get("audit_errors", [])
        if not str(error).startswith(("active_rule_count_below_min:", "active_rule_quota_underfilled:"))
        and str(error) not in {"candidate_recall_active_min_below_threshold", "gap_coverage_active_min_below_threshold"}
    ]
    expected_total = int(data.get("target_total_examples", v3.get("requested_total", 100000)) or 100000)
    if int(manifest.get("total", 0)) != expected_total:
        errors.append(f"dataset_size_below_requested:{manifest.get('total', 0)}!={expected_total}")
    expected_splits = {split: int(count) for split, count in dict(data.get("exact_split_sizes", {}) or {}).items()}
    actual_splits = {split: int(count) for split, count in dict(manifest.get("split_sizes", {}) or {}).items()}
    for split, expected in expected_splits.items():
        if actual_splits.get(split, 0) != expected:
            errors.append(f"split_size_mismatch:{split}:{actual_splits.get(split, 0)}!={expected}")
    composition = dict(manifest.get("composition", {}) or {})
    synthetic_min = int(audit.get("synthetic_min", 70000))
    if int(composition.get(SYNTHETIC_OPEN_CLEAN, 0)) < synthetic_min:
        errors.append(f"synthetic_augmented_from_open_clean_shortage:{composition.get(SYNTHETIC_OPEN_CLEAN, 0)}<{synthetic_min}")
    if int(composition.get(REAL_ERROR_PAIR, 0)) <= 0:
        errors.append("missing_source_type:real_error_pair")
    for split, counts in dict(manifest.get("composition_by_split", {}) or {}).items():
        if int(counts.get(CLEAN_IDENTITY_OPEN, 0)) <= 0:
            errors.append(f"clean_identity_missing_in_split:{split}")
        if int(counts.get(HARD_NEGATIVE_OPEN, 0)) <= 0:
            errors.append(f"hard_negative_missing_in_split:{split}")
    recall = dict(manifest.get("candidate_recall_summary", {}) or {})
    if float(recall.get("active_min_excluding_unknown", 1.0)) < float(audit.get("candidate_recall_min", 0.85)):
        errors.append("candidate_recall_active_min_below_threshold")
    gap = dict(manifest.get("gap_label_coverage_summary", {}) or {})
    if float(gap.get("active_min_excluding_unknown", 1.0)) < float(audit.get("gap_coverage_min", 0.85)):
        errors.append("gap_coverage_active_min_below_threshold")
    underfilled = list(manifest.get("low_count_active_rule_ids", []) or [])
    if underfilled:
        errors.append("active_rule_quota_underfilled:" + ",".join(sorted(str(rule_id) for rule_id in underfilled)))
    leak = dict(manifest.get("template_leakage_summary", {}) or {})
    if float(leak.get("val_overlap_with_train_rate", 0.0)) > 0.03:
        errors.append("template_leakage_val_above_threshold")
    if float(leak.get("test_overlap_with_train_rate", 0.0)) > 0.03:
        errors.append("template_leakage_test_above_threshold")
    if float(manifest.get("synthetic_normalized_pair_duplicate_rate", 0.0)) > 0.25:
        errors.append("synthetic_normalized_duplicate_rate_above_threshold")
    if int(manifest.get("top_normalized_pair_count", 0)) > 20:
        errors.append("top_normalized_pair_count_above_threshold")
    for phrase, count in dict(manifest.get("meta_language_counts", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"meta_language_present:{phrase}")
    for phrase, count in dict(manifest.get("suspicious_template_counts", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"suspicious_template_present:{phrase}")
    errors.extend(_dominance_errors(manifest, config=config))
    return errors


def _dominance_errors(manifest: dict[str, Any], *, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rule_share_limit = float((config.get("data", {}).get("short_dataset_v3", {}) or {}).get("rule_caps", {}).get("max_rule_share_train", 0.10))
    error_share_limit = float((config.get("data", {}).get("short_dataset_v3", {}) or {}).get("rule_caps", {}).get("max_error_type_share_train", 0.35))
    train_size = int(dict(manifest.get("split_sizes", {}) or {}).get("train", 0))
    if train_size <= 0:
        return errors
    train_rule_counts = dict(dict(manifest.get("rule_id_counts_by_split", {}) or {}).get("train", {}) or {})
    for rule_id, count in train_rule_counts.items():
        if str(rule_id) in {"clean_identity", "clean_identity_hard_negative", "unknown", "unknown_real_validated"}:
            continue
        if int(count) > train_size * rule_share_limit:
            errors.append(f"rule_id_train_share_above_limit:{rule_id}:{count}/{train_size}")
    train_error_counts = dict(dict(manifest.get("error_type_counts_by_split", {}) or {}).get("train", {}) or {})
    for error_type, count in train_error_counts.items():
        if int(count) > train_size * error_share_limit:
            errors.append(f"error_type_train_share_above_limit:{error_type}:{count}/{train_size}")
    return errors


def _target_row(
    *,
    rule_id: str,
    tier: str,
    include: bool,
    reason: str,
    candidate_recall: float,
    safety_status: str,
    source: str,
) -> dict[str, Any]:
    quota_min, quota_preferred, quota_max = _quota_for_rule(rule_id, tier=tier, include=include)
    return {
        "rule_id": rule_id,
        "tier": tier,
        "include_in_dataset_v3": bool(include),
        "reason": reason,
        "quota_min": quota_min,
        "quota_preferred": quota_preferred,
        "quota_final": quota_max,
        "candidate_recall": float(candidate_recall),
        "safety_status": safety_status,
        "source": source,
    }


def _quota_for_rule(rule_id: str, *, tier: str, include: bool) -> tuple[int, int, int]:
    if not include or tier == "excluded":
        return 0, 0, 0
    if rule_id in RISKY_LEXICAL_RULE_IDS:
        return 200, 500, 1200
    if tier == "wave1":
        return 500, 1000, 2500
    if tier == "bounded_phase3":
        return 200, 500, 1500
    return 300, 700, 2500


def _stable_v2_rules(v3_config: dict[str, Any]) -> dict[str, float]:
    manifest_path = Path(str(v3_config.get("stable_v2_manifest_path") or "reports/short_dataset_v2/dataset_manifest.json"))
    recall_path = Path(str(v3_config.get("stable_v2_candidate_recall_path") or "reports/short_dataset_v2/candidate_recall_by_rule.csv"))
    manifest = _read_json(manifest_path)
    recall_by_rule = {
        row.get("rule_id", ""): _float(row.get("candidate_recall"), default=0.0)
        for row in _read_csv_dicts(recall_path)
    }
    result: dict[str, float] = {}
    for rule_id in manifest.get("active_rule_ids", []) or []:
        recall = recall_by_rule.get(str(rule_id), 0.0)
        if recall >= 0.85:
            result[str(rule_id)] = recall
    return result


def _verified_wave1_rows(v3_config: dict[str, Any]) -> list[dict[str, str]]:
    path = Path(str(v3_config.get("wave1_verified_path") or "reports/wave1_expansion/wave1_activation_plan_verified.csv"))
    return _read_csv_dicts(path)


def _configured_bounded_phase3_rules(v3_config: dict[str, Any]) -> list[str]:
    return sorted(str(rule_id) for rule_id in v3_config.get("bounded_phase3_rule_ids", []) or [])


def _phase3_excluded_rows(v3_config: dict[str, Any]) -> list[dict[str, str]]:
    path = Path(str(v3_config.get("phase3_activation_path") or "reports/matrix_eval_phase3/next_dataset_activation_plan_v3.csv"))
    return _read_csv_dicts(path)


def _is_wave1_include(row: dict[str, Any]) -> bool:
    return (
        str(row.get("activation_decision") or "").upper() == "INCLUDE"
        and str(row.get("status") or "").upper() == "READY_NEXT_DATASET"
        and str(row.get("candidate_path_exists") or "").lower() == "true"
    )


def _exclusion_reason(row: dict[str, Any]) -> str:
    reason = str(row.get("reason") or "").lower()
    status = str(row.get("status") or "").upper()
    if "under quota" in reason or "keep_under_quota" in reason:
        return "under_quota_nonblocking"
    if status == "NEEDS_CANDIDATE_GENERATOR" or str(row.get("candidate_path_exists") or "").lower() == "false":
        return "no_candidate_path"
    if status == "NEEDS_SYNTAX":
        return "needs_syntax"
    if status == "NEEDS_DICTIONARY":
        return "needs_dictionary"
    if status == "NEEDS_NER":
        return "needs_NER"
    if status == "NEEDS_VALIDATOR":
        return "needs_validator"
    if status == "UNSAFE":
        return "unsafe"
    return "metadata_only"


def _phase3_exclusion_reason(row: dict[str, Any]) -> str:
    decision = str(row.get("phase3_decision") or row.get("wave1_decision") or row.get("blockers") or "").upper()
    blockers = str(row.get("blockers") or row.get("under_quota_decision") or "").upper()
    combined = f"{decision},{blockers}"
    if "NO_CANDIDATE_PATH" in combined or "NEEDS_CANDIDATE_GENERATOR" in combined:
        return "no_candidate_path"
    if "NEEDS_SYNTAX" in combined:
        return "needs_syntax"
    if "NEEDS_DICTIONARY" in combined:
        return "needs_dictionary"
    if "NEEDS_NER" in combined:
        return "needs_NER"
    if "NEEDS_VALIDATOR" in combined:
        return "needs_validator"
    if "UNSAFE" in combined:
        return "unsafe"
    if "UNDER_QUOTA" in combined:
        return "under_quota_nonblocking"
    return "metadata_only"


def _attach_final_counts(active_rows: list[dict[str, Any]], rule_counts: dict[str, int]) -> None:
    for row in active_rows:
        row["quota_final"] = int(rule_counts.get(str(row.get("rule_id")), 0)) if row["include_in_dataset_v3"] else 0


def _write_active_target_rules(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=V3_ACTIVE_TARGET_COLUMNS).to_csv(path, index=False)


def _write_candidate_gate_report(
    active_rows: list[dict[str, Any]],
    candidate_report: pd.DataFrame,
    gap_report: pd.DataFrame,
    path: Path,
    *,
    config: dict[str, Any],
    gap_active_rule_ids: set[str],
) -> None:
    v3 = config.get("data", {}).get("short_dataset_v3", {}) or {}
    audit = dict(v3.get("audit", {}) or {})
    candidate_threshold = float(audit.get("candidate_recall_min", 0.85))
    gap_threshold = float(audit.get("gap_coverage_min", 0.85))
    by_rule = {row["rule_id"]: row for row in active_rows}
    rows: list[dict[str, Any]] = []
    for report, metric, count_column, threshold, gate, allowed_rules in (
        (candidate_report, "candidate_recall", "gold_count", candidate_threshold, "candidate_recall", None),
        (gap_report, "gap_candidate_recall", "gold_gap_count", gap_threshold, "gap_label_coverage", gap_active_rule_ids),
    ):
        if report.empty or metric not in report:
            continue
        for item in report.to_dict("records"):
            rule_id = str(item.get("rule_id") or "")
            target = by_rule.get(rule_id)
            if allowed_rules is not None and rule_id not in allowed_rules:
                continue
            if not target or not target["include_in_dataset_v3"] or int(float(item.get(count_column, 0) or 0)) <= 0:
                continue
            value = float(item.get(metric, 0.0) or 0.0)
            rows.append(
                {
                    "rule_id": rule_id,
                    "gate": gate,
                    "metric_value": value,
                    "threshold": threshold,
                    "passed": value >= threshold,
                    "tier": target["tier"],
                    "reason": target["reason"],
                }
            )
    pd.DataFrame(rows, columns=["rule_id", "gate", "metric_value", "threshold", "passed", "tier", "reason"]).to_csv(path, index=False)


def _write_under_quota_report(rows: list[dict[str, Any]], rule_counts: dict[str, int], path: Path) -> None:
    records = []
    for row in rows:
        if not row["include_in_dataset_v3"]:
            continue
        rule_id = row["rule_id"]
        actual = int(rule_counts.get(rule_id, 0))
        minimum = int(row["quota_min"])
        records.append(
            {
                "rule_id": rule_id,
                "tier": row["tier"],
                "quota_min": minimum,
                "quota_preferred": row["quota_preferred"],
                "actual_total": actual,
                "underfilled": actual < minimum,
                "reason": "" if actual >= minimum else "below_tier_minimum",
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)


def _write_wave1_coverage_report(rows: list[dict[str, Any]], rule_counts: dict[str, int], path: Path) -> None:
    records = []
    for row in rows:
        if row["tier"] != "wave1" and row["rule_id"] not in WAVE1_RULE_IDS:
            continue
        records.append(
            {
                "rule_id": row["rule_id"],
                "tier": row["tier"],
                "include_in_dataset_v3": row["include_in_dataset_v3"],
                "dataset_count": int(rule_counts.get(row["rule_id"], 0)),
                "candidate_recall": row["candidate_recall"],
                "reason": row["reason"],
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)


def _write_generation_report(path: Path, manifest: dict[str, Any]) -> None:
    composition = manifest.get("composition", {}) or {}
    recall = manifest.get("candidate_recall_summary", {}) or {}
    gap = manifest.get("gap_label_coverage_summary", {}) or {}
    lines = [
        "# Short Dataset V3 Generation Report",
        "",
        f"- verdict: {manifest.get('verdict')}",
        f"- total: {manifest.get('total')}",
        f"- requested_total: {manifest.get('requested_total')}",
        f"- split_sizes: {json.dumps(manifest.get('split_sizes', {}), ensure_ascii=False, sort_keys=True)}",
        f"- composition: {json.dumps(composition, ensure_ascii=False, sort_keys=True)}",
        f"- active_target_rule_count: {manifest.get('active_target_rule_count', 0)}",
        f"- stable_v2_core_rule_count: {manifest.get('stable_v2_core_rule_count', 0)}",
        f"- wave1_rule_count: {manifest.get('wave1_rule_count', 0)}",
        f"- bounded_phase3_rule_count: {manifest.get('bounded_phase3_rule_count', 0)}",
        f"- excluded_rule_count: {manifest.get('excluded_rule_count', 0)}",
        f"- synthetic_count: {composition.get(SYNTHETIC_OPEN_CLEAN, 0)}",
        f"- real_pair_count: {composition.get(REAL_ERROR_PAIR, 0)}",
        f"- clean_identity_count: {composition.get(CLEAN_IDENTITY_OPEN, 0)}",
        f"- hard_negative_count: {composition.get(HARD_NEGATIVE_OPEN, 0)}",
        f"- candidate_recall_min_mean: {float(recall.get('active_min_excluding_unknown', 1.0)):.6f} / {float(recall.get('active_mean_excluding_unknown', 1.0)):.6f}",
        f"- gap_coverage_min_mean: {float(gap.get('active_min_excluding_unknown', 1.0)):.6f} / {float(gap.get('active_mean_excluding_unknown', 1.0)):.6f}",
        f"- underfilled_active_rules: {', '.join(manifest.get('low_count_active_rule_ids', []))}",
        "",
        "## Audit Errors",
        "",
    ]
    audit_errors = list(manifest.get("audit_errors", []) or [])
    lines.extend(f"- {error}" for error in audit_errors) if audit_errors else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_constraints(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    v3 = config.get("data", {}).get("short_dataset_v3", {}) or {}
    return {
        "requested_total": v3.get("requested_total", config.get("data", {}).get("target_total_examples")),
        "smoke_total": (v3.get("smoke", {}) or {}).get("requested_total"),
        "real_pair_shortage_reason": manifest.get("real_pair_shortage_reason", ""),
        "missing_external_sources": manifest.get("missing_external_sources", []),
    }


def _punctuation_rule_ids() -> set[str]:
    result: set[str] = set()
    for domain, _group, entry in iter_coverage_entries(load_rules_coverage()):
        if domain == "punctuation":
            result.update(str(rule_id) for rule_id in entry.get("rules", []))
    return result


def _upgrade_result(result: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, Any]:
    upgraded = dict(result)
    if manifest:
        upgraded["total"] = int(manifest.get("total", upgraded.get("total", 0)) or 0)
        upgraded["composition"] = manifest.get("composition", upgraded.get("composition", {}))
        upgraded["splits"] = manifest.get("split_sizes", upgraded.get("splits", {}))
        upgraded["verdict"] = manifest.get("verdict", upgraded.get("verdict", "BLOCKED"))
        return upgraded
    if upgraded.get("verdict") == V2_VERDICT_READY:
        upgraded["verdict"] = "BLOCKED"
    return upgraded


def _is_smoke_config(config: dict[str, Any]) -> bool:
    smoke = ((config.get("data", {}).get("short_dataset_v3", {}) or {}).get("smoke", {}) or {})
    return bool(smoke.get("enabled", False))


def _read_report_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def _active_target_recall_reports(config: dict[str, Any], active_rule_ids: list[str], *, reports_dir: Path) -> dict[str, pd.DataFrame]:
    output_path = Path(str(config.get("data", {}).get("processed_train_path") or ""))
    if not output_path.exists():
        return {
            "candidate_recall_by_rule": _read_report_frame(reports_dir / "candidate_recall_by_rule.csv"),
            "gap_label_coverage_by_rule": _read_report_frame(reports_dir / "gap_label_coverage_by_rule.csv"),
        }
    active_set = set(active_rule_ids)
    frame = pd.read_csv(output_path).fillna("")
    rows = [
        row
        for row in frame.to_dict("records")
        if row.get("source_type") == SYNTHETIC_OPEN_CLEAN and _row_targets_active_rule(row, active_set)
    ]
    return build_candidate_recall_reports(
        rows,
        candidate_generator=CandidateGenerator.from_config(config),
        max_candidates=int(config.get("model", {}).get("max_candidates", 16)),
        rules_config_path="configs/rules.yaml",
    )


def _row_targets_active_rule(row: dict[str, Any], active_rule_ids: set[str]) -> bool:
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        return False
    target_family = str(metadata.get("target_family") or "")
    return bool(metadata.get("candidate_present")) and target_family in active_rule_ids


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _empty_metric_summary() -> dict[str, Any]:
    return {
        "rules_with_gold": 0,
        "active_rules_with_gold": 0,
        "min_excluding_unknown": 1.0,
        "mean_excluding_unknown": 1.0,
        "active_min_excluding_unknown": 1.0,
        "active_mean_excluding_unknown": 1.0,
    }


def _safe_min(frame: pd.DataFrame, column: str) -> float:
    return 1.0 if frame.empty else float(frame[column].min())


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    return 1.0 if frame.empty else float(frame[column].mean())


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _merge_source(existing: Any, addition: str) -> str:
    parts = [part for part in str(existing or "").split("/") if part]
    if addition not in parts:
        parts.append(addition)
    return "/".join(parts)


def _dedupe_errors(errors: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for error in errors:
        if error not in seen:
            seen.add(error)
            result.append(error)
    return result
