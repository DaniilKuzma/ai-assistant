from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.candidates.candidate_generator import CandidateGenerator
from src.data._training_dataset_builder import build_training_dataset_core_from_config
from src.data.training_quality_audit import (
    audit_training_dataset,
    write_artificial_marker_reports,
    write_clean_hard_balance_report,
    write_extended_quality_reports,
    write_generation_strategy_report,
    write_known_quality_bugs_report,
    write_quote_bracket_balance_reports,
    write_rule_semantic_alignment_report,
    write_rule_diversity_report,
)
from src.evaluation.candidate_recall import build_candidate_recall_reports
from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.rules.registry import rule_by_id
from src.rules.syntax_synthetic import (
    SUPPORTED_ORTHOGRAPHY_RULE_IDS,
    SUPPORTED_PUNCTUATION_RULE_IDS,
    SUPPORTED_SYNTAX_RULE_IDS,
)


VERDICT_READY = "READY_FOR_TRAINING_DATASET"
VERDICT_SMOKE_ONLY = "READY_FOR_SMOKE_ONLY"
VERDICT_PARTIAL = "DATASET_PARTIAL"
VERDICT_BLOCKED = "DATASET_BLOCKED"
CORE_VERDICT_READY = "READY_FOR_TRAINING_DATASET"

SYNTHETIC_OPEN_CLEAN = "synthetic_augmented_from_open_clean"
REAL_ERROR_PAIR = "real_error_pair"
CLEAN_IDENTITY_OPEN = "clean_identity_from_open_clean"
HARD_NEGATIVE_OPEN = "hard_negative_from_open_clean"

ACTIVATION_RULE_IDS = frozenset(
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

HIGH_FREQUENCY_LEGACY_RULE_IDS = frozenset(
    {
        "final_punctuation_default",
        "capitalization_sentence_start",
        "frequent_error_exact",
        "hyphen_particles",
        "hyphen_koe_koy",
        "hyphen_po_adverbs",
        "hyphen_whitelist",
    }
)

LEGACY_CANDIDATE_BACKED_RULE_IDS = frozenset(
    {
        "frequent_error_exact",
        "dictionary_fuzzy",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "swapped_letters_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
        "hyphen_particles",
        "hyphen_koe_koy",
        "hyphen_po_adverbs",
        "hyphen_whitelist",
        "pol_polu_compounds",
        "prefix_pre_pri",
        "prefix_s_to_z",
        "prefix_z_to_s",
        "missing_hard_sign",
        "soft_to_hard_sign",
        "pattern_жы_жи",
        "pattern_шы_ши",
        "pattern_чя_ча",
        "pattern_щя_ща",
        "pattern_чю_чу",
        "pattern_щю_щу",
        "pattern_цы_ци",
        "pattern_жо_же",
        "pattern_шо_ше",
        "pattern_чо_че",
        "pattern_що_ще",
        "final_punctuation_default",
        "capitalization_sentence_start",
        "abbreviation_case_protection",
        "sdelat_prefix",
        "cy_exception",
    }
)

BROAD_EXCLUDED_RULE_IDS = frozenset(
    {
        "metadata_only",
        "planned",
        "quote_open",
        "quote_close",
        "capitalization_ner",
        "neural_punctuation",
    }
)

TRAINING_INCLUDE_DECISIONS = frozenset(
    {
        "INCLUDE_NOW",
        "INCLUDE_AFTER_VALIDATOR",
        "INCLUDE_AFTER_THRESHOLD_CALIBRATION",
        "INCLUDE_AFTER_TRAINING",
    }
)

BROAD_ACTIVE_RULE_COLUMNS = [
    "rule_id",
    "source",
    "include",
    "reason",
    "risk_level",
    "needs_validator",
    "needs_threshold_calibration",
    "candidate_path_exists",
    "synthetic_support_exists",
    "hard_negative_support_exists",
    "target_min_examples",
    "target_preferred_examples",
    "final_count",
]

ACTIVE_TARGET_COLUMNS = [
    "rule_id",
    "tier",
    "include_in_dataset",
    "reason",
    "quota_min",
    "quota_preferred",
    "quota_final",
    "candidate_recall",
    "safety_status",
    "source",
]


def build_training_dataset_from_config(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Build canonical training_dataset without promoting any 60k fallback to training-ready."""

    prepared = _as_core_compatible_config(config)
    result = build_training_dataset_core_from_config(prepared, force=force)
    return _finalize_canonical_result(prepared, result)


def resolve_broad_active_training_rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve the canonical broad active rule set from rules.yaml plus syntax support."""

    rules_by_id = _rules_yaml_training_entries()
    candidates = set(rules_by_id) | set(SUPPORTED_SYNTAX_RULE_IDS) | set(LEGACY_CANDIDATE_BACKED_RULE_IDS)
    if not bool(config.get("dictionary", {}).get("yo_e", {}).get("enabled", False)):
        candidates.add("yo_e_candidate")
    candidates.update({"quote_open", "quote_close", "capitalization_ner", "neural_punctuation"})

    rows: list[dict[str, Any]] = []
    for rule_id in sorted(candidates):
        entry = rules_by_id.get(rule_id, {})
        dataset = dict(entry.get("dataset", {}) or {})
        include, reason = _broad_include_decision(rule_id, dataset, config=config)
        source = _broad_rule_source(rule_id, dataset)
        quota_min, quota_preferred = _broad_rule_quota(rule_id, include=include)
        candidate_path = include or bool(dataset.get("current_candidate_path")) or rule_id in SUPPORTED_SYNTAX_RULE_IDS or rule_id in LEGACY_CANDIDATE_BACKED_RULE_IDS
        synthetic_support = include or bool(dataset.get("current_synthetic_support")) or rule_id in SUPPORTED_SYNTAX_RULE_IDS
        hard_negative_support = include or bool(dataset.get("current_hard_negative_support")) or rule_id in SUPPORTED_SYNTAX_RULE_IDS
        decision = str(dataset.get("training_eligibility_decision") or "")
        rows.append(
            {
                "rule_id": rule_id,
                "source": source,
                "include": bool(include),
                "include_in_dataset": bool(include),
                "reason": reason,
                "risk_level": str(dataset.get("risk_level") or _broad_default_risk(rule_id)),
                "needs_validator": decision == "INCLUDE_AFTER_VALIDATOR",
                "needs_threshold_calibration": decision == "INCLUDE_AFTER_THRESHOLD_CALIBRATION",
                "candidate_path_exists": bool(candidate_path),
                "synthetic_support_exists": bool(synthetic_support),
                "hard_negative_support_exists": bool(hard_negative_support),
                "target_min_examples": quota_min,
                "target_preferred_examples": quota_preferred,
                "final_count": 0,
                "tier": source,
                "quota_min": quota_min,
                "quota_preferred": quota_preferred,
                "quota_final": quota_preferred,
                "candidate_recall": float(dataset.get("current_candidate_recall") or (1.0 if include else 0.0)),
                "safety_status": "included" if include else "excluded",
            }
        )
    return rows


def compute_broad_dataset_targets(active_rows: list[dict[str, Any]], *, real_pair_count: int) -> dict[str, Any]:
    included = [row for row in active_rows if bool(row.get("include") or row.get("include_in_dataset"))]
    targeted_synthetic = sum(int(row.get("target_preferred_examples", row.get("quota_preferred", 0)) or 0) for row in included)
    preliminary = targeted_synthetic + max(0, int(real_pair_count))
    clean_identity = max(25_000, int(round(preliminary * 0.10)))
    hard_negative = max(25_000, int(round(preliminary * 0.10)))
    stress = max(10_000, int(round(preliminary * 0.04)))
    synthetic_padding = 0
    hard_negative_padding = 0
    total = targeted_synthetic + clean_identity + hard_negative + stress + max(0, int(real_pair_count))
    if total < 200_000 and included:
        deficit = 200_000 - total
        synthetic_padding = int(round(deficit * 0.70))
        hard_negative_padding = deficit - synthetic_padding
        total += deficit
        hard_negative += hard_negative_padding
    base_without_clean_hard = targeted_synthetic + synthetic_padding + stress + max(0, int(real_pair_count))
    ten_percent_floor = int((base_without_clean_hard + 7) // 8 + 1)
    clean_identity = max(clean_identity, ten_percent_floor)
    hard_negative = max(hard_negative, ten_percent_floor)
    total = targeted_synthetic + synthetic_padding + clean_identity + hard_negative + stress + max(0, int(real_pair_count))
    split_sizes = _exact_80_10_10(total)
    synthetic_total = targeted_synthetic + stress + synthetic_padding
    return {
        "active_rule_count": len(included),
        "targeted_synthetic_target": int(targeted_synthetic + synthetic_padding),
        "targeted_synthetic_base": int(targeted_synthetic),
        "synthetic_padding": int(synthetic_padding),
        "clean_identity_target": int(clean_identity),
        "hard_negative_target": int(hard_negative),
        "hard_negative_padding": int(hard_negative_padding),
        "multi_error_stress_target": int(stress),
        "real_pair_target": int(max(0, real_pair_count)),
        "total_target": int(total),
        "split_sizes": split_sizes,
        "source_type_targets": {
            SYNTHETIC_OPEN_CLEAN: int(synthetic_total),
            REAL_ERROR_PAIR: int(max(0, real_pair_count)),
            CLEAN_IDENTITY_OPEN: int(clean_identity),
            HARD_NEGATIVE_OPEN: int(hard_negative),
        },
        "split_source_type_targets": _split_source_targets(
            {
                SYNTHETIC_OPEN_CLEAN: int(synthetic_total),
                REAL_ERROR_PAIR: int(max(0, real_pair_count)),
                CLEAN_IDENTITY_OPEN: int(clean_identity),
                HARD_NEGATIVE_OPEN: int(hard_negative),
            },
            split_sizes,
        ),
    }


def training_dataset_quality_errors(manifest: dict[str, Any], config: dict[str, Any] | None = None) -> list[str]:
    total = int(manifest.get("total", 0) or 0)
    errors: list[str] = []
    if total < 200_000:
        errors.append("total_below_200000")
        errors.append(f"total_below_200000:{total}")
    splits = {split: int(count) for split, count in dict(manifest.get("split_sizes", {}) or {}).items()}
    if total and splits != _exact_80_10_10(total):
        errors.append(f"split_sizes_not_exact_80_10_10:{splits}")
    composition = _composition_with_aliases(dict(manifest.get("composition", {}) or {}))
    if total and int(composition.get("clean_identity", 0)) < total * 0.10:
        errors.append("clean_identity_below_10_percent")
    if total and int(composition.get("hard_negative", 0)) < total * 0.10:
        errors.append("hard_negative_below_10_percent")
    if int(composition.get("real_error_pair", 0)) <= 0:
        errors.append("missing_real_pairs")
        errors.append("real_pairs_missing")
    stress_count = int(manifest.get("stress_count", composition.get("multi_error_stress", 0)) or 0)
    if total and not (total * 0.03 <= stress_count <= total * 0.05):
        errors.append("stress_ratio_outside_3_5_percent")
    source_counts = dict(manifest.get("source_counts", manifest.get("clean_source_counts", {})) or {})
    for source_name, count in source_counts.items():
        if total and int(count) > total * 0.70:
            errors.append(f"source_dominance_above_70_percent:{source_name}")
    if float((manifest.get("candidate_recall_summary") or {}).get("active_min_excluding_unknown", 1.0)) < 0.85:
        errors.append("candidate_recall_active_min_below_threshold")
    if float((manifest.get("gap_label_coverage_summary") or {}).get("active_min_excluding_unknown", 1.0)) < 0.85:
        errors.append("gap_coverage_active_min_below_threshold")
    rule_counts = {str(key): int(value) for key, value in dict(manifest.get("rule_id_counts", {}) or {}).items()}
    for rule_id in manifest.get("active_rule_ids", []) or []:
        if int(rule_counts.get(str(rule_id), 0)) < 1000:
            errors.append(f"active_rule_under_min:{rule_id}")
    if float(manifest.get("synthetic_normalized_pair_duplicate_rate", 0.0) or 0.0) > 0.25:
        errors.append("synthetic_normalized_duplicate_rate_above_threshold")
    if int(manifest.get("top_normalized_pair_count", 0) or 0) > 20:
        errors.append("top_normalized_pair_count_above_threshold")
    for phrase, count in dict(manifest.get("meta_language_counts", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"meta_language_present:{phrase}")
    for phrase, count in dict(manifest.get("suspicious_template_counts", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"suspicious_template_present:{phrase}")
    if int(manifest.get("hard_negative_accepted_bad_edits", 0) or 0) != 0:
        errors.append("hard_negative_accepted_bad_edits_nonzero")
    if float(manifest.get("corpus_opportunity_share", 1.0) or 0.0) < 0.70:
        errors.append("corpus_opportunity_share_below_threshold")
    if float(manifest.get("fallback_template_share", 0.0) or 0.0) > 0.20:
        errors.append("fallback_template_share_above_threshold")
    for name, count in dict(manifest.get("known_quality_bugs", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"known_quality_bugs_present:{name}")
    for name, count in dict(manifest.get("artificial_marker_counts", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"artificial_marker_present:{name}")
    for name, count in dict(manifest.get("quote_bracket_balance_bugs", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"quote_bracket_balance_present:{name}")
    for name, count in dict(manifest.get("clean_hard_balance_bugs", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"clean_hard_balance_present:{name}")
    if int(dict(manifest.get("rule_semantic_alignment", {}) or {}).get("failed_rows", 0) or 0) != 0:
        errors.append("rule_semantic_alignment_failed")
    if not dict(manifest.get("error_bearing_sentence_source_counts", {}) or {}):
        errors.append("missing_error_bearing_sentence_source_counts")
    if manifest.get("underfilled_rule_ids") or manifest.get("low_count_active_rule_ids"):
        errors.append("active_rule_quota_underfilled")
    if int(dict(manifest.get("rule_diversity_summary", {}) or {}).get("failed_rule_count", 0) or 0) != 0:
        errors.append("rule_diversity_gates_failed")
    if int(dict(manifest.get("extended_quality_audit_summary", {}) or {}).get("blocking_issue_count", 0) or 0) != 0:
        errors.append("extended_quality_audit_blocking_issues")
    source_counts = {str(key): int(value) for key, value in source_counts.items()}
    for source_name, count in source_counts.items():
        if total and count > total * 0.70:
            errors.append(f"unsafe_source_dominance:{source_name}")
    return _dedupe_errors(errors)


def resolve_active_target_rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve active canonical targets for the broad training dataset."""

    return resolve_broad_active_training_rules(config)


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
        if not bool(row.get("include_in_dataset")):
            continue
        rule_id = str(row.get("rule_id") or "")
        quota_min = int(row.get("quota_min") or 0)
        if int(rule_counts.get(rule_id, 0) or 0) < quota_min:
            underfilled.append(rule_id)
    return sorted(underfilled)


def _rules_yaml_training_entries() -> dict[str, dict[str, Any]]:
    coverage = load_rules_coverage()
    result: dict[str, dict[str, Any]] = {}
    for domain, group, entry in iter_coverage_entries(coverage):
        dataset = dict(entry.get("dataset", {}) or {})
        decision = str(dataset.get("training_eligibility_decision") or "")
        eligible = bool(dataset.get("training_eligible_now")) or decision in TRAINING_INCLUDE_DECISIONS
        for raw_rule_id in entry.get("rules", []) or []:
            rule_id = str(raw_rule_id)
            if eligible or rule_id in SUPPORTED_SYNTAX_RULE_IDS or rule_id in LEGACY_CANDIDATE_BACKED_RULE_IDS:
                merged = dict(entry)
                merged["domain"] = domain
                merged["group"] = group
                result[rule_id] = merged
    return result


def _broad_include_decision(rule_id: str, dataset: dict[str, Any], *, config: dict[str, Any]) -> tuple[bool, str]:
    if rule_id in {"quote_open", "quote_close"}:
        return False, "broad_normalization_quote_id_excluded"
    if rule_id == "capitalization_ner":
        return False, "needs_NER:BLOCK_NEEDS_NER:no_safe_current_candidate_path"
    if rule_id == "neural_punctuation":
        return False, "broad_neural_punctuation_excluded"
    if rule_id == "yo_e_candidate" and not bool(config.get("dictionary", {}).get("yo_e", {}).get("enabled", False)):
        return False, "yo_e_disabled_in_dictionary_config"
    if rule_id in BROAD_EXCLUDED_RULE_IDS:
        return False, "excluded_by_broad_dataset_policy"
    if rule_by_id(rule_id) is None:
        return False, "no_registered_rule"
    decision = str(dataset.get("training_eligibility_decision") or "")
    eligible = bool(dataset.get("training_eligible_now")) or decision in TRAINING_INCLUDE_DECISIONS
    if rule_id in SUPPORTED_SYNTAX_RULE_IDS:
        return True, "syntax_supported_candidate_backed"
    if rule_id in LEGACY_CANDIDATE_BACKED_RULE_IDS:
        return True, "legacy_candidate_backed_current_capability"
    if eligible and bool(dataset.get("current_candidate_path", True)):
        return True, "rules_yaml_training_eligible_candidate_backed"
    if eligible:
        return False, "training_eligible_but_no_candidate_path"
    blocker = decision or str(dataset.get("training_eligibility_reason") or "metadata_or_planned")
    return False, blocker


def _broad_rule_source(rule_id: str, dataset: dict[str, Any]) -> str:
    if rule_id in SUPPORTED_SYNTAX_RULE_IDS:
        return "syntax_supported"
    if rule_id in LEGACY_CANDIDATE_BACKED_RULE_IDS:
        return "legacy_stable"
    if dataset:
        return "current_capability"
    return "excluded"


def _broad_rule_quota(rule_id: str, *, include: bool) -> tuple[int, int]:
    if not include:
        return 0, 0
    if rule_id in SUPPORTED_PUNCTUATION_RULE_IDS or rule_id in SUPPORTED_ORTHOGRAPHY_RULE_IDS:
        return 1500, 2500
    if rule_id == "abbreviation_case_protection":
        return 1000, 1500
    if rule_id in RISKY_LEXICAL_RULE_IDS:
        return 1000, 1800
    if rule_id in HIGH_FREQUENCY_LEGACY_RULE_IDS:
        return 1000, 3000
    return 1000, 2500


def _broad_default_risk(rule_id: str) -> str:
    if rule_id in RISKY_LEXICAL_RULE_IDS or rule_id == "abbreviation_case_protection":
        return "medium"
    if rule_id in {"quote_pair_balance", "bracket_pair_balance", "punctuation_delete_replace"}:
        return "medium"
    return "low"


def _exact_80_10_10(total: int) -> dict[str, int]:
    train = int(total * 0.8)
    val = int(total * 0.1)
    return {"train": train, "val": val, "test": int(total - train - val)}


def _split_source_targets(source_targets: dict[str, int], split_sizes: dict[str, int]) -> dict[str, dict[str, int]]:
    result = {split: {source_type: 0 for source_type in source_targets} for split in ("train", "val", "test")}
    total = max(1, sum(split_sizes.values()))
    for source_type, count in source_targets.items():
        train = int(count * split_sizes["train"] / total)
        val = int(count * split_sizes["val"] / total)
        test = int(count) - train - val
        result["train"][source_type] = train
        result["val"][source_type] = val
        result["test"][source_type] = test
    for split in ("train", "val", "test"):
        delta = split_sizes[split] - sum(result[split].values())
        result[split][SYNTHETIC_OPEN_CLEAN] = result[split].get(SYNTHETIC_OPEN_CLEAN, 0) + delta
    return result


def _composition_with_aliases(composition: dict[str, Any]) -> dict[str, int]:
    result = {str(key): int(value) for key, value in composition.items()}
    result.setdefault("clean_identity", int(result.get(CLEAN_IDENTITY_OPEN, 0)))
    result.setdefault("hard_negative", int(result.get(HARD_NEGATIVE_OPEN, 0)))
    result.setdefault("real_error_pair", int(result.get(REAL_ERROR_PAIR, 0)))
    result.setdefault("multi_error_stress", int(result.get("multi_error_stress", 0)))
    return result


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(sum(len(chunk) for chunk in pd.read_csv(path, chunksize=50_000)))


def _as_core_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    data = cloned.setdefault("data", {})
    canonical = copy.deepcopy(data.get("training_dataset", {}) or {})
    active_rows = resolve_active_target_rules(cloned)
    included_rows = [row for row in active_rows if row["include_in_dataset"]]
    real_pair_count = _count_csv_rows(Path(str(data.get("real_error_pairs_validated_path") or "data/processed/real_error_pairs_validated.csv.gz")))
    targets = compute_broad_dataset_targets(active_rows, real_pair_count=real_pair_count)
    configured_targets = dict(canonical.get("source_type_targets", {}) or {})
    configured_total = int(data.get("target_total_examples") or data.get("total_examples") or 0)
    if configured_targets and configured_total >= 200_000 and sum(int(value) for value in configured_targets.values()) == configured_total:
        configured_splits = dict(data.get("exact_split_sizes", {}) or {})
        if sum(int(value) for value in configured_splits.values()) != configured_total:
            configured_splits = _exact_80_10_10(configured_total)
        configured_split_targets = copy.deepcopy(canonical.get("split_source_type_targets", {}) or {})
        if set(configured_split_targets) != {"train", "val", "test"}:
            configured_split_targets = _split_source_targets(configured_targets, configured_splits)
        targets.update(
            {
                "total_target": configured_total,
                "split_sizes": {split: int(count) for split, count in configured_splits.items()},
                "source_type_targets": {source_type: int(count) for source_type, count in configured_targets.items()},
                "split_source_type_targets": configured_split_targets,
                "multi_error_stress_target": int(canonical.get("multi_error_stress_target", targets["multi_error_stress_target"]) or 0),
            }
        )
    split_sizes = targets["split_sizes"]
    data["target_total_examples"] = targets["total_target"]
    data["total_examples"] = targets["total_target"]
    data["train_examples"] = split_sizes["train"]
    data["val_examples"] = split_sizes["val"]
    data["test_examples"] = split_sizes["test"]
    data["exact_split_sizes"] = dict(split_sizes)
    rule_quotas = {
        row["rule_id"]: {
            "min_total": int(row["target_min_examples"]),
            "preferred_total": int(row["target_preferred_examples"]),
            "max_total": max(int(row["target_preferred_examples"]) + 750, int(row["target_min_examples"])),
        }
        for row in included_rows
    }
    core = copy.deepcopy(canonical)
    core["enabled"] = True
    core["requested_total"] = targets["total_target"]
    core["expected_total"] = targets["total_target"]
    core["source_type_targets"] = dict(targets["source_type_targets"])
    core["split_source_type_targets"] = copy.deepcopy(targets["split_source_type_targets"])
    core["multi_error_stress_target"] = int(targets["multi_error_stress_target"])
    core["min_cached_real_pairs"] = 5000
    core["preferred_cached_real_pairs"] = 30000
    core["reuse_clean_sentence_pool_cache"] = True
    core["min_clean_pool_for_ready"] = 300000
    core["min_clean_pool_hard_min"] = 150000
    quota = dict(core.get("active_rule_quota", {}) or {})
    quota.update(
        {
            "enabled": True,
            "rule_ids": [row["rule_id"] for row in included_rows],
            "rule_quotas": rule_quotas,
            "min_total_per_active_rule": min((int(row["target_min_examples"]) for row in included_rows), default=1),
            "preferred_total_per_active_rule": max((int(row["target_preferred_examples"]) for row in included_rows), default=1),
            "split_minimums": {},
        }
    )
    core["active_rule_quota"] = quota
    rule_caps = dict(core.get("rule_caps", {}) or {})
    rule_caps["rule_max_totals"] = {rule_id: quota["max_total"] for rule_id, quota in rule_quotas.items()}
    rule_caps["max_total_per_rule_id"] = max((quota["max_total"] for quota in rule_quotas.values()), default=2500)
    rule_caps["max_train_per_rule_id"] = max((quota["max_total"] for quota in rule_quotas.values()), default=2500)
    rule_caps["max_rule_share_train"] = 0.08
    core["rule_caps"] = rule_caps
    audit = dict(core.get("audit", {}) or {})
    audit.update(
        {
            "candidate_recall_min": 0.85,
            "gap_coverage_min": 0.85,
            "synthetic_min": int(targets["targeted_synthetic_base"]),
            "min_active_rule_count": 1000,
            "min_clean_identity_eval_split": 500,
            "min_hard_negative_eval_split": 500,
            "require_all_source_types": True,
        }
    )
    core["audit"] = audit
    data["training_dataset_core"] = core
    data["config_path"] = data.get("config_path") or "configs/config.yaml"
    training = cloned.setdefault("training", {})
    training["max_train_examples"] = split_sizes["train"]
    training["max_val_examples"] = split_sizes["val"]
    training["max_test_examples"] = split_sizes["test"]
    if str(data.get("processed_train_path")) == "data/processed/correction_dataset.csv.gz":
        cloned.setdefault("paths", {})["reports_dir"] = "reports/dataset_build"
    return cloned


def _finalize_canonical_result(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
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
    if manifest.get("operator_based_generation") is True:
        manifest["final_verdict"] = manifest.get("verdict", result.get("verdict", "DATASET_BLOCKED"))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return _upgrade_result(result, manifest)

    reports_dir = Path(config.get("paths", {}).get("reports_dir") or manifest_path.parent)
    reports_dir.mkdir(parents=True, exist_ok=True)
    active_rows = resolve_active_target_rules(config)
    rule_counts = {str(key): int(value) for key, value in dict(manifest.get("rule_id_counts", {}) or {}).items()}
    _attach_final_counts(active_rows, rule_counts)

    exclusion_reasons = _manifest_active_exclusion_reasons(manifest)
    exclusion_reasons.update(
        {
            rule_id: "excluded_after_quota_underfilled"
            for rule_id in underfilled_active_rule_ids(active_rows, rule_counts)
        }
    )
    _apply_active_rule_exclusions(active_rows, exclusion_reasons)
    _attach_final_counts(active_rows, rule_counts)

    quality_audit = _refresh_quality_audit_reports(
        result=result,
        active_rows=active_rows,
        reports_dir=reports_dir,
    )
    if quality_audit:
        failed_diversity = {
            str(rule_id): "excluded_after_rule_diversity_gate_failed"
            for rule_id in dict(quality_audit.get("rule_diversity_summary", {}) or {}).get("failed_rule_ids", [])
        }
        if failed_diversity:
            _apply_active_rule_exclusions(active_rows, failed_diversity)
            _attach_final_counts(active_rows, rule_counts)
            quality_audit = _refresh_quality_audit_reports(
                result=result,
                active_rows=active_rows,
                reports_dir=reports_dir,
            )
        _apply_quality_audit_to_manifest(manifest, quality_audit)

    underfilled = underfilled_active_rule_ids(active_rows, rule_counts)
    _write_active_target_rules(active_rows, reports_dir / "canonical_active_target_rules.csv")
    _write_broad_active_training_rules(active_rows, reports_dir / "active_training_rules.csv")

    active_rule_ids = [row["rule_id"] for row in active_rows if row["include_in_dataset"]]
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
    _write_candidate_gate_report(
        active_rows,
        candidate_report,
        gap_report,
        reports_dir / "candidate_recall_gate_report.csv",
        config=config,
        gap_active_rule_ids=punctuation_active_rule_ids,
    )
    _write_under_quota_report(active_rows, rule_counts, reports_dir / "under_quota_canonical_report.csv")
    _write_activation_coverage_report(active_rows, rule_counts, reports_dir / "activation_rule_coverage_report.csv")

    manifest = _upgrade_manifest_to_canonical(
        config=config,
        result=result,
        manifest=manifest,
        active_rows=active_rows,
        candidate_summary=candidate_summary,
        gap_summary=gap_summary,
        underfilled=underfilled,
    )
    manifest["audit_errors"] = _dedupe_errors(_canonical_audit_errors(manifest, config=config))
    is_smoke = _is_smoke_config(config)
    if manifest["audit_errors"]:
        manifest["verdict"] = VERDICT_BLOCKED
    elif is_smoke:
        manifest["verdict"] = VERDICT_SMOKE_ONLY
    else:
        manifest["verdict"] = VERDICT_READY
    manifest["final_verdict"] = manifest["verdict"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_generation_report(reports_dir / "dataset_generation_report.md", manifest)
    return _upgrade_result(result, manifest)


def _upgrade_manifest_to_canonical(
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
    canonical = data.get("training_dataset", {}) or {}
    included_rows = [row for row in active_rows if row["include_in_dataset"]]
    excluded_rows = [row for row in active_rows if not row["include_in_dataset"]]
    active_rule_ids = sorted(row["rule_id"] for row in included_rows)
    stable_core_ids = sorted(row["rule_id"] for row in active_rows if "core" in str(row.get("source", "")).split("/") and row["include_in_dataset"])
    activation_ids = sorted(row["rule_id"] for row in included_rows if row["tier"] == "activation")
    bounded_ids = sorted(row["rule_id"] for row in included_rows if row["tier"] == "bounded_activation")

    composition = _composition_with_aliases(dict(manifest.get("composition", {}) or {}))
    stress_count = int(manifest.get("stress_count", composition.get("multi_error_stress", 0)) or 0)
    manifest.update(
        {
            "dataset_version": "training_dataset",
            "requested_total": int(data.get("target_total_examples", canonical.get("requested_total", manifest.get("requested_total", 0))) or 0),
            "actual_total": int(result.get("total", manifest.get("total", 0)) or 0),
            "requested_split_sizes": dict(data.get("exact_split_sizes", {}) or {}),
            "actual_split_sizes": manifest.get("split_sizes", result.get("splits", {})),
            "composition": composition,
            "fallback_used": False,
            "fallback_reason": "",
            "smoke_mode": _is_smoke_config(config),
            "smoke_verdict": VERDICT_SMOKE_ONLY,
            "source_constraints": _source_constraints(config, manifest),
            "candidate_recall_summary": candidate_summary,
            "gap_label_coverage_summary": gap_summary,
            "active_rule_ids": active_rule_ids,
            "active_target_rule_ids": active_rule_ids,
            "stable_core_rule_ids": stable_core_ids,
            "activation_rule_ids": activation_ids,
            "bounded_activation_rule_ids": bounded_ids,
            "excluded_rule_ids": sorted(row["rule_id"] for row in excluded_rows),
            "excluded_active_rule_ids": sorted(row["rule_id"] for row in excluded_rows),
            "active_target_rule_count": len(active_rule_ids),
            "active_rule_count": len(active_rule_ids),
            "stable_core_rule_count": len(stable_core_ids),
            "activation_rule_count": len(activation_ids),
            "bounded_activation_rule_count": len(bounded_ids),
            "excluded_rule_count": len(excluded_rows),
            "low_count_active_rule_ids": underfilled,
            "underfilled_rule_ids": underfilled,
            "real_pair_count": int(composition.get("real_error_pair", 0)),
            "clean_identity_count": int(composition.get("clean_identity", 0)),
            "hard_negative_count": int(composition.get("hard_negative", 0)),
            "stress_count": stress_count,
            "source_counts": dict(manifest.get("clean_source_counts", {}) or {}),
            "hard_negative_accepted_bad_edits": int(manifest.get("hard_negative_accepted_bad_edits", 0) or 0),
            "active_rule_quota_summary": {
                "active_rule_count": len(active_rule_ids),
                "stable_core_rule_count": len(stable_core_ids),
                "activation_rule_count": len(activation_ids),
                "bounded_activation_rule_count": len(bounded_ids),
                "excluded_count": len(excluded_rows),
                "underfilled_count": len(underfilled),
                "strategy": "broad_canonical",
            },
        }
    )
    return manifest


def _manifest_active_exclusion_reasons(manifest: dict[str, Any]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for rule_id in manifest.get("excluded_active_rule_ids", []) or []:
        reasons[str(rule_id)] = "excluded_by_core_builder"
    return reasons


def _apply_active_rule_exclusions(active_rows: list[dict[str, Any]], reasons: dict[str, str]) -> None:
    if not reasons:
        return
    for row in active_rows:
        rule_id = str(row.get("rule_id") or "")
        reason = reasons.get(rule_id)
        if not reason:
            continue
        row["include"] = False
        row["include_in_dataset"] = False
        row["reason"] = reason
        row["safety_status"] = "excluded_after_dataset_quality_audit"


def _refresh_quality_audit_reports(
    *,
    result: dict[str, Any],
    active_rows: list[dict[str, Any]],
    reports_dir: Path,
) -> dict[str, Any]:
    dataset_path = Path(str(result.get("path") or ""))
    if not result.get("path") or not dataset_path.exists() or dataset_path.is_dir():
        return {}
    frame = pd.read_csv(dataset_path, low_memory=False)
    active_rule_ids = [row["rule_id"] for row in active_rows if row["include_in_dataset"]]
    audit = audit_training_dataset(frame, active_rule_ids)
    write_generation_strategy_report(audit, reports_dir / "generation_strategy_report.csv")
    write_rule_diversity_report(audit, reports_dir / "rule_diversity_report.csv")
    write_extended_quality_reports(
        audit,
        reports_dir / "extended_quality_audit.csv",
        reports_dir / "extended_quality_audit.md",
    )
    write_artificial_marker_reports(
        audit,
        reports_dir / "artificial_marker_audit.csv",
        reports_dir / "artificial_marker_audit.md",
    )
    write_quote_bracket_balance_reports(
        audit,
        reports_dir / "quote_bracket_balance_audit.csv",
        reports_dir / "quote_bracket_balance_audit.md",
    )
    write_clean_hard_balance_report(audit, reports_dir / "clean_hard_balance_audit.csv")
    write_rule_semantic_alignment_report(audit, reports_dir / "rule_semantic_alignment_audit.csv")
    write_known_quality_bugs_report(audit, reports_dir / "known_quality_bugs_report.md")
    return audit


def _apply_quality_audit_to_manifest(manifest: dict[str, Any], audit: dict[str, Any]) -> None:
    manifest["corpus_opportunity_share"] = float(audit.get("corpus_opportunity_share", manifest.get("corpus_opportunity_share", 0.0)) or 0.0)
    manifest["fallback_template_share"] = float(audit.get("fallback_template_share", manifest.get("fallback_template_share", 0.0)) or 0.0)
    manifest["known_quality_bugs"] = dict(audit.get("known_quality_bugs", manifest.get("known_quality_bugs", {})) or {})
    manifest["artificial_marker_counts"] = dict(audit.get("artificial_marker_counts", manifest.get("artificial_marker_counts", {})) or {})
    manifest["quote_bracket_balance_bugs"] = dict(
        audit.get("quote_bracket_balance_bugs", manifest.get("quote_bracket_balance_bugs", {})) or {}
    )
    manifest["clean_hard_balance_bugs"] = dict(
        audit.get("clean_hard_balance_bugs", manifest.get("clean_hard_balance_bugs", {})) or {}
    )
    manifest["rule_semantic_alignment"] = dict(
        audit.get("rule_semantic_alignment", manifest.get("rule_semantic_alignment", {})) or {}
    )
    manifest["exact_clean_hard_duplicate_count"] = int(
        audit.get("exact_clean_hard_duplicate_count", manifest.get("exact_clean_hard_duplicate_count", 0)) or 0
    )
    manifest["error_bearing_sentence_source_counts"] = dict(
        audit.get(
            "error_bearing_sentence_source_counts",
            manifest.get("error_bearing_sentence_source_counts", {}),
        )
        or {}
    )
    manifest["rule_diversity_summary"] = dict(audit.get("rule_diversity_summary", manifest.get("rule_diversity_summary", {})) or {})
    manifest["extended_quality_audit_summary"] = dict(
        audit.get(
            "extended_quality_audit_summary",
            manifest.get("extended_quality_audit_summary", {}),
        )
        or {}
    )
    manifest["extended_quality_issue_count"] = int(
        dict(manifest.get("extended_quality_audit_summary", {}) or {}).get("issue_count", 0) or 0
    )

def _canonical_audit_errors(manifest: dict[str, Any], *, config: dict[str, Any]) -> list[str]:
    data = config.get("data", {})
    canonical = data.get("training_dataset", {}) or {}
    audit = dict(canonical.get("audit", {}) or {})
    errors = [
        str(error)
        for error in manifest.get("audit_errors", [])
        if not str(error).startswith(("active_rule_count_below_min:", "active_rule_quota_underfilled:"))
        and not str(error).startswith("synthetic_augmented_from_open_clean_shortage:")
        and str(error)
        not in {
            "candidate_recall_active_min_below_threshold",
            "extended_quality_audit_blocking_issues",
            "gap_coverage_active_min_below_threshold",
            "rule_diversity_gates_failed",
        }
    ]
    expected_total = int(data.get("target_total_examples", canonical.get("requested_total", 200000)) or 200000)
    if int(manifest.get("total", 0)) != expected_total:
        errors.append(f"dataset_size_below_requested:{manifest.get('total', 0)}!={expected_total}")
    if int(manifest.get("total", 0)) < 200000:
        errors.append(f"dataset_size_below_200000:{manifest.get('total', 0)}")
    expected_splits = {split: int(count) for split, count in dict(data.get("exact_split_sizes", {}) or {}).items()}
    actual_splits = {split: int(count) for split, count in dict(manifest.get("split_sizes", {}) or {}).items()}
    for split, expected in expected_splits.items():
        if actual_splits.get(split, 0) != expected:
            errors.append(f"split_size_mismatch:{split}:{actual_splits.get(split, 0)}!={expected}")
    composition = dict(manifest.get("composition", {}) or {})
    if int(composition.get(REAL_ERROR_PAIR, 0)) <= 0:
        errors.append("missing_source_type:real_error_pair")
    total = int(manifest.get("total", 0) or 0)
    if total and int(composition.get("clean_identity", composition.get(CLEAN_IDENTITY_OPEN, 0)) or 0) < total * 0.10:
        errors.append("clean_identity_below_10_percent")
    if total and int(composition.get("hard_negative", composition.get(HARD_NEGATIVE_OPEN, 0)) or 0) < total * 0.10:
        errors.append("hard_negative_below_10_percent")
    stress_count = int(manifest.get("stress_count", composition.get("multi_error_stress", 0)) or 0)
    if total and not (total * 0.03 <= stress_count <= total * 0.05):
        errors.append("stress_ratio_outside_3_5_percent")
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
    if float(manifest.get("corpus_opportunity_share", 0.0) or 0.0) < 0.70:
        errors.append("corpus_opportunity_share_below_threshold")
    if float(manifest.get("fallback_template_share", 1.0)) > 0.20:
        errors.append("fallback_template_share_above_threshold")
    for name, count in dict(manifest.get("known_quality_bugs", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"known_quality_bugs_present:{name}")
    for name, count in dict(manifest.get("artificial_marker_counts", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"artificial_marker_present:{name}")
    for name, count in dict(manifest.get("quote_bracket_balance_bugs", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"quote_bracket_balance_present:{name}")
    for name, count in dict(manifest.get("clean_hard_balance_bugs", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"clean_hard_balance_present:{name}")
    if int(dict(manifest.get("rule_semantic_alignment", {}) or {}).get("failed_rows", 0) or 0) != 0:
        errors.append("rule_semantic_alignment_failed")
    if not dict(manifest.get("error_bearing_sentence_source_counts", {}) or {}):
        errors.append("missing_error_bearing_sentence_source_counts")
    if int(dict(manifest.get("rule_diversity_summary", {}) or {}).get("failed_rule_count", 0) or 0) != 0:
        errors.append("rule_diversity_gates_failed")
    if int(dict(manifest.get("extended_quality_audit_summary", {}) or {}).get("blocking_issue_count", 0) or 0) != 0:
        errors.append("extended_quality_audit_blocking_issues")
    errors.extend(_dominance_errors(manifest, config=config))
    return errors


def _dominance_errors(manifest: dict[str, Any], *, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rule_share_limit = float((config.get("data", {}).get("training_dataset", {}) or {}).get("rule_caps", {}).get("max_rule_share_train", 0.10))
    error_share_limit = float((config.get("data", {}).get("training_dataset", {}) or {}).get("rule_caps", {}).get("max_error_type_share_train", 0.35))
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
        "include_in_dataset": bool(include),
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
    if tier == "activation":
        return 500, 1000, 2500
    if tier == "bounded_activation":
        return 200, 500, 1500
    return 300, 700, 2500


def _stable_core_rules(canonical_config: dict[str, Any]) -> dict[str, float]:
    manifest_path = Path(str(canonical_config.get("stable_core_manifest_path") or "reports/dataset_manifest.json"))
    recall_path = Path(str(canonical_config.get("stable_core_candidate_recall_path") or "reports/candidate_recall_by_rule.csv"))
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


def _verified_activation_rows(canonical_config: dict[str, Any]) -> list[dict[str, str]]:
    path = Path(str(canonical_config.get("activation_verified_path") or "reports/matrix_eval/next_dataset_activation_plan.csv"))
    return _read_csv_dicts(path)


def _configured_bounded_activation_rules(canonical_config: dict[str, Any]) -> list[str]:
    return sorted(str(rule_id) for rule_id in canonical_config.get("bounded_activation_rule_ids", []) or [])


def _activation_excluded_rows(canonical_config: dict[str, Any]) -> list[dict[str, str]]:
    path = Path(str(canonical_config.get("activation_plan_path") or "reports/matrix_eval/next_dataset_activation_plan.csv"))
    return _read_csv_dicts(path)


def _is_activation_include(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or row.get("under_quota_decision") or "").upper()
    candidate_backed = str(row.get("candidate_path_exists") or "").lower() == "true" or _float(row.get("candidate_recall"), default=0.0) > 0.0
    return (
        str(row.get("activation_decision") or "").upper() == "INCLUDE"
        and (not status or status == "READY_NEXT_DATASET")
        and candidate_backed
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


def _activation_exclusion_reason(row: dict[str, Any]) -> str:
    decision = str(row.get("activation_decision") or row.get("activation_decision") or row.get("blockers") or "").upper()
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
        final_count = int(rule_counts.get(str(row.get("rule_id")), 0))
        row["quota_final"] = final_count
        row["final_count"] = final_count


def _write_active_target_rules(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=ACTIVE_TARGET_COLUMNS).to_csv(path, index=False)


def _write_broad_active_training_rules(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        records.append(
            {
                "rule_id": row.get("rule_id", ""),
                "source": row.get("source", ""),
                "include": bool(row.get("include", row.get("include_in_dataset", False))),
                "reason": row.get("reason", ""),
                "risk_level": row.get("risk_level", ""),
                "needs_validator": bool(row.get("needs_validator", False)),
                "needs_threshold_calibration": bool(row.get("needs_threshold_calibration", False)),
                "candidate_path_exists": bool(row.get("candidate_path_exists", False)),
                "synthetic_support_exists": bool(row.get("synthetic_support_exists", False)),
                "hard_negative_support_exists": bool(row.get("hard_negative_support_exists", False)),
                "target_min_examples": int(row.get("target_min_examples", row.get("quota_min", 0)) or 0),
                "target_preferred_examples": int(row.get("target_preferred_examples", row.get("quota_preferred", 0)) or 0),
                "final_count": int(row.get("final_count", 0) or 0),
            }
        )
    pd.DataFrame(records, columns=BROAD_ACTIVE_RULE_COLUMNS).to_csv(path, index=False)


def _write_candidate_gate_report(
    active_rows: list[dict[str, Any]],
    candidate_report: pd.DataFrame,
    gap_report: pd.DataFrame,
    path: Path,
    *,
    config: dict[str, Any],
    gap_active_rule_ids: set[str],
) -> None:
    canonical = config.get("data", {}).get("training_dataset", {}) or {}
    audit = dict(canonical.get("audit", {}) or {})
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
            if not target or not target["include_in_dataset"] or int(float(item.get(count_column, 0) or 0)) <= 0:
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
        if not row["include_in_dataset"]:
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


def _write_activation_coverage_report(rows: list[dict[str, Any]], rule_counts: dict[str, int], path: Path) -> None:
    records = []
    for row in rows:
        if row["tier"] != "activation" and row["rule_id"] not in ACTIVATION_RULE_IDS:
            continue
        records.append(
            {
                "rule_id": row["rule_id"],
                "tier": row["tier"],
                "include_in_dataset": row["include_in_dataset"],
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
        "# Training Dataset Generation Report",
        "",
        f"- verdict: {manifest.get('verdict')}",
        f"- total: {manifest.get('total')}",
        f"- requested_total: {manifest.get('requested_total')}",
        f"- split_sizes: {json.dumps(manifest.get('split_sizes', {}), ensure_ascii=False, sort_keys=True)}",
        f"- composition: {json.dumps(composition, ensure_ascii=False, sort_keys=True)}",
        f"- active_target_rule_count: {manifest.get('active_target_rule_count', 0)}",
        f"- stable_core_rule_count: {manifest.get('stable_core_rule_count', 0)}",
        f"- activation_rule_count: {manifest.get('activation_rule_count', 0)}",
        f"- bounded_activation_rule_count: {manifest.get('bounded_activation_rule_count', 0)}",
        f"- excluded_rule_count: {manifest.get('excluded_rule_count', 0)}",
        f"- synthetic_count: {composition.get(SYNTHETIC_OPEN_CLEAN, 0)}",
        f"- real_pair_count: {composition.get(REAL_ERROR_PAIR, 0)}",
        f"- clean_identity_count: {composition.get(CLEAN_IDENTITY_OPEN, 0)}",
        f"- hard_negative_count: {composition.get(HARD_NEGATIVE_OPEN, 0)}",
        f"- candidate_recall_min_mean: {float(recall.get('active_min_excluding_unknown', 1.0)):.6f} / {float(recall.get('active_mean_excluding_unknown', 1.0)):.6f}",
        f"- gap_coverage_min_mean: {float(gap.get('active_min_excluding_unknown', 1.0)):.6f} / {float(gap.get('active_mean_excluding_unknown', 1.0)):.6f}",
        f"- quote_bracket_balance_bugs: {json.dumps(manifest.get('quote_bracket_balance_bugs', {}), ensure_ascii=False, sort_keys=True)}",
        f"- underfilled_active_rules: {', '.join(manifest.get('low_count_active_rule_ids', []))}",
        "",
        "## Audit Errors",
        "",
    ]
    audit_errors = list(manifest.get("audit_errors", []) or [])
    lines.extend(f"- {error}" for error in audit_errors) if audit_errors else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_constraints(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    canonical = config.get("data", {}).get("training_dataset", {}) or {}
    return {
        "requested_total": canonical.get("requested_total", config.get("data", {}).get("target_total_examples")),
        "smoke_total": (canonical.get("smoke", {}) or {}).get("requested_total"),
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
    if upgraded.get("verdict") == CORE_VERDICT_READY:
        upgraded["verdict"] = "BLOCKED"
    return upgraded


def _is_smoke_config(config: dict[str, Any]) -> bool:
    smoke = ((config.get("data", {}).get("training_dataset", {}) or {}).get("smoke", {}) or {})
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
