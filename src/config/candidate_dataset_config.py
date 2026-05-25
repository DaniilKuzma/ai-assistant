from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


CANONICAL_DATASET_CONFIG_PATH = "data.candidate_opportunity"
DATASET_CONTRACT = "candidate_opportunity"

_DUPLICATE_TOP_LEVEL_SECTIONS = (
    "rule_quota",
    "rule_activation",
    "composition",
    "audit",
    "clean_pool",
    "stress",
    "rule_data_compiler",
    "rule_lab",
)

_CRITICAL_COMPOSITION_RATIOS = (
    "atomic_positive_ratio",
    "atomic_hard_negative_ratio",
    "clean_identity_ratio",
    "stress_multi_error_ratio",
    "real_atomic_train_ratio",
)


def get_candidate_dataset_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_candidate_dataset_config(config)


def normalize_candidate_dataset_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(config.get("data")) if isinstance(config, Mapping) else {}
    raw_candidate = data.get("candidate_opportunity")
    if isinstance(raw_candidate, Mapping):
        candidate = deepcopy(dict(raw_candidate))
    else:
        candidate = _legacy_candidate_dataset_view(config)

    candidate["contract"] = str(candidate.get("contract") or candidate.get("dataset_contract") or DATASET_CONTRACT)
    candidate["totals"] = _normalize_totals(candidate, data)
    candidate["paths"] = _normalize_paths(candidate, data, _mapping(config.get("paths")) if isinstance(config, Mapping) else {})
    candidate["rule_activation"] = dict(_mapping(candidate.get("rule_activation")))
    candidate["composition"] = dict(_mapping(candidate.get("composition")))
    candidate["clean_pool"] = _normalize_clean_pool(candidate)
    candidate["synthetic"] = dict(_mapping(candidate.get("synthetic")))
    candidate["real_pairs"] = dict(_mapping(candidate.get("real_pairs")))
    candidate["stress"] = dict(_mapping(candidate.get("stress")))
    candidate["rule_quota"] = _normalize_rule_quota(candidate)
    candidate["rule_data_compiler"] = dict(_mapping(candidate.get("rule_data_compiler")))
    candidate["rule_lab"] = _normalize_rule_lab(candidate)
    candidate["audit"] = _normalize_audit(candidate)
    if "dataset_build_workers" not in candidate and data.get("dataset_build_workers") is not None:
        candidate["dataset_build_workers"] = data.get("dataset_build_workers")
    if "synthetic_seed" not in candidate and data.get("synthetic_seed") is not None:
        candidate["synthetic_seed"] = data.get("synthetic_seed")
    return candidate


def validate_candidate_dataset_config(config: Mapping[str, Any]) -> list[str]:
    data = _mapping(config.get("data")) if isinstance(config, Mapping) else {}
    errors: list[str] = []

    if "training_dataset" in data:
        errors.append("legacy_dataset_block_present:data.training_dataset")
    if "training_dataset_core" in data:
        errors.append("legacy_dataset_block_present:data.training_dataset_core")

    candidate_present = isinstance(data.get("candidate_opportunity"), Mapping)
    if candidate_present:
        for key in _DUPLICATE_TOP_LEVEL_SECTIONS:
            if key in data and key in _mapping(data["candidate_opportunity"]):
                errors.append(f"duplicate_dataset_config:data.{key}")
        for key in (
            "dataset_contract",
            "target_total_examples",
            "total_examples",
            "train_examples",
            "val_examples",
            "test_examples",
            "exact_split_sizes",
        ):
            if key in data:
                errors.append(f"duplicate_dataset_config:data.{key}")

    candidate = normalize_candidate_dataset_config(config)
    contract = str(candidate.get("contract") or "")
    if contract != DATASET_CONTRACT:
        errors.append(f"invalid_dataset_contract:{contract}")

    totals = candidate_dataset_totals(config)
    total = _int(totals.get("total_examples"), 0)
    train = _int(totals.get("train_examples"), 0)
    val = _int(totals.get("val_examples"), 0)
    test = _int(totals.get("test_examples"), 0)
    if train + val + test != total:
        errors.append(f"dataset_totals_mismatch:{train}+{val}+{test}!={total}")

    composition = candidate_dataset_composition(config)
    ratio_sum = sum(_float(composition.get(key), 0.0) for key in _CRITICAL_COMPOSITION_RATIOS)
    if ratio_sum > 1.0 and not _truthy(composition.get("allow_layer_target_adjustment", False)):
        errors.append(f"composition_ratio_sum_exceeds_one:{ratio_sum:.6f}")

    quota = candidate_dataset_rule_quota(config)
    min_atomic = _int(quota.get("min_atomic_positives_per_active_rule"), 0)
    preferred_atomic = _int(quota.get("preferred_atomic_positives_per_active_rule"), 0)
    max_total = _int(quota.get("max_total_per_rule_id"), 0)
    min_hard = _int(quota.get("min_hard_negatives_per_active_rule"), 0)
    if min_atomic > preferred_atomic:
        errors.append(f"rule_quota_min_atomic_gt_preferred_atomic:{min_atomic}>{preferred_atomic}")
    if preferred_atomic > max_total:
        errors.append(f"rule_quota_preferred_atomic_gt_max_total:{preferred_atomic}>{max_total}")
    if min_hard < 0:
        errors.append(f"rule_quota_min_hard_negatives_negative:{min_hard}")

    activation = candidate_dataset_rule_activation(config)
    expected_final = _int(activation.get("expected_min_final_active_rule_count"), 0)
    target_final = _int(activation.get("target_final_active_rule_count"), 0)
    expected_training = _int(activation.get("expected_min_training_candidate_rule_count"), 0)
    target_training = _int(activation.get("target_training_candidate_rule_count"), 0)
    if expected_final > target_final:
        errors.append(f"rule_activation_expected_final_gt_target:{expected_final}>{target_final}")
    if target_training < expected_training:
        errors.append(f"rule_activation_target_training_below_expected_min:{target_training}<{expected_training}")

    threshold = _float(candidate.get("clean_pool", {}).get("high_confidence_candidate_threshold"), 0.0)
    if threshold < 0.0 or threshold > 1.0:
        errors.append(f"clean_pool_high_confidence_threshold_out_of_range:{threshold:g}")

    return _dedupe(errors)


def candidate_dataset_value(config: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = normalize_candidate_dataset_config(config)
    path = str(dotted_path).strip()
    for prefix in ("data.candidate_opportunity.", "candidate_opportunity."):
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    for part in path.split("."):
        if not part:
            continue
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def candidate_dataset_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(normalize_candidate_dataset_config(config).get("paths", {}) or {})


def candidate_dataset_totals(config: Mapping[str, Any]) -> dict[str, int]:
    totals = normalize_candidate_dataset_config(config).get("totals", {}) or {}
    return {
        "total_examples": _int(totals.get("total_examples"), 0),
        "train_examples": _int(totals.get("train_examples"), 0),
        "val_examples": _int(totals.get("val_examples"), 0),
        "test_examples": _int(totals.get("test_examples"), 0),
    }


def candidate_dataset_rule_quota(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(normalize_candidate_dataset_config(config).get("rule_quota", {}) or {})


def candidate_dataset_rule_activation(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(normalize_candidate_dataset_config(config).get("rule_activation", {}) or {})


def candidate_dataset_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(normalize_candidate_dataset_config(config).get("audit", {}) or {})


def candidate_dataset_composition(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(normalize_candidate_dataset_config(config).get("composition", {}) or {})


def legacy_source_type_targets(config: Mapping[str, Any]) -> dict[str, int]:
    candidate = normalize_candidate_dataset_config(config)
    composition = _mapping(candidate.get("composition"))
    totals = candidate_dataset_totals(config)
    total = totals["total_examples"]
    if total <= 0:
        return {}
    stress = _target_or_ratio(composition, "stress_multi_error_target", "stress_multi_error_ratio", total)
    return {
        "synthetic_augmented_from_open_clean": max(
            0,
            _target_or_ratio(composition, "atomic_positive_target", "atomic_positive_ratio", total),
        ),
        "real_error_pair": max(0, _target_or_ratio(composition, "real_atomic_train_target", "real_atomic_train_ratio", total)),
        "clean_identity_from_open_clean": max(0, _target_or_ratio(composition, "clean_identity_target", "clean_identity_ratio", total)),
        "hard_negative_from_open_clean": max(
            0,
            _target_or_ratio(composition, "atomic_hard_negative_target", "atomic_hard_negative_ratio", total),
        ),
        "multi_error_stress": max(0, stress),
    }


def legacy_split_source_type_targets(config: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    targets = legacy_source_type_targets(config)
    totals = candidate_dataset_totals(config)
    total = totals["total_examples"]
    if total <= 0 or not targets:
        return {}
    splits = {
        "train": totals["train_examples"],
        "val": totals["val_examples"],
        "test": totals["test_examples"],
    }
    result = {split: {} for split in splits}
    for source_type, count in targets.items():
        allocated = 0
        split_items = list(splits.items())
        for split, split_total in split_items[:-1]:
            value = int(round(count * (split_total / total)))
            result[split][source_type] = value
            allocated += value
        result[split_items[-1][0]][source_type] = max(0, count - allocated)
    return result


def candidate_dataset_core_compat_config(config: Mapping[str, Any]) -> dict[str, Any]:
    candidate = normalize_candidate_dataset_config(config)
    quota = candidate_dataset_rule_quota(config)
    audit = candidate_dataset_audit(config)
    stress = dict(candidate.get("stress", {}) or {})
    paths = candidate_dataset_paths(config)
    source_targets = legacy_source_type_targets(config)
    stress_target = int(source_targets.pop("multi_error_stress", 0) or 0)
    split_targets = legacy_split_source_type_targets(config)
    for split_targets_by_type in split_targets.values():
        split_targets_by_type.pop("multi_error_stress", None)
    return {
        "enabled": True,
        "legacy_builder": False,
        "dataset_contract": DATASET_CONTRACT,
        "requested_total": candidate_dataset_totals(config)["total_examples"],
        "expected_total": candidate_dataset_totals(config)["total_examples"],
        "open_corpora_sources_path": paths["open_corpora_sources_config"],
        "real_error_sources_path": paths["real_error_sources_config"],
        "clean_pool_path": paths["clean_pool_path"],
        "source_type_targets": source_targets,
        "split_source_type_targets": split_targets,
        "active_rule_quota": {
            "enabled": True,
            "strategy": "broad_canonical",
            "rule_ids": quota.get("rule_ids", []),
            "min_total_per_active_rule": quota.get("min_atomic_positives_per_active_rule", 0),
            "preferred_total_per_active_rule": quota.get("preferred_atomic_positives_per_active_rule", 0),
            "min_hard_negatives_per_active_rule": quota.get("min_hard_negatives_per_active_rule", 0),
            "disable_rule_if_quota_not_met": quota.get("disable_rule_if_quota_not_met", False),
            "split_minimums": {},
        },
        "rule_caps": {
            "max_total_per_rule_id": quota.get("max_total_per_rule_id", 0),
            "max_train_per_rule_id": quota.get("max_train_per_rule_id", quota.get("max_total_per_rule_id", 0)),
            "max_rule_share_train": quota.get("max_rule_share_train", 0.10),
            "max_error_type_share_train": quota.get("max_error_type_share_train", 0.35),
            "targeted_duplicate_cap": quota.get("targeted_duplicate_cap", 2),
            "stress_generation_rule_cap": quota.get("stress_generation_rule_cap", 5000),
        },
        "pool": {
            "min_clean_sentences": candidate.get("min_clean_pool_for_ready", 150000),
            "max_source_share": candidate.get("max_source_share", 0.6),
            "max_subcorpus_share": candidate.get("max_subcorpus_share", 0.7),
        },
        "audit": {
            "candidate_recall_min": audit.get("candidate_recall_min", audit.get("min_candidate_recall_for_active_rule", 0.95)),
            "gap_coverage_min": audit.get("gap_coverage_min", 0.85),
            "synthetic_min": source_targets.get("synthetic_augmented_from_open_clean", 0),
            "min_active_rule_count": candidate.get("rule_activation", {}).get("expected_min_final_active_rule_count", 0),
            "min_clean_identity_eval_split": audit.get("min_clean_identity_eval_split", 500),
            "min_hard_negative_eval_split": audit.get("min_hard_negative_eval_split", 500),
            "min_rows_for_template_gates": audit.get("min_rows_for_template_gates", 1000),
            "min_rows_for_candidate_gates": audit.get("min_rows_for_candidate_gates", 1000),
            "min_rows_for_eval_source_split_gates": audit.get("min_rows_for_eval_source_split_gates", 1000),
            "require_all_source_types": audit.get("require_all_source_types", True),
            "corpus_opportunity_share_min": audit.get("corpus_opportunity_share_min", 0.70),
            "fallback_template_share_max": audit.get("fallback_template_share_max", 0.20),
            "stress_min_ratio": stress.get("min_ratio", 0.03),
            "stress_max_ratio": stress.get("max_ratio", 0.05),
            "fail_on_stress_under_target": stress.get("fail_on_under_target", False),
            "fail_on_corpus_opportunity_share_below_threshold": audit.get(
                "fail_on_corpus_opportunity_share_below_threshold",
                False,
            ),
            "destructive_diversity_pruning_enabled": audit.get("destructive_diversity_pruning_enabled", False),
        },
        "multi_error_stress_target": stress_target,
        "reuse_clean_sentence_pool_cache": True,
        "min_clean_pool_for_ready": candidate.get("min_clean_pool_for_ready", 300000),
        "min_clean_pool_hard_min": candidate.get("min_clean_pool_hard_min", 150000),
    }


def _legacy_candidate_dataset_view(config: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(config.get("data")) if isinstance(config, Mapping) else {}
    legacy = _mapping(data.get("training_dataset_core")) or _mapping(data.get("training_dataset"))
    paths_config = _mapping(config.get("paths")) if isinstance(config, Mapping) else {}
    candidate = {
        "contract": data.get("dataset_contract") or legacy.get("dataset_contract") or DATASET_CONTRACT,
        "totals": {
            "total_examples": data.get("total_examples") or data.get("target_total_examples") or legacy.get("expected_total") or legacy.get("requested_total"),
            "train_examples": data.get("train_examples"),
            "val_examples": data.get("val_examples"),
            "test_examples": data.get("test_examples"),
        },
        "paths": {
            "processed_dir": data.get("processed_dir"),
            "reports_dir": paths_config.get("reports_dir"),
            "clean_pool_path": data.get("clean_pool_path") or legacy.get("clean_pool_path"),
            "correction_dataset_path": data.get("processed_train_path"),
            "manifest_path": data.get("manifest_path"),
            "open_corpora_sources_config": legacy.get("open_corpora_sources_config") or legacy.get("open_corpora_sources_path"),
            "real_error_sources_config": legacy.get("real_error_sources_config") or legacy.get("real_error_sources_path"),
            "rule_lab_recipes_config": legacy.get("rule_lab_recipes_config"),
        },
        "rule_activation": deepcopy(_mapping(data.get("rule_activation"))),
        "composition": deepcopy(_mapping(data.get("composition"))),
        "clean_pool": deepcopy(_mapping(data.get("clean_pool"))),
        "synthetic": deepcopy(_mapping(data.get("synthetic"))),
        "real_pairs": deepcopy(_mapping(data.get("real_pairs"))),
        "stress": deepcopy(_mapping(data.get("stress"))),
        "rule_quota": deepcopy(_mapping(data.get("rule_quota"))),
        "rule_data_compiler": deepcopy(_mapping(data.get("rule_data_compiler"))),
        "rule_lab": deepcopy(_mapping(data.get("rule_lab"))),
        "audit": deepcopy(_mapping(data.get("audit")) or _mapping(legacy.get("audit"))),
    }
    if not candidate["rule_quota"] and legacy:
        quota = _mapping(legacy.get("active_rule_quota"))
        caps = _mapping(legacy.get("rule_caps"))
        candidate["rule_quota"] = {
            "rule_ids": quota.get("rule_ids", []),
            "min_atomic_positives_per_active_rule": quota.get("min_total_per_active_rule"),
            "preferred_atomic_positives_per_active_rule": quota.get("preferred_total_per_active_rule"),
            "max_total_per_rule_id": caps.get("max_total_per_rule_id"),
            "min_hard_negatives_per_active_rule": quota.get("min_hard_negatives_per_active_rule"),
            "disable_rule_if_quota_not_met": quota.get("disable_rule_if_quota_not_met", False),
        }
    if legacy and "multi_error_stress_target" in legacy:
        candidate["composition"].setdefault("stress_multi_error_target", legacy.get("multi_error_stress_target"))
    return candidate


def _normalize_totals(candidate: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, int]:
    raw = _mapping(candidate.get("totals"))
    exact = _mapping(data.get("exact_split_sizes"))
    total = _int(raw.get("total_examples") or candidate.get("total_examples") or data.get("total_examples") or data.get("target_total_examples"), 0)
    train = _int(raw.get("train_examples") or candidate.get("train_examples") or data.get("train_examples") or exact.get("train"), 0)
    val = _int(raw.get("val_examples") or candidate.get("val_examples") or data.get("val_examples") or exact.get("val"), 0)
    test = _int(raw.get("test_examples") or candidate.get("test_examples") or data.get("test_examples") or exact.get("test"), 0)
    if total <= 0 and train + val + test > 0:
        total = train + val + test
    return {
        "total_examples": total,
        "train_examples": train,
        "val_examples": val,
        "test_examples": test,
    }


def _normalize_paths(candidate: Mapping[str, Any], data: Mapping[str, Any], root_paths: Mapping[str, Any]) -> dict[str, str]:
    raw = _mapping(candidate.get("paths"))
    correction_dataset = str(raw.get("correction_dataset_path") or raw.get("processed_train_path") or data.get("processed_train_path") or "data/processed/correction_dataset.csv.gz")
    processed_dir = str(raw.get("processed_dir") or data.get("processed_dir") or Path(correction_dataset).parent.as_posix())
    reports_dir = str(raw.get("reports_dir") or data.get("reports_dir") or "reports/dataset_build")
    if reports_dir == "reports":
        reports_dir = "reports/dataset_build"
    return {
        "processed_dir": processed_dir,
        "reports_dir": reports_dir,
        "clean_pool_path": str(raw.get("clean_pool_path") or data.get("clean_pool_path") or Path(processed_dir, "clean_sentence_pool.csv.gz").as_posix()),
        "correction_dataset_path": correction_dataset,
        "manifest_path": str(raw.get("manifest_path") or data.get("manifest_path") or Path(processed_dir, "dataset_manifest.json").as_posix()),
        "open_corpora_sources_config": str(
            raw.get("open_corpora_sources_config")
            or raw.get("open_corpora_sources_path")
            or data.get("open_corpora_sources_config")
            or "configs/open_corpora_sources.yaml"
        ),
        "real_error_sources_config": str(
            raw.get("real_error_sources_config")
            or raw.get("real_error_sources_path")
            or data.get("real_error_sources_config")
            or "configs/real_error_sources.yaml"
        ),
        "rule_lab_recipes_config": str(raw.get("rule_lab_recipes_config") or data.get("rule_lab_recipes_config") or "configs/rule_lab_recipes.yaml"),
        "real_error_pairs_validated_path": str(
            raw.get("real_error_pairs_validated_path")
            or data.get("real_error_pairs_validated_path")
            or Path(processed_dir, "real_error_pairs_validated.csv.gz").as_posix()
        ),
        "real_error_pairs_atomic_path": str(
            raw.get("real_error_pairs_atomic_path")
            or data.get("real_error_pairs_atomic_path")
            or Path(processed_dir, "real_error_pairs_atomic.csv.gz").as_posix()
        ),
        "real_error_pairs_stress_path": str(
            raw.get("real_error_pairs_stress_path")
            or data.get("real_error_pairs_stress_path")
            or Path(processed_dir, "real_error_pairs_stress.csv.gz").as_posix()
        ),
    }


def _normalize_clean_pool(candidate: Mapping[str, Any]) -> dict[str, Any]:
    clean_pool = dict(_mapping(candidate.get("clean_pool")))
    clean_pool.setdefault("high_confidence_candidate_threshold", 0.95)
    return clean_pool


def _normalize_rule_quota(candidate: Mapping[str, Any]) -> dict[str, Any]:
    quota = dict(_mapping(candidate.get("rule_quota")))
    if quota.get("min_hard_negatives_per_active_rule") is None:
        quota["min_hard_negatives_per_active_rule"] = 0
    return quota


def _normalize_rule_lab(candidate: Mapping[str, Any]) -> dict[str, Any]:
    rule_lab = dict(_mapping(candidate.get("rule_lab")))
    rule_lab.setdefault("enabled", False)
    return rule_lab


def _normalize_audit(candidate: Mapping[str, Any]) -> dict[str, Any]:
    audit = dict(_mapping(candidate.get("audit")))
    if "candidate_recall_min" in audit and "min_candidate_recall_for_active_rule" not in audit:
        audit["min_candidate_recall_for_active_rule"] = audit["candidate_recall_min"]
    audit.setdefault("destructive_diversity_pruning_enabled", False)
    return audit


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _target_or_ratio(config: Mapping[str, Any], target_key: str, ratio_key: str, total: int) -> int:
    if target_key in config:
        return _int(config.get(target_key), 0)
    return int(round(total * _float(config.get(ratio_key), 0.0)))


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
