from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any, Iterable

import pandas as pd

from src.candidates.candidate_generator import CandidateGenerator
from src.data.atomic_verifier import verify_atomic_positive
from src.data.corruption_operators import (
    Opportunity,
    RuleOperatorRegistry,
    build_default_operator_registry,
    resolve_operator_training_targets,
    write_operator_registry_reports,
)
from src.data.dataset_quality import (
    balance_audit_frames,
    clean_or_hard_quality_pass,
    clean_or_hard_quality_reasons,
    clean_hard_bug_summary,
    duplicate_rate,
    known_quality_bug_summary,
    normalized_pair_hash,
    numeric_punctuation_mismatch,
    numeric_punctuation_mismatch_audit_frame,
    quote_bracket_bug_summary,
)
from src.data.dataset_contract import (
    CLEAN_IDENTITY_OPEN,
    CONTRACT_OPTIONAL_COLUMNS,
    DATASET_CONTRACT,
    HARD_NEGATIVE_OPEN,
    LAYER_ATOMIC_HARD_NEGATIVE,
    LAYER_ATOMIC_POSITIVE,
    LAYER_CLEAN_IDENTITY,
    LAYER_REAL_ATOMIC,
    LAYER_STRESS_MULTI_ERROR,
    REAL_ERROR_PAIR,
    SYNTHETIC_OPEN_CLEAN,
    ensure_contract_columns,
    stable_dataset_hash,
)
from src.data.hard_negative_generation import generate_atomic_hard_negatives, write_hard_negative_reports
from src.data.stress_generation import generate_multi_error_stress_rows
from src.data.training_quality_audit import (
    audit_training_dataset,
    write_artificial_marker_reports,
    write_clean_hard_balance_report,
    write_extended_quality_reports,
    write_generation_strategy_report,
    write_known_quality_bugs_report,
    write_quote_bracket_balance_reports,
    write_report_manifest,
    write_rule_diversity_report,
    write_rule_semantic_alignment_report,
    write_training_quality_gate_reports,
    report_manifest_errors,
)
from src.evaluation.candidate_recall import build_candidate_recall_reports
from src.rules.capabilities import (
    active_rule_ids_for_training,
    capability_manifest_fields,
    capability_training_audit_errors,
    load_rule_capabilities,
    write_rule_capability_reports,
)
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id


TRAINING_INCLUDE_DECISIONS = frozenset(
    {
        "INCLUDE_NOW",
        "INCLUDE_AFTER_VALIDATOR",
        "INCLUDE_AFTER_THRESHOLD_CALIBRATION",
        "INCLUDE_AFTER_TRAINING",
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
        "yo_e_candidate",
    }
)
MISSING_MODULE_RULE_IDS = frozenset({"capitalization_ner", "yo_e_candidate", "neural_punctuation"})
DATASET_COLUMNS = [
    "source",
    "target",
    "split",
    "source_type",
    "error_type",
    "rule_ids",
    "edits",
    "metadata",
    "original_clean_source",
    "source_corpus",
    "source_subcorpus",
    "is_hard_negative",
    "is_real_pair",
    "template_id",
    "normalized_pair_hash",
    "error_types",
    "source_dataset",
    "is_clean",
    "is_synthetic",
    "domain",
    "rule_id",
    "edit_operations",
]
DATASET_COLUMNS.extend(column for column in CONTRACT_OPTIONAL_COLUMNS if column not in DATASET_COLUMNS)
BACKFILL_PRIORITY_RULE_IDS = frozenset(
    {
        "address_comma",
        "introductory_comma",
        "homogeneous_comma",
        "detached_adverbial_comma",
        "detached_participial_comma",
        "comparative_turnover_comma",
        "subject_predicate_dash",
        "direct_speech_colon",
        "direct_speech_dash",
        "direct_speech_quotes",
        "enumeration_colon",
        "explanation_colon",
        "semicolon",
        "quote_pair_balance",
        "bracket_pair_balance",
        "punctuation_delete_replace",
        "dictionary_fuzzy",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
        "swapped_letters_candidate",
        "hyphen_particles",
        "hyphen_koe_koy",
        "hyphen_po_adverbs",
        "context_to_zhe",
        "context_nesmotrya",
        "context_vsledstvie",
        "context_za_to",
        "ne_short_form",
        "ni_particle_context",
        "ni_stable_expression",
        "n_nn_deverbal_adjective",
        "pol_polu_compounds",
        "cy_exception",
    }
)
BACKFILL_TARGET_PER_RULE = 1_250
BACKFILL_MAX_SENTENCES_PER_RULE = 70_000
REQUIRED_BACKFILL_RULE_IDS = frozenset({"hyphen_particles", "detached_participial_comma"})
DEFAULT_CLEAN_POOL_PATH = Path("data/processed/clean_sentence_pool.csv.gz")
DEFAULT_CLEAN_POOL_CHUNKSIZE = 25_000
DEFAULT_LAYER_RATIOS = {
    LAYER_ATOMIC_POSITIVE: 0.45,
    LAYER_ATOMIC_HARD_NEGATIVE: 0.35,
    LAYER_CLEAN_IDENTITY: 0.12,
    LAYER_STRESS_MULTI_ERROR: 0.05,
    LAYER_REAL_ATOMIC: 0.03,
}
LAYER_ORDER = [
    LAYER_ATOMIC_POSITIVE,
    LAYER_ATOMIC_HARD_NEGATIVE,
    LAYER_CLEAN_IDENTITY,
    LAYER_REAL_ATOMIC,
    LAYER_STRESS_MULTI_ERROR,
]
LAYER_FILE_STEMS = {
    LAYER_ATOMIC_POSITIVE: "atomic_positive",
    LAYER_ATOMIC_HARD_NEGATIVE: "atomic_hard_negative",
    LAYER_CLEAN_IDENTITY: "clean_identity",
    LAYER_REAL_ATOMIC: "real_atomic",
    LAYER_STRESS_MULTI_ERROR: "stress_multi_error",
}


@dataclass(frozen=True)
class AtomicPositiveGenerationResult:
    rows: list[dict[str, Any]]
    generation_rows: list[dict[str, Any]]
    rejection_rows: list[dict[str, Any]]
    rules_without_atomic_positive: list[dict[str, Any]]


def build_operator_training_dataset_from_config(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    data_config = config.get("data", {}) or {}
    output_path = Path(str(data_config.get("processed_train_path") or "data/processed/correction_dataset.csv.gz"))
    manifest_path = Path(str(data_config.get("manifest_path") or "data/processed/dataset_manifest.json"))
    reports_root = Path(str((config.get("paths", {}) or {}).get("reports_dir") or "reports"))
    reports_dir = reports_root if reports_root.name == "dataset_build" else reports_root / "dataset_build"
    requested_total = _requested_total_from_config(data_config, output_path=output_path, manifest_path=manifest_path)
    split_sizes = _split_sizes_from_config(data_config, requested_total)
    clean_pool_path = _clean_pool_path_from_config(config)

    if output_path.exists() and manifest_path.exists() and not force:
        manifest = _read_json(manifest_path)
        if manifest.get("operator_based_generation") is True and int(manifest.get("total", 0) or 0) >= requested_total:
            freshness_errors = _existing_report_freshness_errors(manifest, reports_dir, config=config)
            if freshness_errors:
                audit_errors = _dedupe_errors([*list(manifest.get("audit_errors", []) or []), *freshness_errors])
                manifest["audit_errors"] = audit_errors
                manifest["report_freshness"] = {"status": "stale", "errors": freshness_errors}
                manifest["verdict"] = "DATASET_BLOCKED"
                manifest["final_verdict"] = "DATASET_BLOCKED"
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                return {
                    "status": "blocked",
                    "path": str(output_path),
                    "manifest_path": str(manifest_path),
                    "total": int(manifest.get("total", 0)),
                    "verdict": "DATASET_BLOCKED",
                    "dataset_contract": str(manifest.get("dataset_contract") or DATASET_CONTRACT),
                    "dataset_hash": str(manifest.get("dataset_hash") or ""),
                    "audit_errors": list(manifest.get("audit_errors", []) or []),
                    "layer_counts": dict(manifest.get("layer_counts", {}) or {}),
                }
            return {
                "status": "exists",
                "path": str(output_path),
                "manifest_path": str(manifest_path),
                "total": int(manifest.get("total", 0)),
                "verdict": str(manifest.get("verdict", "DATASET_BLOCKED")),
                "dataset_contract": str(manifest.get("dataset_contract") or DATASET_CONTRACT),
                "dataset_hash": str(manifest.get("dataset_hash") or ""),
                "audit_errors": list(manifest.get("audit_errors", []) or []),
                "layer_counts": dict(manifest.get("layer_counts", {}) or {}),
            }

    _reset_dataset_build_reports_dir(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    registry = build_default_operator_registry()
    capabilities = load_rule_capabilities("configs/rules.yaml")
    write_rule_capability_reports(capabilities, reports_dir)
    eligible_rule_ids, pre_excluded = _eligible_rule_ids(config, capabilities=capabilities)
    target_rows = resolve_operator_training_targets(
        candidate_rule_ids=eligible_rule_ids,
        registry=registry,
        excluded_rule_ids=pre_excluded,
    )
    write_operator_registry_reports(target_rows, registry=registry, reports_dir=reports_dir)

    if not clean_pool_path.exists():
        manifest = _blocked_manifest(
            requested_total,
            split_sizes,
            "missing_clean_sentence_pool",
            config=config,
            capabilities=capabilities,
        )
        _write_manifest_and_blocked_reports(manifest, manifest_path, reports_dir)
        return {
            "status": "blocked",
            "verdict": "DATASET_BLOCKED",
            "total": 0,
            "manifest_path": str(manifest_path),
            "dataset_contract": DATASET_CONTRACT,
            "dataset_hash": "",
            "audit_errors": list(manifest.get("audit_errors", []) or []),
            "layer_counts": dict(manifest.get("layer_counts", {}) or {}),
        }

    quota_config = _operator_rule_quota_config(config)
    candidate_rule_ids = {row["rule_id"] for row in target_rows if bool(row.get("include"))}
    configured_rule_ids = set(quota_config.get("rule_ids", []) or [])
    if configured_rule_ids:
        candidate_rule_ids &= configured_rule_ids
    candidate_generator = CandidateGenerator.from_config(config)
    atomic_result = generate_atomic_positive_rows_from_clean_pool(
        clean_pool_path,
        registry=registry,
        candidate_rule_ids=candidate_rule_ids,
        config=config,
        candidate_generator=candidate_generator,
    )
    verified_rows = atomic_result.rows
    rejection_rows = atomic_result.rejection_rows
    rule_min = int(quota_config.get("min_atomic_positives_per_active_rule", 1000))
    verified_counts = Counter(rule_id for row in verified_rows for rule_id in _row_rule_ids(row) if _counts_toward_rule_quota(row))
    active_rule_ids = sorted(rule_id for rule_id, count in verified_counts.items() if count >= rule_min)
    active_set = set(active_rule_ids)
    selected_synthetic = [row for row in verified_rows if any(rule_id in active_set for rule_id in _row_rule_ids(row))]
    selected_synthetic = [_filter_row_rule_ids(row, active_set) for row in selected_synthetic]
    selected_synthetic = _dedupe_pairs(selected_synthetic)
    active_counts = Counter(rule_id for row in selected_synthetic for rule_id in _row_rule_ids(row))
    active_rule_ids = sorted(rule_id for rule_id in active_rule_ids if active_counts.get(rule_id, 0) >= rule_min)
    active_set = set(active_rule_ids)
    selected_synthetic = [row for row in selected_synthetic if any(rule_id in active_set for rule_id in _row_rule_ids(row))]
    selected_synthetic = [_filter_row_rule_ids(row, active_set) for row in selected_synthetic]

    excluded_after_generation = {
        row["rule_id"]: row.get("reason") or "BLOCK_NO_OPERATOR"
        for row in target_rows
        if not bool(row.get("include"))
    }
    for rule_id in sorted(candidate_rule_ids - active_set):
        excluded_after_generation[rule_id] = "insufficient_atomic_positives"

    seed = int(data_config.get("synthetic_seed", 17))
    layer_targets = _layer_targets_from_config(config, requested_total)
    strict_clean_rows = _strict_clean_pool_rows(clean_pool_path, config)
    stress_loss_weight = _stress_loss_weight_from_config(config)
    real_atomic_path = _real_atomic_cache_path(config, output_path=output_path)
    real_stress_path = _real_stress_cache_path(config, output_path=output_path)
    real_atomic_rows = _load_real_atomic_rows_from_cache(
        real_atomic_path,
        target_count=layer_targets[LAYER_REAL_ATOMIC],
        seed=seed,
    )
    real_stress_rows = _load_real_stress_rows_from_cache(
        real_stress_path,
        target_count=layer_targets[LAYER_STRESS_MULTI_ERROR],
        seed=seed + 1,
        loss_weight=stress_loss_weight,
    )
    synthetic_stress_target = max(0, layer_targets[LAYER_STRESS_MULTI_ERROR] - len(real_stress_rows))
    stress_result = generate_multi_error_stress_rows(
        strict_clean_rows,
        registry=registry,
        rule_ids=active_rule_ids,
        candidate_generator=candidate_generator,
        target_count=synthetic_stress_target,
        seed=seed + 2,
        loss_weight=stress_loss_weight,
    )
    stress_rows = real_stress_rows + stress_result.rows

    hard_negative_result = None
    hard_negative_pool: list[dict[str, Any]] = []
    hard_target_with_reserve = layer_targets[LAYER_ATOMIC_HARD_NEGATIVE] + max(
        0,
        layer_targets[LAYER_ATOMIC_POSITIVE] - min(len(selected_synthetic), layer_targets[LAYER_ATOMIC_POSITIVE]),
    )
    if active_rule_ids and hard_target_with_reserve > 0:
        preferred_per_rule = max(1, (hard_target_with_reserve + len(active_rule_ids) - 1) // len(active_rule_ids))
        hard_negative_result = generate_atomic_hard_negatives(
            strict_clean_rows,
            rule_ids=active_rule_ids,
            candidate_generator=candidate_generator,
            min_per_rule=0,
            preferred_per_rule=preferred_per_rule,
            seed=seed + 3,
            max_scan_rows=len(strict_clean_rows),
            fallback_templates_enabled=False,
        )
        hard_negative_pool = hard_negative_result.rows

    clean_identity_pool = _clean_identity_rows_from_clean_pool(strict_clean_rows)
    layer_rows = _compose_layer_rows(
        {
            LAYER_ATOMIC_POSITIVE: selected_synthetic,
            LAYER_ATOMIC_HARD_NEGATIVE: hard_negative_pool,
            LAYER_CLEAN_IDENTITY: clean_identity_pool,
            LAYER_REAL_ATOMIC: real_atomic_rows,
            LAYER_STRESS_MULTI_ERROR: stress_rows,
        },
        targets=layer_targets,
        requested_total=requested_total,
        seed=seed,
    )
    rows = [row for layer in LAYER_ORDER for row in layer_rows.get(layer, [])]
    _assign_layered_splits(layer_rows, split_sizes, seed=seed)
    frame = _frame_from_rows(rows)
    production_gates = _production_audit_enabled(output_path=output_path, manifest_path=manifest_path, requested_total=requested_total)
    if production_gates:
        frame, active_rule_ids, excluded_after_generation, quality_audit = _prune_failed_diversity_rules(
            frame=frame,
            active_rule_ids=active_rule_ids,
            excluded_rule_ids=excluded_after_generation,
            requested_total=requested_total,
            split_sizes=split_sizes,
            seed=int(data_config.get("synthetic_seed", 17)),
            clean_pool_path=clean_pool_path,
            config=config,
        )
    else:
        quality_audit = audit_training_dataset(frame, active_rule_ids)

    frame.to_csv(output_path, index=False)
    _write_split_and_layer_files(frame, output_path)
    recall_reports = _build_candidate_recall_reports(frame, config)

    manifest = _manifest(
        frame,
        config=config,
        recall_reports=recall_reports,
        requested_total=requested_total,
        split_sizes=split_sizes,
        active_rule_ids=active_rule_ids,
        excluded_rule_ids=excluded_after_generation,
        registry_rows=target_rows,
        rejection_rows=rejection_rows,
        quality_audit=quality_audit,
        backfill_attempt_rows=_not_applicable_backfill_rows(candidate_rule_ids),
        capabilities=capabilities,
        production_gates=production_gates,
        low_resource_rule_ids=sorted(candidate_rule_ids - active_set),
        hard_negative_counts_by_rule=_hard_negative_counts_by_target_rule(frame),
        real_stress_count=len(real_stress_rows),
    )
    _write_reports(
        frame,
        manifest,
        reports_dir,
        registry_rows=target_rows,
        rejection_rows=rejection_rows,
        quality_audit=quality_audit,
        backfill_attempt_rows=_not_applicable_backfill_rows(candidate_rule_ids),
        atomic_generation_rows=atomic_result.generation_rows,
        atomic_rejection_rows=atomic_result.rejection_rows,
        rules_without_atomic_positive=atomic_result.rules_without_atomic_positive,
        hard_negative_result=hard_negative_result,
        quota_config=quota_config,
        config=config,
        recall_reports=recall_reports,
    )
    report_manifest = write_report_manifest(
        reports_dir,
        dataset_hash=str(manifest.get("dataset_hash") or ""),
        config_hash=str(manifest.get("config_hash") or ""),
        generated_at=str(manifest.get("generated_at") or ""),
    )
    manifest["report_freshness"] = {"status": "fresh", "errors": [], "report_count": len(report_manifest.get("reports", {}))}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "built" if manifest["verdict"] == "READY_FOR_TRAINING_DATASET" else "blocked",
        "path": str(output_path),
        "manifest_path": str(manifest_path),
        "total": int(len(frame)),
        "verdict": manifest["verdict"],
        "dataset_contract": manifest["dataset_contract"],
        "dataset_hash": manifest["dataset_hash"],
        "audit_errors": manifest["audit_errors"],
        "layer_counts": manifest["layer_counts"],
        "composition": manifest["composition"],
        "splits": manifest["split_sizes"],
    }


def _eligible_rule_ids(config: dict[str, Any], *, capabilities: list[Any]) -> tuple[list[str], dict[str, str]]:
    excluded: dict[str, str] = {}
    result: set[str] = set(active_rule_ids_for_training(capabilities))
    disabled = {str(rule_id) for rule_id in (config.get("synthetic_generation", {}) or {}).get("disabled", {}).keys()}
    for capability in capabilities:
        for rule_id in capability.project_rule_ids:
            if rule_id not in result:
                excluded[rule_id] = capability.training_decision
                continue
            if rule_id in BROAD_EXCLUDED_RULE_IDS or rule_id in disabled:
                excluded[rule_id] = "BLOCK_EXCLUDED_BY_OPERATOR_POLICY"
                result.discard(rule_id)
    if not bool(((config.get("dictionary", {}) or {}).get("yo_e", {}) or {}).get("enabled", False)):
        result.discard("yo_e_candidate")
        excluded["yo_e_candidate"] = "BLOCK_YO_E_DISABLED"
    for rule_id in MISSING_MODULE_RULE_IDS:
        result.discard(rule_id)
        excluded.setdefault(rule_id, "BLOCK_MISSING_MODULE")
    return sorted(result | set(excluded)), excluded


def _requested_total_from_config(data_config: dict[str, Any], *, output_path: Path, manifest_path: Path) -> int:
    configured = int(data_config.get("total_examples") or data_config.get("target_total_examples") or 200_000)
    if _is_canonical_dataset_path(output_path=output_path, manifest_path=manifest_path):
        return max(200_000, configured)
    split_total = sum(
        int(data_config.get(key, 0) or 0)
        for key in ("train_examples", "val_examples", "test_examples")
        if key in data_config
    )
    return split_total if split_total > 0 else max(1, configured)


def _split_sizes_from_config(data_config: dict[str, Any], requested_total: int) -> dict[str, int]:
    keys = {"train": "train_examples", "val": "val_examples", "test": "test_examples"}
    if any(key in data_config for key in keys.values()):
        split_sizes = {split: int(data_config.get(key, 0) or 0) for split, key in keys.items()}
        if sum(split_sizes.values()) == requested_total:
            return split_sizes
    return _exact_split_sizes(requested_total)


def _is_canonical_dataset_path(*, output_path: Path, manifest_path: Path) -> bool:
    return (
        output_path.as_posix() == Path("data/processed/correction_dataset.csv.gz").as_posix()
        and manifest_path.as_posix() == Path("data/processed/dataset_manifest.json").as_posix()
    )


def _production_audit_enabled(*, output_path: Path, manifest_path: Path, requested_total: int) -> bool:
    return _is_canonical_dataset_path(output_path=output_path, manifest_path=manifest_path) or requested_total >= 200_000


def _clean_pool_path_from_config(config: dict[str, Any]) -> Path:
    data_config = config.get("data", {}) or {}
    core_config = data_config.get("training_dataset_core", {}) or {}
    return Path(str(data_config.get("clean_pool_path") or core_config.get("clean_pool_path") or DEFAULT_CLEAN_POOL_PATH))


def _clean_pool_chunksize(config: dict[str, Any]) -> int:
    data_config = config.get("data", {}) or {}
    core_config = data_config.get("training_dataset_core", {}) or {}
    return max(1, int(data_config.get("clean_pool_chunksize") or core_config.get("clean_pool_chunksize") or DEFAULT_CLEAN_POOL_CHUNKSIZE))


def _operator_rule_quota_config(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {}) or {}
    core_config = data_config.get("training_dataset_core", {}) or {}
    old_quota = dict(core_config.get("active_rule_quota", {}) or {})
    old_caps = dict(core_config.get("rule_caps", {}) or {})
    raw = dict(data_config.get("rule_quota", {}) or {})
    min_total = int(raw.get("min_atomic_positives_per_active_rule") or old_quota.get("min_total_per_active_rule") or 1000)
    preferred = int(raw.get("preferred_atomic_positives_per_active_rule") or old_quota.get("preferred_total_per_active_rule") or max(2500, min_total))
    max_total = int(raw.get("max_total_per_rule_id") or old_caps.get("max_total_per_rule_id") or max(preferred, min_total))
    min_hard = int(raw.get("min_hard_negatives_per_active_rule") or old_quota.get("min_hard_negatives_per_active_rule") or 0)
    rule_ids = raw.get("rule_ids") or old_quota.get("rule_ids") or []
    return {
        "min_atomic_positives_per_active_rule": max(0, min_total),
        "preferred_atomic_positives_per_active_rule": max(min_total, preferred),
        "max_total_per_rule_id": max(min_total, max_total),
        "min_hard_negatives_per_active_rule": max(0, min_hard),
        "disable_rule_if_quota_not_met": _truthy(raw.get("disable_rule_if_quota_not_met", old_quota.get("disable_rule_if_quota_not_met", False))),
        "rule_ids": [str(rule_id) for rule_id in rule_ids if str(rule_id)],
    }


def _stress_loss_weight_from_config(config: dict[str, Any]) -> float:
    data_config = config.get("data", {}) or {}
    stress_config = data_config.get("stress", {}) or {}
    core_config = data_config.get("training_dataset_core", {}) or {}
    raw = stress_config.get("loss_weight", core_config.get("stress_loss_weight", 0.4))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.4


def _layer_targets_from_config(config: dict[str, Any], requested_total: int) -> dict[str, int]:
    data_config = config.get("data", {}) or {}
    composition = dict(data_config.get("composition", {}) or {})
    if composition:
        targets: dict[str, int] = {}
        remaining_layers: list[str] = []
        for layer in LAYER_ORDER:
            target_key = _composition_target_key(layer)
            if target_key in composition:
                targets[layer] = max(0, int(composition.get(target_key) or 0))
            else:
                remaining_layers.append(layer)
        remaining_total = max(0, requested_total - sum(targets.values()))
        weights = {
            layer: max(0.0, float(composition.get(_composition_ratio_key(layer), DEFAULT_LAYER_RATIOS[layer]) or 0.0))
            for layer in remaining_layers
        }
        targets.update(_allocate_counts_by_weight(remaining_total, weights))
        return _normalize_layer_targets(targets, requested_total)

    core_config = data_config.get("training_dataset_core", {}) or {}
    legacy_source_targets = dict(core_config.get("source_type_targets", {}) or {})
    if legacy_source_targets:
        stress_target = max(0, int(core_config.get("multi_error_stress_target", 0) or 0))
        synthetic_target = int(legacy_source_targets.get(SYNTHETIC_OPEN_CLEAN, 0) or 0)
        targets = {
            LAYER_ATOMIC_POSITIVE: max(0, synthetic_target - stress_target),
            LAYER_ATOMIC_HARD_NEGATIVE: max(0, int(legacy_source_targets.get(HARD_NEGATIVE_OPEN, 0) or 0)),
            LAYER_CLEAN_IDENTITY: max(0, int(legacy_source_targets.get(CLEAN_IDENTITY_OPEN, 0) or 0)),
            LAYER_REAL_ATOMIC: max(0, int(legacy_source_targets.get(REAL_ERROR_PAIR, 0) or 0)),
            LAYER_STRESS_MULTI_ERROR: stress_target,
        }
        return _normalize_layer_targets(targets, requested_total)

    targets = _allocate_counts_by_weight(requested_total, DEFAULT_LAYER_RATIOS)
    return _normalize_layer_targets(targets, requested_total)


def _composition_target_key(layer: str) -> str:
    if layer == LAYER_REAL_ATOMIC:
        return "real_atomic_train_target"
    return f"{layer}_target"


def _composition_ratio_key(layer: str) -> str:
    if layer == LAYER_REAL_ATOMIC:
        return "real_atomic_train_ratio"
    return f"{layer}_ratio"


def _allocate_counts_by_weight(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0 or not weights:
        return {layer: 0 for layer in weights}
    weight_sum = sum(max(0.0, weight) for weight in weights.values())
    if weight_sum <= 0:
        result = {layer: 0 for layer in weights}
        result[next(iter(weights))] = total
        return result
    raw = {layer: total * max(0.0, weight) / weight_sum for layer, weight in weights.items()}
    result = {layer: int(value) for layer, value in raw.items()}
    remainder = total - sum(result.values())
    for layer, _value in sorted(raw.items(), key=lambda item: (-(item[1] - int(item[1])), LAYER_ORDER.index(item[0]))):
        if remainder <= 0:
            break
        result[layer] += 1
        remainder -= 1
    return result


def _normalize_layer_targets(targets: dict[str, int], requested_total: int) -> dict[str, int]:
    result = {layer: max(0, int(targets.get(layer, 0) or 0)) for layer in LAYER_ORDER}
    delta = requested_total - sum(result.values())
    if delta > 0:
        result[LAYER_ATOMIC_POSITIVE] += delta
    elif delta < 0:
        excess = -delta
        for layer in reversed(LAYER_ORDER):
            removable = min(excess, result[layer])
            result[layer] -= removable
            excess -= removable
            if excess <= 0:
                break
    return result


def _iter_clean_pool_records(clean_pool_path: Path, config: dict[str, Any]) -> Iterable[dict[str, Any]]:
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
    reader = pd.read_csv(
        clean_pool_path,
        usecols=lambda column: column in usecols,
        chunksize=_clean_pool_chunksize(config),
        low_memory=False,
    )
    for chunk in reader:
        for row in chunk.fillna("").to_dict("records"):
            yield row


def _strict_clean_pool_rows(clean_pool_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _iter_clean_pool_records(clean_pool_path, config):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if _contains_artificial_marker(text, text) or not clean_or_hard_quality_pass(text):
            continue
        rows.append(dict(item))
    return rows


def _clean_identity_rows_from_clean_pool(clean_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _identity_row(
            str(row.get("text") or row.get("target") or row.get("source") or "").strip(),
            CLEAN_IDENTITY_OPEN,
            str(row.get("source_name") or row.get("source_corpus") or ""),
            str(row.get("source_subcorpus") or ""),
            str(row.get("domain") or ""),
        )
        for row in clean_rows
        if str(row.get("text") or row.get("target") or row.get("source") or "").strip()
    ]


def _compose_layer_rows(
    pools: dict[str, list[dict[str, Any]]],
    *,
    targets: dict[str, int],
    requested_total: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    seen_hashes: set[str] = set()
    selected: dict[str, list[dict[str, Any]]] = {layer: [] for layer in LAYER_ORDER}
    for layer in LAYER_ORDER:
        selected[layer] = _take_unique_rows(
            _deterministic_rows(pools.get(layer, []), seed=seed + LAYER_ORDER.index(layer)),
            target=targets.get(layer, 0),
            seen_hashes=seen_hashes,
        )

    deficit = max(0, requested_total - sum(len(rows) for rows in selected.values()))
    if deficit:
        selected[LAYER_ATOMIC_HARD_NEGATIVE].extend(
            _take_unique_rows(
                _deterministic_rows(pools.get(LAYER_ATOMIC_HARD_NEGATIVE, []), seed=seed + 101),
                target=deficit,
                seen_hashes=seen_hashes,
            )
        )
    deficit = max(0, requested_total - sum(len(rows) for rows in selected.values()))
    if deficit:
        selected[LAYER_CLEAN_IDENTITY].extend(
            _take_unique_rows(
                _deterministic_rows(pools.get(LAYER_CLEAN_IDENTITY, []), seed=seed + 102),
                target=deficit,
                seen_hashes=seen_hashes,
            )
        )
    return selected


def _deterministic_rows(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    random.Random(seed).shuffle(result)
    return result


def _take_unique_rows(rows: list[dict[str, Any]], *, target: int, seen_hashes: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if len(result) >= target:
            break
        pair_hash = normalized_pair_hash(str(row.get("source", "")), str(row.get("target", "")))
        if pair_hash in seen_hashes:
            continue
        item = dict(row)
        item["normalized_pair_hash"] = pair_hash
        seen_hashes.add(pair_hash)
        result.append(item)
    return result


def generate_atomic_positive_rows_from_clean_pool(
    clean_pool_path: str | Path,
    registry: RuleOperatorRegistry,
    candidate_rule_ids: Iterable[str],
    config: dict[str, Any],
    candidate_generator: Any,
) -> AtomicPositiveGenerationResult:
    path = Path(clean_pool_path)
    quota_config = _operator_rule_quota_config(config)
    preferred = int(quota_config["preferred_atomic_positives_per_active_rule"])
    max_total = int(quota_config["max_total_per_rule_id"])
    target_per_rule = min(preferred, max_total)
    rule_ids = sorted({normalize_rule_id(rule_id) for rule_id in candidate_rule_ids if normalize_rule_id(rule_id)})
    counts: Counter[str] = Counter()
    attempts: Counter[str] = Counter()
    opportunities_seen: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    operators = {rule_id: registry.get(rule_id) for rule_id in rule_ids}
    for rule_id, operator in operators.items():
        if operator is None:
            _reject(rejections, "registry", {"source": "", "target": ""}, rule_id, "BLOCK_NO_OPERATOR")

    active_rule_ids = [rule_id for rule_id in rule_ids if operators.get(rule_id) is not None and target_per_rule > 0]
    if not path.exists() or not active_rule_ids:
        generation_rows = _atomic_generation_rows(rule_ids, counts, attempts, opportunities_seen, rejection_counts, quota_config)
        return AtomicPositiveGenerationResult(
            rows=[],
            generation_rows=generation_rows,
            rejection_rows=rejections,
            rules_without_atomic_positive=_rules_without_atomic_positive(rule_ids, counts, operators=operators),
        )

    for pool_index, item in enumerate(_iter_clean_pool_records(path, config)):
        if all(counts.get(rule_id, 0) >= target_per_rule for rule_id in active_rule_ids):
            break
        target = str(item.get("text", "")).strip()
        if not target:
            _reject(rejections, pool_index, {"source": "", "target": ""}, "", "empty_clean_target")
            continue
        clean_reasons = clean_or_hard_quality_reasons(target)
        if clean_reasons or _contains_artificial_marker(target, target):
            reason = ",".join(clean_reasons or ["artificial_marker"])
            _reject(rejections, pool_index, {"source": target, "target": target}, "", reason)
            continue

        for rule_id in active_rule_ids:
            if counts.get(rule_id, 0) >= target_per_rule:
                continue
            if not _rule_may_have_opportunity(rule_id, target):
                continue
            operator = operators[rule_id]
            if operator is None:
                continue
            attempts[rule_id] += 1
            try:
                opportunities = operator.find_opportunities(target)
            except Exception as exc:
                reason = f"find_opportunities_error:{type(exc).__name__}"
                rejection_counts[(rule_id, reason)] += 1
                _reject(rejections, pool_index, {"source": target, "target": target}, rule_id, reason)
                continue
            if not opportunities:
                continue
            opportunities_seen[rule_id] += len(opportunities)
            for opportunity in opportunities:
                if counts.get(rule_id, 0) >= target_per_rule:
                    break
                try:
                    result = operator.corrupt(target, opportunity)
                except Exception as exc:
                    reason = f"corrupt_error:{type(exc).__name__}"
                    rejection_counts[(rule_id, reason)] += 1
                    _reject(rejections, pool_index, {"source": target, "target": target}, rule_id, reason)
                    continue
                result_rule_id = normalize_rule_id(result.rule_id)
                if result_rule_id != rule_id:
                    reason = "operator_result_rule_mismatch"
                    rejection_counts[(rule_id, reason)] += 1
                    _reject(rejections, pool_index, {"source": result.source, "target": result.target}, rule_id, reason)
                    continue
                verification = verify_atomic_positive(
                    result.source,
                    result.target,
                    result_rule_id,
                    candidate_generator=candidate_generator,
                )
                if not verification.passed:
                    rejection_counts[(rule_id, verification.reason)] += 1
                    _reject(rejections, pool_index, {"source": result.source, "target": result.target}, rule_id, verification.reason)
                    continue
                pair_hash = normalized_pair_hash(result.source, result.target)
                if pair_hash in seen_hashes:
                    reason = "duplicate_pair"
                    rejection_counts[(rule_id, reason)] += 1
                    _reject(rejections, pool_index, {"source": result.source, "target": result.target}, rule_id, reason)
                    continue
                seen_hashes.add(pair_hash)
                row = _row_from_corruption_result(
                    result=result,
                    verification=verification,
                    source_item=item,
                    pair_hash=pair_hash,
                )
                rows.append(row)
                counts[rule_id] += 1

    generation_rows = _atomic_generation_rows(rule_ids, counts, attempts, opportunities_seen, rejection_counts, quota_config)
    return AtomicPositiveGenerationResult(
        rows=rows,
        generation_rows=generation_rows,
        rejection_rows=rejections,
        rules_without_atomic_positive=_rules_without_atomic_positive(rule_ids, counts, operators=operators),
    )


def _atomic_generation_rows(
    rule_ids: list[str],
    counts: Counter[str],
    attempts: Counter[str],
    opportunities_seen: Counter[str],
    rejection_counts: Counter[Any],
    quota_config: dict[str, Any],
) -> list[dict[str, Any]]:
    min_total = int(quota_config["min_atomic_positives_per_active_rule"])
    preferred = int(quota_config["preferred_atomic_positives_per_active_rule"])
    max_total = int(quota_config["max_total_per_rule_id"])
    rows: list[dict[str, Any]] = []
    for rule_id in rule_ids:
        accepted = int(counts.get(rule_id, 0))
        rejected = sum(int(count) for (rejected_rule_id, _reason), count in rejection_counts.items() if rejected_rule_id == rule_id)
        rows.append(
            {
                "rule_id": rule_id,
                "status": "ready" if accepted >= min_total else "underfilled",
                "target_min_total": min_total,
                "preferred_total": preferred,
                "max_total": max_total,
                "accepted_count": accepted,
                "attempted_clean_targets": int(attempts.get(rule_id, 0)),
                "opportunities_seen": int(opportunities_seen.get(rule_id, 0)),
                "rejected_count": rejected,
                "reason": "" if accepted >= min_total else "below_min_atomic_positive_quota",
            }
        )
    return rows


def _rules_without_atomic_positive(
    rule_ids: list[str],
    counts: Counter[str],
    *,
    operators: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule_id in rule_ids:
        accepted = int(counts.get(rule_id, 0))
        if accepted > 0:
            continue
        rows.append(
            {
                "rule_id": rule_id,
                "status": "missing",
                "reason": "BLOCK_NO_OPERATOR" if operators.get(rule_id) is None else "no_atomic_positive_generated",
            }
        )
    return rows


def _not_applicable_backfill_rows(candidate_rule_ids: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": rule_id,
            "initial_verified": 0,
            "generated_corpus": 0,
            "generated_fallback": 0,
            "accepted_after_backfill": 0,
            "activated_after_backfill": False,
            "status": "not_applicable",
            "reason": "canonical_operator_pipeline_generates_from_clean_pool",
        }
        for rule_id in sorted({str(rule_id) for rule_id in candidate_rule_ids if str(rule_id)})
    ]


def _verified_synthetic_rows(
    frame: pd.DataFrame,
    candidate_rule_ids: set[str],
    registry: RuleOperatorRegistry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    synthetic = frame[frame["source_type"].astype(str).eq(SYNTHETIC_OPEN_CLEAN)] if "source_type" in frame else pd.DataFrame()
    for index, raw in synthetic.iterrows():
        row = _normalize_row(raw.to_dict())
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        if _contains_artificial_marker(source, target):
            _reject(rejections, index, row, "", "artificial_marker")
            continue
        metadata = _json_dict(row.get("metadata"))
        if source == target:
            _reject(rejections, index, row, "", "identity_pair")
            continue
        if numeric_punctuation_mismatch(source, target):
            for rule_id in _row_rule_ids(row):
                if rule_id in candidate_rule_ids:
                    _reject(rejections, index, row, rule_id, "numeric_punctuation_mismatch")
            continue
        accepted_rules: list[str] = []
        verification_payloads: dict[str, Any] = {}
        for rule_id in _row_rule_ids(row):
            if rule_id not in candidate_rule_ids:
                continue
            operator = registry.get(rule_id)
            if operator is None:
                _reject(rejections, index, row, rule_id, "BLOCK_NO_OPERATOR")
                continue
            opportunity = Opportunity(
                start=-1,
                end=-1,
                text="",
                rule_id=rule_id,
                evidence={
                    "generation_strategy": metadata.get("generation_strategy") or "corpus_opportunity",
                    "target_family": metadata.get("target_family") or rule_id,
                },
                source_type=str(metadata.get("error_bearing_sentence_source") or "corpus"),
            )
            verification = operator.verify(source, target, opportunity)
            if not verification.passed:
                _reject(rejections, index, row, rule_id, verification.reason)
                continue
            accepted_rules.append(rule_id)
            verification_payloads[rule_id] = asdict(verification)
        if not accepted_rules:
            continue
        accepted_candidate_rule_ids = sorted(
            {
                str(candidate_rule_id)
                for payload in verification_payloads.values()
                for candidate_rule_id in payload.get("candidate_rule_ids", [])
                if str(candidate_rule_id)
            }
        )
        first_verification = verification_payloads[accepted_rules[0]]
        metadata.update(
            {
                "operator_based_generation": True,
                "operator_verify_passed": True,
                "semantic_alignment_pass": True,
                "target_quality_pass": True,
                "candidate_present": True,
                "candidate_rule_ids": accepted_candidate_rule_ids,
                "gold_edit_count": int(first_verification.get("gold_edit_count", 0) or 0),
                "strict_validator_passed": all(bool(payload.get("strict_validator_passed")) for payload in verification_payloads.values()),
                "matched_candidate": first_verification.get("matched_candidate"),
                "operator_verification": verification_payloads,
                "generation_strategy": metadata.get("generation_strategy") or "corpus_opportunity",
                "error_bearing_sentence_source": metadata.get("error_bearing_sentence_source") or "corpus",
            }
        )
        row["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        row["rule_ids"] = json.dumps(accepted_rules, ensure_ascii=False)
        row["rule_id"] = accepted_rules[0]
        row["error_type"] = str(row.get("error_type") or _error_type_for_rules(accepted_rules))
        row["error_types"] = json.dumps([row["error_type"]], ensure_ascii=False)
        rows.append(row)
    return rows, rejections


def _backfill_verified_rows_from_pool(
    *,
    candidate_rule_ids: set[str],
    registry: RuleOperatorRegistry,
    existing_rows: list[dict[str, Any]],
    initial_counts: Counter[str],
    rejection_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool_path = Path("data/processed/clean_sentence_pool.csv.gz")
    report_rule_ids = sorted(
        rule_id
        for rule_id in candidate_rule_ids
        if rule_id in BACKFILL_PRIORITY_RULE_IDS and registry.get(rule_id) is not None
    )
    target_rule_ids = [
        rule_id
        for rule_id in report_rule_ids
        if int(initial_counts.get(rule_id, 0)) > 0 or rule_id in REQUIRED_BACKFILL_RULE_IDS
    ]
    attempts = [
        {
            "rule_id": rule_id,
            "initial_verified": int(initial_counts.get(rule_id, 0)),
            "generated_corpus": 0,
            "generated_fallback": 0,
            "accepted_after_backfill": int(initial_counts.get(rule_id, 0)),
            "activated_after_backfill": False,
            "reason": "",
        }
        for rule_id in report_rule_ids
    ]
    if not target_rule_ids or not pool_path.exists():
        for row in attempts:
            row["reason"] = "missing_clean_sentence_pool" if not pool_path.exists() else "no_seed_opportunities_backfill_deferred"
        return [], attempts

    attempt_by_rule = {row["rule_id"]: row for row in attempts}
    counts = Counter(initial_counts)
    needed = {
        rule_id: BACKFILL_TARGET_PER_RULE
        for rule_id in target_rule_ids
        if int(counts.get(rule_id, 0)) < BACKFILL_TARGET_PER_RULE
    }
    if not needed:
        for row in attempts:
            row["activated_after_backfill"] = int(row["initial_verified"]) >= 1000
            row["reason"] = "already_sufficient"
        return [], attempts

    seen_hashes = {
        normalized_pair_hash(str(row.get("source", "")), str(row.get("target", "")))
        for row in existing_rows
    }
    generated: list[dict[str, Any]] = []
    usecols = {
        "text",
        "source_name",
        "source_subcorpus",
        "domain",
        "license_status",
        "source_doc_id",
        "sentence_id",
        "hash",
    }
    for rule_id in list(needed):
        operator = registry.get(rule_id)
        if operator is None:
            needed.pop(rule_id, None)
            continue
        scanned = 0
        reader = pd.read_csv(pool_path, usecols=lambda column: column in usecols, chunksize=25_000, low_memory=False)
        for chunk in reader:
            if counts.get(rule_id, 0) >= needed.get(rule_id, BACKFILL_TARGET_PER_RULE):
                break
            chunk = chunk.fillna("")
            for item in chunk.to_dict("records"):
                scanned += 1
                if scanned > BACKFILL_MAX_SENTENCES_PER_RULE:
                    break
                if counts.get(rule_id, 0) >= needed.get(rule_id, BACKFILL_TARGET_PER_RULE):
                    break
                target = str(item.get("text", "")).strip()
                if not target or not clean_or_hard_quality_pass(target) or _contains_artificial_marker(target, target):
                    continue
                if not _rule_may_have_opportunity(rule_id, target):
                    continue
                try:
                    opportunities = operator.find_opportunities(target)
                except Exception:
                    continue
                for opportunity in opportunities:
                    if counts.get(rule_id, 0) >= needed.get(rule_id, BACKFILL_TARGET_PER_RULE):
                        break
                    try:
                        result = operator.corrupt(target, opportunity)
                        verification = operator.verify(result.source, result.target, opportunity)
                    except Exception:
                        continue
                    if not verification.passed:
                        _reject(rejection_rows, str(item.get("sentence_id", "")), {"source": result.source, "target": result.target}, rule_id, verification.reason)
                        continue
                    pair_hash = normalized_pair_hash(result.source, result.target)
                    if pair_hash in seen_hashes:
                        continue
                    seen_hashes.add(pair_hash)
                    row = _row_from_corruption_result(
                        result=result,
                        verification=verification,
                        source_item=item,
                        pair_hash=pair_hash,
                    )
                    generated.append(row)
                    counts[rule_id] += 1
                    attempt_by_rule[rule_id]["generated_corpus"] = int(attempt_by_rule[rule_id]["generated_corpus"]) + 1
                    attempt_by_rule[rule_id]["accepted_after_backfill"] = int(counts[rule_id])
            if scanned > BACKFILL_MAX_SENTENCES_PER_RULE:
                break
        if counts.get(rule_id, 0) >= needed.get(rule_id, BACKFILL_TARGET_PER_RULE):
            needed.pop(rule_id, None)

    for row in attempts:
        rule_id = str(row["rule_id"])
        row["accepted_after_backfill"] = int(counts.get(rule_id, 0))
        row["activated_after_backfill"] = int(counts.get(rule_id, 0)) >= 1000
        if not row["reason"]:
            if int(row["generated_corpus"]):
                row["reason"] = "backfilled"
            elif int(counts.get(rule_id, 0)) >= BACKFILL_TARGET_PER_RULE:
                row["reason"] = "already_sufficient"
            else:
                row["reason"] = "no_verified_corpus_opportunities"
    return generated, attempts


def _row_from_corruption_result(
    *,
    result: Any,
    verification: Any,
    source_item: dict[str, Any],
    pair_hash: str,
) -> dict[str, Any]:
    metadata = dict(result.metadata or {})
    verification_payload = asdict(verification)
    accepted_edits = list(getattr(verification, "edits", None) or result.edits)
    metadata.update(
        {
            "dataset_contract": DATASET_CONTRACT,
            "dataset_layer": LAYER_ATOMIC_POSITIVE,
            "is_atomic": True,
            "is_stress": False,
            "count_toward_rule_quota": True,
            "loss_weight": 1.0,
            "operator_based_generation": True,
            "operator_verify_passed": True,
            "semantic_alignment_pass": True,
            "target_quality_pass": True,
            "candidate_present": bool(verification.candidate_present),
            "candidate_rule_ids": verification.candidate_rule_ids,
            "gold_edit_count": int(getattr(verification, "gold_edit_count", 0) or 0),
            "strict_validator_passed": bool(getattr(verification, "strict_validator_passed", False)),
            "matched_candidate": getattr(verification, "matched_candidate", None),
            "operator_verification": {result.rule_id: verification_payload},
            "generation_strategy": result.generation_strategy,
            "error_bearing_sentence_source": "corpus",
            "error_bearing_span": list(result.error_bearing_span),
            "error_form": result.error_form,
            "target_form": result.target_form,
            "target_family": result.rule_id,
            "original_clean_sentence": result.target,
            "carrier_sentence_hash": str(source_item.get("hash") or pair_hash[:24]),
            "clean_hash": str(source_item.get("hash") or ""),
            "source_name": str(source_item.get("source_name") or ""),
            "source_subcorpus": str(source_item.get("source_subcorpus") or ""),
            "source_doc_id": str(source_item.get("source_doc_id") or ""),
            "sentence_id": str(source_item.get("sentence_id") or ""),
            "license_status": str(source_item.get("license_status") or ""),
            "normalized_pair_hash": pair_hash,
            "rule_ids": [result.rule_id],
            "synthetic_source_dataset": f"corpus_backfill_{result.rule_id}",
        }
    )
    error_type = _error_type_for_rules([result.rule_id])
    return {
        "source": result.source,
        "target": result.target,
        "split": "train",
        "source_type": SYNTHETIC_OPEN_CLEAN,
        "error_type": error_type,
        "rule_ids": json.dumps([result.rule_id], ensure_ascii=False),
        "edits": json.dumps(accepted_edits, ensure_ascii=False),
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "original_clean_source": result.target,
        "source_corpus": str(source_item.get("source_name") or ""),
        "source_subcorpus": str(source_item.get("source_subcorpus") or ""),
        "is_hard_negative": False,
        "is_real_pair": False,
        "template_id": "",
        "normalized_pair_hash": pair_hash,
        "error_types": json.dumps([error_type], ensure_ascii=False),
        "source_dataset": str(source_item.get("source_name") or ""),
        "is_clean": False,
        "is_synthetic": True,
        "domain": str(source_item.get("domain") or "open_clean"),
        "rule_id": result.rule_id,
        "edit_operations": json.dumps(accepted_edits, ensure_ascii=False),
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_POSITIVE,
        "is_atomic": True,
        "is_stress": False,
        "count_toward_rule_quota": True,
        "gold_edit_count": 1,
        "loss_weight": 1.0,
        "target_rule_id": result.rule_id,
        "verification_status": "passed",
        "rejection_reason": "",
    }


def _rule_may_have_opportunity(rule_id: str, text: str) -> bool:
    if rule_id.startswith("comma_") or rule_id.endswith("_comma"):
        return "," in text
    if "dash" in rule_id:
        return "—" in text
    if "colon" in rule_id:
        return ":" in text
    if rule_id == "semicolon":
        return ";" in text
    if "quote" in rule_id or "speech" in rule_id:
        return any(char in text for char in ('"', "«", "»", "—", ":"))
    if "bracket" in rule_id:
        return any(char in text for char in "()[]{}")
    if rule_id == "punctuation_delete_replace":
        return any(token in text for token in ("..", "!!", "??", ",,", ";;", "::"))
    return True


def _real_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if "source_type" not in frame:
        return []
    result: list[dict[str, Any]] = []
    for _index, raw in frame[frame["source_type"].astype(str).eq(REAL_ERROR_PAIR)].iterrows():
        row = _normalize_row(raw.to_dict())
        if str(row.get("source", "")) == str(row.get("target", "")):
            continue
        metadata = _json_dict(row.get("metadata"))
        if metadata.get("candidate_present") is False:
            continue
        metadata.update({"candidate_present": True, "operator_based_generation": False})
        row["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        result.append(row)
    return _dedupe_pairs(result)


def _real_atomic_cache_path(config: dict[str, Any], *, output_path: Path) -> Path:
    data_config = config.get("data", {}) or {}
    explicit = data_config.get("real_error_pairs_atomic_path")
    if explicit:
        return Path(str(explicit))
    validated = data_config.get("real_error_pairs_validated_path")
    output_dir = output_path.parent
    local_atomic = output_dir / "real_error_pairs_atomic.csv.gz"
    if local_atomic.exists():
        return local_atomic
    if validated:
        validated_path = Path(str(validated))
        sibling_atomic = validated_path.with_name("real_error_pairs_atomic.csv.gz")
        return sibling_atomic if sibling_atomic.exists() else validated_path
    default_atomic = Path("data/processed/real_error_pairs_atomic.csv.gz")
    return default_atomic if default_atomic.exists() else Path("data/processed/real_error_pairs_validated.csv.gz")


def _real_stress_cache_path(config: dict[str, Any], *, output_path: Path) -> Path:
    data_config = config.get("data", {}) or {}
    explicit = data_config.get("real_error_pairs_stress_path")
    if explicit:
        return Path(str(explicit))
    validated = data_config.get("real_error_pairs_validated_path")
    output_dir = output_path.parent
    local_stress = output_dir / "real_error_pairs_stress.csv.gz"
    if local_stress.exists():
        return local_stress
    if validated:
        return Path(str(validated)).with_name("real_error_pairs_stress.csv.gz")
    return Path("data/processed/real_error_pairs_stress.csv.gz")


def _load_real_atomic_rows_from_cache(path: str | Path, *, target_count: int, seed: int) -> list[dict[str, Any]]:
    rows = _read_real_cache_rows(path)
    accepted = [_normalize_real_layer_row(row, layer=LAYER_REAL_ATOMIC) for row in rows if _is_valid_real_atomic_train_row(row)]
    return _dedupe_pairs(_deterministic_rows([row for row in accepted if row is not None], seed=seed))[: max(0, int(target_count))]


def _load_real_stress_rows_from_cache(path: str | Path, *, target_count: int, seed: int, loss_weight: float = 0.4) -> list[dict[str, Any]]:
    rows = _read_real_cache_rows(path)
    accepted = [
        _normalize_real_layer_row(row, layer=LAYER_STRESS_MULTI_ERROR, loss_weight=loss_weight)
        for row in rows
        if _is_valid_real_stress_row(row)
    ]
    return _dedupe_pairs(_deterministic_rows([row for row in accepted if row is not None], seed=seed))[: max(0, int(target_count))]


def _read_real_cache_rows(path: str | Path) -> list[dict[str, Any]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return []
    return pd.read_csv(cache_path, low_memory=False).fillna("").to_dict("records")


def _is_valid_real_atomic_train_row(row: dict[str, Any]) -> bool:
    if str(row.get("source", "")) == str(row.get("target", "")):
        return False
    if _real_gold_edit_count(row) != 1:
        return False
    if not _row_candidate_present(row):
        return False
    if not _row_strict_validator_passed(row):
        return False
    rule_ids = _row_rule_ids(row)
    return bool(rule_ids) and all(_is_known_training_rule_id(rule_id) for rule_id in rule_ids)


def _is_valid_real_stress_row(row: dict[str, Any]) -> bool:
    if str(row.get("source", "")) == str(row.get("target", "")):
        return False
    if _real_gold_edit_count(row) <= 1:
        return False
    if not _row_candidate_present(row):
        return False
    if not _row_strict_validator_passed(row):
        return False
    rule_ids = _row_rule_ids(row)
    return bool(rule_ids) and all(_is_known_training_rule_id(rule_id) for rule_id in rule_ids)


def _normalize_real_layer_row(row: dict[str, Any], *, layer: str, loss_weight: float | None = None) -> dict[str, Any]:
    result = _normalize_row(dict(row))
    edit_count = _real_gold_edit_count(result)
    result["source_type"] = REAL_ERROR_PAIR
    result["dataset_contract"] = DATASET_CONTRACT
    result["dataset_layer"] = layer
    result["is_real_pair"] = True
    result["is_synthetic"] = False
    result["is_clean"] = False
    result["is_hard_negative"] = False
    result["is_atomic"] = layer == LAYER_REAL_ATOMIC
    result["is_stress"] = layer == LAYER_STRESS_MULTI_ERROR
    result["count_toward_rule_quota"] = False
    default_loss_weight = float(loss_weight if loss_weight is not None else (0.4 if layer == LAYER_STRESS_MULTI_ERROR else 1.0))
    result["loss_weight"] = float(result.get("loss_weight") or default_loss_weight)
    result["gold_edit_count"] = edit_count
    result["normalized_pair_hash"] = normalized_pair_hash(str(result.get("source", "")), str(result.get("target", "")))
    result["verification_status"] = "accepted"
    metadata = _json_dict(result.get("metadata"))
    metadata.update(
        {
            "dataset_contract": DATASET_CONTRACT,
            "dataset_layer": layer,
            "is_stress": layer == LAYER_STRESS_MULTI_ERROR,
            "count_toward_rule_quota": False,
            "loss_weight": result["loss_weight"],
            "gold_edit_count": edit_count,
            "operator_based_generation": False,
        }
    )
    result["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    return result


def _real_gold_edit_count(row: dict[str, Any]) -> int:
    for key in ("gold_edit_count", "edit_count"):
        value = row.get(key)
        if not _is_blank(value):
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                pass
    return _row_gold_edit_count_value(row)


def _row_candidate_present(row: dict[str, Any]) -> bool:
    value = row.get("candidate_present")
    if not _is_blank(value):
        return _truthy(value)
    return _truthy(_json_dict(row.get("metadata")).get("candidate_present", False))


def _row_strict_validator_passed(row: dict[str, Any]) -> bool:
    value = row.get("strict_validator_passed")
    if not _is_blank(value):
        return _truthy(value)
    return _truthy(_json_dict(row.get("metadata")).get("strict_validator_passed", False))


def _is_known_training_rule_id(rule_id: str) -> bool:
    normalized = normalize_rule_id(rule_id)
    return normalized not in {UNKNOWN_RULE_ID, "clean_identity", "clean_identity_hard_negative", "hard_negative"}


def _clean_and_hard_rows(frame: pd.DataFrame, *, clean_target: int, hard_target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean = _identity_rows_from_existing(frame, CLEAN_IDENTITY_OPEN, clean_target)
    hard = _identity_rows_from_existing(frame, HARD_NEGATIVE_OPEN, hard_target)
    if len(clean) < clean_target or len(hard) < hard_target:
        clean_extra, hard_extra = _top_up_clean_hard_from_pool(
            existing_rows=clean + hard,
            clean_needed=max(0, clean_target - len(clean)),
            hard_needed=max(0, hard_target - len(hard)),
        )
        clean.extend(clean_extra)
        hard.extend(hard_extra)
    return clean[:clean_target], hard[:hard_target]


def _identity_rows_from_existing(frame: pd.DataFrame, source_type: str, target_count: int) -> list[dict[str, Any]]:
    if "source_type" not in frame or target_count <= 0:
        return []
    result: list[dict[str, Any]] = []
    for _index, raw in frame[frame["source_type"].astype(str).eq(source_type)].iterrows():
        row = _normalize_row(raw.to_dict())
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        if not source or source != target:
            continue
        if _contains_artificial_marker(source, target) or not clean_or_hard_quality_pass(source):
            continue
        row["rule_id"] = "clean_identity" if source_type == CLEAN_IDENTITY_OPEN else "clean_identity_hard_negative"
        row["rule_ids"] = json.dumps([row["rule_id"]], ensure_ascii=False)
        row["error_type"] = "clean_identity" if source_type == CLEAN_IDENTITY_OPEN else "hard_negative"
        row["error_types"] = json.dumps([row["error_type"]], ensure_ascii=False)
        result.append(row)
        if len(result) >= target_count:
            break
    return result


def _top_up_clean_hard_from_pool(
    *,
    existing_rows: list[dict[str, Any]],
    clean_needed: int,
    hard_needed: int,
    clean_pool_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = config or {}
    path = Path(clean_pool_path or _clean_pool_path_from_config(config))
    if not path.exists() or clean_needed + hard_needed <= 0:
        return [], []
    seen = {str(row.get("source", "")) for row in existing_rows}
    clean: list[dict[str, Any]] = []
    hard: list[dict[str, Any]] = []
    for item in _iter_clean_pool_records(path, config):
        text = str(item.get("text", "")).strip()
        if not text or text in seen:
            continue
        if _contains_artificial_marker(text, text) or not clean_or_hard_quality_pass(text):
            continue
        seen.add(text)
        if len(clean) < clean_needed:
            clean.append(
                _identity_row(
                    text,
                    CLEAN_IDENTITY_OPEN,
                    str(item.get("source_name") or ""),
                    str(item.get("source_subcorpus") or ""),
                    str(item.get("domain") or ""),
                )
            )
        elif len(hard) < hard_needed:
            hard.append(
                _identity_row(
                    text,
                    HARD_NEGATIVE_OPEN,
                    str(item.get("source_name") or ""),
                    str(item.get("source_subcorpus") or ""),
                    str(item.get("domain") or ""),
                )
            )
        if len(clean) >= clean_needed and len(hard) >= hard_needed:
            break
    return clean, hard


def _identity_row(text: str, source_type: str, source_name: str, source_subcorpus: str, domain: str) -> dict[str, Any]:
    rule_id = "clean_identity" if source_type == CLEAN_IDENTITY_OPEN else "clean_identity_hard_negative"
    error_type = "clean_identity" if source_type == CLEAN_IDENTITY_OPEN else "hard_negative"
    layer = LAYER_CLEAN_IDENTITY if source_type == CLEAN_IDENTITY_OPEN else LAYER_ATOMIC_HARD_NEGATIVE
    metadata = {
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": layer,
        "operator_based_generation": False,
        "source_type": source_type,
        "candidate_present": False,
        "is_atomic": source_type == HARD_NEGATIVE_OPEN,
        "is_stress": False,
        "count_toward_rule_quota": False,
        "gold_edit_count": 0,
    }
    return {
        "source": text,
        "target": text,
        "split": "train",
        "source_type": source_type,
        "error_type": error_type,
        "rule_ids": json.dumps([rule_id], ensure_ascii=False),
        "edits": "[]",
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "original_clean_source": text,
        "source_corpus": source_name,
        "source_subcorpus": source_subcorpus,
        "is_hard_negative": source_type == HARD_NEGATIVE_OPEN,
        "is_real_pair": False,
        "template_id": "",
        "normalized_pair_hash": normalized_pair_hash(text, text),
        "error_types": json.dumps([error_type], ensure_ascii=False),
        "source_dataset": source_name,
        "is_clean": source_type == CLEAN_IDENTITY_OPEN,
        "is_synthetic": False,
        "domain": domain or "open_clean",
        "rule_id": rule_id,
        "edit_operations": "[]",
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": layer,
        "is_atomic": source_type == HARD_NEGATIVE_OPEN,
        "is_stress": False,
        "count_toward_rule_quota": False,
        "loss_weight": 1.0,
        "gold_edit_count": 0,
        "target_rule_id": rule_id,
        "candidate_source": "",
        "candidate_replacement": "",
        "candidate_start": -1,
        "candidate_end": -1,
        "verification_status": "",
        "rejection_reason": "",
    }


def _manifest(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    recall_reports: dict[str, pd.DataFrame],
    requested_total: int,
    split_sizes: dict[str, int],
    active_rule_ids: list[str],
    excluded_rule_ids: dict[str, str],
    registry_rows: list[dict[str, Any]],
    rejection_rows: list[dict[str, Any]],
    quality_audit: dict[str, Any],
    backfill_attempt_rows: list[dict[str, Any]],
    capabilities: list[Any],
    production_gates: bool = True,
    low_resource_rule_ids: list[str] | None = None,
    hard_negative_counts_by_rule: dict[str, int] | None = None,
    real_stress_count: int = 0,
) -> dict[str, Any]:
    rule_counts = _rule_counts(frame)
    layer_counts = _value_counts(frame, "dataset_layer")
    composition = _value_counts(frame, "source_type")
    synthetic = frame[frame["source_type"].eq(SYNTHETIC_OPEN_CLEAN)]
    normalized_counts = Counter(frame["normalized_pair_hash"].astype(str).tolist())
    synthetic_norm_counts = Counter(synthetic["normalized_pair_hash"].astype(str).tolist())
    strategy_counts = _metadata_counts(synthetic, "generation_strategy")
    corpus_count = int(strategy_counts.get("corpus_opportunity", 0) + _metadata_counts(synthetic, "error_bearing_sentence_source").get("corpus", 0))
    fallback_count = int(strategy_counts.get("fallback_natural_template", 0) + _metadata_counts(synthetic, "error_bearing_sentence_source").get("fallback_template", 0))
    synthetic_total = max(1, len(synthetic))
    corpus_share = min(1.0, corpus_count / synthetic_total)
    fallback_share = min(1.0, fallback_count / synthetic_total)
    known = known_quality_bug_summary(frame)
    quote = quote_bracket_bug_summary(frame)
    clean_hard = clean_hard_bug_summary(frame)
    numeric_mismatch = int(len(numeric_punctuation_mismatch_audit_frame(frame)))
    composition_for_audit = {**composition, "multi_error_stress": _stress_count(frame)}
    recall_summary = _metric_summary(
        recall_reports["candidate_recall_by_rule"],
        count_column="gold_count",
        metric_column="candidate_recall",
        active_rule_ids=set(active_rule_ids),
    )
    gap_summary = _metric_summary(
        recall_reports["gap_label_coverage_by_rule"],
        count_column="gold_gap_count",
        metric_column="gap_candidate_recall",
        active_rule_ids=set(active_rule_ids) & _punctuation_rule_ids(),
    )
    diversity = dict(quality_audit.get("rule_diversity_summary", {}) or {})
    extended_summary = dict(quality_audit.get("extended_quality_audit_summary", {}) or {})
    semantic_summary = dict(quality_audit.get("rule_semantic_alignment", {}) or {})
    atomic_purity_summary = dict(quality_audit.get("atomic_purity_summary", {}) or {})
    extra_edit_summary = dict(quality_audit.get("extra_edit_summary", {}) or {})
    unknown_rule_train_summary = dict(quality_audit.get("unknown_rule_train_summary", {}) or {})
    mixed_script_clean_summary = dict(quality_audit.get("mixed_script_clean_summary", {}) or {})
    real_pair_atomization_summary = dict(quality_audit.get("real_pair_atomization_summary", {}) or {})
    bearing_counts = dict(quality_audit.get("error_bearing_sentence_source_counts", {}) or {})
    corpus_share = float(quality_audit.get("corpus_opportunity_share", corpus_share) or 0.0)
    fallback_share = float(quality_audit.get("fallback_template_share", fallback_share) or 0.0)
    capability_fields = capability_manifest_fields(capabilities)
    audit_errors = _audit_errors(
        total=len(frame),
        requested_total=requested_total,
        split_sizes=split_sizes,
        composition=composition_for_audit,
        active_rule_ids=active_rule_ids,
        rule_counts=rule_counts,
        known=known,
        quote=quote,
        clean_hard=clean_hard,
        numeric_mismatch=numeric_mismatch,
        corpus_share=corpus_share,
        fallback_share=fallback_share,
        diversity=diversity,
        extended_summary=extended_summary,
        semantic_summary=semantic_summary,
        atomic_purity_summary=atomic_purity_summary,
        extra_edit_summary=extra_edit_summary,
        unknown_rule_train_summary=unknown_rule_train_summary,
        mixed_script_clean_summary=mixed_script_clean_summary,
        real_pair_atomization_summary=real_pair_atomization_summary,
        recall_summary=recall_summary,
        audit_config=_operator_audit_config(config),
        production_gates=production_gates,
    )
    audit_errors.extend(capability_training_audit_errors(rule_counts, capabilities))
    excluded_rows = [{"rule_id": rule_id, "reason": reason} for rule_id, reason in sorted(excluded_rule_ids.items())]
    backfilled_rule_ids = sorted(
        str(row.get("rule_id"))
        for row in backfill_attempt_rows
        if int(row.get("generated_corpus", 0) or 0) + int(row.get("generated_fallback", 0) or 0) > 0
    )
    failed_diversity_rule_ids = sorted(str(rule_id) for rule_id in diversity.get("failed_rule_ids", []) or [])
    manifest = {
        "verdict": "DATASET_BLOCKED" if audit_errors else "READY_FOR_TRAINING_DATASET",
        "dataset_contract": DATASET_CONTRACT,
        "dataset_hash": stable_dataset_hash(frame),
        "config_hash": _config_hash(config),
        "total": int(len(frame)),
        "requested_total": int(requested_total),
        "requested_split_sizes": split_sizes,
        "split_sizes": split_sizes,
        "composition": {**composition, "clean_identity": int(composition.get(CLEAN_IDENTITY_OPEN, 0)), "hard_negative": int(composition.get(HARD_NEGATIVE_OPEN, 0)), "multi_error_stress": _stress_count(frame)},
        "composition_by_split": {split: _value_counts(frame[frame["split"].eq(split)], "source_type") for split in ("train", "val", "test")},
        "layer_counts": {layer: int(layer_counts.get(layer, 0)) for layer in LAYER_ORDER},
        "layer_counts_by_split": {
            split: _value_counts(frame[frame["split"].eq(split)], "dataset_layer") for split in ("train", "val", "test")
        },
        "rule_id_counts": rule_counts,
        "rule_id_counts_atomic_positive_only": rule_counts,
        "hard_negative_counts_by_target_rule": dict(sorted((hard_negative_counts_by_rule or {}).items())),
        "rule_id_counts_by_split": {split: _rule_counts(frame[frame["split"].eq(split)]) for split in ("train", "val", "test")},
        "error_type_counts": _value_counts(frame, "error_type"),
        "operator_based_generation": True,
        "active_rule_count": int(len(active_rule_ids)),
        "active_rule_ids": active_rule_ids,
        "excluded_rule_ids": excluded_rows,
        "excluded_active_rule_ids": sorted(excluded_rule_ids),
        "rules_without_operator": sorted(row["rule_id"] for row in registry_rows if row.get("reason") == "BLOCK_NO_OPERATOR"),
        "underfilled_rule_ids": [],
        "low_count_active_rule_ids": [],
        "operator_acceptance_counts": {rule_id: int(rule_counts.get(rule_id, 0)) for rule_id in active_rule_ids},
        "operator_rejection_counts": dict(Counter(str(row.get("reason", "")) for row in rejection_rows if row.get("reason"))),
        "semantic_alignment_failed_rows": int(semantic_summary.get("failed_rows", 0) or 0),
        "numeric_punctuation_mismatch_count": numeric_mismatch,
        "known_quality_bugs": known,
        "quote_bracket_balance_bugs": quote,
        "clean_hard_balance_bugs": clean_hard,
        "atomic_purity_summary": atomic_purity_summary,
        "extra_edit_summary": extra_edit_summary,
        "unknown_rule_train_summary": unknown_rule_train_summary,
        "mixed_script_clean_summary": mixed_script_clean_summary,
        "real_pair_atomization_summary": real_pair_atomization_summary,
        "rule_semantic_alignment": semantic_summary,
        "candidate_recall_summary": recall_summary,
        "gap_label_coverage_summary": gap_summary,
        "corpus_opportunity_share": float(corpus_share),
        "fallback_template_share": float(fallback_share),
        "error_bearing_sentence_source_counts": bearing_counts,
        "generation_strategy_shares": {key: value / synthetic_total for key, value in strategy_counts.items()},
        "rule_diversity_summary": diversity,
        "extended_quality_audit_summary": extended_summary,
        "extended_quality_issue_count": int(extended_summary.get("issue_count", 0) or 0),
        "failed_diversity_rule_ids": failed_diversity_rule_ids,
        "backfilled_rule_ids": backfilled_rule_ids,
        "still_excluded_after_backfill": excluded_rows,
        "synthetic_normalized_pair_duplicate_rate": duplicate_rate(synthetic["normalized_pair_hash"].astype(str).tolist()),
        "top_normalized_pair_count": int(max(normalized_counts.values()) if normalized_counts else 0),
        "synthetic_top_normalized_pair_count": int(max(synthetic_norm_counts.values()) if synthetic_norm_counts else 0),
        "real_pair_count": int(composition.get(REAL_ERROR_PAIR, 0)),
        "real_error_pair_count": int(composition.get(REAL_ERROR_PAIR, 0)),
        "real_atomic_count": int(layer_counts.get(LAYER_REAL_ATOMIC, 0)),
        "real_stress_count": int(real_stress_count),
        "clean_identity_count": int(composition.get(CLEAN_IDENTITY_OPEN, 0)),
        "hard_negative_count": int(composition.get(HARD_NEGATIVE_OPEN, 0)),
        "stress_count": _stress_count(frame),
        "low_resource_rule_ids": sorted(low_resource_rule_ids or []),
        "eval_only_rule_ids": sorted(capability_fields.get("eval_only_rule_ids", [])),
        "disabled_rule_ids": _disabled_rule_ids(capabilities, excluded_rule_ids=excluded_rule_ids),
        "hard_negative_accepted_bad_edits": 0,
        "audit_errors": audit_errors,
        "warnings": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": int((config.get("data", {}) or {}).get("synthetic_seed", 17)),
    }
    manifest.update(capability_fields)
    manifest["active_rule_ids"] = active_rule_ids
    manifest["active_rule_count"] = int(len(active_rule_ids))
    manifest["excluded_active_rule_ids"] = sorted(set(excluded_rule_ids) | set(capability_fields.get("blocked_rule_ids", [])))
    return manifest


def _write_reports(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    reports_dir: Path,
    *,
    config: dict[str, Any],
    recall_reports: dict[str, pd.DataFrame] | None = None,
    registry_rows: list[dict[str, Any]],
    rejection_rows: list[dict[str, Any]],
    quality_audit: dict[str, Any],
    backfill_attempt_rows: list[dict[str, Any]],
    atomic_generation_rows: list[dict[str, Any]] | None = None,
    atomic_rejection_rows: list[dict[str, Any]] | None = None,
    rules_without_atomic_positive: list[dict[str, Any]] | None = None,
    hard_negative_result: Any | None = None,
    quota_config: dict[str, Any] | None = None,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registry_rows).to_csv(reports_dir / "active_training_rules.csv", index=False)
    pd.DataFrame(
        rejection_rows,
        columns=["row_index", "rule_id", "reason", "source", "target"],
    ).to_csv(reports_dir / "operator_rejection_report.csv", index=False)
    pd.DataFrame(
        [{"rule_id": rule_id, "accepted": count} for rule_id, count in manifest["operator_acceptance_counts"].items()]
    ).to_csv(reports_dir / "operator_generation_report.csv", index=False)
    pd.DataFrame(manifest["excluded_rule_ids"]).to_csv(reports_dir / "future_operator_backlog.csv", index=False)
    pd.DataFrame(backfill_attempt_rows).to_csv(reports_dir / "backfill_attempts_by_rule.csv", index=False)
    pd.DataFrame(manifest["excluded_rule_ids"]).to_csv(reports_dir / "excluded_rules_after_backfill.csv", index=False)
    pd.DataFrame(atomic_generation_rows or []).to_csv(reports_dir / "atomic_positive_generation_report.csv", index=False)
    pd.DataFrame(
        atomic_rejection_rows or [],
        columns=["row_index", "rule_id", "reason", "source", "target"],
    ).to_csv(reports_dir / "atomic_positive_rejection_report.csv", index=False)
    pd.DataFrame(
        rules_without_atomic_positive or [],
        columns=["rule_id", "status", "reason"],
    ).to_csv(reports_dir / "rules_without_atomic_positive.csv", index=False)
    if hard_negative_result is not None:
        write_hard_negative_reports(hard_negative_result, reports_dir)
    else:
        pd.DataFrame(
            [],
            columns=[
                "rule_id",
                "min_per_rule",
                "preferred_per_rule",
                "generated_count",
                "clean_count",
                "fallback_count",
                "rejected_count",
                "status",
            ],
        ).to_csv(reports_dir / "hard_negative_coverage_report.csv", index=False)
    _write_active_rule_quota_report(
        frame,
        active_rule_ids=manifest["active_rule_ids"],
        quota_config=quota_config or {},
        hard_negative_counts_by_rule=dict(manifest.get("hard_negative_counts_by_target_rule", {}) or {}),
        path=reports_dir / "active_rule_quota_report.csv",
    )
    pd.DataFrame([row for row in manifest["excluded_rule_ids"] if "MISSING_MODULE" in row.get("reason", "")]).to_csv(
        reports_dir / "blocked_by_missing_module.csv", index=False
    )
    _candidate_reports(frame, manifest, reports_dir, config=config, recall_reports=recall_reports)
    audits = balance_audit_frames(frame)
    audits["numeric_punctuation_mismatch_audit"].to_csv(reports_dir / "numeric_punctuation_mismatch_audit.csv", index=False)
    audit = quality_audit or audit_training_dataset(frame, manifest["active_rule_ids"])
    write_rule_semantic_alignment_report(audit, reports_dir / "rule_semantic_alignment_audit.csv")
    write_quote_bracket_balance_reports(audit, reports_dir / "quote_bracket_balance_audit.csv", reports_dir / "quote_bracket_balance_audit.md")
    write_clean_hard_balance_report(audit, reports_dir / "clean_hard_balance_audit.csv")
    write_training_quality_gate_reports(audit, reports_dir)
    write_generation_strategy_report(audit, reports_dir / "generation_strategy_report.csv")
    write_rule_diversity_report(audit, reports_dir / "rule_diversity_report.csv")
    write_known_quality_bugs_report(audit, reports_dir / "known_quality_bugs_report.md")
    write_artificial_marker_reports(
        audit,
        reports_dir / "artificial_marker_audit.csv",
        reports_dir / "artificial_marker_audit.md",
    )
    write_extended_quality_reports(audit, reports_dir / "extended_quality_audit.csv", reports_dir / "extended_quality_audit.md")
    _write_dataset_generation_report(manifest, reports_dir / "dataset_generation_report.md")
    # Compatibility reports expected by existing tests/scripts.
    for name in (
        "dataset_balance_by_error_type.csv",
        "dataset_balance_by_rule.csv",
        "dataset_balance_by_split.csv",
        "excluded_active_rules_report.csv",
        "source_usage_report.csv",
        "real_pair_usage_report.csv",
        "template_leakage_report.csv",
        "under_quota_canonical_report.csv",
        "candidate_recall_gate_report.csv",
        "activation_rule_coverage_report.csv",
        "canonical_active_target_rules.csv",
        "missing_external_sources.csv",
        "rejected_backfill_templates.csv",
    ):
        path = reports_dir / name
        if not path.exists():
            pd.DataFrame(
                [
                    {
                        "status": "not_applicable",
                        "reason": "operator_dataset_builder_clean_pool_pipeline",
                    }
                ]
            ).to_csv(path, index=False)


def _write_active_rule_quota_report(
    frame: pd.DataFrame,
    *,
    active_rule_ids: list[str],
    quota_config: dict[str, Any],
    hard_negative_counts_by_rule: dict[str, int],
    path: Path,
) -> None:
    atomic_counts = _rule_counts(frame)
    min_required = int(quota_config.get("min_atomic_positives_per_active_rule", 1000) or 0)
    min_hard_required = int(quota_config.get("min_hard_negatives_per_active_rule", 0) or 0)
    preferred = int(quota_config.get("preferred_atomic_positives_per_active_rule", max(2500, min_required)) or min_required)
    rule_ids = sorted(set(active_rule_ids) | set(atomic_counts) | set(hard_negative_counts_by_rule))
    rows = []
    for rule_id in rule_ids:
        atomic_count = int(atomic_counts.get(rule_id, 0))
        hard_negative_count = int(hard_negative_counts_by_rule.get(rule_id, 0))
        status = "ready" if atomic_count >= min_required and hard_negative_count >= min_hard_required else ("low_resource" if atomic_count > 0 else "disabled")
        if status == "ready":
            reason = ""
        elif atomic_count < min_required:
            reason = "below_min_atomic_positive_quota"
        else:
            reason = "below_min_hard_negative_quota"
        rows.append(
            {
                "rule_id": rule_id,
                "atomic_positive_count": atomic_count,
                "hard_negative_count": hard_negative_count,
                "min_required": min_required,
                "hard_negative_min_required": min_hard_required,
                "preferred": preferred,
                "status": status,
                "reason": reason,
            }
        )
    pd.DataFrame(
        rows,
        columns=[
            "rule_id",
            "atomic_positive_count",
            "hard_negative_count",
            "min_required",
            "hard_negative_min_required",
            "preferred",
            "status",
            "reason",
        ],
    ).to_csv(path, index=False)


def _candidate_reports(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    reports_dir: Path,
    *,
    config: dict[str, Any],
    recall_reports: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    reports = recall_reports or _build_candidate_recall_reports(frame, config)
    candidate_report = reports["candidate_recall_by_rule"]
    gap_report = reports["gap_label_coverage_by_rule"]
    candidate_report.to_csv(reports_dir / "candidate_recall_by_rule.csv", index=False)
    gap_report.to_csv(reports_dir / "gap_label_coverage_by_rule.csv", index=False)
    _write_candidate_recall_gate_report(
        candidate_report,
        reports_dir / "candidate_recall_gate_report.csv",
        active_rule_ids=set(str(rule_id) for rule_id in manifest.get("active_rule_ids", []) or []),
        min_required_recall=_candidate_recall_min(config),
    )
    return reports


def _build_candidate_recall_reports(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return build_candidate_recall_reports(
        frame.to_dict("records"),
        candidate_generator=CandidateGenerator.from_config(config),
        max_candidates=int((config.get("model", {}) or {}).get("max_candidates", 16)),
        rules_config_path="configs/rules.yaml",
        trust_candidate_backed_metadata=False,
    )


def _write_candidate_recall_gate_report(
    candidate_report: pd.DataFrame,
    path: Path,
    *,
    active_rule_ids: set[str],
    min_required_recall: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    report_rows = candidate_report.to_dict("records") if isinstance(candidate_report, pd.DataFrame) else []
    seen: set[str] = set()
    for item in report_rows:
        rule_id = str(item.get("rule_id") or "")
        if not rule_id:
            continue
        seen.add(rule_id)
        gold_count = int(float(item.get("gold_count", 0) or 0))
        candidate_present_count = int(float(item.get("candidate_present_count", 0) or 0))
        recall = float(item.get("candidate_recall", 0.0) or 0.0)
        if gold_count <= 0 or rule_id not in active_rule_ids:
            status = "not_applicable"
        else:
            status = "pass" if recall >= min_required_recall else "fail"
        rows.append(
            {
                "rule_id": rule_id,
                "gold_count": gold_count,
                "candidate_present_count": candidate_present_count,
                "candidate_recall": recall,
                "min_required_recall": float(min_required_recall),
                "status": status,
                "missing_examples": item.get("missing_examples", ""),
            }
        )
    for rule_id in sorted(active_rule_ids - seen):
        rows.append(
            {
                "rule_id": rule_id,
                "gold_count": 0,
                "candidate_present_count": 0,
                "candidate_recall": 0.0,
                "min_required_recall": float(min_required_recall),
                "status": "not_applicable",
                "missing_examples": "",
            }
        )
    output = pd.DataFrame(
        rows,
        columns=[
            "rule_id",
            "gold_count",
            "candidate_present_count",
            "candidate_recall",
            "min_required_recall",
            "status",
            "missing_examples",
        ],
    )
    output.to_csv(path, index=False)
    return output


def _write_dataset_generation_report(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# Operator Dataset Generation Report",
        "",
        f"- verdict: {manifest.get('verdict', 'DATASET_BLOCKED')}",
        f"- total: {manifest.get('total', 0)}",
        f"- split_sizes: {json.dumps(manifest.get('split_sizes', {}), ensure_ascii=False, sort_keys=True)}",
        f"- active_rule_count: {manifest.get('active_rule_count', 0)}",
        f"- operator_based_generation: {manifest.get('operator_based_generation', True)}",
        f"- semantic_alignment_failed_rows: {manifest.get('semantic_alignment_failed_rows', 0)}",
        f"- numeric_punctuation_mismatch_count: {manifest.get('numeric_punctuation_mismatch_count', 0)}",
        f"- corpus_opportunity_share: {float(manifest.get('corpus_opportunity_share', 0.0) or 0.0):.6f}",
        f"- fallback_template_share: {float(manifest.get('fallback_template_share', 0.0) or 0.0):.6f}",
        f"- audit_errors: {json.dumps(manifest.get('audit_errors', []), ensure_ascii=False)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit_errors(
    *,
    total: int,
    requested_total: int,
    split_sizes: dict[str, int],
    composition: dict[str, int],
    active_rule_ids: list[str],
    rule_counts: dict[str, int],
    known: dict[str, int],
    quote: dict[str, int],
    clean_hard: dict[str, int],
    numeric_mismatch: int,
    corpus_share: float,
    fallback_share: float,
    diversity: dict[str, Any],
    extended_summary: dict[str, Any],
    semantic_summary: dict[str, Any],
    atomic_purity_summary: dict[str, Any],
    extra_edit_summary: dict[str, Any],
    unknown_rule_train_summary: dict[str, Any],
    mixed_script_clean_summary: dict[str, Any],
    real_pair_atomization_summary: dict[str, Any],
    recall_summary: dict[str, Any],
    audit_config: dict[str, Any],
    production_gates: bool = True,
) -> list[str]:
    errors: list[str] = []
    if total != requested_total:
        errors.append(f"dataset_size_below_requested:{total}!={requested_total}")
    if production_gates and split_sizes != _exact_split_sizes(total):
        errors.append("split_sizes_not_exact_80_10_10")
    if production_gates:
        if total < 200_000:
            errors.append("total_below_200000")
        if int(composition.get(CLEAN_IDENTITY_OPEN, 0)) < total * 0.10:
            errors.append("clean_identity_below_10_percent")
        if int(composition.get(HARD_NEGATIVE_OPEN, 0)) < total * 0.10:
            errors.append("hard_negative_below_10_percent")
        if int(composition.get(REAL_ERROR_PAIR, 0)) <= 0:
            errors.append("missing_real_pairs")
        stress = int(composition.get("multi_error_stress", 0))
        if not (total * 0.03 <= stress <= total * 0.05):
            errors.append("stress_ratio_outside_3_5_percent")
        for rule_id in active_rule_ids:
            if int(rule_counts.get(rule_id, 0)) < 1000:
                errors.append(f"active_rule_under_min:{rule_id}")
    if numeric_mismatch:
        errors.append("numeric_punctuation_mismatch_present")
    for group, counts in (("known_quality_bugs", known), ("quote_bracket_balance_bugs", quote), ("clean_hard_balance_bugs", clean_hard)):
        for name, count in counts.items():
            if int(count) != 0:
                errors.append(f"{group}_present:{name}")
    if production_gates and corpus_share < 0.70:
        errors.append("corpus_opportunity_share_below_threshold")
    if production_gates and fallback_share > 0.20:
        errors.append("fallback_template_share_above_threshold")
    if production_gates and int(diversity.get("failed_rule_count", 0)) != 0:
        errors.append("rule_diversity_gates_failed")
    if production_gates and int(extended_summary.get("blocking_issue_count", 0) or 0) != 0:
        errors.append("extended_quality_audit_blocking_issues")
    if int(semantic_summary.get("failed_rows", 0) or 0) != 0:
        errors.append("rule_semantic_alignment_failed")
    if int(atomic_purity_summary.get("failed_rows", 0) or 0) != 0:
        errors.append("atomic_positive_gold_edit_count_not_one")
    if int(extra_edit_summary.get("failed_rows", 0) or 0) != 0:
        errors.append("atomic_positive_extra_edits_present")
    if int(unknown_rule_train_summary.get("failed_rows", 0) or 0) != 0:
        errors.append("unknown_rule_in_train")
    if int(mixed_script_clean_summary.get("failed_rows", 0) or 0) != 0:
        errors.append("mixed_script_clean_or_hard_present")
    if int(real_pair_atomization_summary.get("failed_rows", 0) or 0) != 0:
        errors.append("real_pair_atomization_failed")
    if float(recall_summary.get("active_min_excluding_unknown", 1.0) or 0.0) < float(audit_config.get("candidate_recall_min", 0.95)):
        errors.append("candidate_recall_active_min_below_threshold")
    return errors


def _rule_diversity_summary(frame: pd.DataFrame, active_rule_ids: list[str]) -> dict[str, Any]:
    failed: list[str] = []
    for rule_id in active_rule_ids:
        rows = frame[frame["rule_ids"].astype(str).str.contains(f'"{rule_id}"', regex=False, na=False)]
        if len(rows) < 1000:
            failed.append(rule_id)
    return {"active_rule_count": len(active_rule_ids), "failed_rule_count": len(failed), "failed_rule_ids": failed}


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {column: row.get(column, "") for column in DATASET_COLUMNS}
    if not normalized["edits"] and normalized.get("edit_operations"):
        normalized["edits"] = normalized["edit_operations"]
    if not normalized["edit_operations"] and normalized.get("edits"):
        normalized["edit_operations"] = normalized["edits"]
    return normalized


def _row_rule_ids(row: dict[str, Any] | pd.Series) -> list[str]:
    raw = row.get("rule_ids", "[]")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        parsed = raw
    else:
        parsed = []
    result = [normalize_rule_id(str(item)) for item in parsed if str(item)]
    if not result:
        rule_id = normalize_rule_id(str(row.get("rule_id", "")))
        if rule_id:
            result.append(rule_id)
    return result


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(",") if part.strip()]
        return parsed if isinstance(parsed, list) else []
    return []


def _filter_row_rule_ids(row: dict[str, Any], active_rule_ids: set[str]) -> dict[str, Any]:
    filtered = [rule_id for rule_id in _row_rule_ids(row) if rule_id in active_rule_ids]
    result = dict(row)
    result["rule_ids"] = json.dumps(filtered, ensure_ascii=False)
    result["rule_id"] = filtered[0] if filtered else ""
    metadata = _json_dict(result.get("metadata"))
    metadata["rule_ids"] = filtered
    result["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    return result


def _dedupe_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        pair_hash = normalized_pair_hash(str(row.get("source", "")), str(row.get("target", "")))
        if pair_hash in seen:
            continue
        seen.add(pair_hash)
        result.append(row)
    return result


def _clean_hard_targets(total: int, synthetic_count: int, real_count: int) -> tuple[int, int]:
    remaining = max(0, total - synthetic_count - real_count)
    clean = (remaining + 1) // 2
    hard = remaining // 2
    floor = int(total * 0.10)
    if clean < floor:
        clean = floor
    if hard < floor:
        hard = floor
    return clean, hard


def _frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=DATASET_COLUMNS).fillna("")
    if frame.empty:
        return ensure_contract_columns(pd.DataFrame(columns=DATASET_COLUMNS))
    frame["normalized_pair_hash"] = [
        normalized_pair_hash(str(row.source), str(row.target)) for row in frame.itertuples(index=False)
    ]
    return ensure_contract_columns(frame)


def _prune_failed_diversity_rules(
    *,
    frame: pd.DataFrame,
    active_rule_ids: list[str],
    excluded_rule_ids: dict[str, str],
    requested_total: int,
    split_sizes: dict[str, int],
    seed: int,
    clean_pool_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str], dict[str, Any]]:
    active = sorted(set(active_rule_ids))
    excluded = dict(excluded_rule_ids)
    current = frame
    audit = audit_training_dataset(current, active)
    for _iteration in range(8):
        failed = sorted(str(rule_id) for rule_id in dict(audit.get("rule_diversity_summary", {}) or {}).get("failed_rule_ids", []) or [])
        if not failed:
            return current, active, excluded, audit
        failed_set = set(failed)
        for rule_id in failed:
            excluded[rule_id] = "diversity_failed"
        active = [rule_id for rule_id in active if rule_id not in failed_set]
        rows = [
            row
            for row in current.to_dict("records")
            if not (
                str(row.get("source_type", "")) == SYNTHETIC_OPEN_CLEAN
                and any(rule_id in failed_set for rule_id in _row_rule_ids(row))
            )
        ]
        if len(rows) < requested_total:
            clean_top, hard_top = _top_up_clean_hard_from_pool(
                existing_rows=rows,
                clean_needed=(requested_total - len(rows) + 1) // 2,
                hard_needed=(requested_total - len(rows)) // 2,
                clean_pool_path=clean_pool_path,
                config=config,
            )
            rows.extend(clean_top)
            rows.extend(hard_top)
        rows = rows[:requested_total]
        _assign_splits(rows, split_sizes, seed=seed)
        current = _frame_from_rows(rows)
        audit = audit_training_dataset(current, active)
    return current, active, excluded, audit


def _assign_layered_splits(layer_rows: dict[str, list[dict[str, Any]]], split_sizes: dict[str, int], *, seed: int) -> None:
    total = sum(len(rows) for rows in layer_rows.values())
    if total <= 0:
        return
    effective_split_sizes = dict(split_sizes)
    if sum(effective_split_sizes.values()) != total:
        effective_split_sizes = _exact_split_sizes(total)
    for layer in LAYER_ORDER:
        rows = layer_rows.get(layer, [])
        random.Random(seed + 200 + LAYER_ORDER.index(layer)).shuffle(rows)
        counts = _proportional_split_counts(len(rows), effective_split_sizes, total)
        offset = 0
        for split in ("train", "val", "test"):
            count = counts.get(split, 0)
            for row in rows[offset : offset + count]:
                row["split"] = split
            offset += count

    flat_rows = [row for rows in layer_rows.values() for row in rows]
    _rebalance_split_sizes(flat_rows, effective_split_sizes, seed=seed)


def _proportional_split_counts(layer_total: int, split_sizes: dict[str, int], total: int) -> dict[str, int]:
    if layer_total <= 0 or total <= 0:
        return {"train": 0, "val": 0, "test": 0}
    raw = {split: layer_total * split_sizes[split] / total for split in ("train", "val", "test")}
    counts = {split: int(raw[split]) for split in ("train", "val", "test")}
    remainder = layer_total - sum(counts.values())
    for split, _value in sorted(raw.items(), key=lambda item: (-(item[1] - int(item[1])), ("train", "val", "test").index(item[0]))):
        if remainder <= 0:
            break
        counts[split] += 1
        remainder -= 1
    return counts


def _rebalance_split_sizes(rows: list[dict[str, Any]], split_sizes: dict[str, int], *, seed: int) -> None:
    random.Random(seed + 300).shuffle(rows)
    for _iteration in range(len(rows) * 2 + 3):
        counts = Counter(str(row.get("split", "")) for row in rows)
        overfull = [split for split in ("train", "val", "test") if counts.get(split, 0) > split_sizes[split]]
        underfull = [split for split in ("train", "val", "test") if counts.get(split, 0) < split_sizes[split]]
        if not overfull or not underfull:
            break
        from_split = overfull[0]
        to_split = underfull[0]
        for row in rows:
            if str(row.get("split", "")) == from_split:
                row["split"] = to_split
                break


def _write_split_and_layer_files(frame: pd.DataFrame, output_path: Path) -> None:
    for split in ("train", "val", "test"):
        split_frame = frame[frame["split"].eq(split)]
        split_frame.to_csv(output_path.parent / f"{split}.csv", index=False)
        for layer, stem in LAYER_FILE_STEMS.items():
            split_frame[split_frame["dataset_layer"].eq(layer)].to_csv(output_path.parent / f"{split}_{stem}.csv.gz", index=False)


def _assign_splits(rows: list[dict[str, Any]], split_sizes: dict[str, int], *, seed: int) -> None:
    randomizer = random.Random(seed)
    randomizer.shuffle(rows)
    index = 0
    for split in ("train", "val", "test"):
        for row in rows[index : index + split_sizes[split]]:
            row["split"] = split
        index += split_sizes[split]


def _exact_split_sizes(total: int) -> dict[str, int]:
    train = int(total * 0.8)
    val = int(total * 0.1)
    return {"train": train, "val": val, "test": total - train - val}


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].astype(str).value_counts().sort_index().items()}


def _rule_counts(frame: pd.DataFrame) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if frame.empty:
        return {}
    for row in frame.to_dict("records"):
        if not _counts_toward_rule_quota(row):
            continue
        for rule_id in _row_rule_ids(row):
            counter[rule_id] += 1
    return dict(sorted(counter.items()))


def _counts_toward_rule_quota(row: dict[str, Any]) -> bool:
    if _row_dataset_layer_value(row) != LAYER_ATOMIC_POSITIVE:
        return False
    if _row_gold_edit_count_value(row) != 1:
        return False
    value = row.get("count_toward_rule_quota")
    if not _is_blank(value):
        return _truthy(value)
    metadata = _json_dict(row.get("metadata"))
    if "count_toward_rule_quota" in metadata:
        return _truthy(metadata.get("count_toward_rule_quota"))
    return str(row.get("source_type", "")) == SYNTHETIC_OPEN_CLEAN


def _row_dataset_layer_value(row: dict[str, Any]) -> str:
    value = str(row.get("dataset_layer", "") or "").strip()
    if value:
        return value
    metadata = _json_dict(row.get("metadata"))
    value = str(metadata.get("dataset_layer", "") or "").strip()
    if value:
        return value
    if str(row.get("source_type", "")) == SYNTHETIC_OPEN_CLEAN:
        return "stress_multi_error" if _truthy(row.get("is_stress", metadata.get("is_stress", False))) else LAYER_ATOMIC_POSITIVE
    return ""


def _row_gold_edit_count_value(row: dict[str, Any]) -> int:
    value = row.get("gold_edit_count")
    if not _is_blank(value):
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
    metadata = _json_dict(row.get("metadata"))
    if "gold_edit_count" in metadata:
        try:
            return max(0, int(metadata.get("gold_edit_count", 0) or 0))
        except (TypeError, ValueError):
            return 0
    return len(_json_list(row.get("edits") or row.get("edit_operations")))


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "nan"}
    return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "off", "none", "null", "nan"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return bool(value)


def _metadata_counts(frame: pd.DataFrame, key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if frame.empty or "metadata" not in frame:
        return {}
    for value in frame["metadata"].tolist():
        item = _json_dict(value)
        current = item.get(key)
        if current:
            counter[str(current)] += 1
    return dict(counter)


def _stress_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    total = 0
    for row in frame.to_dict("records"):
        layer = str(row.get("dataset_layer", "") or "").strip()
        if not layer:
            layer = str(_json_dict(row.get("metadata")).get("dataset_layer", "") or "").strip()
        if layer == LAYER_STRESS_MULTI_ERROR and _row_gold_edit_count_value(row) >= 2:
            total += 1
    return total


def _hard_negative_counts_by_target_rule(frame: pd.DataFrame) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if frame.empty:
        return {}
    for row in frame.to_dict("records"):
        if _row_dataset_layer_value(row) != LAYER_ATOMIC_HARD_NEGATIVE:
            continue
        target_rule_id = normalize_rule_id(str(row.get("target_rule_id") or _json_dict(row.get("metadata")).get("target_rule_id") or ""))
        if target_rule_id == UNKNOWN_RULE_ID:
            continue
        counter[target_rule_id] += 1
    return dict(sorted(counter.items()))


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _operator_audit_config(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", {}) or {}
    core = data.get("training_dataset_core", {}) or data.get("training_dataset", {}) or {}
    result = dict((core.get("audit", {}) if isinstance(core, dict) else {}) or {})
    top_level = dict(data.get("audit", {}) or {})
    if "min_candidate_recall_for_active_rule" in top_level:
        top_level.setdefault("candidate_recall_min", top_level["min_candidate_recall_for_active_rule"])
    result.update(top_level)
    return result


def _candidate_recall_min(config: dict[str, Any]) -> float:
    return float(_operator_audit_config(config).get("candidate_recall_min", 0.95))


def _metric_summary(
    frame: pd.DataFrame,
    *,
    count_column: str,
    metric_column: str,
    active_rule_ids: set[str],
) -> dict[str, Any]:
    if frame.empty or count_column not in frame or metric_column not in frame:
        return _empty_metric_summary()
    working = frame.copy()
    working[count_column] = pd.to_numeric(working[count_column], errors="coerce").fillna(0)
    working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce").fillna(0.0)
    non_unknown = working[(working["rule_id"].astype(str) != UNKNOWN_RULE_ID) & (working[count_column] > 0)]
    active = non_unknown[non_unknown["rule_id"].astype(str).isin(active_rule_ids)]
    return {
        "rules_with_gold": int(len(non_unknown)),
        "active_rules_with_gold": int(len(active)),
        "min_excluding_unknown": _safe_min(non_unknown, metric_column),
        "mean_excluding_unknown": _safe_mean(non_unknown, metric_column),
        "active_min_excluding_unknown": _safe_min(active, metric_column),
        "active_mean_excluding_unknown": _safe_mean(active, metric_column),
    }


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


def _punctuation_rule_ids() -> set[str]:
    return {
        "comma_subordinate",
        "comma_conjunction",
        "introductory_comma",
        "address_comma",
        "homogeneous_comma",
        "detached_adverbial_comma",
        "detached_participial_comma",
        "apposition_comma",
        "clarification_comma",
        "comparative_turnover_comma",
        "subject_predicate_dash",
        "asyndetic_dash",
        "consequence_dash",
        "explanation_colon",
        "enumeration_colon",
        "enumeration_dash",
        "semicolon",
        "direct_speech_colon",
        "direct_speech_dash",
        "direct_speech_quotes",
        "quote_pair_balance",
        "bracket_pair_balance",
        "punctuation_delete_replace",
        "final_punctuation_default",
    }


def _dedupe_errors(errors: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for error in errors:
        value = str(error)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _reset_dataset_build_reports_dir(reports_dir: Path) -> None:
    if reports_dir.exists() and reports_dir.is_dir() and reports_dir.name == "dataset_build":
        shutil.rmtree(reports_dir)


def _existing_report_freshness_errors(manifest: dict[str, Any], reports_dir: Path, *, config: dict[str, Any]) -> list[str]:
    dataset_hash = str(manifest.get("dataset_hash") or "")
    config_hash = _config_hash(config)
    if not dataset_hash or not config_hash:
        return ["stale_reports_missing_dataset_or_config_hash"]
    errors = report_manifest_errors(reports_dir, dataset_hash=dataset_hash, config_hash=config_hash)
    if str(manifest.get("config_hash") or "") != config_hash:
        errors.append("stale_manifest_config_hash_mismatch")
    return _dedupe_errors(errors)


def _disabled_rule_ids(capabilities: list[Any], *, excluded_rule_ids: dict[str, str] | None = None) -> list[str]:
    result = {
        rule_id
        for capability in capabilities
        if str(getattr(capability, "training_decision", "")) == "BLOCK_DISABLED"
        for rule_id in getattr(capability, "project_rule_ids", [])
    }
    for rule_id, reason in dict(excluded_rule_ids or {}).items():
        if "disabled" in str(reason).lower():
            result.add(rule_id)
    return sorted(result)


def _error_type_for_rules(rule_ids: list[str]) -> str:
    if any("comma" in rule_id or "dash" in rule_id or "colon" in rule_id or "quote" in rule_id or "punctuation" in rule_id for rule_id in rule_ids):
        return "punctuation"
    if any("hyphen" in rule_id or "pol_polu" in rule_id for rule_id in rule_ids):
        return "hyphen"
    if any("context" in rule_id or "ne_" in rule_id for rule_id in rule_ids):
        return "split_join"
    if any("capitalization" in rule_id or "abbreviation" in rule_id for rule_id in rule_ids):
        return "case"
    return "spelling"


def _reject(rejections: list[dict[str, Any]], index: Any, row: dict[str, Any], rule_id: str, reason: str) -> None:
    if len(rejections) >= 100_000:
        return
    rejections.append(
        {
            "row_index": int(index) if isinstance(index, int) else str(index),
            "rule_id": rule_id,
            "reason": reason,
            "source": str(row.get("source", ""))[:300],
            "target": str(row.get("target", ""))[:300],
        }
    )


def _contains_artificial_marker(source: str, target: str) -> bool:
    combined = f"{source}\n{target}".lower()
    return "метка" in combined or "позже редактор проверил запись" in combined or "позже редактор проверил материал" in combined


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _blocked_manifest(
    requested_total: int,
    split_sizes: dict[str, int],
    reason: str,
    *,
    config: dict[str, Any] | None = None,
    capabilities: list[Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "verdict": "DATASET_BLOCKED",
        "dataset_contract": DATASET_CONTRACT,
        "total": 0,
        "requested_total": requested_total,
        "split_sizes": split_sizes,
        "operator_based_generation": True,
        "dataset_hash": "",
        "config_hash": _config_hash(config or {}),
        "layer_counts": {layer: 0 for layer in LAYER_ORDER},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_errors": [reason],
    }
    if capabilities is not None:
        manifest.update(capability_manifest_fields(capabilities))
    return manifest


def _write_manifest_and_blocked_reports(manifest: dict[str, Any], manifest_path: Path, reports_dir: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_generation_report(manifest, reports_dir / "dataset_generation_report.md")
    reason = ",".join(str(item) for item in manifest.get("audit_errors", []) if str(item)) or "blocked"
    not_applicable = [{"status": "not_applicable", "reason": reason}]
    pd.DataFrame(not_applicable).to_csv(reports_dir / "atomic_positive_generation_report.csv", index=False)
    pd.DataFrame(not_applicable).to_csv(reports_dir / "atomic_positive_rejection_report.csv", index=False)
    pd.DataFrame(not_applicable).to_csv(reports_dir / "rules_without_atomic_positive.csv", index=False)
    report_manifest = write_report_manifest(
        reports_dir,
        dataset_hash=str(manifest.get("dataset_hash") or ""),
        config_hash=str(manifest.get("config_hash") or ""),
        generated_at=str(manifest.get("generated_at") or ""),
    )
    manifest["report_freshness"] = {"status": "fresh", "errors": [], "report_count": len(report_manifest.get("reports", {}))}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
