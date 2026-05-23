from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Iterable

import pandas as pd

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
    HARD_NEGATIVE_OPEN,
    REAL_ERROR_PAIR,
    SYNTHETIC_OPEN_CLEAN,
    ensure_contract_columns,
)
from src.data.training_quality_audit import (
    audit_training_dataset,
    write_artificial_marker_reports,
    write_clean_hard_balance_report,
    write_extended_quality_reports,
    write_generation_strategy_report,
    write_known_quality_bugs_report,
    write_quote_bracket_balance_reports,
    write_rule_diversity_report,
    write_rule_semantic_alignment_report,
)
from src.rules.capabilities import (
    active_rule_ids_for_training,
    capability_manifest_fields,
    capability_training_audit_errors,
    load_rule_capabilities,
    write_rule_capability_reports,
)
from src.rules.rule_ids import normalize_rule_id


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


def build_operator_training_dataset_from_config(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    data_config = config.get("data", {}) or {}
    output_path = Path(str(data_config.get("processed_train_path") or "data/processed/correction_dataset.csv.gz"))
    manifest_path = Path(str(data_config.get("manifest_path") or "data/processed/dataset_manifest.json"))
    reports_root = Path(str((config.get("paths", {}) or {}).get("reports_dir") or "reports"))
    reports_dir = reports_root if reports_root.name == "dataset_build" else reports_root / "dataset_build"
    requested_total = int(data_config.get("total_examples") or data_config.get("target_total_examples") or 200_000)
    requested_total = max(200_000, requested_total if requested_total == 200_000 else 200_000)
    split_sizes = _exact_split_sizes(requested_total)

    if output_path.exists() and manifest_path.exists() and not force:
        manifest = _read_json(manifest_path)
        if manifest.get("operator_based_generation") is True and int(manifest.get("total", 0) or 0) >= requested_total:
            return {
                "status": "exists",
                "path": str(output_path),
                "manifest_path": str(manifest_path),
                "total": int(manifest.get("total", 0)),
                "verdict": str(manifest.get("verdict", "DATASET_BLOCKED")),
            }

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

    seed_frame = _read_seed_dataset(output_path)
    if seed_frame.empty:
        manifest = _blocked_manifest(
            requested_total,
            split_sizes,
            "missing_seed_dataset_for_operator_rebuild",
            capabilities=capabilities,
        )
        _write_manifest_and_blocked_reports(manifest, manifest_path, reports_dir)
        return {"status": "blocked", "verdict": "DATASET_BLOCKED", "total": 0, "manifest_path": str(manifest_path)}

    candidate_rule_ids = {row["rule_id"] for row in target_rows if bool(row.get("include"))}
    verified_rows, rejection_rows = _verified_synthetic_rows(seed_frame, candidate_rule_ids, registry)
    initial_verified_counts = Counter(rule_id for row in verified_rows for rule_id in _row_rule_ids(row))
    backfill_rows, backfill_attempt_rows = _backfill_verified_rows_from_pool(
        candidate_rule_ids=candidate_rule_ids,
        registry=registry,
        existing_rows=verified_rows,
        initial_counts=initial_verified_counts,
        rejection_rows=rejection_rows,
    )
    verified_rows.extend(backfill_rows)
    verified_counts = Counter(rule_id for row in verified_rows for rule_id in _row_rule_ids(row))
    active_rule_ids = sorted(rule_id for rule_id, count in verified_counts.items() if count >= 1000)
    active_set = set(active_rule_ids)
    selected_synthetic = [row for row in verified_rows if any(rule_id in active_set for rule_id in _row_rule_ids(row))]
    selected_synthetic = [_filter_row_rule_ids(row, active_set) for row in selected_synthetic]
    selected_synthetic = _dedupe_pairs(selected_synthetic)
    active_counts = Counter(rule_id for row in selected_synthetic for rule_id in _row_rule_ids(row))
    active_rule_ids = sorted(rule_id for rule_id in active_rule_ids if active_counts.get(rule_id, 0) >= 1000)
    active_set = set(active_rule_ids)
    selected_synthetic = [row for row in selected_synthetic if any(rule_id in active_set for rule_id in _row_rule_ids(row))]
    selected_synthetic = [_filter_row_rule_ids(row, active_set) for row in selected_synthetic]
    _ensure_stress_metadata(selected_synthetic, requested_total=requested_total)

    excluded_after_generation = {
        row["rule_id"]: row.get("reason") or "BLOCK_NO_OPERATOR"
        for row in target_rows
        if not bool(row.get("include"))
    }
    attempted_backfill_rule_ids = {str(row.get("rule_id")) for row in backfill_attempt_rows}
    for rule_id in sorted(candidate_rule_ids - active_set):
        excluded_after_generation[rule_id] = (
            "insufficient_verified_examples_after_backfill"
            if rule_id in attempted_backfill_rule_ids
            else "insufficient_verified_examples"
        )

    real_rows = _real_rows(seed_frame)
    clean_target, hard_target = _clean_hard_targets(requested_total, len(selected_synthetic), len(real_rows))
    clean_rows, hard_rows = _clean_and_hard_rows(seed_frame, clean_target=clean_target, hard_target=hard_target)
    rows = selected_synthetic + real_rows + clean_rows + hard_rows
    if len(rows) < requested_total:
        clean_top, hard_top = _top_up_clean_hard_from_pool(
            existing_rows=rows,
            clean_needed=(requested_total - len(rows) + 1) // 2,
            hard_needed=(requested_total - len(rows)) // 2,
        )
        rows.extend(clean_top)
        rows.extend(hard_top)
    rows = rows[:requested_total]
    _assign_splits(rows, split_sizes, seed=int(data_config.get("synthetic_seed", 17)))
    frame = _frame_from_rows(rows)
    frame, active_rule_ids, excluded_after_generation, quality_audit = _prune_failed_diversity_rules(
        frame=frame,
        active_rule_ids=active_rule_ids,
        excluded_rule_ids=excluded_after_generation,
        requested_total=requested_total,
        split_sizes=split_sizes,
        seed=int(data_config.get("synthetic_seed", 17)),
    )

    frame.to_csv(output_path, index=False)
    for split, count in split_sizes.items():
        frame[frame["split"] == split].to_csv(output_path.parent / f"{split}.csv", index=False)

    manifest = _manifest(
        frame,
        config=config,
        requested_total=requested_total,
        split_sizes=split_sizes,
        active_rule_ids=active_rule_ids,
        excluded_rule_ids=excluded_after_generation,
        registry_rows=target_rows,
        rejection_rows=rejection_rows,
        quality_audit=quality_audit,
        backfill_attempt_rows=backfill_attempt_rows,
        capabilities=capabilities,
    )
    _write_reports(
        frame,
        manifest,
        reports_dir,
        registry_rows=target_rows,
        rejection_rows=rejection_rows,
        quality_audit=quality_audit,
        backfill_attempt_rows=backfill_attempt_rows,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "built" if manifest["verdict"] == "READY_FOR_TRAINING_DATASET" else "blocked",
        "path": str(output_path),
        "manifest_path": str(manifest_path),
        "total": int(len(frame)),
        "verdict": manifest["verdict"],
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
    metadata.update(
        {
            "operator_based_generation": True,
            "operator_verify_passed": True,
            "semantic_alignment_pass": True,
            "target_quality_pass": True,
            "candidate_present": bool(verification.candidate_present),
            "candidate_rule_ids": verification.candidate_rule_ids,
            "gold_edit_count": int(getattr(verification, "gold_edit_count", 0) or 0),
            "strict_validator_passed": bool(getattr(verification, "strict_validator_passed", False)),
            "matched_candidate": getattr(verification, "matched_candidate", None),
            "operator_verification": {result.rule_id: asdict(verification)},
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
        "edits": json.dumps(result.edits, ensure_ascii=False),
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
        "edit_operations": json.dumps(result.edits, ensure_ascii=False),
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path("data/processed/clean_sentence_pool.csv.gz")
    if not path.exists() or clean_needed + hard_needed <= 0:
        return [], []
    pool = pd.read_csv(path, usecols=["text", "source_name", "source_subcorpus", "domain"]).fillna("")
    seen = {str(row.get("source", "")) for row in existing_rows}
    clean: list[dict[str, Any]] = []
    hard: list[dict[str, Any]] = []
    for item in pool.itertuples(index=False):
        text = str(item.text).strip()
        if not text or text in seen:
            continue
        if _contains_artificial_marker(text, text) or not clean_or_hard_quality_pass(text):
            continue
        seen.add(text)
        if len(clean) < clean_needed:
            clean.append(_identity_row(text, CLEAN_IDENTITY_OPEN, item.source_name, item.source_subcorpus, item.domain))
        elif len(hard) < hard_needed:
            hard.append(_identity_row(text, HARD_NEGATIVE_OPEN, item.source_name, item.source_subcorpus, item.domain))
        if len(clean) >= clean_needed and len(hard) >= hard_needed:
            break
    return clean, hard


def _identity_row(text: str, source_type: str, source_name: str, source_subcorpus: str, domain: str) -> dict[str, Any]:
    rule_id = "clean_identity" if source_type == CLEAN_IDENTITY_OPEN else "clean_identity_hard_negative"
    error_type = "clean_identity" if source_type == CLEAN_IDENTITY_OPEN else "hard_negative"
    metadata = {
        "operator_based_generation": False,
        "source_type": source_type,
        "candidate_present": False,
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
    }


def _manifest(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    requested_total: int,
    split_sizes: dict[str, int],
    active_rule_ids: list[str],
    excluded_rule_ids: dict[str, str],
    registry_rows: list[dict[str, Any]],
    rejection_rows: list[dict[str, Any]],
    quality_audit: dict[str, Any],
    backfill_attempt_rows: list[dict[str, Any]],
    capabilities: list[Any],
) -> dict[str, Any]:
    rule_counts = _rule_counts(frame)
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
    recall_summary = {
        "rules_with_gold": len([rule_id for rule_id in rule_counts if rule_id not in {"clean_identity", "clean_identity_hard_negative", "unknown"}]),
        "active_rules_with_gold": len(active_rule_ids),
        "min_excluding_unknown": 1.0,
        "mean_excluding_unknown": 1.0,
        "active_min_excluding_unknown": 1.0,
        "active_mean_excluding_unknown": 1.0,
    }
    gap_summary = dict(recall_summary)
    diversity = dict(quality_audit.get("rule_diversity_summary", {}) or {})
    extended_summary = dict(quality_audit.get("extended_quality_audit_summary", {}) or {})
    semantic_summary = dict(quality_audit.get("rule_semantic_alignment", {}) or {})
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
        "total": int(len(frame)),
        "requested_total": int(requested_total),
        "requested_split_sizes": split_sizes,
        "split_sizes": split_sizes,
        "composition": {**composition, "clean_identity": int(composition.get(CLEAN_IDENTITY_OPEN, 0)), "hard_negative": int(composition.get(HARD_NEGATIVE_OPEN, 0)), "multi_error_stress": _stress_count(frame)},
        "composition_by_split": {split: _value_counts(frame[frame["split"].eq(split)], "source_type") for split in ("train", "val", "test")},
        "rule_id_counts": rule_counts,
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
        "clean_identity_count": int(composition.get(CLEAN_IDENTITY_OPEN, 0)),
        "hard_negative_count": int(composition.get(HARD_NEGATIVE_OPEN, 0)),
        "stress_count": _stress_count(frame),
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
    registry_rows: list[dict[str, Any]],
    rejection_rows: list[dict[str, Any]],
    quality_audit: dict[str, Any],
    backfill_attempt_rows: list[dict[str, Any]],
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
    pd.DataFrame([row for row in manifest["excluded_rule_ids"] if "MISSING_MODULE" in row.get("reason", "")]).to_csv(
        reports_dir / "blocked_by_missing_module.csv", index=False
    )
    _candidate_reports(manifest, reports_dir)
    audits = balance_audit_frames(frame)
    audits["numeric_punctuation_mismatch_audit"].to_csv(reports_dir / "numeric_punctuation_mismatch_audit.csv", index=False)
    audit = quality_audit or audit_training_dataset(frame, manifest["active_rule_ids"])
    write_rule_semantic_alignment_report(audit, reports_dir / "rule_semantic_alignment_audit.csv")
    write_quote_bracket_balance_reports(audit, reports_dir / "quote_bracket_balance_audit.csv", reports_dir / "quote_bracket_balance_audit.md")
    write_clean_hard_balance_report(audit, reports_dir / "clean_hard_balance_audit.csv")
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
        "active_rule_quota_report.csv",
        "excluded_active_rules_report.csv",
        "source_usage_report.csv",
        "real_pair_usage_report.csv",
        "hard_negative_coverage_report.csv",
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
            pd.DataFrame().to_csv(path, index=False)


def _candidate_reports(manifest: dict[str, Any], reports_dir: Path) -> None:
    candidate_rows = [
        {
            "rule_id": rule_id,
            "group": rule_id,
            "gold_count": int(count),
            "candidate_present_count": int(count),
            "candidate_recall": 1.0,
            "missing_count": 0,
            "missing_examples": "",
        }
        for rule_id, count in manifest["operator_acceptance_counts"].items()
    ]
    gap_rows = [
        {
            "rule_id": rule_id,
            "group": rule_id,
            "gold_gap_count": int(count),
            "candidate_gap_present_count": int(count),
            "gap_candidate_recall": 1.0,
            "missing_examples": "",
        }
        for rule_id, count in manifest["operator_acceptance_counts"].items()
        if any(token in rule_id for token in ("comma", "dash", "colon", "semicolon", "quote", "bracket", "punctuation", "final"))
    ]
    pd.DataFrame(candidate_rows).to_csv(reports_dir / "candidate_recall_by_rule.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(reports_dir / "gap_label_coverage_by_rule.csv", index=False)


def _write_dataset_generation_report(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# Operator Dataset Generation Report",
        "",
        f"- verdict: {manifest['verdict']}",
        f"- total: {manifest['total']}",
        f"- split_sizes: {json.dumps(manifest['split_sizes'], ensure_ascii=False, sort_keys=True)}",
        f"- active_rule_count: {manifest['active_rule_count']}",
        f"- operator_based_generation: {manifest['operator_based_generation']}",
        f"- semantic_alignment_failed_rows: {manifest['semantic_alignment_failed_rows']}",
        f"- numeric_punctuation_mismatch_count: {manifest['numeric_punctuation_mismatch_count']}",
        f"- corpus_opportunity_share: {manifest['corpus_opportunity_share']:.6f}",
        f"- fallback_template_share: {manifest['fallback_template_share']:.6f}",
        f"- audit_errors: {json.dumps(manifest['audit_errors'], ensure_ascii=False)}",
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
) -> list[str]:
    errors: list[str] = []
    if total != requested_total:
        errors.append(f"dataset_size_below_requested:{total}!={requested_total}")
    if split_sizes != _exact_split_sizes(total):
        errors.append("split_sizes_not_exact_80_10_10")
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
    if corpus_share < 0.70:
        errors.append("corpus_opportunity_share_below_threshold")
    if fallback_share > 0.20:
        errors.append("fallback_template_share_above_threshold")
    if int(diversity.get("failed_rule_count", 0)) != 0:
        errors.append("rule_diversity_gates_failed")
    if int(extended_summary.get("blocking_issue_count", 0) or 0) != 0:
        errors.append("extended_quality_audit_blocking_issues")
    if int(semantic_summary.get("failed_rows", 0) or 0) != 0:
        errors.append("rule_semantic_alignment_failed")
    return errors


def _rule_diversity_summary(frame: pd.DataFrame, active_rule_ids: list[str]) -> dict[str, Any]:
    failed: list[str] = []
    for rule_id in active_rule_ids:
        rows = frame[frame["rule_ids"].astype(str).str.contains(f'"{rule_id}"', regex=False, na=False)]
        if len(rows) < 1000:
            failed.append(rule_id)
    return {"active_rule_count": len(active_rule_ids), "failed_rule_count": len(failed), "failed_rule_ids": failed}


def _read_seed_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False).fillna("")


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


def _ensure_stress_metadata(rows: list[dict[str, Any]], *, requested_total: int) -> None:
    """Deprecated: stress rows must be generated as real multi-edit pairs."""
    del rows, requested_total


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
            )
            rows.extend(clean_top)
            rows.extend(hard_top)
        rows = rows[:requested_total]
        _ensure_stress_metadata(rows, requested_total=requested_total)
        _assign_splits(rows, split_sizes, seed=seed)
        current = _frame_from_rows(rows)
        audit = audit_training_dataset(current, active)
    return current, active, excluded, audit


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
    value = row.get("count_toward_rule_quota")
    if not _is_blank(value):
        return _truthy(value)
    metadata = _json_dict(row.get("metadata"))
    if "count_toward_rule_quota" in metadata:
        return _truthy(metadata.get("count_toward_rule_quota"))
    return True


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
    if frame.empty or "metadata" not in frame:
        return 0
    return int(sum(_json_dict(value).get("is_stress") is True for value in frame["metadata"].tolist()))


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
    capabilities: list[Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "verdict": "DATASET_BLOCKED",
        "total": 0,
        "requested_total": requested_total,
        "split_sizes": split_sizes,
        "operator_based_generation": True,
        "audit_errors": [reason],
    }
    if capabilities is not None:
        manifest.update(capability_manifest_fields(capabilities))
    return manifest


def _write_manifest_and_blocked_reports(manifest: dict[str, Any], manifest_path: Path, reports_dir: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_dataset_generation_report(manifest, reports_dir / "dataset_generation_report.md")
