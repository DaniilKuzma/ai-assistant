from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable

import pandas as pd
import yaml

from src.candidates.candidate_generator import CandidateGenerator
from src.candidates.matching import candidate_matches_edit
from src.data.clean_sentence_pool import (
    CleanSentencePoolResult,
    META_LANGUAGE_PATTERNS,
    build_clean_sentence_pool,
    normalize_template_text,
)
from src.data.real_error_sources import RealErrorLoadResult, load_real_error_pairs
from src.data.sage_sources import prepare_punctuation_jsonl_file, prepare_sage_jsonl_files
from src.data.synthetic_generator import (
    RejectedBackfillTemplate,
    SyntheticExample,
    SyntheticGenerator,
    TargetedBackfillExample,
    TargetedBackfillGenerator,
)
from src.data.training_quality_audit import (
    audit_training_dataset,
    artificial_marker_counts,
    write_extended_quality_reports,
    write_artificial_marker_reports,
    write_generation_strategy_report,
    write_known_quality_bugs_report,
    write_rule_diversity_report,
)
from src.evaluation.candidate_recall import (
    CANDIDATE_RECALL_COLUMNS,
    GAP_LABEL_COVERAGE_COLUMNS,
    build_candidate_recall_reports,
)
from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.rules.registry import rule_by_id
from src.rules.rule_ids import normalize_rule_id
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type


SYNTHETIC_OPEN_CLEAN = "synthetic_augmented_from_open_clean"
REAL_ERROR_PAIR = "real_error_pair"
CLEAN_IDENTITY_OPEN = "clean_identity_from_open_clean"
HARD_NEGATIVE_OPEN = "hard_negative_from_open_clean"
CORE_SOURCE_TYPES = (SYNTHETIC_OPEN_CLEAN, REAL_ERROR_PAIR, CLEAN_IDENTITY_OPEN, HARD_NEGATIVE_OPEN)
ARTIFICIAL_METKA_SUBSTRING = "\u043c\u0435\u0442\u043a\u0430"
ARTIFICIAL_LATER_EDITOR_PATTERNS = (
    "\u043f\u043e\u0437\u0436\u0435 \u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440 "
    "\u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b \u0437\u0430\u043f\u0438\u0441\u044c",
    "\u043f\u043e\u0437\u0436\u0435 \u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440 "
    "\u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b \u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
)
ARTIFICIAL_RANDOM_FILLER_RE = re.compile(
    r"(?:\b(?:\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0435|\u0444\u0430\u0439\u043b\u0435)\s+[\u0430-\u044f\u0451]{2}\.)|"
    r"(?:\b\u043e\u0442\u0447[\u0435\u0451]\u0442\u0435\s+[\u0430-\u044f\u0451]{2}\s+\u0432\u0441\u0442\u0440\u0435\u0442\u0438\u043b\u043e\u0441\u044c)|"
    r"(?:\b\u0437\u0430\u043f\u0438\u0441\u0438\s+[\u0430-\u044f\u0451]{2}\.)|"
    r"(?:\b\u043f\u0438\u0441\u044c\u043c\u0435\s+[\u0430-\u044f\u0451]{2}\s+\u0431\u044b\u043b\u0430)|"
    r"(?:\b\u0437\u0430\u044f\u0432\u043b\u0435\u043d\u0438\u0438\s+[\u0430-\u044f\u0451]{2}\s+\u0443\u043a\u0430\u0437\u0430\u043b\u0438)|"
    r"(?:\u043f\u043e\u0437\u0436\u0435\s+\u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440\s+"
    r"\u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b\s+"
    r"(?:\u0437\u0430\u043f\u0438\u0441\u044c|\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b)\s+"
    r"[\u0430-\u044f\u0451]{2}\b)",
    re.IGNORECASE,
)
MAX_SYNTHETIC_DIFF_CHARS = 320
MAX_SYNTHETIC_DIFF_WORDS = 55
SOURCE_TYPE_ALIASES = {
    "synthetic_augmented": SYNTHETIC_OPEN_CLEAN,
    SYNTHETIC_OPEN_CLEAN: SYNTHETIC_OPEN_CLEAN,
    "real_error_pair": REAL_ERROR_PAIR,
    "clean_identity": CLEAN_IDENTITY_OPEN,
    CLEAN_IDENTITY_OPEN: CLEAN_IDENTITY_OPEN,
    "hard_negative": HARD_NEGATIVE_OPEN,
    HARD_NEGATIVE_OPEN: HARD_NEGATIVE_OPEN,
}
CORE_COLUMNS = [
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
SUSPICIOUS_PHRASES = (
    "проверяет семейство",
    "готовит важный примере",
    "готовит итоговый примере",
    "готовит точный примере",
    "готовит рабочий примере",
)
TECHNICAL_RULE_MARKERS = ("context-pairs", "ne-pos", "n-nn")
SHORT_BROAD_EXCLUDED_RULE_IDS = frozenset(
    {
        "quote_open",
        "quote_close",
        "quote_pair_balance",
        "bracket_pair_balance",
        "quotes_brackets",
        "semicolon",
        "delete_replace",
        "punctuation_delete_replace",
        "punctuation_noise",
        "yo_e_candidate",
        "capitalization_sentence_start",
        "capitalization_ner",
        "abbreviation_case_protection",
        "neural_punctuation",
    }
)
DEFAULT_CAPPED_RULE_IDS = frozenset(
    {
        "n_nn_adjective",
        "n_nn_short_form",
        "final_punctuation_default",
        "pattern_чо_че",
        "prefix_pre_pri",
        "pattern_цы_ци",
        "pattern_жо_же",
    }
)
ACTIVE_SYNTHETIC_STATUSES = frozenset(
    {
        "implemented",
        "partial",
        "deterministic",
        "candidate_only",
        "model_required",
        "dictionary_model_required",
        "syntax_required",
    }
)
_CURRENT_ACTIVE_RULE_IDS: set[str] | None = None


def _progress(stage: str, **payload: Any) -> None:
    if os.environ.get("RUSSIAN_CORRECTOR_DATASET_PROGRESS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    event = {"stage": stage, **payload}
    print("[dataset-build] " + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def build_training_dataset_core_from_config(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    data_config = config.get("data", {})
    core_config = data_config.get("training_dataset_core", {}) or {}
    seed = int(data_config.get("synthetic_seed", core_config.get("seed", 17)))
    output_path = Path(data_config.get("processed_train_path") or "data/processed/correction_dataset.csv.gz")
    output_dir = output_path.parent
    reports_dir = Path(config.get("paths", {}).get("reports_dir") or "reports")
    manifest_path = Path(data_config.get("manifest_path") or reports_dir / "dataset_manifest.json")
    split_sizes = _split_sizes(data_config)
    total = sum(split_sizes.values())
    if total <= 0:
        total = int(data_config.get("target_total_examples", 60_000))
        split_sizes = {"train": 50_000, "val": 5_000, "test": 5_000} if total == 60_000 else _ratio_split(total)
    requested_total = total
    requested_split_sizes = dict(split_sizes)
    source_targets = _source_type_targets(core_config, total)
    _progress(
        "start",
        force=bool(force),
        output_path=str(output_path),
        manifest_path=str(manifest_path),
        requested_total=int(requested_total),
        split_sizes=requested_split_sizes,
    )

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path)
        if len(existing) >= total and manifest_path.exists():
            return {
                "status": "exists",
                "path": str(output_path),
                "manifest_path": str(manifest_path),
                "total": int(len(existing)),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    clean_config = _clean_source_config(config, core_config)
    real_config = _real_source_config(config, core_config)
    clean_pool_path = output_dir / "clean_sentence_pool.csv.gz"
    source_precheck = _precheck_external_sources(clean_config, real_config, reports_dir=reports_dir)
    if not source_precheck["ready"]:
        manifest = _blocked_missing_sources_manifest(
            config=config,
            requested_total=requested_total,
            requested_split_sizes=requested_split_sizes,
            source_precheck=source_precheck,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_blocked_generation_report(reports_dir / "dataset_generation_report.md", manifest)
        return {
            "status": "blocked",
            "path": str(output_path),
            "manifest_path": str(manifest_path),
            "total": 0,
            "composition": {},
            "splits": {},
            "verdict": "BLOCKED_BY_MISSING_EXTERNAL_SOURCES",
        }

    generator = SyntheticGenerator(seed=seed, max_errors_per_sentence=int(core_config.get("max_errors_per_sentence", 2)))
    candidate_generator = CandidateGenerator.from_config(config)

    if bool(core_config.get("reuse_clean_sentence_pool_cache", True)) and clean_pool_path.exists():
        clean_rows = _read_records(clean_pool_path)
        clean_result = _clean_result_from_cache(clean_rows, clean_pool_path, core_config=core_config)
    else:
        _progress("clean_pool_build_start", output_path=str(clean_pool_path))
        clean_result = build_clean_sentence_pool(
            clean_config,
            output_path=clean_pool_path,
            reports_dir=reports_dir,
        )
        clean_rows = _read_records(clean_pool_path)
    _progress("clean_pool_ready", rows=len(clean_rows), path=str(clean_pool_path))

    real_output_path = output_dir / "real_error_pairs_validated.csv.gz"
    _progress("real_pairs_load_start", output_path=str(real_output_path))
    real_result = _load_or_reuse_real_error_pairs(
        real_config,
        candidate_generator=candidate_generator,
        output_path=real_output_path,
        reports_dir=reports_dir,
        core_config=core_config,
    )
    _progress("real_pairs_ready", rows=len(real_result.rows), output_path=str(real_output_path))

    rows: list[dict[str, Any]] = []
    used_clean_hashes: set[str] = set()
    real_target = int(source_targets.get(REAL_ERROR_PAIR, 0))
    rows.extend(_real_rows(real_result.rows[:real_target]))
    accepted_real = len([row for row in rows if row["source_type"] == REAL_ERROR_PAIR])
    real_shortage = max(0, real_target - accepted_real)
    source_targets[SYNTHETIC_OPEN_CLEAN] = max(
        0,
        total
        - accepted_real
        - int(source_targets.get(CLEAN_IDENTITY_OPEN, 0))
        - int(source_targets.get(HARD_NEGATIVE_OPEN, 0)),
    )
    source_targets[REAL_ERROR_PAIR] = accepted_real

    strict_clean_rows = _strict_clean_rows(clean_rows)
    quota_config = _active_rule_quota_config(core_config)
    cap_config = _rule_cap_config(core_config, split_sizes)
    active_rule_ids = _effective_active_rule_ids(config, core_config, quota_config=quota_config)
    global _CURRENT_ACTIVE_RULE_IDS
    _CURRENT_ACTIVE_RULE_IDS = set(active_rule_ids)
    _progress("active_rules_ready", active_rule_count=len(active_rule_ids))
    stress_target = int(core_config.get("multi_error_stress_target", 0) or 0)
    corpus_budget = max(0, int(source_targets.get(SYNTHETIC_OPEN_CLEAN, 0)) - stress_target)
    _progress("corpus_opportunity_start", budget=corpus_budget, clean_rows=len(strict_clean_rows))
    corpus_rows, corpus_state = _corpus_opportunity_rows_for_active_rules(
        existing_rows=rows,
        clean_rows=strict_clean_rows,
        active_rule_ids=active_rule_ids,
        quota_config=quota_config,
        cap_config=cap_config,
        seed=seed,
        synthetic_budget=corpus_budget,
    )
    rows.extend(corpus_rows)
    _progress("corpus_opportunity_done", rows=len(corpus_rows), total_rows=len(rows), state=corpus_state)
    _progress("quota_backfill_start", synthetic_budget=int(source_targets.get(SYNTHETIC_OPEN_CLEAN, 0)))
    current_synthetic_count = len([row for row in rows if row["source_type"] == SYNTHETIC_OPEN_CLEAN])
    quota_budget = max(0, int(source_targets.get(SYNTHETIC_OPEN_CLEAN, 0)) - current_synthetic_count - stress_target)
    quota_rows, quota_state = _build_active_rule_quota_rows(
        rows,
        clean_rows=strict_clean_rows,
        active_rule_ids=active_rule_ids,
        quota_config=quota_config,
        cap_config=cap_config,
        candidate_generator=candidate_generator,
        seed=seed,
        synthetic_budget=quota_budget,
    )
    quota_state["corpus_opportunity_rows"] = len(corpus_rows)
    quota_state["corpus_opportunity_state"] = corpus_state
    rows.extend(quota_rows)
    _progress("quota_backfill_done", rows=len(quota_rows), total_rows=len(rows))
    if stress_target > 0:
        _progress("stress_start", target=stress_target)
        stress_rows = _multi_error_stress_rows(
            existing_rows=rows,
            clean_rows=strict_clean_rows,
            target_count=min(stress_target, max(0, int(source_targets.get(SYNTHETIC_OPEN_CLEAN, 0)) - len([row for row in rows if row["source_type"] == SYNTHETIC_OPEN_CLEAN]))),
            seed=seed + 71,
            cap_config=cap_config,
        )
        rows.extend(stress_rows)
        _progress("stress_done", rows=len(stress_rows), total_rows=len(rows))

    synthetic_target = int(source_targets.get(SYNTHETIC_OPEN_CLEAN, 0))
    synthetic_remaining = max(0, synthetic_target - len([row for row in rows if row["source_type"] == SYNTHETIC_OPEN_CLEAN]))
    general_synthetic_target = min(synthetic_remaining, int(core_config.get("max_general_synthetic_fill", 6000)))
    _progress("general_synthetic_start", target=general_synthetic_target, synthetic_remaining=synthetic_remaining)
    rule_cap_counts = Counter(_rule_counts_from_rows(rows))
    error_cap_counts = Counter(_error_counts_from_rows(rows))
    synthetic_rows = _synthetic_rows_from_clean_pool(
        strict_clean_rows,
        target_count=general_synthetic_target,
        generator=generator,
        seed=seed,
        used_clean_hashes=used_clean_hashes,
        rule_cap_counts=rule_cap_counts,
        error_cap_counts=error_cap_counts,
        max_rule_total=int(cap_config["generation_rule_cap"]),
        max_error_total=int(cap_config["generation_error_type_cap"]),
        rule_max_totals=cap_config.get("rule_max_totals", {}),
    )
    rows.extend(synthetic_rows)
    _progress("general_synthetic_done", rows=len(synthetic_rows), total_rows=len(rows))
    synthetic_remaining = max(0, synthetic_target - len([row for row in rows if row["source_type"] == SYNTHETIC_OPEN_CLEAN]))
    if synthetic_remaining:
        fallback_share_max = float((core_config.get("audit", {}) or {}).get("fallback_template_share_max", 0.20))
        fallback_budget = _remaining_fallback_template_budget(rows, max_share=fallback_share_max)
        targeted_fill_target = min(synthetic_remaining, fallback_budget)
        _progress(
            "targeted_fill_start",
            target=targeted_fill_target,
            requested=synthetic_remaining,
            fallback_budget=fallback_budget,
            fallback_share_max=fallback_share_max,
        )
    if synthetic_remaining and targeted_fill_target > 0:
        fill_rows, fill_state = _fill_remaining_with_targeted_rows(
            clean_rows=strict_clean_rows,
            active_rule_ids=active_rule_ids,
            target_count=targeted_fill_target,
            candidate_generator=candidate_generator,
            seed=seed + 101,
            existing_rows=rows,
            cap_config=cap_config,
        )
        rows.extend(fill_rows)
        quota_state["rejected_templates"].extend(fill_state["rejected_templates"])
        _progress("targeted_fill_done", rows=len(fill_rows), total_rows=len(rows))
    elif synthetic_remaining:
        _progress("targeted_fill_skipped", requested=synthetic_remaining, reason="fallback_share_budget_exhausted")

    clean_identity_target = int(source_targets.get(CLEAN_IDENTITY_OPEN, 0))
    _progress("clean_identity_start", target=clean_identity_target)
    rows.extend(
        _identity_rows_from_clean_pool(
            strict_clean_rows,
            target_count=clean_identity_target,
            source_type=CLEAN_IDENTITY_OPEN,
            used_clean_hashes=used_clean_hashes,
        )
    )
    _progress("clean_identity_done", total_rows=len(rows))

    hard_negative_target = int(source_targets.get(HARD_NEGATIVE_OPEN, 0))
    _progress("hard_negative_start", target=hard_negative_target)
    rows.extend(
        _hard_negative_rows_from_clean_pool(
            clean_rows,
            target_count=hard_negative_target,
            used_clean_hashes=used_clean_hashes,
        )
    )
    _progress("hard_negative_done", total_rows=len(rows))

    before_marker_filter = len(rows)
    rows = [row for row in rows if not _row_contains_artificial_marker(row)]
    _progress("artificial_marker_filter_done", removed=before_marker_filter - len(rows), total_rows=len(rows))

    top_up_needed = max(0, total - len(rows))
    if top_up_needed:
        _progress("safe_clean_hard_top_up_start", target=top_up_needed, total_rows=len(rows))
        top_up_rows = _safe_clean_hard_top_up_rows(
            rows,
            clean_rows=strict_clean_rows,
            target_total=total,
            used_clean_hashes=used_clean_hashes,
        )
        rows.extend(top_up_rows)
        _progress("safe_clean_hard_top_up_done", rows=len(top_up_rows), total_rows=len(rows))

    shortage_errors = _target_shortage_errors(rows, _quality_source_minimum_targets(source_targets, total))
    rows = rows[:total]
    _attach_template_fields(rows)
    effective_split_sizes = split_sizes if len(rows) == total else _proportional_targets(len(rows), split_sizes)
    split_source_targets = _actual_split_source_targets(core_config, effective_split_sizes, _source_counts_from_rows(rows))
    split_core_config = dict(core_config) if len(rows) == total else {}
    if split_source_targets:
        split_core_config["split_source_type_targets"] = split_source_targets
    _assign_core_splits(rows, effective_split_sizes, core_config=split_core_config, seed=seed)
    _attach_template_fields(rows)

    frame = pd.DataFrame(rows, columns=CORE_COLUMNS)
    _progress("write_dataset_start", rows=len(frame), output_path=str(output_path))
    frame.to_csv(output_path, index=False)
    for split in ("train", "val", "test"):
        frame[frame["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)
    _progress("write_dataset_done", rows=len(frame))

    _progress("recall_reports_start", rows=len(frame))
    recall_reports = build_candidate_recall_reports(
        frame.to_dict("records"),
        candidate_generator=candidate_generator,
        rules_config_path="configs/rules.yaml",
    )
    recall_reports["candidate_recall_by_rule"].to_csv(reports_dir / "candidate_recall_by_rule.csv", index=False)
    recall_reports["gap_label_coverage_by_rule"].to_csv(reports_dir / "gap_label_coverage_by_rule.csv", index=False)
    _progress("recall_reports_done")
    _write_balance_reports(frame, reports_dir)
    _write_source_usage_report(frame, reports_dir / "source_usage_report.csv")
    _write_real_pair_usage_report(frame, reports_dir / "real_pair_usage_report.csv", real_result=real_result, real_target=real_target)
    _write_hard_negative_coverage_report(frame, reports_dir / "hard_negative_coverage_report.csv")
    quota_state = _finalize_quota_state(frame, quota_state, quota_config=quota_config, cap_config=cap_config)
    _write_active_rule_quota_report(quota_state["quota_rows"], reports_dir / "active_rule_quota_report.csv")
    _write_excluded_active_rules_report(quota_state["excluded_rows"], reports_dir / "excluded_active_rules_report.csv")
    _write_rejected_backfill_templates(quota_state["rejected_templates"], reports_dir / "rejected_backfill_templates.csv")
    template_leakage = _write_template_leakage_report(frame, reports_dir / "template_leakage_report.csv")
    template_quality = _write_template_quality_report(frame, reports_dir / "template_quality_report.md")
    _progress("quality_audit_start")
    quality_audit = audit_training_dataset(frame, quota_state["active_rule_ids"])
    write_generation_strategy_report(quality_audit, reports_dir / "generation_strategy_report.csv")
    write_rule_diversity_report(quality_audit, reports_dir / "rule_diversity_report.csv")
    write_extended_quality_reports(
        quality_audit,
        reports_dir / "extended_quality_audit.csv",
        reports_dir / "extended_quality_audit.md",
    )
    write_artificial_marker_reports(
        quality_audit,
        reports_dir / "artificial_marker_audit.csv",
        reports_dir / "artificial_marker_audit.md",
    )
    write_known_quality_bugs_report(quality_audit, reports_dir / "known_quality_bugs_report.md")
    _progress("quality_audit_done", artificial_marker_counts=quality_audit.get("artificial_marker_counts", {}))

    manifest = _manifest(
        frame,
        config=config,
        core_config=core_config,
        clean_result=clean_result,
        real_result=real_result,
        recall_reports=recall_reports,
        template_leakage=template_leakage,
        template_quality=template_quality,
        real_target=real_target,
        real_shortage=real_shortage,
        shortage_errors=shortage_errors,
        requested_total=requested_total,
        requested_split_sizes=requested_split_sizes,
        reports_dir=reports_dir,
        quota_state=quota_state,
        quality_audit=quality_audit,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_generation_report(frame, reports_dir / "dataset_generation_report.md", manifest)
    _progress("done", verdict=manifest["verdict"], total=len(frame), audit_errors=manifest.get("audit_errors", []))

    return {
        "status": "built",
        "path": str(output_path),
        "manifest_path": str(manifest_path),
        "total": int(len(frame)),
        "composition": manifest["composition"],
        "splits": manifest["split_sizes"],
        "verdict": manifest["verdict"],
    }


def template_id_for_pair(source: str, target: str) -> str:
    normalized = _normalized_pair(source, target, entity_normalize=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def normalized_pair_hash(source: str, target: str) -> str:
    normalized = _normalized_pair(source, target, entity_normalize=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def assign_template_disjoint_splits(rows: list[dict[str, Any]], split_sizes: dict[str, int], seed: int = 17) -> None:
    for row in rows:
        row["template_id"] = template_id_for_pair(str(row.get("source", "")), str(row.get("target", "")))
        row["normalized_pair_hash"] = normalized_pair_hash(str(row.get("source", "")), str(row.get("target", "")))
        row["split"] = ""
    units = _template_units(rows, seed)
    remaining = {split: int(split_sizes.get(split, 0)) for split in ("train", "val", "test")}
    for unit in units:
        placed = False
        for split in _preferred_splits(unit, remaining):
            if len(unit) <= remaining[split]:
                for row in unit:
                    row["split"] = split
                remaining[split] -= len(unit)
                placed = True
                break
        if not placed:
            raise ValueError(f"template-disjoint split cannot fit unit of size {len(unit)} into remaining {remaining}")
    if any(value != 0 for value in remaining.values()):
        raise ValueError(f"template-disjoint split size mismatch: {remaining}")


def _clean_source_config(config: dict[str, Any], core_config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(core_config.get("open_corpora_sources"), dict):
        source_config = dict(core_config["open_corpora_sources"])
    else:
        path = Path(str(core_config.get("open_corpora_sources_path") or "configs/open_corpora_sources.yaml"))
        with path.open("r", encoding="utf-8") as handle:
            source_config = yaml.safe_load(handle) or {}
    pool = dict(source_config.get("pool", {}) or {})
    pool.update(dict(core_config.get("pool", {}) or {}))
    source_config["pool"] = pool
    return source_config


def _real_source_config(config: dict[str, Any], core_config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(core_config.get("real_error_sources"), dict):
        return dict(core_config["real_error_sources"])
    path = Path(str(core_config.get("real_error_sources_path") or "configs/real_error_sources.yaml"))
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _precheck_external_sources(clean_config: dict[str, Any], real_config: dict[str, Any], *, reports_dir: Path) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    downloads_allowed = _downloads_allowed(clean_config) or _downloads_allowed(real_config)
    if downloads_allowed:
        _prepare_materialized_real_sources(real_config)
    clean_missing = _missing_source_specs(clean_config, key="clean_sources")
    real_missing = _missing_source_specs(real_config, key="real_sources")
    missing.extend(clean_missing)
    missing.extend(real_missing)
    if missing:
        pd.DataFrame(missing).to_csv(reports_dir / "missing_external_sources.csv", index=False)
    clean_policy = dict(clean_config.get("sources", {}).get("download_policy", {}) or clean_config.get("download_policy", {}) or {})
    clean_pool_policy = dict(clean_config.get("pool", {}) or {})
    clean_missing_blocks = bool(clean_policy.get("fail_if_insufficient_sources", clean_pool_policy.get("fail_if_insufficient_sources", False)))
    blocking_missing = list(real_missing) + (list(clean_missing) if clean_missing_blocks else [])
    return {"ready": not blocking_missing, "missing": missing, "blocking_missing": blocking_missing, "downloads_allowed": downloads_allowed}


def _clean_result_from_cache(clean_rows: list[dict[str, Any]], output_path: Path, *, core_config: dict[str, Any]) -> CleanSentencePoolResult:
    source_counts = Counter(str(row.get("source_name") or row.get("source_corpus") or "cached_clean_pool") for row in clean_rows)
    subcorpus_counts = Counter(str(row.get("source_subcorpus") or row.get("source_name") or "cached_clean_pool") for row in clean_rows)
    min_clean = int(core_config.get("min_clean_pool_for_ready", 300_000))
    shortage = "" if len(clean_rows) >= min_clean else f"accepted_clean_sentences_below_min:{len(clean_rows)}<{min_clean}"
    return CleanSentencePoolResult(
        accepted_count=len(clean_rows),
        total_seen=len(clean_rows),
        output_path=str(output_path),
        source_counts=dict(sorted(source_counts.items())),
        subcorpus_counts=dict(sorted(subcorpus_counts.items())),
        rejection_reason_counts={},
        source_reports=[
            {
                "source_name": "clean_sentence_pool_cache",
                "status": "loaded",
                "reason": "reused_existing_clean_sentence_pool",
                "accepted": len(clean_rows),
            }
        ],
        dominance_violations=[],
        shortage_reason=shortage,
    )


def _downloads_allowed(config: dict[str, Any]) -> bool:
    policy = dict(config.get("download_policy", {}) or config.get("sources", {}).get("download_policy", {}) or {})
    env_name = str(policy.get("allow_downloads_env") or "RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS")
    return os.environ.get(env_name, "").strip().lower() in {"1", "true", "yes", "on"}


def _prepare_materialized_real_sources(real_config: dict[str, Any]) -> None:
    specs = _source_specs_from_config(real_config, "real_sources")
    sage_paths = [
        Path(str(spec.get("local_path") or ""))
        for _name, spec in specs
        if str(spec.get("type") or "") == "sage_hf_or_local" and spec.get("local_path")
    ]
    if sage_paths and not all(path.exists() and path.stat().st_size > 0 for path in sage_paths):
        prepare_sage_jsonl_files(output_dir=sage_paths[0].parent)
    for _name, spec in specs:
        if str(spec.get("type") or "") != "local_jsonl":
            continue
        if str(spec.get("hf_id") or "") != "ai-forever/spellcheck_punctuation_benchmark":
            continue
        path = Path(str(spec.get("local_path") or ""))
        if not path.exists() or path.stat().st_size <= 0:
            prepare_punctuation_jsonl_file(output_path=path)


def _missing_source_specs(config: dict[str, Any], *, key: str) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    downloads_allowed = _downloads_allowed(config)
    for source_name, spec in _source_specs_from_config(config, key):
        if spec.get("enabled") is False:
            continue
        source_type = str(spec.get("type") or "local_jsonl")
        local_path = Path(str(spec.get("local_path") or spec.get("path") or "")) if spec.get("local_path") or spec.get("path") else None
        archive_path = Path(str(spec.get("archive_path") or "")) if spec.get("archive_path") else None
        if any(path and path.exists() and (path.is_dir() or path.stat().st_size > 0) for path in (local_path, archive_path)):
            continue
        if source_type in {"hf_dataset", "huggingface_dataset"} and _has_materialized_sage_sources(config):
            continue
        can_download = bool(spec.get("url") or spec.get("hf_id") or source_type == "sage_hf_or_local") and downloads_allowed
        if can_download:
            continue
        missing.append(
            {
                "source_name": source_name,
                "type": source_type,
                "local_path": str(local_path or ""),
                "archive_path": str(archive_path or ""),
                "reason": "downloads_disabled" if (spec.get("url") or spec.get("hf_id")) and not downloads_allowed else "missing_local_path",
            }
        )
    return missing


def _source_specs_from_config(config: dict[str, Any], key: str) -> list[tuple[str, dict[str, Any]]]:
    raw = config.get(key, config.get("sources", []))
    if isinstance(raw, dict):
        return [(str(name), dict(spec or {})) for name, spec in raw.items()]
    if isinstance(raw, list):
        return [(str(spec.get("name") or f"source_{index}"), dict(spec)) for index, spec in enumerate(raw) if isinstance(spec, dict)]
    return []


def _has_materialized_sage_sources(config: dict[str, Any]) -> bool:
    paths = [
        Path(str(spec.get("local_path") or ""))
        for _name, spec in _source_specs_from_config(config, "real_sources")
        if str(spec.get("type") or "") == "sage_hf_or_local" and spec.get("local_path")
    ]
    return bool(paths) and all(path.exists() and path.stat().st_size > 0 for path in paths)


def _blocked_missing_sources_manifest(
    *,
    config: dict[str, Any],
    requested_total: int,
    requested_split_sizes: dict[str, int],
    source_precheck: dict[str, Any],
) -> dict[str, Any]:
    return {
        "total": 0,
        "requested_total": int(requested_total),
        "requested_split_sizes": requested_split_sizes,
        "split_sizes": {"train": 0, "val": 0, "test": 0},
        "composition": {},
        "composition_by_split": {"train": {}, "val": {}, "test": {}},
        "source_type_counts_by_split": {"train": {}, "val": {}, "test": {}},
        "missing_external_sources": source_precheck["missing"],
        "audit_errors": ["missing_external_sources"],
        "verdict": "BLOCKED_BY_MISSING_EXTERNAL_SOURCES",
        "config_path": str(config.get("data", {}).get("config_path", "configs/config.yaml")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_blocked_generation_report(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Dataset Generation Report",
        "",
        f"- verdict: {manifest['verdict']}",
        f"- total: {manifest['total']}",
        "",
        "## Missing External Sources",
        "",
    ]
    for item in manifest.get("missing_external_sources", []):
        lines.append(f"- {item.get('source_name')}: {item.get('reason')} ({item.get('local_path') or item.get('archive_path')})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _active_rule_quota_config(core_config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(core_config.get("active_rule_quota", {}) or {})
    audit = dict(core_config.get("audit", {}) or {})
    if raw.get("enabled") is False or int(audit.get("min_active_rule_count", 1)) <= 0:
        raw["enabled"] = False
        raw["min_total_per_active_rule"] = 0
        raw["preferred_total_per_active_rule"] = 0
        raw["split_minimums"] = {}
        return raw
    raw.setdefault("enabled", True)
    raw.setdefault("min_total_per_active_rule", max(80, int(audit.get("min_active_rule_count", 80))))
    raw.setdefault("preferred_total_per_active_rule", max(200, int(raw["min_total_per_active_rule"])))
    raw.setdefault("split_minimums", {"train": 60, "val": 10, "test": 10})
    return raw


def _rule_cap_config(core_config: dict[str, Any], split_sizes: dict[str, int]) -> dict[str, Any]:
    raw = dict(core_config.get("rule_caps", {}) or {})
    max_total = int(raw.get("max_total_per_rule_id", 2500))
    max_train = int(raw.get("max_train_per_rule_id", 2000))
    max_error_share = float(raw.get("max_error_type_share_train", 0.35))
    rule_max_totals = {
        str(rule_id): int(value)
        for rule_id, value in dict(raw.get("rule_max_totals", {}) or {}).items()
        if int(value) > 0
    }
    quota_by_rule = dict((core_config.get("active_rule_quota", {}) or {}).get("rule_quotas", {}) or {})
    for rule_id, quota in quota_by_rule.items():
        if isinstance(quota, dict) and int(quota.get("max_total", 0) or 0) > 0:
            rule_max_totals[str(rule_id)] = int(quota["max_total"])
    return {
        "max_total_per_rule_id": max_total,
        "max_train_per_rule_id": max_train,
        "max_rule_share_train": float(raw.get("max_rule_share_train", 0.10)),
        "max_error_type_share_train": max_error_share,
        "generation_rule_cap": min(max_total, max_train),
        "stress_generation_rule_cap": int(raw.get("stress_generation_rule_cap", max(min(max_total, max_train) * 2, 5000))),
        "generation_error_type_cap": max(1, int(split_sizes.get("train", 0) * max_error_share)),
        "rule_max_totals": rule_max_totals,
        "targeted_duplicate_cap": int(raw.get("targeted_duplicate_cap", 2)),
    }


def _effective_active_rule_ids(
    config: dict[str, Any],
    core_config: dict[str, Any],
    *,
    quota_config: dict[str, Any],
) -> list[str]:
    configured = quota_config.get("rule_ids")
    if isinstance(configured, list) and configured:
        return sorted(str(rule_id) for rule_id in configured if rule_by_id(str(rule_id)) is not None)
    coverage = load_rules_coverage()
    disabled = set(str(rule_id) for rule_id in coverage.get("synthetic_generation", {}).get("disabled", {}).keys())
    disabled.update(str(rule_id) for rule_id in config.get("synthetic_generation", {}).get("disabled", {}).keys())
    excluded = set(SHORT_BROAD_EXCLUDED_RULE_IDS) | disabled
    if bool(core_config.get("include_yo_e_candidate", False)) and bool(config.get("dictionary", {}).get("yo_e", {}).get("enabled", False)):
        excluded.discard("yo_e_candidate")
    result: set[str] = set()
    for _domain, _group, entry in iter_coverage_entries(coverage):
        status = str(entry.get("status") or "")
        if status not in ACTIVE_SYNTHETIC_STATUSES:
            continue
        for raw_rule_id in entry.get("rules", []):
            rule_id = normalize_rule_id(raw_rule_id)
            if rule_id in excluded or rule_by_id(rule_id) is None:
                continue
            result.add(rule_id)
    return sorted(result)


def _split_sizes(data_config: dict[str, Any]) -> dict[str, int]:
    keys = {"train": "train_examples", "val": "val_examples", "test": "test_examples"}
    if any(key in data_config for key in keys.values()):
        return {split: int(data_config.get(key, 0)) for split, key in keys.items()}
    raw = data_config.get("exact_split_sizes")
    if isinstance(raw, dict):
        return {split: int(raw.get(split, 0)) for split in ("train", "val", "test")}
    return {}


def _ratio_split(total: int) -> dict[str, int]:
    val = int(round(total * 0.0833333333))
    test = int(round(total * 0.0833333333))
    return {"train": total - val - test, "val": val, "test": test}


def _source_type_targets(core_config: dict[str, Any], total: int) -> dict[str, int]:
    raw = dict(core_config.get("source_type_targets", {}) or {})
    if not raw and total == 60_000:
        raw = {
            SYNTHETIC_OPEN_CLEAN: 38_400,
            REAL_ERROR_PAIR: 6_000,
            CLEAN_IDENTITY_OPEN: 7_800,
            HARD_NEGATIVE_OPEN: 7_800,
        }
    if not raw:
        raw = {
            SYNTHETIC_OPEN_CLEAN: int(round(total * 0.64)),
            REAL_ERROR_PAIR: int(round(total * 0.10)),
            CLEAN_IDENTITY_OPEN: int(round(total * 0.13)),
        }
        raw[HARD_NEGATIVE_OPEN] = total - sum(raw.values())
    normalized_raw: dict[str, int] = {}
    for source_type, value in raw.items():
        normalized = SOURCE_TYPE_ALIASES.get(str(source_type), str(source_type))
        normalized_raw[normalized] = normalized_raw.get(normalized, 0) + int(value)
    result = {source_type: int(normalized_raw.get(source_type, 0)) for source_type in CORE_SOURCE_TYPES}
    delta = total - sum(result.values())
    result[SYNTHETIC_OPEN_CLEAN] += delta
    return result


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict("records")


def _build_construction_backed_recall_reports(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rule_groups = _rule_group_map()
    gold_counter: Counter[str] = Counter()
    present_counter: Counter[str] = Counter()
    gap_counter: Counter[str] = Counter()
    present_gap_counter: Counter[str] = Counter()
    for _idx, row in frame.iterrows():
        candidate_present = _row_candidate_present(row)
        for edit in _json_list(row.get("edits") or row.get("edit_operations")):
            if not isinstance(edit, dict):
                continue
            rule_id = normalize_rule_id(str(edit.get("rule_id") or "unknown"))
            if rule_id == "unknown":
                continue
            gold_counter[rule_id] += 1
            if candidate_present:
                present_counter[rule_id] += 1
            if str(edit.get("edit_type") or "") in {
                "punctuation_insert",
                "punctuation_delete",
                "punctuation_replace",
                "final_punctuation",
            }:
                gap_counter[rule_id] += 1
                if candidate_present:
                    present_gap_counter[rule_id] += 1
    candidate_rows = [
        {
            "rule_id": rule_id,
            "group": rule_groups.get(rule_id, "unknown"),
            "gold_count": gold_counter[rule_id],
            "candidate_present_count": present_counter[rule_id],
            "candidate_recall": _safe_rate(present_counter[rule_id], gold_counter[rule_id]),
            "missing_count": max(0, gold_counter[rule_id] - present_counter[rule_id]),
            "missing_examples": "[]",
        }
        for rule_id in sorted(gold_counter)
    ]
    gap_rows = [
        {
            "rule_id": rule_id,
            "group": rule_groups.get(rule_id, "unknown"),
            "gold_gap_count": gap_counter[rule_id],
            "candidate_gap_present_count": present_gap_counter[rule_id],
            "gap_candidate_recall": _safe_rate(present_gap_counter[rule_id], gap_counter[rule_id]),
            "missing_examples": "[]",
        }
        for rule_id in sorted(gap_counter)
    ]
    return {
        "candidate_recall_by_rule": pd.DataFrame(candidate_rows, columns=CANDIDATE_RECALL_COLUMNS),
        "gap_label_coverage_by_rule": pd.DataFrame(gap_rows, columns=GAP_LABEL_COVERAGE_COLUMNS),
    }


def _row_candidate_present(row: pd.Series) -> bool:
    metadata = _json_dict(row.get("metadata"))
    if "candidate_present" in metadata:
        return bool(metadata["candidate_present"])
    return str(row.get("source_type") or "") in {SYNTHETIC_OPEN_CLEAN, REAL_ERROR_PAIR}


def _rule_group_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for _domain, group, entry in iter_coverage_entries(load_rules_coverage()):
        for rule_id in entry.get("rules", []):
            result[str(rule_id)] = group
    return result


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _strict_clean_rows(clean_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strict = [row for row in clean_rows if str(row.get("accepted_reason") or "") == "passed_quality_filters"]
    return strict or clean_rows


def _real_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        if _contains_artificial_marker_text(source, target):
            continue
        edits = _json_list(row.get("edits") or row.get("edit_operations"))
        rule_ids = _rule_ids_from_edits(edits) or _json_list(row.get("rule_ids")) or [str(row.get("rule_id") or "unknown")]
        error_types = _error_types_from_edits(edits) or _json_list(row.get("error_types")) or [str(row.get("error_type") or "unknown")]
        metadata = _json_dict(row.get("metadata"))
        metadata.update(
            {
                "source_type": REAL_ERROR_PAIR,
                "candidate_present": bool(row.get("candidate_present", True)),
                "generation_strategy": "real_pair",
                "error_bearing_sentence_source": "real_pair",
            }
        )
        result.append(
            _core_row(
                source=source,
                target=target,
                source_type=REAL_ERROR_PAIR,
                error_type=str(error_types[0] if error_types else "unknown"),
                rule_ids=[str(rule_id) for rule_id in rule_ids],
                edits=edits,
                metadata=metadata,
                original_clean_source="",
                source_corpus=str(row.get("source_dataset") or "real_error_pair"),
                source_subcorpus="",
                is_hard_negative=False,
                is_real_pair=True,
                is_clean=False,
                is_synthetic=False,
                domain=str(row.get("domain") or "real_error_pair"),
                error_types=[str(error_type) for error_type in error_types],
            )
        )
    return result


def _load_or_reuse_real_error_pairs(
    real_config: dict[str, Any],
    *,
    candidate_generator: CandidateGenerator,
    output_path: Path,
    reports_dir: Path,
    core_config: dict[str, Any],
) -> RealErrorLoadResult:
    min_cached = int(core_config.get("min_cached_real_pairs", 5000))
    preferred_cached = int(core_config.get("preferred_cached_real_pairs", 30000))
    if bool(core_config.get("reuse_validated_real_pairs_cache", True)) and output_path.exists():
        cached_rows = _read_records(output_path)
        if cached_rows and (reports_dir / "real_pair_filter_report.csv").exists():
            return RealErrorLoadResult(
                rows=cached_rows,
                accepted_count=len(cached_rows),
                rejected_count=0,
                source_reports=[
                    {
                        "source_dataset": "validated_real_pair_cache",
                        "status": "loaded",
                        "reason": "reused_validated_cache_after_prior_refresh",
                        "accepted": len(cached_rows),
                    }
                ],
                rejection_reason_counts={},
                output_path=str(output_path),
            )
        if not should_refresh_real_pair_cache(output_path, min_cached=min_cached, preferred_cached=preferred_cached):
            return RealErrorLoadResult(
                rows=cached_rows,
                accepted_count=len(cached_rows),
                rejected_count=0,
                source_reports=[
                    {
                        "source_dataset": "validated_real_pair_cache",
                        "status": "loaded",
                        "reason": "reused_validated_cache",
                        "accepted": len(cached_rows),
                    }
                ],
                rejection_reason_counts={},
                output_path=str(output_path),
            )
    result = load_real_error_pairs(
        real_config,
        candidate_generator=candidate_generator,
        output_path=output_path,
        reports_dir=reports_dir,
    )
    if result.accepted_count <= 0 and output_path.exists():
        cached_rows = _read_records(output_path)
        if cached_rows:
            return RealErrorLoadResult(
                rows=cached_rows,
                accepted_count=len(cached_rows),
                rejected_count=result.rejected_count,
                source_reports=[
                    *result.source_reports,
                    {
                        "source_dataset": "validated_real_pair_cache",
                        "status": "loaded",
                        "reason": "reused_cache_after_refresh_shortage",
                        "accepted": len(cached_rows),
                    },
                ],
                rejection_reason_counts=result.rejection_reason_counts,
                output_path=str(output_path),
            )
    return result


def should_refresh_real_pair_cache(
    output_path: Path,
    *,
    min_cached: int = 5000,
    preferred_cached: int = 30000,
    cached_count: int | None = None,
    minimum: int | None = None,
    preferred: int | None = None,
) -> bool:
    if minimum is not None:
        min_cached = int(minimum)
    if preferred is not None:
        preferred_cached = int(preferred)
    if not output_path.exists():
        return True
    if cached_count is None:
        try:
            count = int(sum(len(chunk) for chunk in pd.read_csv(output_path, chunksize=50_000)))
        except Exception:
            return True
    else:
        count = int(cached_count)
    if count < int(min_cached):
        return True
    return False


def _corpus_opportunity_rows_for_active_rules(
    *,
    existing_rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    active_rule_ids: list[str],
    quota_config: dict[str, Any],
    cap_config: dict[str, Any],
    seed: int,
    synthetic_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if synthetic_budget <= 0 or not active_rule_ids or not clean_rows:
        return [], {"scan_count": 0}
    min_total = int(quota_config.get("min_total_per_active_rule", 1000))
    preferred = int(quota_config.get("preferred_total_per_active_rule", 2500))
    quota_by_rule = dict(quota_config.get("rule_quotas", {}) or {})
    target_by_rule: dict[str, int] = {}
    for rule_id in active_rule_ids:
        _rule_min, rule_preferred, rule_max = _quota_values_for_rule(
            rule_id,
            min_total=min_total,
            preferred=preferred,
            quota_by_rule=quota_by_rule,
        )
        target_by_rule[rule_id] = min(rule_preferred, rule_max)

    randomizer = random.Random(seed + 31)
    pool = list(clean_rows)
    randomizer.shuffle(pool)
    generator = SyntheticGenerator(seed=seed + 37, max_errors_per_sentence=3)
    analyzer = DiffAnalyzer()
    active = set(active_rule_ids)
    rows: list[dict[str, Any]] = []
    seen_pairs = {(str(row.get("source", "")), str(row.get("target", ""))) for row in existing_rows}
    rule_counts = Counter(_rule_counts_from_rows(existing_rows))
    error_counts = Counter(_error_counts_from_rows(existing_rows))
    normalized_counts = Counter(normalized_pair_hash(str(row.get("source", "")), str(row.get("target", ""))) for row in existing_rows)
    clean_attempt_limit = min(len(pool), int(quota_config.get("corpus_opportunity_scan_limit", 8_000)))
    specific_attempt_limit = min(len(pool), int(quota_config.get("corpus_opportunity_specific_scan_limit", 80_000)))

    def add_row(row: dict[str, Any]) -> bool:
        if len(rows) >= synthetic_budget:
            return False
        if _row_contains_artificial_marker(row):
            return False
        pair_key = (row["source"], row["target"])
        if pair_key in seen_pairs:
            return False
        normalized_hash = normalized_pair_hash(row["source"], row["target"])
        if normalized_counts[normalized_hash] >= 1:
            return False
        row_rules = [rule_id for rule_id in _json_list(row.get("rule_ids")) if rule_id in active]
        if not row_rules:
            return False
        if not any(rule_counts.get(rule_id, 0) < target_by_rule.get(rule_id, preferred) for rule_id in row_rules):
            return False
        if _row_exceeds_caps(
            row,
            rule_counts,
            error_counts,
            max_rule_total=max(int(cap_config.get("generation_rule_cap", 2500)), 3500),
            max_error_total=int(cap_config["generation_error_type_cap"]),
            rule_max_totals=cap_config.get("rule_max_totals", {}),
        ):
            return False
        seen_pairs.add(pair_key)
        normalized_counts[normalized_hash] += 1
        _increment_row_caps(row, rule_counts, error_counts)
        rows.append(row)
        return True

    scan_count = 0
    for clean in _cycled(pool, max_iterations=clean_attempt_limit):
        if len(rows) >= synthetic_budget:
            break
        if all(rule_counts.get(rule_id, 0) >= target_by_rule.get(rule_id, preferred) for rule_id in active_rule_ids):
            break
        target = str(clean.get("text", "")).strip()
        if not target:
            continue
        scan_count += 1
        if scan_count == 1 or scan_count % 1000 == 0:
            _progress(
                "corpus_opportunity_scan",
                scanned=scan_count,
                rows=len(rows),
                synthetic_budget=synthetic_budget,
            )
        for example in generator.generate_variants_from_clean(target, max_variants=12):
            example_rules = [rule_id for rule_id in (example.rule_ids or []) if rule_id in active]
            if not example_rules:
                continue
            if not any(rule_counts.get(rule_id, 0) < target_by_rule.get(rule_id, preferred) for rule_id in example_rules):
                continue
            row = _row_from_synthetic_example(example, clean, analyzer)
            if row is not None:
                add_row(row)

    missing = [rule_id for rule_id in active_rule_ids if rule_counts.get(rule_id, 0) < target_by_rule.get(rule_id, preferred)]
    if missing and len(rows) < synthetic_budget:
        for rule_id in sorted(missing, key=lambda item: rule_counts.get(item, 0)):
            if rule_counts.get(rule_id, 0) >= target_by_rule.get(rule_id, preferred):
                continue
            attempts = 0
            _progress(
                "corpus_specific_rule_start",
                rule_id=rule_id,
                current=rule_counts.get(rule_id, 0),
                target=target_by_rule.get(rule_id, preferred),
                scan_limit=specific_attempt_limit,
            )
            for clean in _cycled(pool, max_iterations=specific_attempt_limit):
                if len(rows) >= synthetic_budget or rule_counts.get(rule_id, 0) >= target_by_rule.get(rule_id, preferred):
                    break
                target = str(clean.get("text", "")).strip()
                if not target:
                    continue
                if attempts and attempts % 5000 == 0:
                    _progress(
                        "corpus_specific_rule_scan",
                        rule_id=rule_id,
                        attempts=attempts,
                        current=rule_counts.get(rule_id, 0),
                        rows=len(rows),
                    )
                pair = _rule_specific_corpus_pair(rule_id, target, attempts)
                attempts += 1
                if pair is None:
                    continue
                source, corrected = pair
                row = _row_from_corpus_pair(
                    rule_id=rule_id,
                    source=source,
                    target=corrected,
                    clean=clean,
                    analyzer=analyzer,
                    template_hint=f"corpus_{rule_id}",
                )
                if row is not None:
                    add_row(row)
            _progress(
                "corpus_specific_rule_done",
                rule_id=rule_id,
                attempts=attempts,
                current=rule_counts.get(rule_id, 0),
                rows=len(rows),
            )

    return rows, {
        "scan_count": scan_count,
        "rule_counts": dict(sorted((rule_id, rule_counts.get(rule_id, 0)) for rule_id in active_rule_ids)),
    }


def _rule_specific_corpus_pair(rule_id: str, target: str, attempt_index: int) -> tuple[str, str] | None:
    if len(target) < 20 or len(target) > 260:
        return None
    if _contains_known_bad_text(target):
        return None
    if rule_id == "final_punctuation_default":
        stripped = target.rstrip()
        if stripped.endswith((".", "!", "?")):
            return stripped[:-1] + target[len(stripped) :], target
        return None
    if rule_id in {
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
    }:
        return _remove_nth_punctuation(target, ",", attempt_index)
    if rule_id in {"asyndetic_dash", "consequence_dash", "enumeration_dash", "subject_predicate_dash"}:
        if " — " not in target:
            return None
        return target.replace(" — ", " ", 1), target
    if rule_id in {"explanation_colon", "enumeration_colon", "direct_speech_colon"}:
        if ":" not in target or re.search(r"\d:\d|https?://", target):
            return None
        return _remove_nth_punctuation(target, ":", attempt_index)
    if rule_id == "semicolon":
        return _remove_nth_punctuation(target, ";", attempt_index)
    if rule_id == "direct_speech_dash":
        if "» — " in target:
            return target.replace("» — ", "» ", 1), target
        if "\" — " in target:
            return target.replace("\" — ", "\" ", 1), target
        return None
    if rule_id in {"direct_speech_quotes", "quote_pair_balance"}:
        for char in ("«", "»", "\""):
            if char in target:
                return target.replace(char, "", 1), target
        return None
    if rule_id == "bracket_pair_balance":
        for char in ("(", ")", "[", "]"):
            if char in target:
                return target.replace(char, "", 1), target
        return None
    if rule_id == "punctuation_delete_replace":
        if "," in target:
            return target.replace(",", ",,", 1), target
        stripped = target.rstrip()
        if stripped.endswith("."):
            return stripped + "." + target[len(stripped) :], target
        return None
    if rule_id == "capitalization_sentence_start":
        stripped = target.lstrip()
        offset = len(target) - len(stripped)
        if stripped and "А" <= stripped[0] <= "Я":
            return target[:offset] + stripped[0].lower() + stripped[1:], target
        return None
    if rule_id == "abbreviation_case_protection":
        for token in ("ООО", "АО", "ИП", "РФ", "США", "НББ"):
            if re.search(rf"\b{token}\b", target):
                return target.replace(token, token.lower(), 1), target
        return None
    if rule_id == "hyphen_po_adverbs":
        for token in ("по-русски", "по-дружески", "по-новому", "по-старому"):
            if token in target.lower() and not _hyphen_po_followed_by_nounish_context(target, token):
                match = re.search(re.escape(token), target, flags=re.IGNORECASE)
                if match:
                    return target[: match.start()] + target[match.start() : match.end()].replace("-", " ", 1) + target[match.end() :], target
        return None
    hyphen_tokens = {
        "hyphen_particles": ("кто-то", "что-то", "где-либо", "когда-нибудь"),
        "hyphen_koe_koy": ("кое-кто", "кое-где", "кое-как", "кой-кто"),
        "hyphen_whitelist": ("по-русски", "по-английски", "кто-нибудь", "кое-кто"),
        "pol_polu_compounds": ("пол-лимона", "пол-яблока", "пол-Москвы", "пол-Европы"),
    }
    for token in hyphen_tokens.get(rule_id, ()):
        match = re.search(re.escape(token), target, flags=re.IGNORECASE)
        if match:
            return target[: match.start()] + target[match.start() : match.end()].replace("-", " ", 1) + target[match.end() :], target
    context_tokens = {
        "context_tak_zhe": (("так же", "также"), ("также", "так же")),
        "context_to_zhe": (("то же", "тоже"), ("тоже", "то же")),
        "context_chto_by": (("что бы", "чтобы"), ("чтобы", "что бы")),
        "context_za_to": (("зато", "за то"), ("за то", "зато")),
        "context_vsledstvie": (("вследствие", "в следствие"), ("в следствие", "вследствие")),
        "context_nesmotrya": (("несмотря", "не смотря"), ("не смотря", "несмотря")),
        "ni_stable_expression": (("ни разу", "не разу"), ("ни в коем случае", "не в коем случае")),
        "ni_particle_context": (("ни сказал", "не сказал"), ("ни решил", "не решил"), ("ни было", "не было")),
    }
    lower_target = target.lower()
    for clean_form, dirty_form in context_tokens.get(rule_id, ()):
        match = re.search(rf"\b{re.escape(clean_form)}\b", lower_target)
        if match:
            return target[: match.start()] + _match_case(target[match.start() : match.end()], dirty_form) + target[match.end() :], target
    safe_replacements = _safe_orthography_replacements(rule_id)
    lower = target.lower()
    for clean_form, dirty_form in safe_replacements:
        match = re.search(rf"\b{re.escape(clean_form)}\b", lower)
        if not match:
            continue
        original = target[match.start() : match.end()]
        dirty = _match_case(original, dirty_form)
        return target[: match.start()] + dirty + target[match.end() :], target
    return _generic_word_corpus_pair(rule_id, target, attempt_index)


def _generic_word_corpus_pair(rule_id: str, target: str, attempt_index: int) -> tuple[str, str] | None:
    generic_rules = {
        "dictionary_fuzzy",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "swapped_letters_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
    }
    if rule_id not in generic_rules:
        return None
    matches = [
        match
        for match in re.finditer(r"\b[А-Яа-яЁё]{6,14}\b", target)
        if not target[match.start() : match.end()].istitle()
    ]
    if not matches:
        return None
    match = matches[attempt_index % len(matches)]
    word = target[match.start() : match.end()]
    dirty = _generic_dirty_word(rule_id, word)
    if not dirty or dirty == word:
        return None
    return target[: match.start()] + dirty + target[match.end() :], target


def _generic_dirty_word(rule_id: str, word: str) -> str:
    lower = word.lower()
    if rule_id == "missing_letter_candidate" and len(word) > 5:
        index = max(1, len(word) // 2)
        dirty = word[:index] + word[index + 1 :]
        return dirty
    if rule_id == "extra_letter_candidate" and len(word) > 4:
        index = max(1, len(word) // 2)
        return word[:index] + word[index] + word[index:]
    if rule_id == "swapped_letters_candidate" and len(word) > 5:
        index = max(1, len(word) // 2)
        return word[:index] + word[index + 1] + word[index] + word[index + 2 :]
    if rule_id == "double_consonant_candidate":
        for index, char in enumerate(lower[1:-1], start=1):
            if char in "бвгджзклмнпрстфхцчшщ":
                return word[:index] + word[index] + word[index:]
    if rule_id == "keyboard_typo_candidate":
        replacements = {"о": "л", "а": "с", "е": "н", "и": "ш", "р": "о"}
        for index, char in enumerate(lower):
            if char in replacements:
                return word[:index] + _match_case(word[index], replacements[char]) + word[index + 1 :]
    if rule_id == "dictionary_fuzzy":
        for index, char in enumerate(lower):
            if char in "оеаия":
                replacement = {"о": "а", "е": "и", "а": "о", "и": "е", "я": "е"}[char]
                return word[:index] + _match_case(word[index], replacement) + word[index + 1 :]
    return ""


def _row_from_corpus_pair(
    *,
    rule_id: str,
    source: str,
    target: str,
    clean: dict[str, Any],
    analyzer: DiffAnalyzer,
    template_hint: str,
) -> dict[str, Any] | None:
    if (
        not source
        or not target
        or source == target
        or _diff_pair_too_expensive(source, target)
        or _contains_known_bad_text(source)
        or _contains_known_bad_text(target)
        or _contains_artificial_marker_text(source, target)
    ):
        return None
    edits = [edit for edit in analyzer.analyze(source, target, candidates=[]) if is_allowed_edit_type(edit.edit_type)]
    edits = _ensure_rule_ids(edits, [rule_id])
    if not edits:
        return None
    edit_dicts = [asdict(edit) for edit in edits]
    error_types = sorted({coarse_error_type(edit.edit_type) for edit in edits if coarse_error_type(edit.edit_type) != "unknown"})
    metadata = _clean_metadata(clean)
    metadata.update(
        _generation_metadata(
            source,
            target,
            edit_dicts,
            [rule_id],
            generation_strategy="corpus_opportunity",
            error_bearing_sentence_source="corpus",
        )
    )
    metadata["synthetic_source_dataset"] = template_hint
    return _core_row(
        source=source,
        target=target,
        source_type=SYNTHETIC_OPEN_CLEAN,
        error_type=error_types[0] if error_types else "unknown",
        rule_ids=_rule_ids_from_edits(edit_dicts) or [rule_id],
        edits=edit_dicts,
        metadata=metadata,
        original_clean_source=target,
        source_corpus=str(clean.get("source_name") or ""),
        source_subcorpus=str(clean.get("source_subcorpus") or ""),
        is_hard_negative=False,
        is_real_pair=False,
        is_clean=False,
        is_synthetic=True,
        domain=str(clean.get("domain") or "open_clean"),
        error_types=error_types,
    )


def _remove_nth_punctuation(target: str, char: str, attempt_index: int) -> tuple[str, str] | None:
    positions = [match.start() for match in re.finditer(re.escape(char), target)]
    if not positions:
        return None
    index = positions[attempt_index % len(positions)]
    return target[:index] + target[index + 1 :], target


def _safe_orthography_replacements(rule_id: str) -> tuple[tuple[str, str], ...]:
    pairs: dict[str, tuple[tuple[str, str], ...]] = {
        "frequent_error_exact": (("сделал", "зделал"), ("вообще", "вобще"), ("предварительный", "предворительный")),
        "dictionary_fuzzy": (("библиотека", "библеотека"), ("корова", "карова"), ("молоко", "малако"), ("собака", "сабака"), ("территория", "тирритория")),
        "double_consonant_candidate": (("грамматика", "граматика"), ("территория", "територия"), ("профессия", "проффесия"), ("комиссия", "комисия")),
        "keyboard_typo_candidate": (("молоко", "молокл"), ("корова", "клрова"), ("грамматика", "грсмматика"), ("собака", "слбака")),
        "swapped_letters_candidate": (("корова", "коорва"), ("библиотека", "бибилотека"), ("молоко", "молкоо"), ("собака", "соабка")),
        "missing_letter_candidate": (("молоко", "млоко"), ("корова", "корва"), ("библиотека", "библотека"), ("собака", "сбака")),
        "extra_letter_candidate": (("собака", "собакаа"), ("молоко", "молокоо"), ("корова", "коорова"), ("библиотека", "библиотекаа")),
        "missing_hard_sign": (("объявление", "обявление"), ("подъезд", "подезд"), ("съезд", "сезд"), ("объект", "обект"), ("разъяснение", "разяснение")),
        "soft_to_hard_sign": (("объявление", "обьявление"), ("подъезд", "подьезд"), ("съезд", "сьезд"), ("объект", "обьект")),
        "sdelat_prefix": (("сделать", "зделать"), ("сделал", "зделал"), ("сделали", "зделали"), ("сделано", "зделано")),
        "prefix_pre_pri": (("превосходный", "привосходный"), ("прибытие", "пребытие"), ("предел", "придел")),
        "prefix_s_to_z": (("бездарный", "бесдарный"), ("разбудить", "расбудить"), ("издалека", "исдалека")),
        "prefix_z_to_s": (("бесполезный", "безполезный"), ("рассказать", "разсказать"), ("исправить", "изправить")),
        "pattern_жы_жи": (("жизнь", "жызнь"), ("житель", "жыитель"), ("пружина", "пружына")),
        "pattern_шы_ши": (("машина", "машына"), ("ширина", "шырина"), ("тишина", "тышина")),
        "pattern_чя_ча": (("часть", "чясть"), ("чайник", "чяйник"), ("задача", "задачя")),
        "pattern_щя_ща": (("щавель", "щявель"), ("площадь", "площядь"), ("прощание", "прощяние")),
        "pattern_щю_щу": (("щука", "щюка"), ("щуплый", "щюплый"), ("щуриться", "щюриться")),
        "pattern_чю_чу": (("чудо", "чюдо"), ("чувство", "чювство"), ("чужой", "чюжой")),
        "pattern_цы_ци": (("цифра", "цыфра"), ("цирк", "цырк"), ("лекция", "лекцыя")),
        "pattern_жо_же": (("желтый", "жолтый"), ("жесткий", "жосткий"), ("железо", "жолезо")),
        "pattern_шо_ше": (("шестой", "шостой"), ("шелест", "шолест"), ("шепот", "шопот")),
        "pattern_чо_че": (("черный", "чорный"), ("чертеж", "чортеж"), ("человек", "чоловек")),
        "pattern_що_ще": (("щедрый", "щодрый"), ("щетка", "щотка"), ("щенок", "щонок")),
        "cy_exception": (("цыган", "циган"), ("цыпленок", "ципленок"), ("цыпочки", "ципочки")),
        "n_nn_adjective": (("длинный", "длиный"), ("ценный", "ценый"), ("странный", "страный")),
        "n_nn_participle": (("подписанный", "подписаный"), ("проверенный", "провереный"), ("согласованный", "согласованый")),
        "n_nn_deverbal_adjective": (("жареный", "жаренный"), ("ветреный", "ветренный"), ("раненый", "раненный")),
        "ne_short_form": (("неясен", "не ясен"), ("непонятен", "не понятен"), ("недоволен", "не доволен")),
        "ne_predicative": (("невозможно", "не возможно"), ("необходимо", "не обходимо"), ("неизвестно", "не известно"), ("непонятно", "не понятно"), ("неясно", "не ясно")),
        "n_nn_short_form": (("уверены", "уверенны"), ("подготовлены", "подготовленны"), ("проверены", "проверенны"), ("согласованы", "согласованны")),
        "tsya_soft_delete": (("учится", "учиться"), ("готовится", "готовиться"), ("строится", "строиться")),
        "tsya_soft_insert": (("появиться", "появится"), ("вернуться", "вернутся"), ("учиться", "учится")),
    }
    return pairs.get(rule_id, ())


def _hyphen_po_followed_by_nounish_context(text: str, token: str) -> bool:
    match = re.search(re.escape(token), text, flags=re.IGNORECASE)
    if not match:
        return False
    after = text[match.end() : match.end() + 24].lower()
    return bool(re.match(r"\s+(?:плану|план|договору|договор|вариант|адресу|адрес|отчету|отчет)\b", after))


def _contains_known_bad_text(text: str) -> bool:
    if re.search(r"\S\u2014\s|\s\u2014\S", text):
        return True
    lower = text.lower()
    bad = (
        "несогласен с выводом",
        "ненужно комиссии",
        "по-старому плану",
        "по-новому вариант",
        "по-новому договору",
        "по-старому адресу",
        "сохранил территория",
        "записал житель",
        "читал свежая сводка",
        "в закрытая заявка",
    )
    return any(phrase in lower for phrase in bad)


def _contains_artificial_marker_text(*texts: str) -> bool:
    combined = "\n".join(str(text or "") for text in texts).lower()
    if ARTIFICIAL_METKA_SUBSTRING in combined:
        return True
    if any(pattern in combined for pattern in ARTIFICIAL_LATER_EDITOR_PATTERNS):
        return True
    return bool(ARTIFICIAL_RANDOM_FILLER_RE.search(combined))


def _row_contains_artificial_marker(row: dict[str, Any]) -> bool:
    return _contains_artificial_marker_text(str(row.get("source", "")), str(row.get("target", "")))


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement

def _build_active_rule_quota_rows(
    existing_rows: list[dict[str, Any]],
    *,
    clean_rows: list[dict[str, Any]],
    active_rule_ids: list[str],
    quota_config: dict[str, Any],
    cap_config: dict[str, Any],
    candidate_generator: CandidateGenerator,
    seed: int,
    synthetic_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not quota_config.get("enabled", True) or not active_rule_ids:
        return [], {
            "active_rule_ids": [] if not quota_config.get("enabled", True) else active_rule_ids,
            "excluded_active_rule_ids": [],
            "quota_rows": [],
            "excluded_rows": [],
            "rejected_templates": [],
            "rule_caps_applied": [],
            "error_type_caps_applied": [],
        }
    min_total = int(quota_config.get("min_total_per_active_rule", 80))
    preferred = int(quota_config.get("preferred_total_per_active_rule", max(200, min_total)))
    backfill_generator = TargetedBackfillGenerator(candidate_generator, seed=seed)
    analyzer_rows: list[dict[str, Any]] = []
    rejected_templates: list[RejectedBackfillTemplate] = []
    excluded_rows: list[dict[str, Any]] = []
    quota_rows: list[dict[str, Any]] = []
    active_result: list[str] = []
    clean_cycle = _targeted_carrier_rows(clean_rows) or list(clean_rows) or [{"text": "", "source_name": "", "source_subcorpus": "", "domain": "open_clean"}]
    seen_pairs = {(str(row.get("source", "")), str(row.get("target", ""))) for row in existing_rows}
    current_counts = Counter(_rule_counts_from_rows(existing_rows))
    rule_cap_counts = Counter(current_counts)
    error_cap_counts = Counter(_error_counts_from_rows(existing_rows))
    normalized_counts = Counter(normalized_pair_hash(str(row.get("source", "")), str(row.get("target", ""))) for row in existing_rows)
    budget_left = max(0, int(synthetic_budget))
    quota_by_rule = dict(quota_config.get("rule_quotas", {}) or {})

    for rule_id in active_rule_ids:
        current = int(current_counts.get(rule_id, 0))
        rule_min, rule_preferred, rule_max = _quota_values_for_rule(
            rule_id,
            min_total=min_total,
            preferred=preferred,
            quota_by_rule=quota_by_rule,
        )
        needed = max(0, min(rule_preferred, rule_max) - current)
        generated_count = 0
        action = "ok"
        reason = ""
        if needed > 0 and budget_left > 0:
            needed = min(needed, budget_left)
            _progress(
                "targeted_fill_rule_start",
                rule_id=rule_id,
                current=current,
                needed=needed,
                budget_left=budget_left,
            )
            pool, rejected, reason = _candidate_backed_example_pool_for_rule(
                rule_id,
                required_count=min(needed, 260),
                backfill_generator=backfill_generator,
                candidate_generator=candidate_generator,
            )
            rejected_templates.extend(rejected)
            row_attempts = 0
            duplicate_cap = int(cap_config.get("targeted_duplicate_cap", 20))
            while pool and generated_count < needed and budget_left > 0 and row_attempts < max(needed * 4, 1000):
                if row_attempts and row_attempts % 1000 == 0:
                    _progress(
                        "targeted_fill_rule_scan",
                        rule_id=rule_id,
                        attempts=row_attempts,
                        generated=generated_count,
                        needed=needed,
                        budget_left=budget_left,
                    )
                example = pool[row_attempts % len(pool)]
                clean = clean_cycle[(len(analyzer_rows) + len(existing_rows) + row_attempts) % len(clean_cycle)]
                if budget_left <= 0:
                    break
                row = _row_from_targeted_example(example, clean)
                if row is None:
                    row_attempts += 1
                    continue
                if _row_contains_artificial_marker(row):
                    row_attempts += 1
                    continue
                pair_key = (row["source"], row["target"])
                normalized_hash = normalized_pair_hash(row["source"], row["target"])
                if pair_key in seen_pairs and normalized_counts[normalized_hash] >= duplicate_cap:
                    row_attempts += 1
                    continue
                if _row_exceeds_caps(
                    row,
                    rule_cap_counts,
                    error_cap_counts,
                    max_rule_total=min(int(cap_config["generation_rule_cap"]), rule_max),
                    max_error_total=None,
                    rule_max_totals=cap_config.get("rule_max_totals", {}),
                ):
                    row_attempts += 1
                    continue
                seen_pairs.add(pair_key)
                normalized_counts[normalized_hash] += 1
                _increment_row_caps(row, rule_cap_counts, error_cap_counts)
                analyzer_rows.append(row)
                budget_left -= 1
                generated_count += 1
                row_attempts += 1
            action = "backfilled" if generated_count else "ok"
            if generated_count > 0:
                current_counts.update(_rule_counts_from_rows(analyzer_rows[-generated_count:]))
            _progress(
                "targeted_fill_rule_done",
                rule_id=rule_id,
                generated=generated_count,
                attempts=row_attempts,
                final_count=int(current_counts.get(rule_id, 0)),
                budget_left=budget_left,
            )
        final_count = int(current_counts.get(rule_id, 0))
        if final_count < rule_min or (needed > 0 and generated_count == 0 and reason):
            excluded_rows.append(
                {
                    "rule_id": rule_id,
                    "reason": reason or "no_existing_candidate_backed_pattern",
                    "previous_status": "active",
                    "new_status": "excluded_from_synthetic_target",
                    "why_not_generated": reason or "could_not_reach_minimum_with_existing_candidates",
                }
            )
            action = "excluded"
        else:
            active_result.append(rule_id)
        quota_rows.append(
            {
                "rule_id": rule_id,
                "status": "active" if rule_id in active_result else "excluded",
                "target_min_total": rule_min,
                "preferred_total": rule_preferred,
                "final_total": final_count,
                "train_count": 0,
                "val_count": 0,
                "test_count": 0,
                "source": "targeted/backfill" if generated_count else "real/synthetic",
                "action": action,
                "reason": reason,
            }
        )

    return analyzer_rows, {
        "active_rule_ids": sorted(active_result),
        "excluded_active_rule_ids": sorted(row["rule_id"] for row in excluded_rows),
        "quota_rows": quota_rows,
        "excluded_rows": excluded_rows,
        "rejected_templates": rejected_templates,
        "rule_caps_applied": [],
        "error_type_caps_applied": [],
    }


def _quota_values_for_rule(
    rule_id: str,
    *,
    min_total: int,
    preferred: int,
    quota_by_rule: dict[str, Any],
) -> tuple[int, int, int]:
    raw = quota_by_rule.get(rule_id, {}) if isinstance(quota_by_rule, dict) else {}
    if isinstance(raw, dict) and raw:
        rule_min = int(raw.get("min_total", raw.get("quota_min", min_total)) or min_total)
        rule_preferred = int(raw.get("preferred_total", raw.get("quota_preferred", preferred)) or preferred)
        rule_max = int(raw.get("max_total", raw.get("quota_max", max(rule_preferred, rule_min))) or max(rule_preferred, rule_min))
        return rule_min, max(rule_min, rule_preferred), max(rule_min, rule_max)
    return min_total, _preferred_total_for_rule(rule_id, min_total=min_total, preferred=preferred), max(preferred, min_total)


def _preferred_total_for_rule(rule_id: str, *, min_total: int, preferred: int) -> int:
    risky = {
        "dictionary_fuzzy",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "swapped_letters_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
    }
    if rule_id in risky:
        return max(min_total, min(preferred, min_total))
    return preferred


def _candidate_backed_example_pool_for_rule(
    rule_id: str,
    *,
    required_count: int,
    backfill_generator: TargetedBackfillGenerator,
    candidate_generator: CandidateGenerator,
) -> tuple[list[TargetedBackfillExample], list[RejectedBackfillTemplate], str]:
    from src.rules.syntax_synthetic import SUPPORTED_SYNTAX_RULE_IDS, build_syntax_eval_examples

    result = backfill_generator.generate_for_rule(rule_id, required_count, seen_pairs=set())
    examples = list(result.examples)
    rejected = list(result.rejected)
    reason = result.excluded_reason

    if len(examples) < max(1, min(required_count, 20)):
        for source, target in _bounded_candidate_pairs(rule_id):
            validation = _validate_candidate_backed_pair(rule_id, source, target, candidate_generator=candidate_generator)
            if validation is None:
                validation = backfill_generator._validate_pair(rule_id, source, target)
            if isinstance(validation, RejectedBackfillTemplate):
                if len(rejected) < 500:
                    rejected.append(validation)
                continue
            if validation not in examples:
                examples.append(validation)
            if len(examples) >= required_count:
                break

    if rule_id in SUPPORTED_SYNTAX_RULE_IDS and len(examples) < max(1, min(required_count, 40)):
        syntax_needed = max(40, min(required_count - len(examples), required_count))
        syntax_rows = build_syntax_eval_examples(
            selected_rule_ids=[rule_id],
            min_examples_per_rule=syntax_needed,
            clean_pool_path="__templates_only_for_training_dataset__.csv.gz",
            candidate_generator=candidate_generator,
        )
        for row in syntax_rows.to_dict("records"):
            converted = _syntax_eval_row_to_targeted_example(row)
            if converted is not None:
                examples.append(converted)
            if len(examples) >= required_count:
                break

    if len(examples) < required_count:
        for forced in _forced_targeted_examples(rule_id, required_count - len(examples)):
            if forced not in examples:
                examples.append(forced)
            if len(examples) >= required_count:
                break

    if examples:
        return examples, rejected, ""
    return examples, rejected, reason or "no_existing_candidate_backed_pattern"


def _forced_targeted_examples(rule_id: str, required_count: int) -> list[TargetedBackfillExample]:
    if required_count <= 0:
        return []
    pairs = _forced_candidate_pairs(rule_id, required_count)
    if not pairs:
        return []
    analyzer = DiffAnalyzer()
    result: list[TargetedBackfillExample] = []
    for source, target in pairs:
        edits = [edit for edit in analyzer.analyze(source, target, candidates=[]) if is_allowed_edit_type(edit.edit_type)]
        edits = _ensure_rule_ids(edits, [rule_id])
        if not edits:
            continue
        edit_dicts = [asdict(edit) for edit in edits]
        result.append(
            TargetedBackfillExample(
                source=source,
                target=target,
                rule_ids=_rule_ids_from_edits(edit_dicts) or [rule_id],
                edits=edit_dicts,
                source_dataset=f"forced_training_{rule_id}",
                metadata={
                    "target_family": rule_id,
                    "candidate_present": True,
                    "source_type": SYNTHETIC_OPEN_CLEAN,
                    "candidate_rule_ids": [rule_id],
                },
            )
        )
    return result


def _forced_candidate_pairs(rule_id: str, required_count: int) -> list[tuple[str, str]]:
    templates: dict[str, tuple[tuple[str, str], ...]] = {
        "asyndetic_dash": (
            ("Солнце село город затих {marker}.", "Солнце село — город затих {marker}."),
            ("Звонок прозвучал совещание началось {marker}.", "Звонок прозвучал — совещание началось {marker}."),
        ),
        "consequence_dash": (
            ("Начался дождь встречу перенесли {marker}.", "Начался дождь — встречу перенесли {marker}."),
            ("Срок истек заявку вернули {marker}.", "Срок истек — заявку вернули {marker}."),
        ),
        "enumeration_dash": (
            ("Отчет, договор, заявка все готовы {marker}.", "Отчет, договор, заявка — все готовы {marker}."),
            ("Сроки, подписи, даты все согласованы {marker}.", "Сроки, подписи, даты — все согласованы {marker}."),
        ),
        "abbreviation_case_protection": (
            ("ооо представило отчет {marker}.", "ООО представило отчет {marker}."),
            ("ао обновило график {marker}.", "АО обновило график {marker}."),
            ("ип подал заявку {marker}.", "ИП подал заявку {marker}."),
        ),
        "hyphen_whitelist": (
            ("Во первых редактор проверил документ {marker}.", "Во-первых редактор проверил документ {marker}."),
            ("Кто нибудь отправит отчет {marker}.", "Кто-нибудь отправит отчет {marker}."),
            ("Кое кто сохранил таблицу {marker}.", "Кое-кто сохранил таблицу {marker}."),
        ),
    }
    selected = templates.get(rule_id)
    if not selected:
        return []
    contexts = (
        "сегодня",
        "к вечеру",
        "после совещания",
        "в рабочем отчете",
        "для служебной справки",
        "в итоговой сводке",
        "перед отправкой",
        "после проверки",
        "для протокола",
        "в новом разделе",
    )
    pairs: list[tuple[str, str]] = []
    for index in range(required_count):
        source, target = selected[index % len(selected)]
        context = contexts[(index // max(1, len(selected))) % len(contexts)]
        pairs.append((source.format(marker=context), target.format(marker=context)))
    return pairs


def _validate_candidate_backed_pair(
    rule_id: str,
    source: str,
    target: str,
    *,
    candidate_generator: CandidateGenerator,
) -> TargetedBackfillExample | None:
    if source == target:
        return None
    candidates = candidate_generator.generate(source)
    analyzer = DiffAnalyzer()
    backed_edits: list[Edit] = []
    seen: set[tuple[int, int, str, str, str]] = set()
    for edit in analyzer.analyze(source, target, candidates=candidates):
        if not is_allowed_edit_type(edit.edit_type):
            continue
        matches = [candidate for candidate in candidates if candidate.rule_id == rule_id and candidate_matches_edit(candidate, edit)]
        if not matches:
            continue
        candidate = matches[0]
        backed = Edit(
            source=edit.source,
            replacement=edit.replacement,
            edit_type=edit.edit_type,
            start=edit.start,
            end=edit.end,
            status=edit.status,
            reason=edit.reason,
            confidence=getattr(candidate, "confidence", 1.0),
            rule_id=rule_id,
        )
        key = (backed.start, backed.end, backed.replacement, backed.edit_type, backed.rule_id)
        if key not in seen:
            seen.add(key)
            backed_edits.append(backed)
    if not backed_edits:
        return None
    edits = [asdict(edit) for edit in backed_edits]
    return TargetedBackfillExample(
        source=source,
        target=target,
        rule_ids=_rule_ids_from_edits(edits) or [rule_id],
        edits=edits,
        source_dataset=f"bounded_training_{rule_id}",
        metadata={
            "target_family": rule_id,
            "candidate_present": True,
            "source_type": SYNTHETIC_OPEN_CLEAN,
            "candidate_rule_ids": sorted({candidate.rule_id for candidate in candidates if candidate.rule_id}),
        },
    )


def _syntax_eval_row_to_targeted_example(row: dict[str, Any]) -> TargetedBackfillExample | None:
    rule_id = normalize_rule_id(str(row.get("rule_id") or ""))
    if not rule_id:
        return None
    metadata = _json_dict(row.get("metadata"))
    edits = metadata.get("edit_operations")
    if not isinstance(edits, list):
        edits = []
    if not edits:
        return None
    metadata.update(
        {
            "target_family": rule_id,
            "candidate_present": True,
            "source_type": SYNTHETIC_OPEN_CLEAN,
            "syntax_family": str(row.get("syntax_family") or metadata.get("syntax_family") or ""),
        }
    )
    rule_ids = _rule_ids_from_edits(edits) or [rule_id]
    return TargetedBackfillExample(
        source=str(row.get("source") or ""),
        target=str(row.get("target") or ""),
        rule_ids=rule_ids,
        edits=[dict(edit) for edit in edits if isinstance(edit, dict)],
        source_dataset=f"syntax_training_{rule_id}",
        metadata=metadata,
    )


def _bounded_candidate_pairs(rule_id: str) -> tuple[tuple[str, str], ...]:
    if rule_id == "hyphen_whitelist":
        from src.candidates.frequent_errors import HYPHEN_WHITELIST

        rows = []
        for index, (source, target) in enumerate((item for item in HYPHEN_WHITELIST.items() if item[0] != item[1])):
            rows.append(
                (
                    f"\u0420\u0435\u0434\u0430\u043a\u0442\u043e\u0440 \u0432\u0441\u0442\u0430\u0432\u0438\u043b {source} \u0432 \u043e\u0442\u0447\u0435\u0442 {index}.",
                    f"\u0420\u0435\u0434\u0430\u043a\u0442\u043e\u0440 \u0432\u0441\u0442\u0430\u0432\u0438\u043b {target} \u0432 \u043e\u0442\u0447\u0435\u0442 {index}.",
                )
            )
        return tuple(rows)
    pairs: dict[str, tuple[tuple[str, str], ...]] = {
        "final_punctuation_default": (
            ("\u041a\u043e\u043c\u0438\u0441\u0441\u0438\u044f \u043f\u043e\u0434\u043f\u0438\u0441\u0430\u043b\u0430 \u043e\u0442\u0447\u0435\u0442", "\u041a\u043e\u043c\u0438\u0441\u0441\u0438\u044f \u043f\u043e\u0434\u043f\u0438\u0441\u0430\u043b\u0430 \u043e\u0442\u0447\u0435\u0442."),
            ("\u0420\u0435\u0434\u0430\u043a\u0442\u043e\u0440 \u0441\u0432\u0435\u0440\u0438\u043b \u0441\u043f\u0438\u0441\u043e\u043a", "\u0420\u0435\u0434\u0430\u043a\u0442\u043e\u0440 \u0441\u0432\u0435\u0440\u0438\u043b \u0441\u043f\u0438\u0441\u043e\u043a."),
            ("\u041e\u0442\u0434\u0435\u043b \u043e\u0442\u043f\u0440\u0430\u0432\u0438\u043b \u0441\u0432\u043e\u0434\u043a\u0443", "\u041e\u0442\u0434\u0435\u043b \u043e\u0442\u043f\u0440\u0430\u0432\u0438\u043b \u0441\u0432\u043e\u0434\u043a\u0443."),
        ),
        "capitalization_sentence_start": (
            ("\u043a\u043e\u043c\u0438\u0441\u0441\u0438\u044f \u043f\u043e\u0434\u043f\u0438\u0441\u0430\u043b\u0430 \u043e\u0442\u0447\u0435\u0442.", "\u041a\u043e\u043c\u0438\u0441\u0441\u0438\u044f \u043f\u043e\u0434\u043f\u0438\u0441\u0430\u043b\u0430 \u043e\u0442\u0447\u0435\u0442."),
            ("\u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440 \u0441\u0432\u0435\u0440\u0438\u043b \u0441\u043f\u0438\u0441\u043e\u043a.", "\u0420\u0435\u0434\u0430\u043a\u0442\u043e\u0440 \u0441\u0432\u0435\u0440\u0438\u043b \u0441\u043f\u0438\u0441\u043e\u043a."),
            ("\u043e\u0442\u0434\u0435\u043b \u043e\u0442\u043f\u0440\u0430\u0432\u0438\u043b \u0441\u0432\u043e\u0434\u043a\u0443.", "\u041e\u0442\u0434\u0435\u043b \u043e\u0442\u043f\u0440\u0430\u0432\u0438\u043b \u0441\u0432\u043e\u0434\u043a\u0443."),
        ),
        "abbreviation_case_protection": (
            ("\u043e\u043e\u043e \u043f\u0440\u0435\u0434\u0441\u0442\u0430\u0432\u0438\u043b\u043e \u043e\u0442\u0447\u0435\u0442.", "\u041e\u041e\u041e \u043f\u0440\u0435\u0434\u0441\u0442\u0430\u0432\u0438\u043b\u043e \u043e\u0442\u0447\u0435\u0442."),
            ("\u0430\u043e \u043e\u0431\u043d\u043e\u0432\u0438\u043b\u043e \u0433\u0440\u0430\u0444\u0438\u043a.", "\u0410\u041e \u043e\u0431\u043d\u043e\u0432\u0438\u043b\u043e \u0433\u0440\u0430\u0444\u0438\u043a."),
            ("\u0438\u043f \u043f\u043e\u0434\u0430\u043b \u0437\u0430\u044f\u0432\u043a\u0443.", "\u0418\u041f \u043f\u043e\u0434\u0430\u043b \u0437\u0430\u044f\u0432\u043a\u0443."),
        ),
        "frequent_error_exact": (
            ("\u041e\u043d \u0437\u0434\u0435\u043b\u0430\u043b \u043e\u0442\u0447\u0435\u0442.", "\u041e\u043d \u0441\u0434\u0435\u043b\u0430\u043b \u043e\u0442\u0447\u0435\u0442."),
            ("\u0420\u0435\u0434\u0430\u043a\u0442\u043e\u0440 \u0432\u043e\u0431\u0449\u0435 \u043d\u0435 \u0441\u043f\u043e\u0440\u0438\u043b.", "\u0420\u0435\u0434\u0430\u043a\u0442\u043e\u0440 \u0432\u043e\u043e\u0431\u0449\u0435 \u043d\u0435 \u0441\u043f\u043e\u0440\u0438\u043b."),
            ("\u041e\u0442\u0434\u0435\u043b \u043f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u0438\u043b \u043f\u0440\u0435\u0434\u0432\u0430\u0440\u0438\u0442\u0435\u043b\u043d\u044b\u0439 \u0440\u0430\u0441\u0447\u0435\u0442.", "\u041e\u0442\u0434\u0435\u043b \u043f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u0438\u043b \u043f\u0440\u0435\u0434\u0432\u0430\u0440\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0439 \u0440\u0430\u0441\u0447\u0435\u0442."),
        ),
        "prefix_pre_pri": (
            ("\u041f\u0440\u0438\u0432\u043e\u0441\u0445\u043e\u0434\u043d\u044b\u0439 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0443\u0434\u0438\u0432\u0438\u043b \u0432\u0441\u0435\u0445.", "\u041f\u0440\u0435\u0432\u043e\u0441\u0445\u043e\u0434\u043d\u044b\u0439 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0443\u0434\u0438\u0432\u0438\u043b \u0432\u0441\u0435\u0445."),
            ("\u041f\u0440\u0435\u0431\u044b\u0442\u0438\u0435 \u043f\u043e\u0435\u0437\u0434\u0430 \u043e\u0431\u044a\u044f\u0432\u0438\u043b\u0438 \u0443\u0442\u0440\u043e\u043c.", "\u041f\u0440\u0438\u0431\u044b\u0442\u0438\u0435 \u043f\u043e\u0435\u0437\u0434\u0430 \u043e\u0431\u044a\u044f\u0432\u0438\u043b\u0438 \u0443\u0442\u0440\u043e\u043c."),
            ("\u041f\u0440\u0438\u0434\u0435\u043b \u0442\u0435\u0440\u043f\u0435\u043d\u0438\u044f \u0431\u044b\u043b \u0431\u043b\u0438\u0437\u043e\u043a.", "\u041f\u0440\u0435\u0434\u0435\u043b \u0442\u0435\u0440\u043f\u0435\u043d\u0438\u044f \u0431\u044b\u043b \u0431\u043b\u0438\u0437\u043e\u043a."),
        ),
    }
    return pairs.get(rule_id, ())


def _targeted_carrier_rows(clean_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in clean_rows:
        text = str(row.get("text") or "").strip()
        if not text or len(text) > 140:
            continue
        lower = text.lower()
        if any(pattern in lower for pattern in META_LANGUAGE_PATTERNS):
            continue
        if any(phrase in lower for phrase in SUSPICIOUS_PHRASES):
            continue
        key = normalize_template_text(text)
        if key in seen:
            continue
        seen.add(key)
        eligible.append(row)
        if len(eligible) >= 240_000:
            break
    return _balanced_clean_rows(eligible, limit=120_000)


def _balanced_clean_rows(clean_rows: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        buckets[str(row.get("source_name") or row.get("source_corpus") or "unknown")].append(row)
    result: list[dict[str, Any]] = []
    sources = sorted(buckets)
    index = 0
    while sources and (limit is None or len(result) < limit):
        source = sources[index % len(sources)]
        bucket = buckets[source]
        if bucket:
            result.append(bucket.pop(0))
            if limit is not None and len(result) >= limit:
                break
        if not bucket:
            sources.remove(source)
            if not sources:
                break
            index %= len(sources)
        else:
            index += 1
    return result


def _balanced_clean_items(items: list[tuple[dict[str, Any], list[str]]], *, limit: int | None = None) -> list[tuple[dict[str, Any], list[str]]]:
    buckets: dict[str, list[tuple[dict[str, Any], list[str]]]] = defaultdict(list)
    for item in items:
        row = item[0]
        buckets[str(row.get("source_name") or row.get("source_corpus") or "unknown")].append(item)
    result: list[tuple[dict[str, Any], list[str]]] = []
    sources = sorted(buckets)
    index = 0
    while sources and (limit is None or len(result) < limit):
        source = sources[index % len(sources)]
        bucket = buckets[source]
        if bucket:
            result.append(bucket.pop(0))
            if limit is not None and len(result) >= limit:
                break
        if not bucket:
            sources.remove(source)
            if not sources:
                break
            index %= len(sources)
        else:
            index += 1
    return result


def _fill_remaining_with_targeted_rows(
    *,
    clean_rows: list[dict[str, Any]],
    active_rule_ids: list[str],
    target_count: int,
    candidate_generator: CandidateGenerator,
    seed: int,
    existing_rows: list[dict[str, Any]],
    cap_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_count <= 0 or not active_rule_ids:
        return [], {"rejected_templates": []}
    generator = TargetedBackfillGenerator(candidate_generator, seed=seed)
    clean_cycle = _targeted_carrier_rows(clean_rows) or list(clean_rows) or [{"text": "", "source_name": "", "source_subcorpus": "", "domain": "open_clean"}]
    seen_pairs = {(str(row.get("source", "")), str(row.get("target", ""))) for row in existing_rows}
    rule_cap_counts = Counter(_rule_counts_from_rows(existing_rows))
    error_cap_counts = Counter(_error_counts_from_rows(existing_rows))
    normalized_counts = Counter(normalized_pair_hash(str(row.get("source", "")), str(row.get("target", ""))) for row in existing_rows)
    rows: list[dict[str, Any]] = []
    rejected: list[RejectedBackfillTemplate] = []
    example_pools: dict[str, list[TargetedBackfillExample]] = {}
    for rule_id in active_rule_ids:
        pool, pool_rejected, _reason = _candidate_backed_example_pool_for_rule(
            rule_id,
            required_count=220,
            backfill_generator=generator,
            candidate_generator=candidate_generator,
        )
        rejected.extend(pool_rejected[:5])
        if pool:
            example_pools[rule_id] = pool
    if not example_pools:
        return rows, {"rejected_templates": rejected}

    attempts = 0
    duplicate_cap = int(cap_config.get("targeted_duplicate_cap", 20))
    ordered_rules = sorted(example_pools, key=lambda item: rule_cap_counts.get(item, 0))
    rule_attempts = Counter()
    while len(rows) < target_count and attempts < max(target_count * 8, 1000):
        ordered_rules = sorted(ordered_rules, key=lambda item: rule_cap_counts.get(item, 0))
        rule_id = _targeted_fill_rule_for_attempt(ordered_rules, rule_cap_counts, attempt_index=attempts)
        pool = example_pools[rule_id]
        example = pool[rule_attempts[rule_id] % len(pool)]
        rule_attempts[rule_id] += 1
        row = _row_from_targeted_example(example, clean_cycle[(len(rows) + len(existing_rows)) % len(clean_cycle)])
        if row is None:
            attempts += 1
            continue
        if _row_contains_artificial_marker(row):
            attempts += 1
            continue
        pair_key = (row["source"], row["target"])
        normalized_hash = normalized_pair_hash(row["source"], row["target"])
        if pair_key in seen_pairs and normalized_counts[normalized_hash] >= duplicate_cap:
            attempts += 1
            continue
        if _row_exceeds_caps(
            row,
            rule_cap_counts,
            error_cap_counts,
            max_rule_total=int(cap_config["generation_rule_cap"]),
            max_error_total=int(cap_config["generation_error_type_cap"]),
            rule_max_totals=cap_config.get("rule_max_totals", {}),
        ):
            attempts += 1
            continue
        seen_pairs.add(pair_key)
        normalized_counts[normalized_hash] += 1
        _increment_row_caps(row, rule_cap_counts, error_cap_counts)
        rows.append(row)
        attempts += 1
    return rows, {"rejected_templates": rejected}


def _targeted_fill_rule_for_attempt(ordered_rules: list[str], rule_counts: dict[str, int] | Counter[str], *, attempt_index: int) -> str:
    if not ordered_rules:
        return ""
    return ordered_rules[int(attempt_index) % len(ordered_rules)]


def _row_from_targeted_example(example: TargetedBackfillExample, clean: dict[str, Any]) -> dict[str, Any] | None:
    metadata = _clean_metadata(clean)
    metadata.update(example.metadata)
    error_types = sorted({coarse_error_type(str(edit.get("edit_type") or "")) for edit in example.edits})
    error_types = [error_type for error_type in error_types if error_type != "unknown"]
    source = str(example.source).strip()
    target = str(example.target).strip()
    if (
        not source
        or not target
        or source == target
        or _contains_known_bad_text(source)
        or _contains_known_bad_text(target)
        or _contains_artificial_marker_text(source, target)
        or ("hyphen_po_adverbs" in example.rule_ids and _hyphen_po_bad_positive(source, target))
        or ("n_nn_short_form" in example.rule_ids and _n_nn_short_form_bad_positive(source, target))
    ):
        return None
    if any(rule_id in {"asyndetic_dash", "consequence_dash", "enumeration_dash"} for rule_id in example.rule_ids):
        if re.search(r"\S—\s|\s—\S", target):
            return None
    metadata.update(
        _generation_metadata(
            source,
            target,
            example.edits,
            example.rule_ids,
            generation_strategy="fallback_natural_template",
            error_bearing_sentence_source="fallback_template",
        )
    )
    metadata["template_id"] = str(example.source_dataset or metadata.get("target_family") or "")
    return _core_row(
        source=source,
        target=target,
        source_type=SYNTHETIC_OPEN_CLEAN,
        error_type=error_types[0] if error_types else "unknown",
        rule_ids=example.rule_ids,
        edits=example.edits,
        metadata=metadata,
        original_clean_source=target,
        source_corpus=str(clean.get("source_name") or "targeted_open_clean"),
        source_subcorpus=str(clean.get("source_subcorpus") or ""),
        is_hard_negative=False,
        is_real_pair=False,
        is_clean=False,
        is_synthetic=True,
        domain=str(clean.get("domain") or "open_clean"),
        error_types=error_types,
    )


def _fallback_natural_suffix(clean: dict[str, Any]) -> str:
    del clean
    return ""


def _targeted_generation_strategy(source: str, target: str, clean: dict[str, Any]) -> str:
    del source, target, clean
    return "fallback_natural_template"
    basis = f"{source}\n{target}\n{clean.get('hash') or clean.get('sentence_id') or clean.get('text') or ''}"
    value = int(hashlib.sha256(basis.encode("utf-8")).hexdigest()[:4], 16)
    return "fallback_natural_template" if value % 10 < 1 else "corpus_opportunity"


def _targeted_carrier_sentence(clean: dict[str, Any], target: str) -> str:
    text = str(clean.get("text") or "").strip()
    if not text or text == target:
        return ""
    if len(text) > 140:
        return ""
    lower = text.lower()
    if any(pattern in lower for pattern in META_LANGUAGE_PATTERNS):
        return ""
    if any(phrase in lower for phrase in SUSPICIOUS_PHRASES):
        return ""
    if _contains_artificial_marker_text(text):
        return ""
    return text


def _append_targeted_carrier(text: str, carrier: str) -> str:
    if not carrier:
        return text
    value = text.strip()
    if not value.endswith((".", "!", "?", "…")):
        value += "."
    return f"{value} {carrier}"


def _multi_error_stress_rows(
    *,
    existing_rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    target_count: int,
    seed: int,
    cap_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    pool = [
        row
        for row in existing_rows
        if row.get("source_type") == SYNTHETIC_OPEN_CLEAN
        and not _json_dict(row.get("metadata")).get("is_stress")
        and bool(_json_dict(row.get("metadata")).get("candidate_present"))
        and _json_list(row.get("edits"))
    ]
    if len(pool) < 2:
        return []
    randomizer = random.Random(seed)
    randomizer.shuffle(pool)
    clean_cycle = _targeted_carrier_rows(clean_rows) or clean_rows or [{"text": "", "source_name": "multi_error_stress", "source_subcorpus": "", "domain": "open_clean"}]
    distribution = [2] * 35 + [3] * 35 + [4] * 20 + [5] * 10
    rule_cap_counts = Counter(_rule_counts_from_rows(existing_rows))
    error_cap_counts = Counter(_error_counts_from_rows(existing_rows))
    normalized_counts = Counter(normalized_pair_hash(str(row.get("source", "")), str(row.get("target", ""))) for row in existing_rows)
    rows: list[dict[str, Any]] = []
    cursor = 0
    attempts = 0
    while len(rows) < target_count and attempts < max(target_count * 40, 1000):
        error_count = distribution[(len(rows) + attempts) % len(distribution)]
        components = [pool[(cursor + offset) % len(pool)] for offset in range(error_count)]
        cursor = (cursor + error_count) % len(pool)
        row = _combine_stress_components(components, clean_cycle[(len(rows) + attempts) % len(clean_cycle)], error_count=error_count)
        attempts += 1
        if row is None:
            continue
        normalized_hash = normalized_pair_hash(row["source"], row["target"])
        if normalized_counts[normalized_hash] >= 20:
            continue
        if _row_exceeds_caps(
            row,
            rule_cap_counts,
            error_cap_counts,
            max_rule_total=int(cap_config.get("stress_generation_rule_cap", cap_config["generation_rule_cap"])),
            max_error_total=None,
            rule_max_totals=cap_config.get("rule_max_totals", {}),
        ):
            continue
        normalized_counts[normalized_hash] += 1
        _increment_row_caps(row, rule_cap_counts, error_cap_counts)
        rows.append(row)
    return rows


def _combine_stress_components(components: list[dict[str, Any]], clean: dict[str, Any], *, error_count: int) -> dict[str, Any] | None:
    source_parts: list[str] = []
    target_parts: list[str] = []
    edits: list[dict[str, Any]] = []
    rule_ids: list[str] = []
    error_types: list[str] = []
    component_bearing_sources: list[str] = []
    source_offset = 0
    for component in components:
        source = str(component.get("source") or "").strip()
        target = str(component.get("target") or "").strip()
        if not source or not target or source == target:
            return None
        if source_parts:
            source_offset += 1
        for edit in _json_list(component.get("edits")):
            if not isinstance(edit, dict):
                continue
            adjusted = dict(edit)
            if int(adjusted.get("start", -1) or -1) >= 0:
                adjusted["start"] = int(adjusted.get("start", 0)) + source_offset
            if int(adjusted.get("end", -1) or -1) >= 0:
                adjusted["end"] = int(adjusted.get("end", 0)) + source_offset
            edits.append(adjusted)
        for rule_id in _json_list(component.get("rule_ids")):
            rule_id = str(rule_id)
            if rule_id and rule_id not in rule_ids:
                rule_ids.append(rule_id)
        component_error_types = _json_list(component.get("error_types")) or [component.get("error_type")]
        for error_type in component_error_types:
            error_type = str(error_type)
            if error_type and error_type != "unknown" and error_type not in error_types:
                error_types.append(error_type)
        bearing_source = str(_json_dict(component.get("metadata")).get("error_bearing_sentence_source") or "")
        if bearing_source:
            component_bearing_sources.append(bearing_source)
        source_parts.append(source)
        target_parts.append(target)
        source_offset += len(source)
    if not rule_ids or not edits:
        return None
    source_text = " ".join(source_parts)
    target_text = " ".join(target_parts)
    if _contains_artificial_marker_text(source_text, target_text):
        return None
    metadata = _clean_metadata(clean)
    error_bearing_source = "corpus" if component_bearing_sources and all(source == "corpus" for source in component_bearing_sources) else "fallback_template"
    metadata.update(
        {
            "source_type": SYNTHETIC_OPEN_CLEAN,
            "candidate_present": True,
            "generation_strategy": "multi_error_stress",
            "error_bearing_sentence_source": error_bearing_source,
            "target_family": rule_ids[0],
            "is_stress": True,
            "error_count": int(error_count),
            "rule_ids": rule_ids,
            "original_clean_sentence": target_text,
            "carrier_sentence_hash": hashlib.sha256(target_text.encode("utf-8")).hexdigest()[:24],
            "candidate_rule_ids": sorted(set(rule_ids)),
        }
    )
    return _core_row(
        source=source_text,
        target=target_text,
        source_type=SYNTHETIC_OPEN_CLEAN,
        error_type=error_types[0] if error_types else "mixed",
        rule_ids=rule_ids,
        edits=edits,
        metadata=metadata,
        original_clean_source=target_text,
        source_corpus=str(clean.get("source_name") or "multi_error_stress"),
        source_subcorpus=str(clean.get("source_subcorpus") or ""),
        is_hard_negative=False,
        is_real_pair=False,
        is_clean=False,
        is_synthetic=True,
        domain=str(clean.get("domain") or "open_clean"),
        error_types=error_types or ["mixed"],
    )


def _synthetic_rows_from_clean_pool(
    clean_rows: list[dict[str, Any]],
    *,
    target_count: int,
    generator: SyntheticGenerator,
    seed: int,
    used_clean_hashes: set[str],
    rule_cap_counts: Counter[str] | None = None,
    error_cap_counts: Counter[str] | None = None,
    max_rule_total: int | None = None,
    max_error_total: int | None = None,
    rule_max_totals: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    pool = list(clean_rows)
    randomizer.shuffle(pool)
    result: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    normalized_counts: Counter[str] = Counter()
    normalized_cap = 1 if target_count >= 1000 else 4
    analyzer = DiffAnalyzer()
    rule_cap_counts = rule_cap_counts if rule_cap_counts is not None else Counter()
    error_cap_counts = error_cap_counts if error_cap_counts is not None else Counter()
    clean_attempt_limit = _synthetic_clean_attempt_limit(len(pool), target_count)
    attempts = 0
    rejected_expensive = 0
    for clean in _cycled(pool, max_iterations=clean_attempt_limit):
        if len(result) >= target_count or not pool:
            break
        attempts += 1
        if attempts == 1 or attempts % 5000 == 0:
            _progress(
                "general_synthetic_scan",
                attempts=attempts,
                rows=len(result),
                target=target_count,
                rejected_expensive=rejected_expensive,
            )
        target = str(clean.get("text", "")).strip()
        if not target:
            continue
        if _diff_pair_too_expensive(target, target):
            rejected_expensive += 1
            continue
        variants = generator.generate_variants_from_clean(target, max_variants=30)
        for example in variants:
            if len(result) >= target_count:
                break
            row = _row_from_synthetic_example(example, clean, analyzer)
            if row is None:
                continue
            if _row_exceeds_caps(
                row,
                rule_cap_counts,
                error_cap_counts,
                max_rule_total=max_rule_total,
                max_error_total=max_error_total,
                rule_max_totals=rule_max_totals,
            ):
                continue
            pair_key = (row["source"], row["target"])
            if pair_key in seen_pairs:
                continue
            normalized_hash = normalized_pair_hash(row["source"], row["target"])
            if normalized_counts[normalized_hash] >= normalized_cap:
                continue
            seen_pairs.add(pair_key)
            normalized_counts[normalized_hash] += 1
            used_clean_hashes.add(str(clean.get("hash") or ""))
            _increment_row_caps(row, rule_cap_counts, error_cap_counts)
            result.append(row)
    if len(result) < target_count:
        retry_attempt_limit = _synthetic_clean_attempt_limit(len(pool), target_count - len(result), retry=True)
        retry_attempts = 0
        for clean in _cycled(pool, max_iterations=retry_attempt_limit):
            if len(result) >= target_count or not pool:
                break
            retry_attempts += 1
            if retry_attempts == 1 or retry_attempts % 5000 == 0:
                _progress(
                    "general_synthetic_retry_scan",
                    attempts=retry_attempts,
                    rows=len(result),
                    target=target_count,
                    rejected_expensive=rejected_expensive,
                )
            target = str(clean.get("text", "")).strip()
            if not target:
                continue
            if _diff_pair_too_expensive(target, target):
                rejected_expensive += 1
                continue
            for example in generator.generate_variants_from_clean(target, max_variants=30):
                if len(result) >= target_count:
                    break
                row = _row_from_synthetic_example(example, clean, analyzer)
                if row is None:
                    continue
                if _row_exceeds_caps(
                    row,
                    rule_cap_counts,
                    error_cap_counts,
                    max_rule_total=max_rule_total,
                    max_error_total=max_error_total,
                    rule_max_totals=rule_max_totals,
                ):
                    continue
                pair_key = (row["source"], row["target"])
                if pair_key in seen_pairs:
                    continue
                normalized_hash = normalized_pair_hash(row["source"], row["target"])
                if normalized_counts[normalized_hash] >= int(4):
                    continue
                seen_pairs.add(pair_key)
                normalized_counts[normalized_hash] += 1
                used_clean_hashes.add(str(clean.get("hash") or ""))
                _increment_row_caps(row, rule_cap_counts, error_cap_counts)
                result.append(row)
    return result


def _row_from_synthetic_example(
    example: SyntheticExample,
    clean: dict[str, Any],
    analyzer: DiffAnalyzer,
) -> dict[str, Any] | None:
    source = str(example.source).strip()
    target = str(example.target).strip()
    if (
        not source
        or source == target
        or _diff_pair_too_expensive(source, target)
        or _contains_known_bad_text(source)
        or _contains_known_bad_text(target)
        or _contains_artificial_marker_text(source, target)
    ):
        return None
    rule_ids = [rule_id for rule_id in (example.rule_ids or []) if _is_active_rule(rule_id)]
    if not rule_ids:
        return None
    if "hyphen_po_adverbs" in rule_ids and _hyphen_po_bad_positive(source, target):
        return None
    if "n_nn_short_form" in rule_ids and _n_nn_short_form_bad_positive(source, target):
        return None
    if any(rule_id in {"asyndetic_dash", "consequence_dash", "enumeration_dash"} for rule_id in rule_ids):
        if re.search(r"\S—\s|\s—\S", target):
            return None
    edits = _lightweight_synthetic_edits(source, target, rule_ids)
    if not edits:
        candidates = []
        edits = [
            edit
            for edit in analyzer.analyze(source, target, candidates=candidates)
            if is_allowed_edit_type(edit.edit_type) and (not edit.rule_id or _is_active_rule(edit.rule_id))
        ]
    edits = _ensure_rule_ids(edits, rule_ids)
    if not edits:
        return None
    error_types = sorted({coarse_error_type(edit.edit_type) for edit in edits if coarse_error_type(edit.edit_type) != "unknown"})
    if not error_types:
        return None
    metadata = _clean_metadata(clean)
    edit_dicts = [asdict(edit) for edit in edits]
    metadata.update(
        _generation_metadata(
            source,
            target,
            edit_dicts,
            _rule_ids_from_edits(edit_dicts) or rule_ids,
            generation_strategy="corpus_opportunity",
            error_bearing_sentence_source="corpus",
        )
    )
    metadata.update({"source_type": SYNTHETIC_OPEN_CLEAN, "synthetic_source_dataset": example.source_dataset})
    return _core_row(
        source=source,
        target=target,
        source_type=SYNTHETIC_OPEN_CLEAN,
        error_type=error_types[0],
        rule_ids=_rule_ids_from_edits(edit_dicts) or rule_ids,
        edits=edit_dicts,
        metadata=metadata,
        original_clean_source=target,
        source_corpus=str(clean.get("source_name") or ""),
        source_subcorpus=str(clean.get("source_subcorpus") or ""),
        is_hard_negative=False,
        is_real_pair=False,
        is_clean=False,
        is_synthetic=True,
        domain=str(clean.get("domain") or "open_clean"),
        error_types=error_types,
    )


def _lightweight_synthetic_edits(source: str, target: str, rule_ids: list[str]) -> list[Edit]:
    """Recover gold edits for generated corpus variants without candidate sweeps."""
    if not rule_ids:
        return []
    opcodes = [opcode for opcode in difflib.SequenceMatcher(a=source, b=target, autojunk=False).get_opcodes() if opcode[0] != "equal"]
    if not opcodes or len(opcodes) > max(5, len(rule_ids) * 2):
        return []
    edits: list[Edit] = []
    for index, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        source_part = source[i1:i2]
        target_part = target[j1:j2]
        rule_id = rule_ids[min(index, len(rule_ids) - 1)]
        edit_type = _lightweight_edit_type(rule_id, tag, source_part, target_part, source_index=i1, source_len=len(source))
        if not is_allowed_edit_type(edit_type):
            return []
        edits.append(
            Edit(
                source_part,
                target_part,
                edit_type,
                i1,
                i2,
                confidence=0.95,
                rule_id=rule_id,
            )
        )
    return edits


def _lightweight_edit_type(
    rule_id: str,
    tag: str,
    source_part: str,
    target_part: str,
    *,
    source_index: int,
    source_len: int,
) -> str:
    combined = f"{source_part}{target_part}"
    if rule_id == "final_punctuation_default" or (
        tag == "insert"
        and source_index >= max(0, source_len - 1)
        and target_part.strip() in {".", "!", "?", "…"}
    ):
        return "final_punctuation"
    if _rule_id_is_punctuation_like(rule_id) or _diff_is_punctuation_only(source_part, target_part):
        if tag == "insert":
            return "punctuation_insert"
        if tag == "delete":
            return "punctuation_delete"
        return "punctuation_replace"
    if rule_id.startswith("capitalization") or (
        source_part
        and target_part
        and source_part.lower() == target_part.lower()
        and source_part != target_part
    ):
        return "case_change"
    if "hyphen" in rule_id or "-" in combined:
        return "hyphen_change"
    if source_part.replace(" ", "") == target_part.replace(" ", "") and source_part.count(" ") != target_part.count(" "):
        return "join_words" if source_part.count(" ") > target_part.count(" ") else "split_word"
    return "spelling_replace"


def _rule_id_is_punctuation_like(rule_id: str) -> bool:
    return any(
        marker in rule_id
        for marker in (
            "comma",
            "dash",
            "colon",
            "semicolon",
            "speech",
            "quote",
            "bracket",
            "punctuation",
        )
    )


def _diff_is_punctuation_only(source_part: str, target_part: str) -> bool:
    text = f"{source_part}{target_part}".strip()
    return bool(text) and not re.search(r"[A-Za-zА-Яа-яЁё0-9]", text)


def _clean_row_key(clean: dict[str, Any]) -> str:
    text = str(clean.get("text") or "")
    return str(clean.get("hash") or hashlib.sha256(text.encode("utf-8")).hexdigest())


def _identity_rows_from_clean_pool(
    clean_rows: list[dict[str, Any]],
    *,
    target_count: int,
    source_type: str,
    used_clean_hashes: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    balanced = _balanced_clean_rows(clean_rows)
    for allow_used in (False, True):
        for clean in balanced:
            if len(result) >= target_count:
                break
            text = str(clean.get("text", "")).strip()
            if not text:
                continue
            if _contains_artificial_marker_text(text):
                continue
            key = _clean_row_key(clean)
            if key in used_clean_hashes and not allow_used:
                continue
            used_clean_hashes.add(key)
            metadata = _clean_metadata(clean)
            metadata["source_type"] = source_type
            metadata["generation_strategy"] = "clean_identity"
            metadata["error_bearing_sentence_source"] = "clean_identity"
            metadata["original_clean_sentence"] = text
            metadata["carrier_sentence_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
            result.append(
                _core_row(
                    source=text,
                    target=text,
                    source_type=source_type,
                    error_type="clean_identity",
                    rule_ids=["clean_identity"],
                    edits=[],
                    metadata=metadata,
                    original_clean_source=text,
                    source_corpus=str(clean.get("source_name") or ""),
                    source_subcorpus=str(clean.get("source_subcorpus") or ""),
                    is_hard_negative=False,
                    is_real_pair=False,
                    is_clean=True,
                    is_synthetic=False,
                    domain=str(clean.get("domain") or "open_clean"),
                    error_types=[],
                )
            )
        if len(result) >= target_count:
            break
    return result


def _hard_negative_rows_from_clean_pool(
    clean_rows: list[dict[str, Any]],
    *,
    target_count: int,
    used_clean_hashes: set[str],
) -> list[dict[str, Any]]:
    forced_traps = [
        ({"text": "Команда работала по старому плану.", "source_name": "bounded_hard_negative", "source_subcorpus": "hyphen_po_adverbs", "domain": "open_clean"}, ["hyphen_po_adverbs_adjective_guard"]),
        ({"text": "Юрист проверил по новому договору несколько пунктов.", "source_name": "bounded_hard_negative", "source_subcorpus": "hyphen_po_adverbs", "domain": "open_clean"}, ["hyphen_po_adverbs_adjective_guard"]),
        ({"text": "Письмо отправили по старому адресу.", "source_name": "bounded_hard_negative", "source_subcorpus": "hyphen_po_adverbs", "domain": "open_clean"}, ["hyphen_po_adverbs_adjective_guard"]),
        ({"text": "Он не согласен с выводом комиссии.", "source_name": "bounded_hard_negative", "source_subcorpus": "ne_short_form", "domain": "open_clean"}, ["ne_short_form_guard"]),
        ({"text": "Редактор не готов подписать документ.", "source_name": "bounded_hard_negative", "source_subcorpus": "ne_short_form", "domain": "open_clean"}, ["ne_short_form_guard"]),
        ({"text": "Секретарь не обязан менять формулировку.", "source_name": "bounded_hard_negative", "source_subcorpus": "ne_short_form", "domain": "open_clean"}, ["ne_short_form_guard"]),
    ]
    positive: list[tuple[dict[str, Any], list[str]]] = []
    fallback: list[tuple[dict[str, Any], list[str]]] = []
    used_positive: list[tuple[dict[str, Any], list[str]]] = []
    used_fallback: list[tuple[dict[str, Any], list[str]]] = []
    for clean in clean_rows:
        text = str(clean.get("text", "")).strip()
        if not text:
            continue
        if _contains_artificial_marker_text(text):
            continue
        traps = detect_hard_negative_traps(text)
        clean_hash = _clean_row_key(clean)
        bucket_pair = (clean, traps if traps else ["natural_clean_guard"])
        if clean_hash and clean_hash in used_clean_hashes:
            if traps:
                used_positive.append(bucket_pair)
            else:
                used_fallback.append(bucket_pair)
        elif traps:
            positive.append(bucket_pair)
        else:
            fallback.append(bucket_pair)
    selected = forced_traps[:target_count]
    selected.extend(_balanced_clean_items(positive, limit=max(0, target_count - len(selected))))
    if len(selected) < target_count:
        selected.extend(_balanced_clean_items(fallback, limit=target_count - len(selected)))
    if len(selected) < target_count:
        selected.extend(_balanced_clean_items(used_positive, limit=target_count - len(selected)))
    if len(selected) < target_count:
        selected.extend(_balanced_clean_items(used_fallback, limit=target_count - len(selected)))
    result: list[dict[str, Any]] = []
    for clean, traps in selected:
        text = str(clean.get("text", "")).strip()
        if _contains_artificial_marker_text(text):
            continue
        used_clean_hashes.add(_clean_row_key(clean))
        metadata = _clean_metadata(clean)
        hard_negative_kind = "trap_candidate" if traps and traps != ["natural_clean_guard"] else "guard_only"
        metadata.update(
            {
                "source_type": HARD_NEGATIVE_OPEN,
                "generation_strategy": "hard_negative",
                "error_bearing_sentence_source": "hard_negative",
                "trap_types": traps,
                "is_hard_negative": True,
                "hard_negative_kind": hard_negative_kind,
                "expected_accepted_edits": 0,
                "trap_rule_id": traps[0] if traps else "",
                "guard_family": traps[0] if traps else "natural_clean_guard",
                "original_clean_sentence": text,
                "carrier_sentence_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:24],
            }
        )
        result.append(
            _core_row(
                source=text,
                target=text,
                source_type=HARD_NEGATIVE_OPEN,
                error_type="hard_negative",
                rule_ids=["clean_identity_hard_negative"],
                edits=[],
                metadata=metadata,
                original_clean_source=text,
                source_corpus=str(clean.get("source_name") or ""),
                source_subcorpus=str(clean.get("source_subcorpus") or ""),
                is_hard_negative=True,
                is_real_pair=False,
                is_clean=True,
                is_synthetic=False,
                domain=str(clean.get("domain") or "open_clean"),
                error_types=[],
            )
        )
    return result


def detect_hard_negative_traps(text: str) -> list[str]:
    lower = text.lower()
    traps: list[str] = []
    if re.search(r"\b\w+(?:ться|тся)\b", lower):
        traps.append("tsya_correct")
    if re.search(r"\b\w*(?:нн|н)\w*\b", lower):
        traps.append("n_nn_correct")
    if re.search(r"\b(?:не|ни)\s+[а-яё]+", lower):
        traps.append("ne_ni_correct")
    if any(marker in lower for marker in ("также", "так же", "тоже", "то же", "чтобы", "что бы", "зато", "за то")):
        traps.append("context_pair_correct")
    if re.search(r"\b[а-яё]+-[а-яё0-9]+", lower):
        traps.append("hyphen_correct")
    if re.search(r"https?://|www\.|[\w.+-]+@[\w-]+\.[\w.-]+", text):
        traps.append("url_email_protected")
    if re.search(r"\d+(?:[,.]\d+)?\s?%|\d+[,.]\d+|\d+-[а-яё]+", lower):
        traps.append("numbers_percent_decimals")
    if re.search(r"\b(?:США|РФ|ООО|АО|ИП|г\.|ул\.|т\.д\.|т\.п\.)\b", text):
        traps.append("abbreviation_correct")
    if "как " in lower or re.search(r"\bчто\b.*\bесли\b|\bесли\b.*\bто\b", lower):
        traps.append("punctuation_trap")
    if any(char in text for char in "«»()[]") or "..." in text or text.endswith(("!", "?")):
        traps.append("quotes_brackets_final_punctuation")
    return sorted(set(traps))


def _ensure_rule_ids(edits: list[Edit], rule_ids: list[str]) -> list[Edit]:
    if not edits:
        return []
    result: list[Edit] = []
    for index, edit in enumerate(edits):
        if edit.rule_id and _is_active_rule(edit.rule_id):
            result.append(edit)
            continue
        rule_id = rule_ids[min(index, len(rule_ids) - 1)]
        result.append(
            Edit(
                source=edit.source,
                replacement=edit.replacement,
                edit_type=edit.edit_type,
                start=edit.start,
                end=edit.end,
                status=edit.status,
                reason=edit.reason,
                confidence=edit.confidence,
                rule_id=rule_id,
            )
        )
    return result


def _is_active_rule(rule_id: str) -> bool:
    if _CURRENT_ACTIVE_RULE_IDS is not None:
        return rule_id in _CURRENT_ACTIVE_RULE_IDS
    from src.data.full_dataset_builder import SHORT_ACTIVE_RULE_IDS, SHORT_EXCLUDED_SYNTHETIC_RULE_IDS

    return rule_id in SHORT_ACTIVE_RULE_IDS and rule_id not in SHORT_EXCLUDED_SYNTHETIC_RULE_IDS


def _active_rule_ids() -> list[str]:
    if _CURRENT_ACTIVE_RULE_IDS is not None:
        return sorted(_CURRENT_ACTIVE_RULE_IDS)
    from src.data.full_dataset_builder import SHORT_ACTIVE_RULE_IDS

    return sorted(SHORT_ACTIVE_RULE_IDS)


def _excluded_rule_ids() -> list[str]:
    from src.data.full_dataset_builder import SHORT_ACTIVE_RULE_IDS, SHORT_EXCLUDED_SYNTHETIC_RULE_IDS

    coverage_rule_ids: set[str] = set()
    inactive: set[str] = set()
    active_statuses = {"implemented", "partial", "deterministic", "candidate_only"}
    for _domain, _group, entry in iter_coverage_entries(load_rules_coverage()):
        status = str(entry.get("status", ""))
        for rule_id in entry.get("rules", []):
            coverage_rule_ids.add(str(rule_id))
            if status not in active_statuses:
                inactive.add(str(rule_id))
    return sorted(SHORT_EXCLUDED_SYNTHETIC_RULE_IDS | inactive | (coverage_rule_ids - SHORT_ACTIVE_RULE_IDS))


def _core_row(
    *,
    source: str,
    target: str,
    source_type: str,
    error_type: str,
    rule_ids: list[str],
    edits: list[dict[str, Any]],
    metadata: dict[str, Any],
    original_clean_source: str,
    source_corpus: str,
    source_subcorpus: str,
    is_hard_negative: bool,
    is_real_pair: bool,
    is_clean: bool,
    is_synthetic: bool,
    domain: str,
    error_types: list[str],
) -> dict[str, Any]:
    rule_ids = [str(rule_id) for rule_id in rule_ids if str(rule_id)] or ["unknown"]
    metadata = dict(metadata)
    metadata.setdefault("source_type", source_type)
    metadata.setdefault("rule_ids", rule_ids)
    source_type = SOURCE_TYPE_ALIASES.get(source_type, source_type)
    return {
        "source": source,
        "target": target,
        "split": "train",
        "source_type": source_type,
        "error_type": error_type,
        "rule_ids": json.dumps(rule_ids, ensure_ascii=False),
        "edits": json.dumps(edits, ensure_ascii=False),
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "original_clean_source": original_clean_source,
        "source_corpus": source_corpus,
        "source_subcorpus": source_subcorpus,
        "is_hard_negative": bool(is_hard_negative),
        "is_real_pair": bool(is_real_pair),
        "template_id": "",
        "normalized_pair_hash": "",
        "error_types": json.dumps(sorted(set(error_types)), ensure_ascii=False),
        "source_dataset": source_corpus or source_type,
        "is_clean": bool(is_clean),
        "is_synthetic": bool(is_synthetic),
        "domain": domain,
        "rule_id": rule_ids[0],
        "edit_operations": json.dumps(edits, ensure_ascii=False),
    }


def _attach_template_fields(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        row["template_id"] = template_id_for_pair(source, target)
        row["normalized_pair_hash"] = normalized_pair_hash(source, target)


def _assign_core_splits(rows: list[dict[str, Any]], split_sizes: dict[str, int], *, core_config: dict[str, Any], seed: int) -> None:
    split_source_targets = core_config.get("split_source_type_targets")
    if isinstance(split_source_targets, dict) and split_source_targets:
        _assign_template_disjoint_splits_with_source_targets(rows, split_sizes, split_source_targets, seed=seed)
    else:
        assign_template_disjoint_splits(rows, split_sizes, seed=seed)
    remaining = {split: int(split_sizes.get(split, 0)) - sum(row.get("split") == split for row in rows) for split in ("train", "val", "test")}
    if any(value != 0 for value in remaining.values()):
        _rebalance_splits(rows, split_sizes, seed=seed)


def _assign_template_disjoint_splits_with_source_targets(
    rows: list[dict[str, Any]],
    split_sizes: dict[str, int],
    split_source_targets: dict[str, Any],
    *,
    seed: int,
) -> None:
    for row in rows:
        row["template_id"] = template_id_for_pair(str(row.get("source", "")), str(row.get("target", "")))
        row["normalized_pair_hash"] = normalized_pair_hash(str(row.get("source", "")), str(row.get("target", "")))
        row["split"] = ""
    source_targets = _normalize_split_source_targets(split_source_targets)
    if not _split_source_targets_match(rows, source_targets):
        assign_template_disjoint_splits(rows, split_sizes, seed=seed)
        return
    remaining_total = {split: int(split_sizes.get(split, 0)) for split in ("train", "val", "test")}
    remaining_source = {split: dict(source_targets.get(split, {})) for split in ("train", "val", "test")}
    for unit in _template_units(rows, seed):
        unit_counts = Counter(SOURCE_TYPE_ALIASES.get(str(row.get("source_type")), str(row.get("source_type"))) for row in unit)
        candidates = [split for split in ("train", "val", "test") if remaining_total[split] >= len(unit)]
        if not candidates:
            raise ValueError(f"template-disjoint split cannot fit unit of size {len(unit)} into remaining {remaining_total}")
        split = min(candidates, key=lambda name: _source_target_penalty(unit_counts, remaining_source[name], remaining_total[name]))
        for row in unit:
            row["split"] = split
        remaining_total[split] -= len(unit)
        for source_type, count in unit_counts.items():
            remaining_source[split][source_type] = remaining_source[split].get(source_type, 0) - count
    if any(value != 0 for value in remaining_total.values()):
        raise ValueError(f"template-disjoint split size mismatch: {remaining_total}")


def _normalize_split_source_targets(raw: dict[str, Any]) -> dict[str, dict[str, int]]:
    normalized: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        values = raw.get(split, {}) or {}
        normalized[split] = {source_type: 0 for source_type in CORE_SOURCE_TYPES}
        for source_type, count in values.items():
            canonical = SOURCE_TYPE_ALIASES.get(str(source_type), str(source_type))
            normalized[split][canonical] = normalized[split].get(canonical, 0) + int(count)
    return normalized


def _split_source_targets_match(rows: list[dict[str, Any]], targets: dict[str, dict[str, int]]) -> bool:
    actual = Counter(SOURCE_TYPE_ALIASES.get(str(row.get("source_type")), str(row.get("source_type"))) for row in rows)
    requested = Counter()
    for values in targets.values():
        requested.update(values)
    return all(actual.get(source_type, 0) == requested.get(source_type, 0) for source_type in CORE_SOURCE_TYPES)


def _source_target_penalty(unit_counts: Counter[str], remaining_source: dict[str, int], remaining_total: int) -> tuple[int, int, int]:
    overshoot = 0
    underfill = 0
    for source_type, count in unit_counts.items():
        after = remaining_source.get(source_type, 0) - count
        if after < 0:
            overshoot += abs(after)
        else:
            underfill += after
    return (overshoot, underfill, -remaining_total)


def _rebalance_splits(rows: list[dict[str, Any]], split_sizes: dict[str, int], *, seed: int) -> None:
    randomizer = random.Random(seed)
    groups = _template_units(rows, seed)
    randomizer.shuffle(groups)
    for row in rows:
        row["split"] = ""
    remaining = {split: int(split_sizes.get(split, 0)) for split in ("train", "val", "test")}
    for group in groups:
        eligible = [split for split, size in remaining.items() if size >= len(group)]
        if not eligible:
            for row in group:
                row["split"] = max(remaining, key=remaining.get)
                remaining[row["split"]] -= 1
            continue
        split = max(eligible, key=lambda name: remaining[name])
        for row in group:
            row["split"] = split
        remaining[split] -= len(group)
    over = [split for split, value in remaining.items() if value < 0]
    under = [split for split, value in remaining.items() if value > 0]
    for split in over:
        movable = [row for row in rows if row["split"] == split]
        randomizer.shuffle(movable)
        while remaining[split] < 0 and under and movable:
            target = under[0]
            row = movable.pop()
            row["split"] = target
            remaining[split] += 1
            remaining[target] -= 1
            if remaining[target] <= 0:
                under.pop(0)
    if any(value != 0 for value in remaining.values()):
        raise ValueError(f"training_dataset_core split assignment failed: {remaining}")


def _template_units(rows: list[dict[str, Any]], seed: int) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("template_id") or template_id_for_pair(row.get("source", ""), row.get("target", "")))].append(row)
    units = list(grouped.values())
    random.Random(seed).shuffle(units)
    units.sort(key=len, reverse=True)
    return units


def _preferred_splits(unit: list[dict[str, Any]], remaining: dict[str, int]) -> list[str]:
    return sorted(remaining, key=lambda split: (-remaining[split], split))


def _proportional_targets(total: int, split_sizes: dict[str, int]) -> dict[str, int]:
    all_total = sum(split_sizes.values()) or 1
    train = int(round(total * split_sizes.get("train", 0) / all_total))
    val = int(round(total * split_sizes.get("val", 0) / all_total))
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def _normalized_pair(source: str, target: str, *, entity_normalize: bool) -> str:
    return f"{_normalize_template_component(source, entity_normalize=entity_normalize)}|||{_normalize_template_component(target, entity_normalize=entity_normalize)}"


def _normalize_template_component(text: str, *, entity_normalize: bool) -> str:
    value = normalize_template_text(str(text or ""))
    value = value.lower().replace("ё", "е")
    value = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", "<date>", value)
    value = re.sub(r"\b\d{4}\s*(?:год[ауе]?|г\.)\b", "<date>", value)
    value = re.sub(r"\d+(?:[,.]\d+)?", "<n>", value)
    value = re.sub(r"[«»„“”\"']", "\"", value)
    if entity_normalize:
        value = re.sub(r"\b[А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z]+){0,2}\b", "<ent>", value)
    value = re.sub(r"\b(?:lenta|лента|taiga|тайга|opencorpora|ruwiki|wiki|wikipedia)\b", "<source>", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return [parsed]
    if value is None:
        return []
    return [value]


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _rule_ids_from_edits(edits: list[Any]) -> list[str]:
    result: list[str] = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        rule_id = str(edit.get("rule_id") or "")
        if rule_id and rule_id not in result:
            result.append(rule_id)
    return result


def _error_types_from_edits(edits: list[Any]) -> list[str]:
    result: list[str] = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        error_type = coarse_error_type(str(edit.get("edit_type") or ""))
        if error_type != "unknown" and error_type not in result:
            result.append(error_type)
    return result


def _clean_metadata(clean: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_name": str(clean.get("source_name") or ""),
        "source_subcorpus": str(clean.get("source_subcorpus") or ""),
        "license_status": str(clean.get("license_status") or clean.get("license/status") or ""),
        "source_doc_id": str(clean.get("source_doc_id") or ""),
        "sentence_id": str(clean.get("sentence_id") or ""),
        "clean_hash": str(clean.get("hash") or ""),
    }


def _generation_metadata(
    source: str,
    target: str,
    edits: list[dict[str, Any]],
    rule_ids: list[str],
    *,
    generation_strategy: str,
    error_bearing_sentence_source: str,
) -> dict[str, Any]:
    first = next((edit for edit in edits if isinstance(edit, dict)), {})
    start = int(first.get("start", -1) or -1) if first else -1
    end = int(first.get("end", -1) or -1) if first else -1
    error_form = str(first.get("source") or "")
    target_form = str(first.get("replacement") or "")
    return {
        "generation_strategy": generation_strategy,
        "error_bearing_sentence_source": error_bearing_sentence_source,
        "candidate_present": True,
        "candidate_rule_ids": sorted(set(str(rule_id) for rule_id in rule_ids if str(rule_id))),
        "target_family": str(rule_ids[0]) if rule_ids else "",
        "original_clean_sentence": target,
        "carrier_sentence_hash": hashlib.sha256(target.encode("utf-8")).hexdigest()[:24],
        "error_bearing_span": [start, end],
        "error_form": error_form,
        "target_form": target_form,
        "normalized_pair_hash": normalized_pair_hash(source, target),
    }


def _hyphen_po_bad_positive(source: str, target: str) -> bool:
    del source
    lower = target.lower()
    bad_patterns = (
        "по-старому плану",
        "по-новому вариант",
        "по-новому договору",
        "по-старому адресу",
    )
    return any(pattern in lower for pattern in bad_patterns)


def _diff_pair_too_expensive(source: str, target: str) -> bool:
    """Avoid quadratic difflib work on long synthetic pairs."""
    if max(len(source), len(target)) > MAX_SYNTHETIC_DIFF_CHARS:
        return True
    if max(len(source.split()), len(target.split())) > MAX_SYNTHETIC_DIFF_WORDS:
        return True
    return False


def _n_nn_short_form_bad_positive(source: str, target: str) -> bool:
    return bool(
        re.search(r"\b(?:цены|страны)\b", source, flags=re.IGNORECASE)
        and re.search(r"\b(?:ценны|странны)\b", target, flags=re.IGNORECASE)
    )


def _synthetic_clean_attempt_limit(pool_size: int, target_count: int, *, retry: bool = False) -> int:
    if target_count <= 0 or pool_size <= 0:
        return 0
    multiplier = 8 if retry else 16
    return min(max(target_count * multiplier, target_count + 1000), max(pool_size * (2 if retry else 4), target_count))


def _cycled(rows: list[dict[str, Any]], *, max_iterations: int | None = None) -> Iterable[dict[str, Any]]:
    if not rows:
        return
    index = 0
    max_iterations = max_iterations if max_iterations is not None else max(len(rows) * 40, len(rows))
    while index < max_iterations:
        yield rows[index % len(rows)]
        index += 1


def _target_shortage_errors(rows: list[dict[str, Any]], targets: dict[str, int]) -> list[str]:
    counts = Counter(str(row.get("source_type")) for row in rows)
    return [
        f"{source_type}_shortage:{counts.get(source_type, 0)}<{target}"
        for source_type, target in targets.items()
        if counts.get(source_type, 0) < target
    ]


def _source_counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {source_type: sum(row.get("source_type") == source_type for row in rows) for source_type in CORE_SOURCE_TYPES}


def _metadata_bearing_source(row: dict[str, Any]) -> str:
    return str(_json_dict(row.get("metadata")).get("error_bearing_sentence_source") or "")


def _synthetic_bearing_counts_from_rows(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("source_type") != SYNTHETIC_OPEN_CLEAN:
            continue
        source = _metadata_bearing_source(row)
        if source:
            counts[source] += 1
    return counts


def _remaining_fallback_template_budget(rows: list[dict[str, Any]], *, max_share: float) -> int:
    counts = _synthetic_bearing_counts_from_rows(rows)
    corpus = int(counts.get("corpus", 0))
    fallback = int(counts.get("fallback_template", 0))
    if max_share <= 0 or max_share >= 1:
        return 0
    # Keep (fallback + x) / (corpus + fallback + x) <= max_share.
    return max(0, int((max_share * corpus / (1.0 - max_share)) - fallback))


def _quality_source_minimum_targets(source_targets: dict[str, int], total: int) -> dict[str, int]:
    minimum = (int(total) + 9) // 10
    real_minimum = max(1, int(source_targets.get(REAL_ERROR_PAIR, 0)))
    return {
        REAL_ERROR_PAIR: real_minimum,
        CLEAN_IDENTITY_OPEN: minimum,
        HARD_NEGATIVE_OPEN: minimum,
    }


def _safe_clean_hard_top_up_rows(
    existing_rows: list[dict[str, Any]],
    *,
    clean_rows: list[dict[str, Any]],
    target_total: int,
    used_clean_hashes: set[str],
) -> list[dict[str, Any]]:
    remaining = max(0, int(target_total) - len(existing_rows))
    if remaining <= 0:
        return []
    counts = _source_counts_from_rows(existing_rows)
    minimum = (int(target_total) + 9) // 10
    clean_needed = max(0, minimum - int(counts.get(CLEAN_IDENTITY_OPEN, 0)))
    hard_needed = max(0, minimum - int(counts.get(HARD_NEGATIVE_OPEN, 0)))
    required = clean_needed + hard_needed
    if required > remaining:
        clean_needed = min(clean_needed, remaining)
        hard_needed = max(0, remaining - clean_needed)
        required = clean_needed + hard_needed
    extra = remaining - required
    clean_needed += extra // 2 + extra % 2
    hard_needed += extra // 2

    result: list[dict[str, Any]] = []
    if clean_needed:
        result.extend(
            _identity_rows_from_clean_pool(
                clean_rows,
                target_count=clean_needed,
                source_type=CLEAN_IDENTITY_OPEN,
                used_clean_hashes=used_clean_hashes,
            )
        )
    if hard_needed:
        result.extend(
            _hard_negative_rows_from_clean_pool(
                clean_rows,
                target_count=hard_needed,
                used_clean_hashes=used_clean_hashes,
            )
        )
    if len(result) < remaining:
        result.extend(
            _identity_rows_from_clean_pool(
                clean_rows,
                target_count=remaining - len(result),
                source_type=CLEAN_IDENTITY_OPEN,
                used_clean_hashes=used_clean_hashes,
            )
        )
    return result[:remaining]


def _actual_split_source_targets(
    core_config: dict[str, Any],
    split_sizes: dict[str, int],
    source_counts: dict[str, int],
) -> dict[str, dict[str, int]]:
    configured = _normalize_split_source_targets(core_config.get("split_source_type_targets", {}) or {})
    if not configured:
        configured = {
            split: {
                source_type: int(round(source_counts.get(source_type, 0) * split_sizes.get(split, 0) / max(1, sum(split_sizes.values()))))
                for source_type in CORE_SOURCE_TYPES
            }
            for split in ("train", "val", "test")
        }
    result = {split: {source_type: 0 for source_type in CORE_SOURCE_TYPES} for split in ("train", "val", "test")}
    for source_type in CORE_SOURCE_TYPES:
        total = int(source_counts.get(source_type, 0))
        configured_total = sum(int(configured.get(split, {}).get(source_type, 0)) for split in ("train", "val", "test"))
        if configured_total <= 0:
            allocations = _proportional_source_split(total, split_sizes)
        elif configured_total == total:
            allocations = {split: int(configured[split].get(source_type, 0)) for split in ("train", "val", "test")}
        else:
            allocations = _scale_split_values({split: int(configured[split].get(source_type, 0)) for split in ("train", "val", "test")}, total)
        for split, count in allocations.items():
            result[split][source_type] = count
    for split in ("train", "val", "test"):
        fixed = sum(result[split][source_type] for source_type in CORE_SOURCE_TYPES if source_type != SYNTHETIC_OPEN_CLEAN)
        result[split][SYNTHETIC_OPEN_CLEAN] = max(0, int(split_sizes.get(split, 0)) - fixed)
    synthetic_delta = int(source_counts.get(SYNTHETIC_OPEN_CLEAN, 0)) - sum(result[split][SYNTHETIC_OPEN_CLEAN] for split in ("train", "val", "test"))
    if synthetic_delta:
        _adjust_split_delta(result, SYNTHETIC_OPEN_CLEAN, synthetic_delta, split_sizes)
    return result


def _proportional_source_split(total: int, split_sizes: dict[str, int]) -> dict[str, int]:
    split_total = max(1, sum(split_sizes.values()))
    values = {split: int(total * split_sizes.get(split, 0) / split_total) for split in ("train", "val", "test")}
    remainder = total - sum(values.values())
    for split in sorted(("train", "val", "test"), key=lambda item: -split_sizes.get(item, 0)):
        if remainder <= 0:
            break
        values[split] += 1
        remainder -= 1
    return values


def _scale_split_values(values: dict[str, int], total: int) -> dict[str, int]:
    raw_total = max(1, sum(values.values()))
    scaled = {split: int(total * values.get(split, 0) / raw_total) for split in ("train", "val", "test")}
    remainder = total - sum(scaled.values())
    for split in sorted(("train", "val", "test"), key=lambda item: -values.get(item, 0)):
        if remainder <= 0:
            break
        scaled[split] += 1
        remainder -= 1
    return scaled


def _adjust_split_delta(
    targets: dict[str, dict[str, int]],
    source_type: str,
    delta: int,
    split_sizes: dict[str, int],
) -> None:
    order = sorted(("train", "val", "test"), key=lambda item: -split_sizes.get(item, 0))
    step = 1 if delta > 0 else -1
    remaining = abs(delta)
    while remaining:
        changed = False
        for split in order:
            if remaining <= 0:
                break
            if step < 0 and targets[split].get(source_type, 0) <= 0:
                continue
            targets[split][source_type] = targets[split].get(source_type, 0) + step
            remaining -= 1
            changed = True
        if not changed:
            break


def _row_exceeds_caps(
    row: dict[str, Any],
    rule_counts: Counter[str],
    error_counts: Counter[str],
    *,
    max_rule_total: int | None,
    max_error_total: int | None,
    rule_max_totals: dict[str, int] | None = None,
) -> bool:
    rule_max_totals = rule_max_totals or {}
    if max_rule_total:
        for rule_id in _json_list(row.get("rule_ids")):
            if str(rule_id) in {"clean_identity", "clean_identity_hard_negative", "unknown", "unknown_real_validated"}:
                continue
            limit = int(rule_max_totals.get(str(rule_id), max_rule_total))
            if rule_counts.get(str(rule_id), 0) >= limit:
                return True
    if max_error_total:
        error_type = str(row.get("error_type") or "")
        if error_type not in {"clean_identity", "hard_negative"} and error_counts.get(error_type, 0) >= max_error_total:
            return True
    return False


def _increment_row_caps(row: dict[str, Any], rule_counts: Counter[str], error_counts: Counter[str]) -> None:
    for rule_id in _json_list(row.get("rule_ids")):
        rule_counts[str(rule_id)] += 1
    error_counts[str(row.get("error_type") or "unknown")] += 1


def _rule_counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for rule_id in _json_list(row.get("rule_ids")):
            counter[str(rule_id)] += 1
    return dict(counter)


def _error_counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("error_type") or "unknown") for row in rows))


def _quota_row_min(row: Any, default: int) -> int:
    if isinstance(row, dict):
        return int(row.get("target_min_total", row.get("min_total", default)) or default)
    return int(default)


def _finalize_quota_state(
    frame: pd.DataFrame,
    quota_state: dict[str, Any],
    *,
    quota_config: dict[str, Any],
    cap_config: dict[str, Any],
) -> dict[str, Any]:
    rule_counts = _rule_id_counts(frame)
    split_counts = _rule_id_counts_by_split(frame)
    min_total = int(quota_config.get("min_total_per_active_rule", 0))
    enforce_split_minimums = len(frame) >= int(quota_config.get("min_rows_for_split_minimums", 1000))
    split_minimums = (
        {
            split: int(count)
            for split, count in dict(quota_config.get("split_minimums", {}) or {}).items()
            if split in {"train", "val", "test"}
        }
        if enforce_split_minimums
        else {}
    )
    excluded = set(quota_state.get("excluded_active_rule_ids", []))
    active_rule_ids = [rule_id for rule_id in list(quota_state.get("active_rule_ids", [])) if rule_id not in excluded]
    quota_by_rule = {row["rule_id"]: dict(row) for row in quota_state.get("quota_rows", [])}
    for rule_id in sorted(set(active_rule_ids) | set(quota_by_rule)):
        row = quota_by_rule.setdefault(
            rule_id,
            {
                "rule_id": rule_id,
                "status": "active",
                "target_min_total": min_total,
                "preferred_total": int(quota_config.get("preferred_total_per_active_rule", min_total)),
                "source": "real/synthetic",
                "action": "ok",
                "reason": "",
            },
        )
        row["final_total"] = int(rule_counts.get(rule_id, 0))
        row["train_count"] = int(split_counts.get("train", {}).get(rule_id, 0))
        row["val_count"] = int(split_counts.get("val", {}).get(rule_id, 0))
        row["test_count"] = int(split_counts.get("test", {}).get(rule_id, 0))
        row_min_total = int(row.get("target_min_total", min_total) or min_total)
        if row["final_total"] < row_min_total and rule_id not in excluded:
            row["status"] = "underfilled"
            row["action"] = "underfilled"
            row["reason"] = row.get("reason") or "below_minimum_after_split"
        elif rule_id not in excluded:
            missing_split = [
                split
                for split, threshold in split_minimums.items()
                if int(row.get(f"{split}_count", 0)) < threshold
            ]
            if missing_split:
                row["status"] = "underfilled"
                row["action"] = "underfilled"
                row["reason"] = "split_minimum_below_threshold:" + ",".join(sorted(missing_split))
    low_active = sorted(
        rule_id
        for rule_id in active_rule_ids
        if int(rule_counts.get(rule_id, 0)) < _quota_row_min(quota_by_rule.get(rule_id, {}), min_total)
        or any(int(split_counts.get(split, {}).get(rule_id, 0)) < threshold for split, threshold in split_minimums.items())
    )
    rule_max_totals = dict(cap_config.get("rule_max_totals", {}) or {})
    capped_rules = sorted(
        rule_id
        for rule_id, count in rule_counts.items()
        if rule_id not in {"clean_identity", "clean_identity_hard_negative", "unknown", "unknown_real_validated"}
        and int(count) >= int(rule_max_totals.get(rule_id, cap_config["generation_rule_cap"]))
    )
    train_errors = _value_counts(frame[frame["split"] == "train"], "error_type")
    capped_errors = sorted(
        error_type
        for error_type, count in train_errors.items()
        if error_type not in {"clean_identity", "hard_negative"} and int(count) >= int(cap_config["generation_error_type_cap"])
    )
    quota_state["quota_rows"] = [quota_by_rule[rule_id] for rule_id in sorted(quota_by_rule)]
    quota_state["active_rule_ids"] = sorted(active_rule_ids)
    quota_state["low_count_active_rule_ids"] = low_active
    quota_state["rule_caps_applied"] = sorted(set(quota_state.get("rule_caps_applied", [])) | set(capped_rules) | (set(DEFAULT_CAPPED_RULE_IDS) & set(rule_counts)))
    quota_state["error_type_caps_applied"] = sorted(set(quota_state.get("error_type_caps_applied", [])) | set(capped_errors))
    quota_state["active_rule_quota_summary"] = {
        "active_rule_count": len(active_rule_ids),
        "excluded_count": len(quota_state.get("excluded_active_rule_ids", [])),
        "underfilled_count": len(low_active),
        "min_total_per_active_rule": min_total,
        "preferred_total_per_active_rule": int(quota_config.get("preferred_total_per_active_rule", min_total)),
    }
    return quota_state


def _write_active_rule_quota_report(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "rule_id",
        "status",
        "target_min_total",
        "preferred_total",
        "final_total",
        "train_count",
        "val_count",
        "test_count",
        "source",
        "action",
        "reason",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _write_excluded_active_rules_report(rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["rule_id", "reason", "previous_status", "new_status", "why_not_generated"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _write_rejected_backfill_templates(rows: list[RejectedBackfillTemplate], path: Path) -> None:
    columns = [
        "rule_id",
        "proposed_source",
        "proposed_target",
        "reason",
        "candidate_count",
        "matching_candidate_found",
        "generated_candidate_rule_ids",
    ]
    records = [
        {
            "rule_id": row.rule_id,
            "proposed_source": row.proposed_source,
            "proposed_target": row.proposed_target,
            "reason": row.reason,
            "candidate_count": row.candidate_count,
            "matching_candidate_found": row.matching_candidate_found,
            "generated_candidate_rule_ids": json.dumps(row.generated_candidate_rule_ids, ensure_ascii=False),
        }
        for row in rows
    ]
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)


def _write_balance_reports(frame: pd.DataFrame, reports_dir: Path) -> None:
    rule_rows: list[dict[str, Any]] = []
    for (split, source_type, error_type), group in frame.groupby(["split", "source_type", "error_type"], dropna=False):
        counter: Counter[str] = Counter()
        for value in group["rule_ids"].tolist():
            for rule_id in _json_list(value):
                counter[str(rule_id)] += 1
        for rule_id, count in sorted(counter.items()):
            rule_rows.append(
                {
                    "split": split,
                    "rule_id": rule_id,
                    "source_type": source_type,
                    "error_type": error_type,
                    "count": count,
                }
            )
    pd.DataFrame(rule_rows).to_csv(reports_dir / "dataset_balance_by_rule.csv", index=False)
    frame.groupby(["split", "error_type", "source_type"], dropna=False).size().reset_index(name="count").to_csv(
        reports_dir / "dataset_balance_by_error_type.csv",
        index=False,
    )
    split_rows = []
    for split in ("train", "val", "test"):
        split_frame = frame[frame["split"] == split]
        row = {"split": split, "total": int(len(split_frame))}
        row.update({source_type: int((split_frame["source_type"] == source_type).sum()) for source_type in CORE_SOURCE_TYPES})
        split_rows.append(row)
    pd.DataFrame(split_rows).to_csv(reports_dir / "dataset_balance_by_split.csv", index=False)


def _write_source_usage_report(frame: pd.DataFrame, path: Path) -> None:
    rows = []
    if not frame.empty:
        grouped = frame.groupby(["source_type", "source_corpus", "source_subcorpus"], dropna=False).size().reset_index(name="count")
        rows = grouped.to_dict("records")
    pd.DataFrame(rows, columns=["source_type", "source_corpus", "source_subcorpus", "count"]).to_csv(path, index=False)


def _write_real_pair_usage_report(frame: pd.DataFrame, path: Path, *, real_result: Any, real_target: int) -> None:
    real = frame[frame["source_type"] == REAL_ERROR_PAIR] if not frame.empty else pd.DataFrame()
    rows = []
    if not real.empty:
        rows = real.groupby(["source_corpus"], dropna=False).size().reset_index(name="included_count").to_dict("records")
    rows.append(
        {
            "source_corpus": "__summary__",
            "included_count": int(len(real)),
            "accepted_count": int(getattr(real_result, "accepted_count", len(real))),
            "rejected_count": int(getattr(real_result, "rejected_count", 0)),
            "target": int(real_target),
            "shortage": max(0, int(real_target) - int(len(real))),
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_hard_negative_coverage_report(frame: pd.DataFrame, path: Path) -> None:
    rows = []
    hard = frame[frame["source_type"] == HARD_NEGATIVE_OPEN] if not frame.empty else pd.DataFrame()
    for _idx, row in hard.iterrows():
        metadata = _json_dict(row.get("metadata"))
        traps = _json_list(metadata.get("trap_types")) or ["natural_clean_guard"]
        kind = "trap_candidate" if traps and traps != ["natural_clean_guard"] else "guard_only"
        for trap in traps:
            rows.append(
                {
                    "hard_negative_kind": kind,
                    "trap_rule_id": str(trap),
                    "guard_family": str(trap),
                    "count": 1,
                    "expected_accepted_edits": 0,
                }
            )
    if rows:
        report = pd.DataFrame(rows).groupby(["hard_negative_kind", "trap_rule_id", "guard_family", "expected_accepted_edits"], dropna=False).size().reset_index(name="count")
    else:
        report = pd.DataFrame(columns=["hard_negative_kind", "trap_rule_id", "guard_family", "expected_accepted_edits", "count"])
    report.to_csv(path, index=False)


def _write_template_leakage_report(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    train = set(frame.loc[frame["split"] == "train", "template_id"].astype(str))
    rows = []
    summary = {}
    for split in ("val", "test"):
        split_ids = set(frame.loc[frame["split"] == split, "template_id"].astype(str))
        overlap = split_ids & train
        rate = len(overlap) / max(1, len(split_ids))
        summary[f"{split}_overlap_with_train_rate"] = float(rate)
        rows.append(
            {
                "split": split,
                "template_count": len(split_ids),
                "overlap_with_train_count": len(overlap),
                "overlap_with_train_rate": rate,
                "sample_template_ids": json.dumps(sorted(overlap)[:20], ensure_ascii=False),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return summary


def _write_template_quality_report(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    synthetic = frame[frame["source_type"] == SYNTHETIC_OPEN_CLEAN]
    synthetic_text = "\n".join((synthetic["source"].astype(str) + "\n" + synthetic["target"].astype(str)).tolist()).lower()
    all_text = "\n".join((frame["source"].astype(str) + "\n" + frame["target"].astype(str)).tolist()).lower()
    meta_counts = {phrase: int(synthetic_text.count(phrase.lower())) for phrase in META_LANGUAGE_PATTERNS}
    meta_counts["technical_rule_names"] = sum(int(synthetic_text.count(marker)) for marker in TECHNICAL_RULE_MARKERS)
    suspicious = {phrase: int(all_text.count(phrase.lower())) for phrase in SUSPICIOUS_PHRASES}
    examples = []
    for _idx, row in frame.iterrows():
        combined = f"{row['source']} {row['target']}".lower()
        if any(phrase in combined for phrase in META_LANGUAGE_PATTERNS + SUSPICIOUS_PHRASES):
            examples.append({"source": row["source"], "target": row["target"], "source_type": row["source_type"]})
        if len(examples) >= 10:
            break
    verdict = "passed" if all(count == 0 for count in suspicious.values()) else "failed"
    lines = [
        "# Template Quality Report",
        "",
        f"- verdict: {verdict}",
        f"- meta_language_counts: {json.dumps(meta_counts, ensure_ascii=False, sort_keys=True)}",
        f"- suspicious_template_counts: {json.dumps(suspicious, ensure_ascii=False, sort_keys=True)}",
        "",
        "## Examples",
        "",
        json.dumps(examples, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"meta_language_counts": meta_counts, "suspicious_template_counts": suspicious, "examples": examples, "verdict": verdict}


def _manifest(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    core_config: dict[str, Any],
    clean_result: Any,
    real_result: Any,
    recall_reports: dict[str, pd.DataFrame],
    template_leakage: dict[str, Any],
    template_quality: dict[str, Any],
    real_target: int,
    real_shortage: int,
    shortage_errors: list[str],
    requested_total: int,
    requested_split_sizes: dict[str, int],
    reports_dir: Path,
    quota_state: dict[str, Any] | None = None,
    quality_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quota_state = quota_state or {}
    quality_audit = quality_audit or {}
    active_rule_ids = sorted(str(rule_id) for rule_id in (quota_state.get("active_rule_ids") or _active_rule_ids()))
    excluded_active_rule_ids = sorted(str(rule_id) for rule_id in quota_state.get("excluded_active_rule_ids", []))
    excluded_rule_ids = sorted(set(_excluded_rule_ids()) | set(excluded_active_rule_ids))
    inactive_rule_ids = sorted(set(excluded_rule_ids) - set(active_rule_ids))
    normalized_counts = Counter(frame["normalized_pair_hash"].astype(str))
    synthetic = frame[frame["source_type"] == SYNTHETIC_OPEN_CLEAN]
    synthetic_norm_counts = Counter(synthetic["normalized_pair_hash"].astype(str))
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
    composition = _value_counts(frame, "source_type", keys=CORE_SOURCE_TYPES)
    stress_count = _stress_row_count(frame)
    composition_with_aliases = dict(composition)
    composition_with_aliases["clean_identity"] = int(composition.get(CLEAN_IDENTITY_OPEN, 0))
    composition_with_aliases["hard_negative"] = int(composition.get(HARD_NEGATIVE_OPEN, 0))
    composition_with_aliases["real_error_pair"] = int(composition.get(REAL_ERROR_PAIR, 0))
    composition_with_aliases["multi_error_stress"] = int(stress_count)
    manifest = {
        "total": int(len(frame)),
        "requested_total": int(requested_total),
        "requested_split_sizes": requested_split_sizes,
        "split_sizes": _value_counts(frame, "split", keys=("train", "val", "test")),
        "composition": composition_with_aliases,
        "composition_by_split": _counts_by_split(frame, "source_type", keys=CORE_SOURCE_TYPES),
        "source_type_counts_by_split": _counts_by_split(frame, "source_type", keys=CORE_SOURCE_TYPES),
        "error_type_counts": _value_counts(frame, "error_type"),
        "error_type_counts_by_split": _counts_by_split(frame, "error_type"),
        "rule_id_counts": _rule_id_counts(frame),
        "rule_id_counts_by_split": _rule_id_counts_by_split(frame),
        "clean_source_counts": _value_counts(frame[frame["source_type"] != REAL_ERROR_PAIR], "source_corpus"),
        "clean_source_counts_by_split": _counts_by_split(frame[frame["source_type"] != REAL_ERROR_PAIR], "source_corpus"),
        "real_source_counts": _value_counts(frame[frame["source_type"] == REAL_ERROR_PAIR], "source_corpus"),
        "real_source_counts_by_split": _counts_by_split(frame[frame["source_type"] == REAL_ERROR_PAIR], "source_corpus"),
        "hard_negative_count": int(composition.get(HARD_NEGATIVE_OPEN, 0)),
        "clean_identity_count": int(composition.get(CLEAN_IDENTITY_OPEN, 0)),
        "real_pair_count": int(composition.get(REAL_ERROR_PAIR, 0)),
        "stress_count": int(stress_count),
        "hard_negative_accepted_bad_edits": 0,
        "candidate_recall_summary": recall_summary,
        "gap_label_coverage_summary": gap_summary,
        "template_leakage_summary": template_leakage,
        "normalized_pair_unique_count": int(len(normalized_counts)),
        "normalized_pair_duplicate_rate": _duplicate_rate(normalized_counts, len(frame)),
        "synthetic_normalized_pair_duplicate_rate": _duplicate_rate(synthetic_norm_counts, len(synthetic)),
        "top_normalized_pair_count": int(max(normalized_counts.values()) if normalized_counts else 0),
        "meta_language_counts": template_quality["meta_language_counts"],
        "suspicious_template_counts": template_quality["suspicious_template_counts"],
        "corpus_opportunity_share": float(quality_audit.get("corpus_opportunity_share", 0.0)),
        "fallback_template_share": float(quality_audit.get("fallback_template_share", 1.0)),
        "known_quality_bugs": dict(quality_audit.get("known_quality_bugs", {}) or {}),
        "artificial_marker_counts": dict(
            quality_audit.get(
                "artificial_marker_counts",
                artificial_marker_counts(frame),
            )
            or {}
        ),
        "exact_clean_hard_duplicate_count": int(quality_audit.get("exact_clean_hard_duplicate_count", 0) or 0),
        "error_bearing_sentence_source_counts": dict(quality_audit.get("error_bearing_sentence_source_counts", {}) or {}),
        "rule_diversity_summary": dict(quality_audit.get("rule_diversity_summary", {}) or {}),
        "extended_quality_audit_summary": dict(quality_audit.get("extended_quality_audit_summary", {}) or {}),
        "extended_quality_issue_count": int(dict(quality_audit.get("extended_quality_audit_summary", {}) or {}).get("issue_count", 0) or 0),
        "real_pair_acceptance_rate": real_result.accepted_count / max(1, real_result.accepted_count + real_result.rejected_count),
        "rejected_real_pair_reasons": dict(sorted(real_result.rejection_reason_counts.items())),
        "source_ingestion_summary": {
            "accepted_clean_sentences": clean_result.accepted_count,
            "total_seen": clean_result.total_seen,
            "shortage_reason": clean_result.shortage_reason,
            "dominance_violations": clean_result.dominance_violations,
            "source_reports": clean_result.source_reports,
            "real_target": real_target,
            "real_shortage": real_shortage,
            "accepted_real_pairs": real_result.accepted_count,
            "rejected_real_pairs": real_result.rejected_count,
        },
        "real_pair_shortage_reason": f"accepted_real_pairs_below_target:{real_target - real_shortage}<{real_target}" if real_shortage else "",
        "active_rule_ids": active_rule_ids,
        "inactive_rule_ids": inactive_rule_ids,
        "excluded_rule_ids": excluded_rule_ids,
        "active_rule_quota_summary": quota_state.get("active_rule_quota_summary", {}),
        "low_count_active_rule_ids": quota_state.get("low_count_active_rule_ids", []),
        "underfilled_rule_ids": quota_state.get("low_count_active_rule_ids", []),
        "excluded_active_rule_ids": excluded_active_rule_ids,
        "rule_caps_applied": quota_state.get("rule_caps_applied", []),
        "error_type_caps_applied": quota_state.get("error_type_caps_applied", []),
        "unknown_count": int(_rule_id_counts(frame).get("unknown", 0)),
        "seed": int(config.get("data", {}).get("synthetic_seed", 17)),
        "config_path": str(config.get("data", {}).get("config_path", "configs/config.yaml")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    audit_errors = _audit_errors(
        frame,
        manifest=manifest,
        core_config=core_config,
        clean_result=clean_result,
        shortage_errors=shortage_errors,
        quality_audit=quality_audit,
    )
    manifest["warnings"] = _audit_warnings(manifest=manifest, core_config=core_config, clean_result=clean_result)
    manifest["audit_errors"] = audit_errors
    manifest["verdict"] = "DATASET_BLOCKED" if audit_errors else "READY_FOR_TRAINING_DATASET"
    return manifest


def _audit_errors(
    frame: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    core_config: dict[str, Any],
    clean_result: Any,
    shortage_errors: list[str],
    quality_audit: dict[str, Any] | None = None,
) -> list[str]:
    audit = dict(core_config.get("audit", {}) or {})
    errors = list(shortage_errors)
    expected_total = int(manifest.get("requested_total", manifest["total"]))
    if manifest["total"] != expected_total:
        errors.append(f"dataset_size_below_requested:{manifest['total']}!={expected_total}")
    preferred_clean_min = int(core_config.get("min_clean_pool_for_ready", 150_000))
    hard_clean_min = min(preferred_clean_min, int(core_config.get("min_clean_pool_hard_min", min(60_000, preferred_clean_min))))
    if clean_result.accepted_count < hard_clean_min:
        errors.append(f"clean_pool_below_min:{clean_result.accepted_count}")
    errors.extend(clean_result.dominance_violations)
    enforce_template_gates = manifest["total"] >= int(audit.get("min_rows_for_template_gates", 1000))
    if enforce_template_gates:
        if float(manifest["template_leakage_summary"].get("val_overlap_with_train_rate", 0.0)) > 0.03:
            errors.append("template_leakage_val_above_threshold")
        if float(manifest["template_leakage_summary"].get("test_overlap_with_train_rate", 0.0)) > 0.03:
            errors.append("template_leakage_test_above_threshold")
        if float(manifest["synthetic_normalized_pair_duplicate_rate"]) > 0.25:
            errors.append("synthetic_normalized_duplicate_rate_above_threshold")
        if int(manifest["top_normalized_pair_count"]) > 20:
            errors.append("top_normalized_pair_count_above_threshold")
    if any(int(count) != 0 for count in manifest["suspicious_template_counts"].values()):
        errors.append("suspicious_template_phrase_present")
    if float(manifest.get("corpus_opportunity_share", 0.0) or 0.0) < float(audit.get("corpus_opportunity_share_min", 0.70)):
        errors.append("corpus_opportunity_share_below_threshold")
    if float(manifest.get("fallback_template_share", 1.0)) > float(audit.get("fallback_template_share_max", 0.20)):
        errors.append("fallback_template_share_above_threshold")
    for name, count in dict(manifest.get("known_quality_bugs", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"known_quality_bugs_present:{name}")
    for name, count in dict(manifest.get("artificial_marker_counts", {}) or {}).items():
        if int(count) != 0:
            errors.append(f"artificial_marker_present:{name}")
    if not dict(manifest.get("error_bearing_sentence_source_counts", {}) or {}):
        errors.append("missing_error_bearing_sentence_source_counts")
    if int(dict(manifest.get("rule_diversity_summary", {}) or {}).get("failed_rule_count", 0) or 0) != 0:
        errors.append("rule_diversity_gates_failed")
    if int(dict(manifest.get("extended_quality_audit_summary", {}) or {}).get("blocking_issue_count", 0) or 0) != 0:
        errors.append("extended_quality_audit_blocking_issues")
    enforce_candidate_gates = manifest["total"] >= int(audit.get("min_rows_for_candidate_gates", 1000))
    if enforce_candidate_gates:
        if float(manifest["candidate_recall_summary"].get("active_min_excluding_unknown", 1.0)) < float(audit.get("candidate_recall_min", 0.85)):
            errors.append("candidate_recall_active_min_below_threshold")
        if float(manifest["gap_label_coverage_summary"].get("active_min_excluding_unknown", 1.0)) < float(audit.get("gap_coverage_min", 0.85)):
            errors.append("gap_coverage_active_min_below_threshold")
    min_active_rule_count = int(audit.get("min_active_rule_count", 50))
    if min_active_rule_count > 0:
        low = [
            rule_id
            for rule_id in manifest["active_rule_ids"]
            if int(manifest["rule_id_counts"].get(rule_id, 0)) < min_active_rule_count
        ]
        if low:
            errors.append("active_rule_count_below_min:" + ",".join(low))
    if manifest.get("low_count_active_rule_ids"):
        errors.append("active_rule_quota_underfilled:" + ",".join(manifest["low_count_active_rule_ids"]))
    for source_name, count in dict(manifest.get("clean_source_counts", {}) or {}).items():
        if manifest["total"] and int(count) > manifest["total"] * 0.70:
            errors.append(f"source_dominance_above_70_percent:{source_name}")
    if manifest["total"] >= int(audit.get("min_rows_for_eval_source_split_gates", 1000)):
        eval_clean_min = int(audit.get("min_clean_identity_eval_split", 500))
        eval_hard_min = int(audit.get("min_hard_negative_eval_split", 500))
        by_split = manifest["composition_by_split"]
        for split in ("val", "test"):
            if int(by_split.get(split, {}).get(CLEAN_IDENTITY_OPEN, 0)) < eval_clean_min:
                errors.append(f"clean_identity_eval_split_below_min:{split}")
            if int(by_split.get(split, {}).get(HARD_NEGATIVE_OPEN, 0)) < eval_hard_min:
                errors.append(f"hard_negative_eval_split_below_min:{split}")
    if bool(audit.get("require_all_source_types", True)):
        for source_type in CORE_SOURCE_TYPES:
            if int(manifest["composition"].get(source_type, 0)) <= 0:
                errors.append(f"missing_source_type:{source_type}")
    return errors


def _audit_warnings(*, manifest: dict[str, Any], core_config: dict[str, Any], clean_result: Any) -> list[str]:
    warnings: list[str] = []
    preferred_clean_min = int(core_config.get("min_clean_pool_for_ready", 150_000))
    hard_clean_min = min(preferred_clean_min, int(core_config.get("min_clean_pool_hard_min", min(60_000, preferred_clean_min))))
    if hard_clean_min <= clean_result.accepted_count < preferred_clean_min:
        warnings.append(f"clean_pool_below_preferred:{clean_result.accepted_count}<{preferred_clean_min}")
    if manifest.get("real_pair_shortage_reason"):
        warnings.append(str(manifest["real_pair_shortage_reason"]))
    return warnings


def _metric_summary(frame: pd.DataFrame, *, count_column: str, metric_column: str, active_rule_ids: set[str]) -> dict[str, Any]:
    if frame.empty or count_column not in frame:
        return {
            "rules_with_gold": 0,
            "active_rules_with_gold": 0,
            "min_excluding_unknown": 1.0,
            "mean_excluding_unknown": 1.0,
            "active_min_excluding_unknown": 1.0,
            "active_mean_excluding_unknown": 1.0,
        }
    working = frame.copy()
    working[count_column] = pd.to_numeric(working[count_column], errors="coerce").fillna(0)
    working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce").fillna(0.0)
    non_unknown = working[(working["rule_id"] != "unknown") & (working[count_column] > 0)]
    active = non_unknown[non_unknown["rule_id"].isin(active_rule_ids)]
    return {
        "rules_with_gold": int(len(non_unknown)),
        "active_rules_with_gold": int(len(active)),
        "min_excluding_unknown": _safe_min(non_unknown, metric_column),
        "mean_excluding_unknown": _safe_mean(non_unknown, metric_column),
        "active_min_excluding_unknown": _safe_min(active, metric_column),
        "active_mean_excluding_unknown": _safe_mean(active, metric_column),
    }


def _punctuation_rule_ids() -> set[str]:
    result: set[str] = set()
    for domain, _group, entry in iter_coverage_entries(load_rules_coverage()):
        if domain == "punctuation":
            result.update(str(rule_id) for rule_id in entry.get("rules", []))
    return result


def _safe_min(frame: pd.DataFrame, column: str) -> float:
    return 1.0 if frame.empty else float(frame[column].min())


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    return 1.0 if frame.empty else float(frame[column].mean())


def _value_counts(frame: pd.DataFrame, column: str, keys: Iterable[str] | None = None) -> dict[str, int]:
    counts = Counter(str(value) for value in frame[column].fillna("").tolist()) if not frame.empty and column in frame else Counter()
    result = {str(key): int(counts.get(str(key), 0)) for key in keys} if keys else {}
    for key, count in sorted(counts.items()):
        if key:
            result[key] = int(count)
    return result


def _counts_by_split(frame: pd.DataFrame, column: str, keys: Iterable[str] | None = None) -> dict[str, dict[str, int]]:
    return {
        split: _value_counts(frame[frame["split"] == split], column, keys=keys)
        for split in ("train", "val", "test")
    }


def _rule_id_counts(frame: pd.DataFrame) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in frame.get("rule_ids", pd.Series(dtype=str)).tolist():
        for rule_id in _json_list(value):
            counter[str(rule_id)] += 1
    return dict(sorted(counter.items()))


def _rule_id_counts_by_split(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {split: _rule_id_counts(frame[frame["split"] == split]) for split in ("train", "val", "test")}


def _stress_row_count(frame: pd.DataFrame) -> int:
    if frame.empty or "metadata" not in frame:
        return 0
    return int(sum(bool(_json_dict(value).get("is_stress")) for value in frame["metadata"].tolist()))


def _duplicate_rate(counts: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    duplicates = sum(max(0, count - 1) for count in counts.values())
    return float(duplicates / total)


def _write_generation_report(frame: pd.DataFrame, path: Path, manifest: dict[str, Any]) -> None:
    rule_counts = manifest["rule_id_counts"]
    low_count_active = list(manifest.get("low_count_active_rule_ids", []))
    lines = [
        "# Training Dataset Generation Report",
        "",
        f"- verdict: {manifest['verdict']}",
        f"- total: {manifest['total']}",
        f"- split_sizes: {json.dumps(manifest['split_sizes'], ensure_ascii=False, sort_keys=True)}",
        f"- composition: {json.dumps(manifest['composition'], ensure_ascii=False, sort_keys=True)}",
        f"- clean_source_counts: {json.dumps(manifest['clean_source_counts'], ensure_ascii=False, sort_keys=True)}",
        f"- real_source_counts: {json.dumps(manifest['real_source_counts'], ensure_ascii=False, sort_keys=True)}",
        f"- real_pair_acceptance_rate: {manifest['real_pair_acceptance_rate']:.6f}",
        f"- candidate_recall_min_mean: {manifest['candidate_recall_summary'].get('active_min_excluding_unknown', 1.0):.6f} / {manifest['candidate_recall_summary'].get('active_mean_excluding_unknown', 1.0):.6f}",
        f"- gap_coverage_min_mean: {manifest['gap_label_coverage_summary'].get('active_min_excluding_unknown', 1.0):.6f} / {manifest['gap_label_coverage_summary'].get('active_mean_excluding_unknown', 1.0):.6f}",
        f"- template_leakage: {json.dumps(manifest['template_leakage_summary'], ensure_ascii=False, sort_keys=True)}",
        f"- synthetic_normalized_duplicate_rate: {manifest['synthetic_normalized_pair_duplicate_rate']:.6f}",
        f"- top_normalized_pair_count: {manifest['top_normalized_pair_count']}",
        f"- meta_language_counts: {json.dumps(manifest['meta_language_counts'], ensure_ascii=False, sort_keys=True)}",
        f"- suspicious_template_counts: {json.dumps(manifest['suspicious_template_counts'], ensure_ascii=False, sort_keys=True)}",
        f"- corpus_opportunity_share: {manifest.get('corpus_opportunity_share', 0.0):.6f}",
        f"- fallback_template_share: {manifest.get('fallback_template_share', 0.0):.6f}",
        f"- known_quality_bugs: {json.dumps(manifest.get('known_quality_bugs', {}), ensure_ascii=False, sort_keys=True)}",
        f"- rule_diversity_summary: {json.dumps(manifest.get('rule_diversity_summary', {}), ensure_ascii=False, sort_keys=True)}",
        f"- extended_quality_audit_summary: {json.dumps(manifest.get('extended_quality_audit_summary', {}), ensure_ascii=False, sort_keys=True)}",
        f"- active_rule_quota_summary: {json.dumps(manifest.get('active_rule_quota_summary', {}), ensure_ascii=False, sort_keys=True)}",
        f"- low_count_active_rule_ids: {', '.join(low_count_active)}",
        f"- excluded_active_rule_ids: {', '.join(manifest.get('excluded_active_rule_ids', []))}",
        f"- rule_caps_applied: {', '.join(manifest.get('rule_caps_applied', []))}",
        f"- error_type_caps_applied: {', '.join(manifest.get('error_type_caps_applied', []))}",
        "",
        "## Top 30 Rule Counts",
        "",
    ]
    for rule_id, count in sorted(rule_counts.items(), key=lambda item: (-item[1], item[0]))[:30]:
        lines.append(f"- {rule_id}: {count}")
    if manifest["audit_errors"]:
        lines.extend(["", "## Audit Errors", ""])
        lines.extend(f"- {error}" for error in manifest["audit_errors"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
