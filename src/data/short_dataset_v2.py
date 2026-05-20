from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
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
from src.data.clean_sentence_pool import (
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
V2_SOURCE_TYPES = (SYNTHETIC_OPEN_CLEAN, REAL_ERROR_PAIR, CLEAN_IDENTITY_OPEN, HARD_NEGATIVE_OPEN)
SOURCE_TYPE_ALIASES = {
    "synthetic_augmented": SYNTHETIC_OPEN_CLEAN,
    SYNTHETIC_OPEN_CLEAN: SYNTHETIC_OPEN_CLEAN,
    "real_error_pair": REAL_ERROR_PAIR,
    "clean_identity": CLEAN_IDENTITY_OPEN,
    CLEAN_IDENTITY_OPEN: CLEAN_IDENTITY_OPEN,
    "hard_negative": HARD_NEGATIVE_OPEN,
    HARD_NEGATIVE_OPEN: HARD_NEGATIVE_OPEN,
}
V2_COLUMNS = [
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


def build_short_dataset_v2_from_config(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    data_config = config.get("data", {})
    v2_config = data_config.get("short_dataset_v2", {}) or {}
    seed = int(data_config.get("synthetic_seed", v2_config.get("seed", 17)))
    output_path = Path(data_config.get("processed_train_path") or "data/processed/short_dataset_v2/correction_dataset.csv.gz")
    output_dir = output_path.parent
    reports_dir = Path(config.get("paths", {}).get("reports_dir") or "reports/short_dataset_v2")
    manifest_path = Path(data_config.get("manifest_path") or reports_dir / "dataset_manifest.json")
    split_sizes = _split_sizes(data_config)
    total = sum(split_sizes.values())
    if total <= 0:
        total = int(data_config.get("target_total_examples", 60_000))
        split_sizes = {"train": 50_000, "val": 5_000, "test": 5_000} if total == 60_000 else _ratio_split(total)
    requested_total = total
    requested_split_sizes = dict(split_sizes)
    source_targets = _source_type_targets(v2_config, total)

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
    clean_config = _clean_source_config(config, v2_config)
    real_config = _real_source_config(config, v2_config)
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

    generator = SyntheticGenerator(seed=seed, max_errors_per_sentence=int(v2_config.get("max_errors_per_sentence", 2)))
    candidate_generator = CandidateGenerator.from_config(config)

    clean_result = build_clean_sentence_pool(
        clean_config,
        output_path=output_dir / "clean_sentence_pool.csv.gz",
        reports_dir=reports_dir,
    )
    clean_rows = _read_records(output_dir / "clean_sentence_pool.csv.gz")

    real_output_path = output_dir / "real_error_pairs_validated.csv.gz"
    real_result = _load_or_reuse_real_error_pairs(
        real_config,
        candidate_generator=candidate_generator,
        output_path=real_output_path,
        reports_dir=reports_dir,
        v2_config=v2_config,
    )

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
    quota_config = _active_rule_quota_config(v2_config)
    cap_config = _rule_cap_config(v2_config, split_sizes)
    active_rule_ids = _effective_active_rule_ids(config, v2_config, quota_config=quota_config)
    quota_rows, quota_state = _build_active_rule_quota_rows(
        rows,
        clean_rows=strict_clean_rows,
        active_rule_ids=active_rule_ids,
        quota_config=quota_config,
        cap_config=cap_config,
        candidate_generator=candidate_generator,
        seed=seed,
        synthetic_budget=int(source_targets.get(SYNTHETIC_OPEN_CLEAN, 0)),
    )
    rows.extend(quota_rows)

    synthetic_target = int(source_targets.get(SYNTHETIC_OPEN_CLEAN, 0))
    synthetic_remaining = max(0, synthetic_target - len([row for row in rows if row["source_type"] == SYNTHETIC_OPEN_CLEAN]))
    general_synthetic_target = min(synthetic_remaining, int(v2_config.get("max_general_synthetic_fill", 6000)))
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
    )
    rows.extend(synthetic_rows)
    synthetic_remaining = max(0, synthetic_target - len([row for row in rows if row["source_type"] == SYNTHETIC_OPEN_CLEAN]))
    if synthetic_remaining:
        fill_rows, fill_state = _fill_remaining_with_targeted_rows(
            clean_rows=strict_clean_rows,
            active_rule_ids=quota_state["active_rule_ids"],
            target_count=synthetic_remaining,
            candidate_generator=candidate_generator,
            seed=seed + 101,
            existing_rows=rows,
            cap_config=cap_config,
        )
        rows.extend(fill_rows)
        quota_state["rejected_templates"].extend(fill_state["rejected_templates"])

    clean_identity_target = int(source_targets.get(CLEAN_IDENTITY_OPEN, 0))
    rows.extend(
        _identity_rows_from_clean_pool(
            strict_clean_rows,
            target_count=clean_identity_target,
            source_type=CLEAN_IDENTITY_OPEN,
            used_clean_hashes=used_clean_hashes,
        )
    )

    hard_negative_target = int(source_targets.get(HARD_NEGATIVE_OPEN, 0))
    rows.extend(
        _hard_negative_rows_from_clean_pool(
            clean_rows,
            target_count=hard_negative_target,
            used_clean_hashes=used_clean_hashes,
        )
    )

    shortage_errors = _target_shortage_errors(rows, source_targets)
    rows = rows[:total]
    _attach_template_fields(rows)
    effective_split_sizes = split_sizes if len(rows) == total else _proportional_targets(len(rows), split_sizes)
    split_source_targets = _actual_split_source_targets(v2_config, effective_split_sizes, _source_counts_from_rows(rows))
    split_v2_config = dict(v2_config) if len(rows) == total else {}
    if split_source_targets:
        split_v2_config["split_source_type_targets"] = split_source_targets
    _assign_v2_splits(rows, effective_split_sizes, v2_config=split_v2_config, seed=seed)
    _attach_template_fields(rows)

    frame = pd.DataFrame(rows, columns=V2_COLUMNS)
    frame.to_csv(output_path, index=False)
    for split in ("train", "val", "test"):
        frame[frame["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)

    recall_reports = build_candidate_recall_reports(
        frame.to_dict("records"),
        candidate_generator=candidate_generator,
        rules_config_path="configs/rules.yaml",
    )
    recall_reports["candidate_recall_by_rule"].to_csv(reports_dir / "candidate_recall_by_rule.csv", index=False)
    recall_reports["gap_label_coverage_by_rule"].to_csv(reports_dir / "gap_label_coverage_by_rule.csv", index=False)
    _write_balance_reports(frame, reports_dir)
    quota_state = _finalize_quota_state(frame, quota_state, quota_config=quota_config, cap_config=cap_config)
    _write_active_rule_quota_report(quota_state["quota_rows"], reports_dir / "active_rule_quota_report.csv")
    _write_excluded_active_rules_report(quota_state["excluded_rows"], reports_dir / "excluded_active_rules_report.csv")
    _write_rejected_backfill_templates(quota_state["rejected_templates"], reports_dir / "rejected_backfill_templates.csv")
    template_leakage = _write_template_leakage_report(frame, reports_dir / "template_leakage_report.csv")
    template_quality = _write_template_quality_report(frame, reports_dir / "template_quality_report.md")

    manifest = _manifest(
        frame,
        config=config,
        v2_config=v2_config,
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
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_generation_report(frame, reports_dir / "dataset_generation_report.md", manifest)

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


def _clean_source_config(config: dict[str, Any], v2_config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(v2_config.get("open_corpora_sources"), dict):
        source_config = dict(v2_config["open_corpora_sources"])
    else:
        path = Path(str(v2_config.get("open_corpora_sources_path") or "configs/open_corpora_sources.yaml"))
        with path.open("r", encoding="utf-8") as handle:
            source_config = yaml.safe_load(handle) or {}
    pool = dict(source_config.get("pool", {}) or {})
    pool.update(dict(v2_config.get("pool", {}) or {}))
    source_config["pool"] = pool
    return source_config


def _real_source_config(config: dict[str, Any], v2_config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(v2_config.get("real_error_sources"), dict):
        return dict(v2_config["real_error_sources"])
    path = Path(str(v2_config.get("real_error_sources_path") or "configs/real_error_sources.yaml"))
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _precheck_external_sources(clean_config: dict[str, Any], real_config: dict[str, Any], *, reports_dir: Path) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    downloads_allowed = _downloads_allowed(clean_config) or _downloads_allowed(real_config)
    if downloads_allowed:
        _prepare_materialized_real_sources(real_config)
    missing.extend(_missing_source_specs(clean_config, key="clean_sources"))
    missing.extend(_missing_source_specs(real_config, key="real_sources"))
    if missing:
        pd.DataFrame(missing).to_csv(reports_dir / "missing_external_sources.csv", index=False)
    return {"ready": not missing, "missing": missing, "downloads_allowed": downloads_allowed}


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
        "config_path": str(config.get("data", {}).get("config_path", "configs/config.short_dataset_v2.yaml")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_blocked_generation_report(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Short Dataset V2 Generation Report",
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


def _active_rule_quota_config(v2_config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(v2_config.get("active_rule_quota", {}) or {})
    audit = dict(v2_config.get("audit", {}) or {})
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


def _rule_cap_config(v2_config: dict[str, Any], split_sizes: dict[str, int]) -> dict[str, Any]:
    raw = dict(v2_config.get("rule_caps", {}) or {})
    max_total = int(raw.get("max_total_per_rule_id", 2500))
    max_train = int(raw.get("max_train_per_rule_id", 2000))
    max_error_share = float(raw.get("max_error_type_share_train", 0.35))
    return {
        "max_total_per_rule_id": max_total,
        "max_train_per_rule_id": max_train,
        "max_rule_share_train": float(raw.get("max_rule_share_train", 0.10)),
        "max_error_type_share_train": max_error_share,
        "generation_rule_cap": min(max_total, max_train),
        "generation_error_type_cap": max(1, int(split_sizes.get("train", 0) * max_error_share)),
    }


def _effective_active_rule_ids(
    config: dict[str, Any],
    v2_config: dict[str, Any],
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
    if bool(v2_config.get("include_yo_e_candidate", False)) and bool(config.get("dictionary", {}).get("yo_e", {}).get("enabled", False)):
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


def _source_type_targets(v2_config: dict[str, Any], total: int) -> dict[str, int]:
    raw = dict(v2_config.get("source_type_targets", {}) or {})
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
    result = {source_type: int(normalized_raw.get(source_type, 0)) for source_type in V2_SOURCE_TYPES}
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
        edits = _json_list(row.get("edits") or row.get("edit_operations"))
        rule_ids = _rule_ids_from_edits(edits) or _json_list(row.get("rule_ids")) or [str(row.get("rule_id") or "unknown")]
        error_types = _error_types_from_edits(edits) or _json_list(row.get("error_types")) or [str(row.get("error_type") or "unknown")]
        metadata = _json_dict(row.get("metadata"))
        metadata.update({"source_type": REAL_ERROR_PAIR, "candidate_present": bool(row.get("candidate_present", True))})
        result.append(
            _v2_row(
                source=str(row.get("source", "")),
                target=str(row.get("target", "")),
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
    v2_config: dict[str, Any],
) -> RealErrorLoadResult:
    min_cached = int(v2_config.get("min_cached_real_pairs", 1000))
    if bool(v2_config.get("reuse_validated_real_pairs_cache", True)) and output_path.exists():
        cached_rows = _read_records(output_path)
        if len(cached_rows) >= min_cached:
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
    return load_real_error_pairs(
        real_config,
        candidate_generator=candidate_generator,
        output_path=output_path,
        reports_dir=reports_dir,
    )


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
    clean_cycle = list(clean_rows) or [{"text": "", "source_name": "", "source_subcorpus": "", "domain": "open_clean"}]
    seen_pairs = {(str(row.get("source", "")), str(row.get("target", ""))) for row in existing_rows}
    current_counts = Counter(_rule_counts_from_rows(existing_rows))
    rule_cap_counts = Counter(current_counts)
    error_cap_counts = Counter(_error_counts_from_rows(existing_rows))
    budget_left = max(0, int(synthetic_budget))

    for rule_id in active_rule_ids:
        current = int(current_counts.get(rule_id, 0))
        rule_preferred = _preferred_total_for_rule(rule_id, min_total=min_total, preferred=preferred)
        needed = max(0, rule_preferred - current)
        generated_count = 0
        action = "ok"
        reason = ""
        if needed > 0 and budget_left > 0:
            needed = min(needed, budget_left)
            result = backfill_generator.generate_for_rule(rule_id, needed, seen_pairs=seen_pairs)
            rejected_templates.extend(result.rejected)
            for example in result.examples:
                if budget_left <= 0:
                    break
                row = _row_from_targeted_example(example, clean_cycle[(len(analyzer_rows) + len(existing_rows)) % len(clean_cycle)])
                if _row_exceeds_caps(
                    row,
                    rule_cap_counts,
                    error_cap_counts,
                    max_rule_total=int(cap_config["generation_rule_cap"]),
                    max_error_total=int(cap_config["generation_error_type_cap"]),
                ):
                    continue
                _increment_row_caps(row, rule_cap_counts, error_cap_counts)
                analyzer_rows.append(row)
                budget_left -= 1
                generated_count += 1
            action = "backfilled" if generated_count else "ok"
            reason = result.excluded_reason
            current_counts.update(_rule_counts_from_rows(analyzer_rows[-generated_count:]))
        final_count = int(current_counts.get(rule_id, 0))
        if final_count < min_total or (needed > 0 and generated_count == 0 and reason):
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
                "target_min_total": min_total,
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
    clean_cycle = list(clean_rows) or [{"text": "", "source_name": "", "source_subcorpus": "", "domain": "open_clean"}]
    seen_pairs = {(str(row.get("source", "")), str(row.get("target", ""))) for row in existing_rows}
    rule_cap_counts = Counter(_rule_counts_from_rows(existing_rows))
    error_cap_counts = Counter(_error_counts_from_rows(existing_rows))
    normalized_counts = Counter(normalized_pair_hash(str(row.get("source", "")), str(row.get("target", ""))) for row in existing_rows)
    rows: list[dict[str, Any]] = []
    rejected: list[RejectedBackfillTemplate] = []
    example_pools: dict[str, list[TargetedBackfillExample]] = {}
    for rule_id in active_rule_ids:
        result = generator.generate_for_rule(rule_id, 220, seen_pairs=set())
        rejected.extend(result.rejected[:5])
        if result.examples:
            example_pools[rule_id] = result.examples
    if not example_pools:
        return rows, {"rejected_templates": rejected}

    attempts = 0
    duplicate_cap = int(cap_config.get("targeted_duplicate_cap", 20))
    ordered_rules = sorted(example_pools, key=lambda item: rule_cap_counts.get(item, 0))
    while len(rows) < target_count and attempts < max(target_count * 8, 1000):
        rule_id = min(ordered_rules, key=lambda item: rule_cap_counts.get(item, 0))
        pool = example_pools[rule_id]
        example = pool[attempts % len(pool)]
        row = _row_from_targeted_example(example, clean_cycle[(len(rows) + len(existing_rows)) % len(clean_cycle)])
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
        ):
            attempts += 1
            continue
        seen_pairs.add(pair_key)
        normalized_counts[normalized_hash] += 1
        _increment_row_caps(row, rule_cap_counts, error_cap_counts)
        rows.append(row)
        attempts += 1
    return rows, {"rejected_templates": rejected}


def _row_from_targeted_example(example: TargetedBackfillExample, clean: dict[str, Any]) -> dict[str, Any]:
    metadata = _clean_metadata(clean)
    metadata.update(example.metadata)
    error_types = sorted({coarse_error_type(str(edit.get("edit_type") or "")) for edit in example.edits})
    error_types = [error_type for error_type in error_types if error_type != "unknown"]
    carrier = _targeted_carrier_sentence(clean, example.target)
    source = _append_targeted_carrier(example.source, carrier)
    target = _append_targeted_carrier(example.target, carrier)
    return _v2_row(
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
    return text


def _append_targeted_carrier(text: str, carrier: str) -> str:
    if not carrier:
        return text
    value = text.strip()
    if not value.endswith((".", "!", "?", "…")):
        value += "."
    return f"{value} {carrier}"


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
    for clean in _cycled(pool, max_iterations=clean_attempt_limit):
        if len(result) >= target_count or not pool:
            break
        target = str(clean.get("text", "")).strip()
        if not target:
            continue
        variants = generator.generate_variants_from_clean(target, max_variants=30)
        for example in variants:
            if len(result) >= target_count:
                break
            row = _row_from_synthetic_example(example, clean, analyzer)
            if row is None:
                continue
            if _row_exceeds_caps(row, rule_cap_counts, error_cap_counts, max_rule_total=max_rule_total, max_error_total=max_error_total):
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
        for clean in _cycled(pool, max_iterations=retry_attempt_limit):
            if len(result) >= target_count or not pool:
                break
            target = str(clean.get("text", "")).strip()
            if not target:
                continue
            for example in generator.generate_variants_from_clean(target, max_variants=30):
                if len(result) >= target_count:
                    break
                row = _row_from_synthetic_example(example, clean, analyzer)
                if row is None:
                    continue
                if _row_exceeds_caps(row, rule_cap_counts, error_cap_counts, max_rule_total=max_rule_total, max_error_total=max_error_total):
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
    if not source or source == target:
        return None
    rule_ids = [rule_id for rule_id in (example.rule_ids or []) if _is_active_rule(rule_id)]
    if not rule_ids:
        return None
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
    metadata.update({"source_type": SYNTHETIC_OPEN_CLEAN, "synthetic_source_dataset": example.source_dataset})
    return _v2_row(
        source=source,
        target=target,
        source_type=SYNTHETIC_OPEN_CLEAN,
        error_type=error_types[0],
        rule_ids=_rule_ids_from_edits([asdict(edit) for edit in edits]) or rule_ids,
        edits=[asdict(edit) for edit in edits],
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


def _identity_rows_from_clean_pool(
    clean_rows: list[dict[str, Any]],
    *,
    target_count: int,
    source_type: str,
    used_clean_hashes: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for clean in clean_rows:
        if len(result) >= target_count:
            break
        text = str(clean.get("text", "")).strip()
        if not text:
            continue
        used_clean_hashes.add(str(clean.get("hash") or ""))
        metadata = _clean_metadata(clean)
        metadata["source_type"] = source_type
        result.append(
            _v2_row(
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
    return result


def _hard_negative_rows_from_clean_pool(
    clean_rows: list[dict[str, Any]],
    *,
    target_count: int,
    used_clean_hashes: set[str],
) -> list[dict[str, Any]]:
    positive: list[tuple[dict[str, Any], list[str]]] = []
    fallback: list[tuple[dict[str, Any], list[str]]] = []
    for clean in clean_rows:
        text = str(clean.get("text", "")).strip()
        if not text:
            continue
        traps = detect_hard_negative_traps(text)
        if traps:
            positive.append((clean, traps))
        else:
            fallback.append((clean, ["natural_clean_guard"]))
    selected = (positive + fallback)[:target_count]
    result: list[dict[str, Any]] = []
    for clean, traps in selected:
        text = str(clean.get("text", "")).strip()
        used_clean_hashes.add(str(clean.get("hash") or ""))
        metadata = _clean_metadata(clean)
        metadata.update({"source_type": HARD_NEGATIVE_OPEN, "trap_types": traps})
        result.append(
            _v2_row(
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
    from src.data.full_dataset_builder import SHORT_ACTIVE_RULE_IDS, SHORT_EXCLUDED_SYNTHETIC_RULE_IDS

    return rule_id in SHORT_ACTIVE_RULE_IDS and rule_id not in SHORT_EXCLUDED_SYNTHETIC_RULE_IDS


def _active_rule_ids() -> list[str]:
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


def _v2_row(
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


def _assign_v2_splits(rows: list[dict[str, Any]], split_sizes: dict[str, int], *, v2_config: dict[str, Any], seed: int) -> None:
    split_source_targets = v2_config.get("split_source_type_targets")
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
        normalized[split] = {source_type: 0 for source_type in V2_SOURCE_TYPES}
        for source_type, count in values.items():
            canonical = SOURCE_TYPE_ALIASES.get(str(source_type), str(source_type))
            normalized[split][canonical] = normalized[split].get(canonical, 0) + int(count)
    return normalized


def _split_source_targets_match(rows: list[dict[str, Any]], targets: dict[str, dict[str, int]]) -> bool:
    actual = Counter(SOURCE_TYPE_ALIASES.get(str(row.get("source_type")), str(row.get("source_type"))) for row in rows)
    requested = Counter()
    for values in targets.values():
        requested.update(values)
    return all(actual.get(source_type, 0) == requested.get(source_type, 0) for source_type in V2_SOURCE_TYPES)


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
        raise ValueError(f"short_dataset_v2 split assignment failed: {remaining}")


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
    return {source_type: sum(row.get("source_type") == source_type for row in rows) for source_type in V2_SOURCE_TYPES}


def _actual_split_source_targets(
    v2_config: dict[str, Any],
    split_sizes: dict[str, int],
    source_counts: dict[str, int],
) -> dict[str, dict[str, int]]:
    configured = _normalize_split_source_targets(v2_config.get("split_source_type_targets", {}) or {})
    if not configured:
        configured = {
            split: {
                source_type: int(round(source_counts.get(source_type, 0) * split_sizes.get(split, 0) / max(1, sum(split_sizes.values()))))
                for source_type in V2_SOURCE_TYPES
            }
            for split in ("train", "val", "test")
        }
    result = {split: {source_type: 0 for source_type in V2_SOURCE_TYPES} for split in ("train", "val", "test")}
    for source_type in V2_SOURCE_TYPES:
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
        fixed = sum(result[split][source_type] for source_type in V2_SOURCE_TYPES if source_type != SYNTHETIC_OPEN_CLEAN)
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
) -> bool:
    if max_rule_total:
        for rule_id in _json_list(row.get("rule_ids")):
            if str(rule_id) in {"clean_identity", "clean_identity_hard_negative", "unknown", "unknown_real_validated"}:
                continue
            if rule_counts.get(str(rule_id), 0) >= max_rule_total:
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
    active_rule_ids = list(quota_state.get("active_rule_ids", []))
    excluded = set(quota_state.get("excluded_active_rule_ids", []))
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
        if row["final_total"] < min_total and rule_id not in excluded:
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
        if int(rule_counts.get(rule_id, 0)) < min_total
        or any(int(split_counts.get(split, {}).get(rule_id, 0)) < threshold for split, threshold in split_minimums.items())
    )
    capped_rules = sorted(
        rule_id
        for rule_id, count in rule_counts.items()
        if rule_id not in {"clean_identity", "clean_identity_hard_negative", "unknown", "unknown_real_validated"}
        and int(count) >= int(cap_config["generation_rule_cap"])
    )
    train_errors = _value_counts(frame[frame["split"] == "train"], "error_type")
    capped_errors = sorted(
        error_type
        for error_type, count in train_errors.items()
        if error_type not in {"clean_identity", "hard_negative"} and int(count) >= int(cap_config["generation_error_type_cap"])
    )
    quota_state["quota_rows"] = [quota_by_rule[rule_id] for rule_id in sorted(quota_by_rule)]
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
        row.update({source_type: int((split_frame["source_type"] == source_type).sum()) for source_type in V2_SOURCE_TYPES})
        split_rows.append(row)
    pd.DataFrame(split_rows).to_csv(reports_dir / "dataset_balance_by_split.csv", index=False)


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
    v2_config: dict[str, Any],
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
) -> dict[str, Any]:
    quota_state = quota_state or {}
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
    composition = _value_counts(frame, "source_type", keys=V2_SOURCE_TYPES)
    manifest = {
        "total": int(len(frame)),
        "requested_total": int(requested_total),
        "requested_split_sizes": requested_split_sizes,
        "split_sizes": _value_counts(frame, "split", keys=("train", "val", "test")),
        "composition": composition,
        "composition_by_split": _counts_by_split(frame, "source_type", keys=V2_SOURCE_TYPES),
        "source_type_counts_by_split": _counts_by_split(frame, "source_type", keys=V2_SOURCE_TYPES),
        "error_type_counts": _value_counts(frame, "error_type"),
        "error_type_counts_by_split": _counts_by_split(frame, "error_type"),
        "rule_id_counts": _rule_id_counts(frame),
        "rule_id_counts_by_split": _rule_id_counts_by_split(frame),
        "clean_source_counts": _value_counts(frame[frame["source_type"] != REAL_ERROR_PAIR], "source_corpus"),
        "clean_source_counts_by_split": _counts_by_split(frame[frame["source_type"] != REAL_ERROR_PAIR], "source_corpus"),
        "real_source_counts": _value_counts(frame[frame["source_type"] == REAL_ERROR_PAIR], "source_corpus"),
        "real_source_counts_by_split": _counts_by_split(frame[frame["source_type"] == REAL_ERROR_PAIR], "source_corpus"),
        "hard_negative_count": int(composition.get(HARD_NEGATIVE_OPEN, 0)),
        "candidate_recall_summary": recall_summary,
        "gap_label_coverage_summary": gap_summary,
        "template_leakage_summary": template_leakage,
        "normalized_pair_unique_count": int(len(normalized_counts)),
        "normalized_pair_duplicate_rate": _duplicate_rate(normalized_counts, len(frame)),
        "synthetic_normalized_pair_duplicate_rate": _duplicate_rate(synthetic_norm_counts, len(synthetic)),
        "top_normalized_pair_count": int(max(normalized_counts.values()) if normalized_counts else 0),
        "meta_language_counts": template_quality["meta_language_counts"],
        "suspicious_template_counts": template_quality["suspicious_template_counts"],
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
        "excluded_active_rule_ids": excluded_active_rule_ids,
        "rule_caps_applied": quota_state.get("rule_caps_applied", []),
        "error_type_caps_applied": quota_state.get("error_type_caps_applied", []),
        "unknown_count": int(_rule_id_counts(frame).get("unknown", 0)),
        "seed": int(config.get("data", {}).get("synthetic_seed", 17)),
        "config_path": str(config.get("data", {}).get("config_path", "configs/config.short_dataset_v2.yaml")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    audit_errors = _audit_errors(
        frame,
        manifest=manifest,
        v2_config=v2_config,
        clean_result=clean_result,
        shortage_errors=shortage_errors,
    )
    manifest["warnings"] = _audit_warnings(manifest=manifest, v2_config=v2_config, clean_result=clean_result)
    manifest["audit_errors"] = audit_errors
    manifest["verdict"] = "BLOCKED" if audit_errors else "READY_FOR_SHORT_TRAINING_DATASET_V2"
    return manifest


def _audit_errors(
    frame: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    v2_config: dict[str, Any],
    clean_result: Any,
    shortage_errors: list[str],
) -> list[str]:
    audit = dict(v2_config.get("audit", {}) or {})
    errors = list(shortage_errors)
    expected_total = int(manifest.get("requested_total", manifest["total"]))
    if manifest["total"] != expected_total:
        errors.append(f"dataset_size_below_requested:{manifest['total']}!={expected_total}")
    preferred_clean_min = int(v2_config.get("min_clean_pool_for_ready", 150_000))
    hard_clean_min = min(preferred_clean_min, int(v2_config.get("min_clean_pool_hard_min", min(60_000, preferred_clean_min))))
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
        for source_type in V2_SOURCE_TYPES:
            if int(manifest["composition"].get(source_type, 0)) <= 0:
                errors.append(f"missing_source_type:{source_type}")
    return errors


def _audit_warnings(*, manifest: dict[str, Any], v2_config: dict[str, Any], clean_result: Any) -> list[str]:
    warnings: list[str] = []
    preferred_clean_min = int(v2_config.get("min_clean_pool_for_ready", 150_000))
    hard_clean_min = min(preferred_clean_min, int(v2_config.get("min_clean_pool_hard_min", min(60_000, preferred_clean_min))))
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


def _duplicate_rate(counts: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    duplicates = sum(max(0, count - 1) for count in counts.values())
    return float(duplicates / total)


def _write_generation_report(frame: pd.DataFrame, path: Path, manifest: dict[str, Any]) -> None:
    rule_counts = manifest["rule_id_counts"]
    low_count_active = list(manifest.get("low_count_active_rule_ids", []))
    lines = [
        "# Short Dataset V2 Generation Report",
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
