from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
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
from src.data.real_error_sources import load_real_error_pairs
from src.data.synthetic_generator import SyntheticExample, SyntheticGenerator
from src.evaluation.candidate_recall import (
    CANDIDATE_RECALL_COLUMNS,
    GAP_LABEL_COVERAGE_COLUMNS,
    build_candidate_recall_reports,
)
from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type


V2_SOURCE_TYPES = ("synthetic_augmented", "real_error_pair", "clean_identity", "hard_negative")
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
    generator = SyntheticGenerator(seed=seed, max_errors_per_sentence=int(v2_config.get("max_errors_per_sentence", 2)))
    candidate_generator = CandidateGenerator.from_config(config)

    clean_config = _clean_source_config(config, v2_config)
    clean_result = build_clean_sentence_pool(
        clean_config,
        output_path=output_dir / "clean_sentence_pool.csv.gz",
        reports_dir=reports_dir,
    )
    clean_rows = _read_records(output_dir / "clean_sentence_pool.csv.gz")

    real_config = _real_source_config(config, v2_config)
    real_result = load_real_error_pairs(
        real_config,
        candidate_generator=candidate_generator,
        output_path=output_dir / "real_error_pairs_validated.csv.gz",
        reports_dir=reports_dir,
    )

    rows: list[dict[str, Any]] = []
    used_clean_hashes: set[str] = set()
    real_target = int(source_targets.get("real_error_pair", 0))
    rows.extend(_real_rows(real_result.rows[:real_target]))
    accepted_real = len([row for row in rows if row["source_type"] == "real_error_pair"])
    real_shortage = max(0, real_target - accepted_real)
    source_targets["synthetic_augmented"] = int(source_targets.get("synthetic_augmented", 0)) + real_shortage
    source_targets["real_error_pair"] = accepted_real

    synthetic_target = int(source_targets.get("synthetic_augmented", 0))
    synthetic_rows = _synthetic_rows_from_clean_pool(
        clean_rows,
        target_count=synthetic_target,
        generator=generator,
        seed=seed,
        used_clean_hashes=used_clean_hashes,
    )
    rows.extend(synthetic_rows)

    clean_identity_target = int(source_targets.get("clean_identity", 0))
    rows.extend(
        _identity_rows_from_clean_pool(
            clean_rows,
            target_count=clean_identity_target,
            source_type="clean_identity",
            used_clean_hashes=used_clean_hashes,
        )
    )

    hard_negative_target = int(source_targets.get("hard_negative", 0))
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
    _assign_v2_splits(rows, effective_split_sizes, v2_config=v2_config if len(rows) == total else {}, seed=seed)
    _attach_template_fields(rows)

    frame = pd.DataFrame(rows, columns=V2_COLUMNS)
    frame.to_csv(output_path, index=False)
    for split in ("train", "val", "test"):
        frame[frame["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)

    recall_reports = build_candidate_recall_reports(
        frame.to_dict("records"),
        candidate_generator=candidate_generator,
        max_candidates=int(config.get("model", {}).get("max_candidates", 16)),
    )
    recall_reports["candidate_recall_by_rule"].to_csv(reports_dir / "candidate_recall_by_rule.csv", index=False)
    recall_reports["gap_label_coverage_by_rule"].to_csv(reports_dir / "gap_label_coverage_by_rule.csv", index=False)
    _write_balance_reports(frame, reports_dir)
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
            "synthetic_augmented": 38_400,
            "real_error_pair": 6_000,
            "clean_identity": 7_800,
            "hard_negative": 7_800,
        }
    if not raw:
        raw = {
            "synthetic_augmented": int(round(total * 0.64)),
            "real_error_pair": int(round(total * 0.10)),
            "clean_identity": int(round(total * 0.13)),
        }
        raw["hard_negative"] = total - sum(raw.values())
    result = {source_type: int(raw.get(source_type, 0)) for source_type in V2_SOURCE_TYPES}
    delta = total - sum(result.values())
    result["synthetic_augmented"] += delta
    return result


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict("records")


def _real_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        edits = _json_list(row.get("edits") or row.get("edit_operations"))
        rule_ids = _rule_ids_from_edits(edits) or _json_list(row.get("rule_ids")) or [str(row.get("rule_id") or "unknown")]
        error_types = _error_types_from_edits(edits) or _json_list(row.get("error_types")) or [str(row.get("error_type") or "unknown")]
        metadata = _json_dict(row.get("metadata"))
        metadata.update({"source_type": "real_error_pair", "candidate_present": bool(row.get("candidate_present", True))})
        result.append(
            _v2_row(
                source=str(row.get("source", "")),
                target=str(row.get("target", "")),
                source_type="real_error_pair",
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


def _synthetic_rows_from_clean_pool(
    clean_rows: list[dict[str, Any]],
    *,
    target_count: int,
    generator: SyntheticGenerator,
    seed: int,
    used_clean_hashes: set[str],
) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    pool = list(clean_rows)
    randomizer.shuffle(pool)
    result: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    analyzer = DiffAnalyzer()
    for clean in _cycled(pool):
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
            pair_key = (row["source"], row["target"])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            used_clean_hashes.add(str(clean.get("hash") or ""))
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
    metadata.update({"source_type": "synthetic_augmented", "synthetic_source_dataset": example.source_dataset})
    return _v2_row(
        source=source,
        target=target,
        source_type="synthetic_augmented",
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
        metadata.update({"source_type": "hard_negative", "trap_types": traps})
        result.append(
            _v2_row(
                source=text,
                target=text,
                source_type="hard_negative",
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
        for row in rows:
            row["split"] = ""
        for source_type in V2_SOURCE_TYPES:
            source_rows = [row for row in rows if row.get("source_type") == source_type]
            targets = {
                split: int((split_source_targets.get(split, {}) or {}).get(source_type, 0))
                for split in ("train", "val", "test")
            }
            if sum(targets.values()) != len(source_rows):
                targets = _proportional_targets(len(source_rows), split_sizes)
            assign_template_disjoint_splits(source_rows, targets, seed=seed + len(source_type))
    else:
        assign_template_disjoint_splits(rows, split_sizes, seed=seed)
    remaining = {split: int(split_sizes.get(split, 0)) - sum(row.get("split") == split for row in rows) for split in ("train", "val", "test")}
    if any(value != 0 for value in remaining.values()):
        _rebalance_splits(rows, split_sizes, seed=seed)


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
        "license_status": str(clean.get("license/status") or ""),
        "source_doc_id": str(clean.get("source_doc_id") or ""),
        "sentence_id": str(clean.get("sentence_id") or ""),
        "clean_hash": str(clean.get("hash") or ""),
    }


def _cycled(rows: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    if not rows:
        return
    index = 0
    max_iterations = max(len(rows) * 40, len(rows))
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
    synthetic = frame[frame["source_type"] == "synthetic_augmented"]
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
) -> dict[str, Any]:
    active_rule_ids = _active_rule_ids()
    excluded_rule_ids = _excluded_rule_ids()
    inactive_rule_ids = sorted(set(excluded_rule_ids) - set(active_rule_ids))
    normalized_counts = Counter(frame["normalized_pair_hash"].astype(str))
    synthetic = frame[frame["source_type"] == "synthetic_augmented"]
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
        active_rule_ids=set(active_rule_ids),
    )
    composition = _value_counts(frame, "source_type", keys=V2_SOURCE_TYPES)
    manifest = {
        "total": int(len(frame)),
        "requested_total": int(requested_total),
        "requested_split_sizes": requested_split_sizes,
        "split_sizes": _value_counts(frame, "split", keys=("train", "val", "test")),
        "composition": composition,
        "composition_by_split": _counts_by_split(frame, "source_type", keys=V2_SOURCE_TYPES),
        "error_type_counts": _value_counts(frame, "error_type"),
        "error_type_counts_by_split": _counts_by_split(frame, "error_type"),
        "rule_id_counts": _rule_id_counts(frame),
        "rule_id_counts_by_split": _rule_id_counts_by_split(frame),
        "clean_source_counts": _value_counts(frame[frame["source_type"] != "real_error_pair"], "source_corpus"),
        "clean_source_counts_by_split": _counts_by_split(frame[frame["source_type"] != "real_error_pair"], "source_corpus"),
        "real_source_counts": _value_counts(frame[frame["source_type"] == "real_error_pair"], "source_corpus"),
        "real_source_counts_by_split": _counts_by_split(frame[frame["source_type"] == "real_error_pair"], "source_corpus"),
        "hard_negative_count": int(composition.get("hard_negative", 0)),
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
        "active_rule_ids": active_rule_ids,
        "inactive_rule_ids": inactive_rule_ids,
        "excluded_rule_ids": excluded_rule_ids,
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
    if clean_result.accepted_count < int(v2_config.get("min_clean_pool_for_ready", 150_000)):
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
    if float(manifest["candidate_recall_summary"].get("active_min_excluding_unknown", 1.0)) < float(audit.get("candidate_recall_min", 0.85)):
        errors.append("candidate_recall_active_min_below_threshold")
    if float(manifest["gap_label_coverage_summary"].get("active_min_excluding_unknown", 1.0)) < float(audit.get("gap_coverage_min", 0.85)):
        errors.append("gap_coverage_active_min_below_threshold")
    min_active_rule_count = int(audit.get("min_active_rule_count", 50))
    if min_active_rule_count > 0:
        low = [
            rule_id
            for rule_id in manifest["active_rule_ids"]
            if 0 < int(manifest["rule_id_counts"].get(rule_id, 0)) < min_active_rule_count
        ]
        if low:
            errors.append("active_rule_count_below_min:" + ",".join(low))
    if bool(audit.get("require_all_source_types", True)):
        for source_type in V2_SOURCE_TYPES:
            if int(manifest["composition"].get(source_type, 0)) <= 0:
                errors.append(f"missing_source_type:{source_type}")
    return errors


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
    low_count_active = [
        rule_id
        for rule_id in manifest["active_rule_ids"]
        if 0 < int(rule_counts.get(rule_id, 0)) < 50
    ]
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
        f"- low_count_active_rule_ids: {', '.join(low_count_active)}",
        "",
        "## Top 20 Rule Counts",
        "",
    ]
    for rule_id, count in sorted(rule_counts.items(), key=lambda item: (-item[1], item[0]))[:20]:
        lines.append(f"- {rule_id}: {count}")
    if manifest["audit_errors"]:
        lines.extend(["", "## Audit Errors", ""])
        lines.extend(f"- {error}" for error in manifest["audit_errors"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
