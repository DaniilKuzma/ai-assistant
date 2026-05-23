from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import csv
import gzip
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd
import yaml
from rapidfuzz.distance import Levenshtein

from src.candidates.candidate_generator import CandidateGenerator
from src.candidates.matching import candidate_matches_edit
from src.data.clean_sentence_pool import cyrillic_ratio
from src.data.sage_sources import downloads_enabled
from src.data.source_downloads import DownloadBudget, SourceDownloadResult, download_if_allowed
from src.preprocessing.protected_spans import find_protected_spans
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type
from src.validation.strict_validator import StrictValidator


REAL_PAIR_COLUMNS = [
    "source",
    "target",
    "source_dataset",
    "source_subdataset",
    "domain",
    "detected_error_types",
    "candidate_present",
    "candidate_rule_ids",
    "edit_count",
    "char_edit_ratio",
    "token_edit_ratio",
    "is_real_pair",
    "metadata",
    "raw_id",
    "raw_source_path",
    "error_types",
    "error_type",
    "source_type",
    "is_clean",
    "is_hard_negative",
    "is_synthetic",
    "split",
    "rule_id",
    "rule_ids",
    "edit_operations",
    "edits",
    "dataset_layer",
    "is_stress",
    "count_toward_rule_quota",
    "loss_weight",
    "routing_category",
    "routing_reason",
    "strict_validator_passed",
]

REJECTED_PAIR_COLUMNS = [
    "source_dataset",
    "source",
    "target",
    "reason",
    "detected_error_types",
    "candidate_present",
    "char_edit_ratio",
    "token_edit_ratio",
    "notes",
    "edit_summary",
]

SOURCE_COLUMNS = ("source", "input", "input_text", "incorrect", "erroneous", "original", "error_text", "corrupted", "src")
TARGET_COLUMNS = ("target", "target_text", "correction", "correct", "corrected", "output", "correct_text", "corrected_text", "tgt")
DISALLOWED_REAL_MARKERS = (
    " чувак",
    " чувиха",
    " лол",
    " кек",
    " хрен",
    " блин",
    " фиг",
    " имхо",
    " епт",
    " ёп",
    " нах",
    " хуй",
    " пизд",
    " бляд",
    " сука",
    " говн",
    "github",
    " pull request",
    " commit",
    "анамнез",
)


@dataclass(frozen=True)
class RealPairValidation:
    accepted: bool
    reason: str
    row: dict[str, Any] | None
    edit_summary: str
    detected_error_types: list[str]
    candidate_present: bool
    char_edit_ratio: float = 0.0
    token_edit_ratio: float = 0.0
    routing_category: str = "rejected"
    strict_validator_passed: bool = False


@dataclass(frozen=True)
class RealErrorLoadResult:
    rows: list[dict[str, Any]]
    accepted_count: int
    rejected_count: int
    source_reports: list[dict[str, Any]]
    rejection_reason_counts: dict[str, int]
    output_path: str
    stress_rows: list[dict[str, Any]] = field(default_factory=list)
    mining_rows: list[dict[str, Any]] = field(default_factory=list)
    holdout_rows: list[dict[str, Any]] = field(default_factory=list)
    atomic_output_path: str = ""
    holdout_output_path: str = ""
    stress_output_path: str = ""
    mining_output_path: str = ""
    rejected_output_path: str = ""

    def __post_init__(self) -> None:
        if not self.atomic_output_path:
            object.__setattr__(self, "atomic_output_path", self.output_path)


def load_real_error_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_real_error_pair(
    source: str,
    target: str,
    *,
    source_dataset: str,
    candidate_generator: CandidateGenerator | None = None,
    domain: str = "real_error_pair",
    max_char_edit_ratio: float = 0.25,
    max_token_edit_ratio: float = 0.30,
    min_tokens: int = 5,
    max_tokens: int = 45,
    require_all_edits_candidate_covered: bool = True,
    require_strict_validator: bool = True,
) -> RealPairValidation:
    source = _normalize_text(source)
    target = _normalize_text(target)
    reason = _basic_pair_rejection(source, target, min_tokens=min_tokens, max_tokens=max_tokens)
    if reason:
        return RealPairValidation(False, reason, None, "", [], False)

    char_ratio = Levenshtein.distance(source, target) / max(1, max(len(source), len(target)))
    if char_ratio > max_char_edit_ratio:
        return RealPairValidation(False, "char_edit_distance_too_high", None, "", [], False, char_ratio, 0.0)
    token_ratio = _token_edit_ratio(source, target)
    if token_ratio > max_token_edit_ratio:
        return RealPairValidation(False, "token_edit_distance_too_high", None, "", [], False, char_ratio, token_ratio)

    generator = candidate_generator or CandidateGenerator()
    candidates = generator.generate(source)
    edits = _deduplicate_logical_real_edits(DiffAnalyzer().analyze(source, target, candidates=candidates), target)
    if not edits:
        return RealPairValidation(False, "unsupported_edit_type", None, "", [], False, char_ratio, token_ratio)

    edit_matches = _candidate_matches_by_edit(edits, candidates)
    resolved_rule_ids = _resolved_rule_ids(edits, edit_matches)
    candidate_present = all(bool(matches) for matches in edit_matches)
    error_types = sorted(
        {
            coarse_error_type(edit.edit_type)
            for edit in edits
            if coarse_error_type(edit.edit_type) != "unknown"
        }
    )
    unsupported_or_unknown = any(_is_unknown_or_unsupported_edit(edit) for edit in edits)
    detected_error_types = sorted(set(error_types) | ({"unknown"} if unsupported_or_unknown else set())) or ["unknown"]
    dirty_span = any(_is_dirty_real_edit_span(source, edit) for edit in edits if not _is_unknown_or_unsupported_edit(edit))
    has_unknown_rule = any(rule_id == UNKNOWN_RULE_ID for rule_id in resolved_rule_ids)
    if dirty_span:
        return RealPairValidation(False, "protected_or_invalid_edit_span", None, _edit_summary(edits), detected_error_types, candidate_present, char_ratio, token_ratio)
    if unsupported_or_unknown or has_unknown_rule:
        row = _real_pair_row(
            source,
            target,
            source_dataset=source_dataset,
            domain=domain,
            error_types=detected_error_types,
            candidate_present=candidate_present,
            candidate_rule_ids=_candidate_rule_ids_for_matches(edit_matches),
            edits=edits,
            rule_ids=resolved_rule_ids,
            char_ratio=char_ratio,
            token_ratio=token_ratio,
            routing_category="mining",
            routing_reason="unknown_rule_mining_only",
            strict_validator_passed=False,
        )
        return RealPairValidation(
            False,
            "unknown_rule_mining_only",
            row,
            _edit_summary(edits),
            detected_error_types,
            candidate_present,
            char_ratio,
            token_ratio,
            routing_category="mining",
            strict_validator_passed=False,
        )
    if not candidate_present:
        return RealPairValidation(False, "candidate_missing", None, _edit_summary(edits), detected_error_types, False, char_ratio, token_ratio)
    if require_all_edits_candidate_covered and not candidate_present:
        return RealPairValidation(False, "candidate_missing", None, _edit_summary(edits), detected_error_types, False, char_ratio, token_ratio)

    strict_validator_passed = True
    if require_strict_validator:
        strict_result = StrictValidator().validate(source, target, trusted_edits=candidates)
        strict_validator_passed = strict_result.apply_accepted() == target
        if not strict_validator_passed:
            reasons = sorted({edit.reason for edit in strict_result.rejected_edits if edit.reason})
            reason = "strict_validator_rejected" + (f":{','.join(reasons[:3])}" if reasons else "")
            return RealPairValidation(False, reason, None, _edit_summary(edits), detected_error_types, candidate_present, char_ratio, token_ratio)

    row = _real_pair_row(
        source,
        target,
        source_dataset=source_dataset,
        domain=domain,
        error_types=detected_error_types,
        candidate_present=candidate_present,
        candidate_rule_ids=_candidate_rule_ids_for_matches(edit_matches),
        edits=edits,
        rule_ids=resolved_rule_ids,
        char_ratio=char_ratio,
        token_ratio=token_ratio,
        routing_category="accepted_known_rule",
        routing_reason="known_rule_candidate_covered",
        strict_validator_passed=strict_validator_passed,
    )
    return RealPairValidation(
        True,
        "",
        row,
        _edit_summary(edits),
        detected_error_types,
        True,
        char_ratio,
        token_ratio,
        routing_category="accepted_known_rule",
        strict_validator_passed=strict_validator_passed,
    )


def load_real_error_pairs(
    config: dict[str, Any] | str | Path,
    *,
    candidate_generator: CandidateGenerator | None = None,
    output_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> RealErrorLoadResult:
    if isinstance(config, str | Path):
        config = load_real_error_config(config)
    source_specs = _source_specs(config)
    validation_config = dict(config.get("validation", {}) or {})
    policy = _download_policy(config)
    budget = DownloadBudget()
    generator = candidate_generator or CandidateGenerator()
    rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    mining_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    atomization_rows: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    stress_loss_weight = float(validation_config.get("stress_loss_weight", 0.4))
    require_all_covered = bool(validation_config.get("require_all_edits_candidate_covered", True))
    require_strict_validator = bool(validation_config.get("require_strict_validator", True))

    for source_name, spec in source_specs:
        if spec.get("enabled") is False:
            source_reports.append(_source_report(source_name, spec, status="skipped", reason="disabled"))
            continue
        source_type = str(spec.get("type") or "local_jsonl")
        if not _is_supported_source_type(source_type):
            source_reports.append(_source_report(source_name, spec, status="skipped", reason="skipped_format_unknown"))
            continue
        if _should_skip_materialized_hf_source(source_name, spec, source_specs):
            source_reports.append(_source_report(source_name, spec, status="skipped", reason="materialized_to_sage_local_jsonl"))
            continue
        download = download_if_allowed(source_name, spec, policy, budget)
        path = Path(download.path) if download.path else None
        if not download.used and source_type not in {"hf_dataset", "huggingface_dataset"}:
            source_reports.append(_source_report(source_name, spec, status="skipped", reason=download.reason or download.mode, download=download))
            continue
        if source_type in {"hf_dataset", "huggingface_dataset"} and not download.used:
            source_reports.append(_source_report(source_name, spec, status="skipped", reason=download.reason or download.mode, download=download))
            continue
        accepted_before = len(rows)
        holdout_before = len(holdout_rows)
        stress_before = len(stress_rows)
        mining_before = len(mining_rows)
        rejected_before = len(rejected_rows)
        seen = 0
        max_raw_pairs = int(spec.get("max_raw_pairs") or spec.get("max_seen") or 0)
        max_accepted_pairs = int(spec.get("max_pairs") or spec.get("max_examples") or 10_000)
        try:
            for source, target, metadata in _iter_pairs(spec, path):
                seen += 1
                validation = validate_real_error_pair(
                    source,
                    target,
                    source_dataset=source_name,
                    candidate_generator=generator,
                    domain=str(metadata.get("domain") or spec.get("domain") or "real_error_pair"),
                    max_char_edit_ratio=float(validation_config.get("max_char_edit_ratio", 0.25)),
                    max_token_edit_ratio=float(validation_config.get("max_token_edit_ratio", 0.30)),
                    min_tokens=int(validation_config.get("min_tokens", 5)),
                    max_tokens=int(validation_config.get("max_tokens", 45)),
                    require_all_edits_candidate_covered=require_all_covered,
                    require_strict_validator=require_strict_validator,
                )
                allowed_error_types = set(str(item) for item in spec.get("allowed_error_types", []))
                if validation.row is not None:
                    candidate_row = _attach_raw_real_pair_metadata(validation.row, metadata, spec, path)
                else:
                    candidate_row = None
                if (
                    validation.accepted
                    and candidate_row is not None
                    and allowed_error_types
                    and not set(validation.detected_error_types) <= allowed_error_types
                ):
                    rejection_counts["disallowed_error_type"] += 1
                    rejected = _rejected_row(
                        source_name,
                        source,
                        target,
                        "disallowed_error_type",
                        json.dumps(validation.detected_error_types, ensure_ascii=False),
                        validation.candidate_present,
                        validation.char_edit_ratio,
                        validation.token_edit_ratio,
                        validation.edit_summary,
                    )
                    rejected_rows.append(rejected)
                    atomization_rows.append(_atomization_row(source_name, source, target, validation, "rejected", "disallowed_error_type", rejected))
                elif validation.routing_category == "mining" and candidate_row is not None:
                    mining_row = _mining_real_row(candidate_row, validation.reason)
                    mining_rows.append(mining_row)
                    atomization_rows.append(_atomization_row(source_name, source, target, validation, "mining", validation.reason, mining_row))
                elif validation.accepted and candidate_row is not None:
                    routed_category, routed_row = _route_known_real_row(
                        candidate_row,
                        stress_loss_weight=stress_loss_weight,
                    )
                    if routed_category == "atomic_train":
                        rows.append(routed_row)
                    elif routed_category == "holdout":
                        holdout_rows.append(routed_row)
                    elif routed_category == "stress":
                        stress_rows.append(routed_row)
                    else:
                        rejection_counts[routed_row["routing_reason"]] += 1
                        rejected = _rejected_row(
                            source_name,
                            source,
                            target,
                            str(routed_row["routing_reason"]),
                            json.dumps(validation.detected_error_types, ensure_ascii=False),
                            validation.candidate_present,
                            validation.char_edit_ratio,
                            validation.token_edit_ratio,
                            validation.edit_summary,
                        )
                        rejected_rows.append(rejected)
                        atomization_rows.append(_atomization_row(source_name, source, target, validation, "rejected", str(routed_row["routing_reason"]), rejected))
                        continue
                    atomization_rows.append(_atomization_row(source_name, source, target, validation, routed_category, str(routed_row.get("routing_reason", "")), routed_row))
                else:
                    rejection_counts[validation.reason] += 1
                    rejected = _rejected_row(
                        source_name,
                        source,
                        target,
                        validation.reason,
                        json.dumps(validation.detected_error_types, ensure_ascii=False),
                        validation.candidate_present,
                        validation.char_edit_ratio,
                        validation.token_edit_ratio,
                        validation.edit_summary,
                    )
                    rejected_rows.append(rejected)
                    atomization_rows.append(_atomization_row(source_name, source, target, validation, "rejected", validation.reason, rejected))
                if len(rows) - accepted_before >= max_accepted_pairs:
                    break
                if max_raw_pairs and seen >= max_raw_pairs:
                    break
        except Exception as exc:
            source_reports.append(_source_report(source_name, spec, status="skipped", reason=f"loader_error:{exc.__class__.__name__}", download=download))
            continue
        source_reports.append(
            _source_report(
                source_name,
                spec,
                status="loaded" if seen > 0 else "skipped",
                reason="" if seen > 0 else "no_pairs_loaded",
                total_seen=seen,
                accepted=len(rows) - accepted_before,
                holdout=len(holdout_rows) - holdout_before,
                stress=len(stress_rows) - stress_before,
                mining=len(mining_rows) - mining_before,
                rejected=len(rejected_rows) - rejected_before,
                download=download,
            )
        )

    rows = _deduplicate_rows(rows)
    holdout_rows = _deduplicate_rows(holdout_rows)
    stress_rows = _deduplicate_rows(stress_rows)
    mining_rows = _deduplicate_rows(mining_rows)
    rows, cap_rejections, cap_rejection_counts = _apply_source_caps(rows, _cap_shares(source_specs))
    if cap_rejections:
        rejected_rows.extend(cap_rejections)
        rejection_counts.update(cap_rejection_counts)
        for rejected in cap_rejections:
            atomization_rows.append(
                _atomization_row(
                    str(rejected.get("source_dataset") or ""),
                    str(rejected.get("source") or ""),
                    str(rejected.get("target") or ""),
                    None,
                    "rejected",
                    "source_share_cap",
                    rejected,
                )
            )
    _refresh_source_report_counts(source_reports, rows, rejected_rows, holdout_rows=holdout_rows, stress_rows=stress_rows, mining_rows=mining_rows)
    output = Path(output_path or "data/processed/real_error_pairs_validated.csv.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_output = output.with_name("real_error_pairs_atomic.csv.gz")
    holdout_output = output.with_name("real_error_pairs_holdout.csv.gz")
    stress_output = output.with_name("real_error_pairs_stress.csv.gz")
    mining_output = output.with_name("real_error_pairs_mining.csv.gz")
    rejected_output = output.with_name("real_error_pairs_rejected.csv.gz")
    pd.DataFrame(rows, columns=REAL_PAIR_COLUMNS).to_csv(output, index=False)
    pd.DataFrame(rows, columns=REAL_PAIR_COLUMNS).to_csv(atomic_output, index=False)
    pd.DataFrame(holdout_rows, columns=REAL_PAIR_COLUMNS).to_csv(holdout_output, index=False)
    pd.DataFrame(stress_rows, columns=REAL_PAIR_COLUMNS).to_csv(stress_output, index=False)
    pd.DataFrame(mining_rows, columns=REAL_PAIR_COLUMNS).to_csv(mining_output, index=False)
    pd.DataFrame(rejected_rows, columns=REJECTED_PAIR_COLUMNS).to_csv(rejected_output, index=False)
    report_dir = Path(reports_dir or "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_real_pair_filter_rows(source_reports, rejection_counts)).to_csv(report_dir / "real_pair_filter_report.csv", index=False)
    pd.DataFrame(atomization_rows).to_csv(report_dir / "real_pair_atomization_report.csv", index=False)
    pd.DataFrame(rejected_rows, columns=REJECTED_PAIR_COLUMNS).to_csv(
        report_dir / "rejected_real_pairs.csv",
        index=False,
    )
    pd.DataFrame(
        [{"reason": reason, "count": count} for reason, count in sorted(rejection_counts.items())],
        columns=["reason", "count"],
    ).to_csv(report_dir / "rejected_real_pair_reasons.csv", index=False)
    _write_real_source_report(
        report_dir / "real_error_source_report.md",
        source_reports,
        rejection_counts,
        downloads_are_enabled=downloads_enabled(str(policy.get("allow_downloads_env") or "RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS")),
        loader_method=str(config.get("_loader_method") or "mixed_local_hf_jsonl"),
    )
    return RealErrorLoadResult(
        rows=rows,
        accepted_count=len(rows),
        rejected_count=len(rejected_rows),
        source_reports=source_reports,
        rejection_reason_counts=dict(sorted(rejection_counts.items())),
        output_path=str(output),
        stress_rows=stress_rows,
        mining_rows=mining_rows,
        holdout_rows=holdout_rows,
        atomic_output_path=str(atomic_output),
        holdout_output_path=str(holdout_output),
        stress_output_path=str(stress_output),
        mining_output_path=str(mining_output),
        rejected_output_path=str(rejected_output),
    )


def _attach_raw_real_pair_metadata(
    row: dict[str, Any],
    metadata: dict[str, Any],
    spec: dict[str, Any],
    path: Path | None,
) -> dict[str, Any]:
    result = dict(row)
    raw_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    row_metadata = _json_dict(result.get("metadata"))
    row_metadata.update(raw_metadata)
    raw_split = _explicit_source_split(metadata, raw_metadata)
    if raw_split:
        row_metadata.setdefault("raw_split", raw_split)
        row_metadata.setdefault("source_split", raw_split)
    row_metadata.setdefault("raw_id", str(metadata.get("raw_id") or ""))
    result["metadata"] = json.dumps(row_metadata, ensure_ascii=False, sort_keys=True)
    result["source_subdataset"] = str(
        metadata.get("source_subdataset")
        or raw_metadata.get("dataset")
        or spec.get("dataset_name")
        or spec.get("name_in_dataset")
        or ""
    )
    result["raw_id"] = str(metadata.get("raw_id") or "")
    result["raw_source_path"] = str(metadata.get("raw_source_path") or path or "")
    return result


def _route_known_real_row(
    row: dict[str, Any],
    *,
    stress_loss_weight: float,
) -> tuple[str, dict[str, Any]]:
    edit_count = int(row.get("edit_count") or 0)
    raw_split = _json_dict(row.get("metadata")).get("raw_split", "")
    if edit_count == 1:
        if _is_explicit_holdout_split(raw_split):
            return "holdout", _holdout_real_row(row, str(raw_split))
        return "atomic_train", _atomic_real_row(row)
    if edit_count > 1:
        return "stress", _stress_real_row(row, stress_loss_weight)
    rejected = dict(row)
    rejected["routing_category"] = "rejected"
    rejected["routing_reason"] = "empty_edit_set"
    return "rejected", rejected


def _atomic_real_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["split"] = "train"
    result["dataset_layer"] = "real_atomic"
    result["is_stress"] = False
    result["count_toward_rule_quota"] = False
    result["loss_weight"] = 1.0
    result["routing_category"] = "atomic_train"
    result["routing_reason"] = "single_edit_known_rule"
    result["metadata"] = _metadata_with_updates(
        result.get("metadata"),
        routing_category="atomic_train",
        routing_reason="single_edit_known_rule",
        count_toward_rule_quota=False,
    )
    return result


def _holdout_real_row(row: dict[str, Any], raw_split: str) -> dict[str, Any]:
    result = dict(row)
    result["split"] = raw_split
    result["dataset_layer"] = "real_holdout"
    result["is_stress"] = False
    result["count_toward_rule_quota"] = False
    result["loss_weight"] = 1.0
    result["routing_category"] = "holdout"
    result["routing_reason"] = "explicit_source_split"
    result["metadata"] = _metadata_with_updates(
        result.get("metadata"),
        source_split=raw_split,
        raw_split=raw_split,
        holdout_reason="explicit_source_split",
        routing_category="holdout",
        routing_reason="explicit_source_split",
    )
    return result


def _stress_real_row(row: dict[str, Any], stress_loss_weight: float) -> dict[str, Any]:
    result = dict(row)
    result["split"] = "stress"
    result["dataset_layer"] = "stress_multi_error"
    result["is_stress"] = True
    result["count_toward_rule_quota"] = False
    result["loss_weight"] = float(stress_loss_weight)
    result["routing_category"] = "stress"
    result["routing_reason"] = "multi_edit_known_rule_stress"
    result["metadata"] = _metadata_with_updates(
        result.get("metadata"),
        dataset_layer="stress_multi_error",
        is_stress=True,
        count_toward_rule_quota=False,
        routing_category="stress",
        routing_reason="multi_edit_known_rule_stress",
    )
    return result


def _mining_real_row(row: dict[str, Any], reason: str) -> dict[str, Any]:
    result = dict(row)
    result["split"] = "mining"
    result["dataset_layer"] = "real_mining"
    result["is_stress"] = False
    result["count_toward_rule_quota"] = False
    result["loss_weight"] = 0.0
    result["routing_category"] = "mining"
    result["routing_reason"] = reason
    result["metadata"] = _metadata_with_updates(
        result.get("metadata"),
        dataset_layer="real_mining",
        count_toward_rule_quota=False,
        routing_category="mining",
        routing_reason=reason,
    )
    return result


def _metadata_with_updates(value: Any, **updates: Any) -> str:
    metadata = _json_dict(value)
    metadata.update(updates)
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def _explicit_source_split(metadata: dict[str, Any], raw_metadata: dict[str, Any]) -> str:
    for value in (
        metadata.get("raw_split"),
        metadata.get("source_split"),
        metadata.get("split"),
        raw_metadata.get("raw_split"),
        raw_metadata.get("source_split"),
        raw_metadata.get("split"),
    ):
        text = str(value or "").strip().lower()
        if text:
            return text
    return ""


def _is_explicit_holdout_split(split: Any) -> bool:
    return str(split or "").strip().lower() in {"eval", "evaluation", "val", "valid", "validation", "dev", "test", "holdout"}


def _atomization_row(
    source_dataset: str,
    source: str,
    target: str,
    validation: RealPairValidation | None,
    routing_category: str,
    routing_reason: str,
    row: dict[str, Any] | None,
) -> dict[str, Any]:
    row = row or {}
    return {
        "source_dataset": source_dataset,
        "raw_id": str(row.get("raw_id") or ""),
        "routing_category": routing_category,
        "routing_reason": routing_reason,
        "source": _normalize_text(source),
        "target": _normalize_text(target),
        "edit_count": int(row.get("edit_count") or 0),
        "rule_id": str(row.get("rule_id") or UNKNOWN_RULE_ID),
        "rule_ids": str(row.get("rule_ids") or json.dumps([UNKNOWN_RULE_ID])),
        "detected_error_types": str(row.get("detected_error_types") or "[]"),
        "candidate_present": bool(validation.candidate_present) if validation is not None else bool(row.get("candidate_present", False)),
        "strict_validator_passed": bool(validation.strict_validator_passed) if validation is not None else bool(row.get("strict_validator_passed", False)),
        "char_edit_ratio": float(validation.char_edit_ratio) if validation is not None else _as_float(row.get("char_edit_ratio"), 0.0),
        "token_edit_ratio": float(validation.token_edit_ratio) if validation is not None else _as_float(row.get("token_edit_ratio"), 0.0),
        "edit_summary": validation.edit_summary if validation is not None else str(row.get("edit_summary") or row.get("notes") or ""),
    }


def _download_policy(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("download_policy", {}) or config.get("sources", {}).get("download_policy", {}) or {})


def _source_specs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw = config.get("real_sources", config.get("sources", []))
    if isinstance(raw, dict):
        return [(str(name), dict(spec or {})) for name, spec in raw.items()]
    if isinstance(raw, list):
        return [(str(spec.get("name") or f"source_{index}"), dict(spec)) for index, spec in enumerate(raw) if isinstance(spec, dict)]
    return []


def _is_supported_source_type(source_type: str) -> bool:
    return source_type in {
        "sage_hf_or_local",
        "local_jsonl",
        "jsonl",
        "local_csv",
        "csv",
        "tsv",
        "table",
        "m2",
        "m2_local",
        "hf_dataset",
        "huggingface_dataset",
    }


def _iter_pairs(spec: dict[str, Any], path: Path | None) -> Iterable[tuple[str, str, dict[str, Any]]]:
    source_type = str(spec.get("type") or "local_jsonl")
    if source_type == "sage_hf_or_local":
        source_type = "local_jsonl"
    if source_type in {"local_jsonl", "jsonl"}:
        if path is None:
            return
        with _open_text(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                yield _first_value(item, SOURCE_COLUMNS), _first_value(item, TARGET_COLUMNS), {
                    "domain": str(item.get("domain") or ""),
                    "raw_id": str(item.get("raw_id") or item.get("id") or ""),
                    "source_subdataset": str(item.get("dataset") or item.get("source_subdataset") or ""),
                    "raw_split": str(item.get("raw_split") or item.get("source_split") or item.get("split") or ""),
                    "raw_source_path": str(path),
                    "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                }
    elif source_type in {"local_csv", "csv", "tsv", "table"}:
        if path is None:
            return
        delimiter = "\t" if str(path).endswith((".tsv", ".tab")) else ","
        with _open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for item in reader:
                yield _first_value(item, SOURCE_COLUMNS), _first_value(item, TARGET_COLUMNS), {
                    "domain": str(item.get("domain") or ""),
                    "raw_id": str(item.get("raw_id") or item.get("id") or ""),
                    "raw_split": str(item.get("raw_split") or item.get("source_split") or item.get("split") or ""),
                    "raw_source_path": str(path),
                }
    elif source_type in {"m2", "m2_local"}:
        if path is None:
            return
        from src.data.external_sources import _m2_pairs

        for file_path in _iter_m2_files(path):
            for index, (source, target) in enumerate(_m2_pairs(file_path)):
                yield source, target, {"domain": "m2", "raw_id": f"{file_path.name}:{index}", "raw_source_path": str(file_path)}
    elif source_type in {"hf_dataset", "huggingface_dataset"}:
        yield from _iter_hf_pairs(spec)


def _iter_hf_pairs(spec: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    from datasets import load_dataset

    dataset_id = str(spec.get("hf_id") or spec.get("repo"))
    requested_configs: list[str | None] = []
    for key in ("name_in_dataset", "dataset_name"):
        if spec.get(key):
            requested_configs.append(str(spec[key]))
    requested_configs.extend(str(item) for item in spec.get("configs", []) if item)
    if not requested_configs:
        requested_configs = [None, "RUSpellRU", "MultidomainGold", "MedSpellchecker", "MedSpellChecker", "GitHubTypoCorpusRu"]
    configs: list[str | None] = []
    for config_name in requested_configs:
        if config_name not in configs:
            configs.append(config_name)
    splits = [str(spec["split"])] if spec.get("split") else [str(item) for item in spec.get("splits", ("train", "test"))]
    loaded_any = False
    for config_name in configs:
        for split in splits:
            kwargs: dict[str, Any] = {"split": split}
            if config_name:
                kwargs["name"] = config_name
            try:
                try:
                    dataset = load_dataset(dataset_id, trust_remote_code=True, **kwargs)
                except TypeError:
                    dataset = load_dataset(dataset_id, **kwargs)
            except Exception:
                continue
            loaded_any = True
            for row_index, item in enumerate(dataset):
                if not isinstance(item, dict):
                    continue
                yield _first_value(item, SOURCE_COLUMNS), _first_value(item, TARGET_COLUMNS), {
                    "domain": str(item.get("domain") or config_name or ""),
                    "raw_id": str(item.get("raw_id") or item.get("id") or f"{split}:{row_index}"),
                    "source_subdataset": str(config_name or ""),
                    "raw_split": split,
                    "metadata": {"split": split, "hf_id": dataset_id, "hf_config": config_name or ""},
                    "raw_source_path": dataset_id,
                }
    if not loaded_any:
        return


def _basic_pair_rejection(source: str, target: str, *, min_tokens: int, max_tokens: int) -> str:
    if not source or not target:
        return "empty_source_or_target"
    if source == target:
        return "source_equals_target"
    if min(cyrillic_ratio(source), cyrillic_ratio(target)) < 0.65:
        return "low_cyrillic_ratio"
    source_tokens = _tokens(source)
    target_tokens = _tokens(target)
    if min(len(source_tokens), len(target_tokens)) < min_tokens:
        return "too_few_tokens"
    if max(len(source_tokens), len(target_tokens)) > max_tokens:
        return "too_many_tokens"
    lower = f" {source.lower()} {target.lower()} "
    combined = f"{source}\n{target}"
    if re.search(r"(?m)^\s{0,3}#{1,6}\s|```|`|<[^>]+>|\{\{|}}|\[\[|]]", combined):
        return "markup_or_code_fragment"
    if re.search(r"(?m)^\s*[A-Za-z_][\w.-]{1,32}\s*:", combined):
        return "markup_or_code_fragment"
    if re.search(r"\b(?:src|tests?|docs?|github|commit|pull request|api)\b|/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", lower):
        return "markup_or_code_fragment"
    if any(marker in lower for marker in DISALLOWED_REAL_MARKERS):
        return "forbidden_domain_noise"
    if re.search(r"https?://|www\.|@\w+|#[\wа-яё]+|```|/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source + " " + target):
        return "code_or_social_marker"
    if re.search(r"\b(?:анамнез|пациент|диагноз|симптом|терапия|таблетк|инъекц)\b", lower):
        return "forbidden_domain_noise"
    if re.search(r"^\s*(?:\d{1,2}:\d{2}(?::\d{2})?|\[[^\]]+\])", combined):
        return "subtitles_or_dialogue_fragment"
    if re.search(r"^\s*[—-]\s+", combined):
        return "literary_or_dialogue_fragment"
    if any(ord(char) > 0xFFFF for char in source + target):
        return "emoji"
    return ""


def _iter_m2_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() == ".m2":
            yield path
        return
    if path.is_dir():
        for file_path in sorted(path.rglob("*.m2")):
            if file_path.is_file():
                yield file_path


def _is_real_edit_supported(source_text: str, edit: Edit) -> bool:
    if not is_allowed_edit_type(edit.edit_type):
        return False
    if coarse_error_type(edit.edit_type) == "unknown":
        return False
    if edit.start < 0 or edit.end < edit.start:
        return False
    span_end = edit.end if edit.end > edit.start else edit.start + 1
    return not any(edit.start < span.end and span.start < span_end for span in find_protected_spans(source_text))


def _deduplicate_logical_real_edits(edits: list[Edit], target: str) -> list[Edit]:
    result: list[Edit] = []
    index_by_key: dict[tuple[int, int, str, str, str], int] = {}
    for edit in edits:
        key = (
            edit.start,
            edit.end,
            edit.edit_type,
            edit.source.lower(),
            edit.replacement.lower(),
        )
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(result)
            result.append(edit)
            continue
        existing = result[existing_index]
        if _edit_case_score(edit, target) > _edit_case_score(existing, target):
            result[existing_index] = edit
    return result


def _edit_case_score(edit: Edit, target: str) -> tuple[float, int, int]:
    return (
        float(edit.confidence),
        1 if edit.replacement and edit.replacement in target else 0,
        sum(1 for char in edit.replacement if char.isupper()),
    )


def _is_unknown_or_unsupported_edit(edit: Edit) -> bool:
    return not is_allowed_edit_type(edit.edit_type) or coarse_error_type(edit.edit_type) == "unknown"


def _is_dirty_real_edit_span(source_text: str, edit: Edit) -> bool:
    if edit.start < 0 or edit.end < edit.start:
        return True
    span_end = edit.end if edit.end > edit.start else edit.start + 1
    return any(edit.start < span.end and span.start < span_end for span in find_protected_spans(source_text))


def _candidate_matches_by_edit(edits: list[Edit], candidates: Iterable[Any]) -> list[list[Any]]:
    candidate_list = list(candidates)
    return [
        [candidate for candidate in candidate_list if candidate_matches_edit(candidate, edit)]
        for edit in edits
    ]


def _resolved_rule_ids(edits: list[Edit], matches_by_edit: list[list[Any]]) -> list[str]:
    result: list[str] = []
    for edit, matches in zip(edits, matches_by_edit, strict=True):
        rule_id = normalize_rule_id(edit.rule_id)
        if rule_id == UNKNOWN_RULE_ID:
            for candidate in matches:
                rule_id = normalize_rule_id(getattr(candidate, "rule_id", ""))
                if rule_id != UNKNOWN_RULE_ID:
                    break
        result.append(rule_id)
    return result


def _candidate_rule_ids_for_matches(matches_by_edit: list[list[Any]]) -> list[str]:
    result: list[str] = []
    for matches in matches_by_edit:
        for candidate in matches:
            rule_id = normalize_rule_id(getattr(candidate, "rule_id", ""))
            if rule_id != UNKNOWN_RULE_ID and rule_id not in result:
                result.append(rule_id)
    return sorted(result)


def _real_pair_row(
    source: str,
    target: str,
    *,
    source_dataset: str,
    domain: str,
    error_types: list[str],
    candidate_present: bool,
    candidate_rule_ids: list[str],
    edits: list[Edit],
    rule_ids: list[str],
    char_ratio: float,
    token_ratio: float,
    routing_category: str,
    routing_reason: str,
    strict_validator_passed: bool,
) -> dict[str, Any]:
    normalized_rule_ids = _unique_rule_ids(rule_ids)
    metadata = {
        "source_type": "real_error_pair",
        "source_dataset": source_dataset,
        "candidate_present": bool(candidate_present),
        "routing_category": routing_category,
        "routing_reason": routing_reason,
        "strict_validator_passed": bool(strict_validator_passed),
        "count_toward_rule_quota": False,
    }
    primary_rule_id = normalized_rule_ids[0] if normalized_rule_ids else UNKNOWN_RULE_ID
    return {
        "source": source,
        "target": target,
        "source_dataset": source_dataset,
        "source_subdataset": "",
        "domain": domain,
        "detected_error_types": json.dumps(error_types, ensure_ascii=False),
        "candidate_present": bool(candidate_present),
        "candidate_rule_ids": json.dumps(candidate_rule_ids, ensure_ascii=False),
        "edit_count": len(edits),
        "char_edit_ratio": char_ratio,
        "token_edit_ratio": token_ratio,
        "is_real_pair": True,
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "raw_id": "",
        "raw_source_path": "",
        "error_types": json.dumps(error_types, ensure_ascii=False),
        "error_type": error_types[0] if error_types else "unknown",
        "source_type": "real_error_pair",
        "is_clean": False,
        "is_hard_negative": False,
        "is_synthetic": False,
        "split": "train",
        "rule_id": primary_rule_id,
        "rule_ids": json.dumps(normalized_rule_ids or [UNKNOWN_RULE_ID], ensure_ascii=False),
        "edit_operations": json.dumps([asdict(edit) for edit in edits], ensure_ascii=False),
        "edits": json.dumps([asdict(edit) for edit in edits], ensure_ascii=False),
        "dataset_layer": "real_atomic",
        "is_stress": False,
        "count_toward_rule_quota": False,
        "loss_weight": 1.0,
        "routing_category": routing_category,
        "routing_reason": routing_reason,
        "strict_validator_passed": bool(strict_validator_passed),
    }


def _token_edit_ratio(source: str, target: str) -> float:
    source_tokens = _tokens(source)
    target_tokens = _tokens(target)
    if len(source_tokens) == len(target_tokens) and source_tokens:
        total = 0.0
        for source_token, target_token in zip(source_tokens, target_tokens, strict=True):
            if source_token == target_token:
                continue
            total += min(1.0, Levenshtein.distance(source_token, target_token) / max(1, max(len(source_token), len(target_token))))
        return total / len(source_tokens)
    return Levenshtein.distance(source_tokens, target_tokens) / max(1, max(len(source_tokens), len(target_tokens)))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)?|\d+(?:[,.]\d+)?", text)


def _rule_ids(edits: list[Edit]) -> list[str]:
    return _unique_rule_ids([edit.rule_id for edit in edits])


def _unique_rule_ids(rule_ids: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for raw_rule_id in rule_ids:
        rule_id = normalize_rule_id(raw_rule_id)
        if rule_id not in result:
            result.append(rule_id)
    return result


def _edit_summary(edits: list[Edit]) -> str:
    return "; ".join(f"{edit.edit_type}:{edit.source}->{edit.replacement}:{edit.rule_id or 'unknown'}" for edit in edits)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\xa0", " ")).strip()


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {"raw_metadata": value}
        return dict(loaded) if isinstance(loaded, dict) else {}
    return {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_value(item: dict[str, Any], names: tuple[str, ...]) -> str:
    lowered = {str(key).lower(): value for key, value in item.items()}
    for name in names:
        value = lowered.get(name)
        if value is not None:
            return str(value).strip()
    return ""


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8-sig", errors="replace")


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["source"], row["target"])
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _cap_shares(source_specs: list[tuple[str, dict[str, Any]]]) -> dict[str, float]:
    caps: dict[str, float] = {}
    for source_name, spec in source_specs:
        if spec.get("enabled") is False:
            continue
        if "cap_share" not in spec:
            continue
        cap = float(spec.get("cap_share") or 1.0)
        caps[source_name] = cap
        dataset_name = str(spec.get("dataset_name") or "").strip()
        if dataset_name:
            caps[dataset_name] = cap
    return caps


def _apply_source_caps(
    rows: list[dict[str, Any]],
    caps: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    if not rows or not caps:
        return rows, [], Counter()
    source_counts = Counter(str(row.get("source_dataset") or "") for row in rows)
    use_ceil_caps = _should_use_ceil_caps(source_counts, caps)
    effective_caps = _effective_caps_for_active_sources(source_counts, caps)
    target_counts = _target_counts_for_caps(source_counts, effective_caps, use_ceil=use_ceil_caps)
    if all(source_counts[source] <= target_counts.get(source, source_counts[source]) for source in source_counts):
        return rows, [], Counter()
    kept_counts: Counter[str] = Counter()
    result: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        source_name = str(row.get("source_dataset") or "")
        allowed = target_counts.get(source_name, source_counts[source_name])
        if kept_counts[source_name] < allowed:
            result.append(row)
            kept_counts[source_name] += 1
            continue
        rejected.append(
            _rejected_row(
                source_name,
                str(row.get("source", "")),
                str(row.get("target", "")),
                "source_share_cap",
                str(row.get("error_types", "[]")),
                bool(row.get("candidate_present", True)),
                _as_float(row.get("char_edit_ratio"), 0.0),
                _as_float(row.get("token_edit_ratio"), 0.0),
                str(row.get("edit_operations", "")),
            )
        )
        counts["source_share_cap"] += 1
    return result, rejected, counts


def _rejected_row(
    source_dataset: str,
    source: str,
    target: str,
    reason: str,
    detected_error_types: str,
    candidate_present: bool,
    char_edit_ratio: float,
    token_edit_ratio: float,
    notes: str,
) -> dict[str, Any]:
    return {
        "source_dataset": source_dataset,
        "source": _normalize_text(source),
        "target": _normalize_text(target),
        "reason": _normalize_text(reason),
        "detected_error_types": _normalize_text(detected_error_types),
        "candidate_present": bool(candidate_present),
        "char_edit_ratio": float(char_edit_ratio),
        "token_edit_ratio": float(token_edit_ratio),
        "notes": _normalize_text(notes),
        "edit_summary": _normalize_text(notes),
    }


def _effective_caps_for_active_sources(source_counts: Counter[str], caps: dict[str, float]) -> dict[str, float]:
    effective = dict(caps)
    active_capped_sources = [
        source_name
        for source_name in source_counts
        if source_name in caps and 0 < caps[source_name] < 1
    ]
    if not active_capped_sources:
        return effective
    active_uncapped_sources = [
        source_name
        for source_name in source_counts
        if source_name not in caps or caps.get(source_name, 1.0) >= 1
    ]
    if active_uncapped_sources:
        return effective
    cap_sum = sum(caps[source_name] for source_name in active_capped_sources)
    if cap_sum >= 1.0:
        return effective
    for source_name in active_capped_sources:
        effective[source_name] = min(1.0, caps[source_name] / max(0.000001, cap_sum))
    return effective


def _should_use_ceil_caps(source_counts: Counter[str], caps: dict[str, float]) -> bool:
    active_capped_sources = [
        source_name
        for source_name in source_counts
        if source_name in caps and 0 < caps[source_name] < 1
    ]
    if not active_capped_sources:
        return False
    active_uncapped_sources = [
        source_name
        for source_name in source_counts
        if source_name not in caps or caps.get(source_name, 1.0) >= 1
    ]
    if active_uncapped_sources:
        return False
    return sum(caps[source_name] for source_name in active_capped_sources) < 1.0


def _target_counts_for_caps(source_counts: Counter[str], caps: dict[str, float], *, use_ceil: bool) -> dict[str, int]:
    if use_ceil:
        total = sum(source_counts.values())
        targets = {
            source_name: min(count, _allowed_for_total(count, caps.get(source_name, 1.0), total, use_ceil=True))
            for source_name, count in source_counts.items()
        }
        surplus = sum(targets.values()) - total
        if surplus > 0:
            for source_name, _ in sorted(targets.items(), key=lambda item: (-item[1], item[0])):
                if surplus <= 0:
                    break
                removable = min(surplus, max(0, targets[source_name] - 1))
                targets[source_name] -= removable
                surplus -= removable
        return targets
    total = sum(source_counts.values())
    low = 0
    high = total
    while low < high:
        mid = (low + high + 1) // 2
        if _cap_capacity(source_counts, caps, mid, use_ceil=use_ceil) >= mid:
            low = mid
        else:
            high = mid - 1
    target_total = low
    targets = {
        source_name: min(count, _allowed_for_total(count, caps.get(source_name, 1.0), target_total, use_ceil=use_ceil))
        for source_name, count in source_counts.items()
    }
    surplus = sum(targets.values()) - target_total
    if surplus > 0:
        for source_name, _ in sorted(targets.items(), key=lambda item: (-item[1], item[0])):
            if surplus <= 0:
                break
            removable = min(surplus, max(0, targets[source_name] - 1))
            targets[source_name] -= removable
            surplus -= removable
    return targets


def _cap_capacity(source_counts: Counter[str], caps: dict[str, float], total: int, *, use_ceil: bool) -> int:
    return sum(
        min(count, _allowed_for_total(count, caps.get(source_name, 1.0), total, use_ceil=use_ceil))
        for source_name, count in source_counts.items()
    )


def _allowed_for_total(count: int, cap: float, total: int, *, use_ceil: bool) -> int:
    if count <= 0 or total <= 0:
        return 0
    if cap <= 0:
        return 0
    if cap >= 1:
        return count
    raw_allowed = math.ceil(cap * total) if use_ceil else math.floor(cap * total)
    return min(count, max(1, int(raw_allowed)))


def _refresh_source_report_counts(
    source_reports: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    *,
    holdout_rows: list[dict[str, Any]] | None = None,
    stress_rows: list[dict[str, Any]] | None = None,
    mining_rows: list[dict[str, Any]] | None = None,
) -> None:
    holdout_rows = holdout_rows or []
    stress_rows = stress_rows or []
    mining_rows = mining_rows or []
    accepted_counts = Counter(str(row.get("source_dataset") or "") for row in rows)
    holdout_counts = Counter(str(row.get("source_dataset") or "") for row in holdout_rows)
    stress_counts = Counter(str(row.get("source_dataset") or "") for row in stress_rows)
    mining_counts = Counter(str(row.get("source_dataset") or "") for row in mining_rows)
    rejected_counts = Counter(str(row.get("source_dataset") or "") for row in rejected_rows)
    accepted_candidate_counts = Counter(
        str(row.get("source_dataset") or "")
        for row in rows
        if str(row.get("candidate_present", "")).lower() in {"true", "1", "yes"} or row.get("candidate_present") is True
    )
    cap_counts = Counter(str(row.get("source_dataset") or "") for row in rejected_rows if row.get("reason") == "source_share_cap")
    for report in source_reports:
        if report.get("status") != "loaded":
            continue
        source_name = str(report["source_dataset"])
        accepted = int(accepted_counts.get(source_name, 0))
        report["accepted"] = accepted
        report["atomic_train"] = accepted
        report["holdout"] = int(holdout_counts.get(source_name, 0))
        report["stress"] = int(stress_counts.get(source_name, 0))
        report["mining"] = int(mining_counts.get(source_name, 0))
        report["rejected"] = int(rejected_counts.get(source_name, 0))
        report["cap_rejections"] = int(cap_counts.get(source_name, 0))
        report["candidate_coverage"] = (accepted_candidate_counts.get(source_name, 0) / accepted) if accepted else 0.0
        report["candidate_present_rate"] = report["candidate_coverage"]


def _should_skip_materialized_hf_source(
    source_name: str,
    spec: dict[str, Any],
    source_specs: list[tuple[str, dict[str, Any]]],
) -> bool:
    source_type = str(spec.get("type") or "")
    hf_id = str(spec.get("hf_id") or spec.get("repo") or "")
    if source_type not in {"hf_dataset", "huggingface_dataset"}:
        return False
    if hf_id != "ai-forever/spellcheck_benchmark":
        return False
    expected = [
        Path(str(item.get("local_path") or ""))
        for _, item in source_specs
        if str(item.get("type") or "") == "sage_hf_or_local" and item.get("local_path")
    ]
    return bool(expected) and all(path.exists() and path.stat().st_size > 0 for path in expected)


def _source_report(
    source_name: str,
    spec: dict[str, Any],
    *,
    status: str,
    reason: str,
    total_seen: int = 0,
    accepted: int = 0,
    holdout: int = 0,
    stress: int = 0,
    mining: int = 0,
    rejected: int = 0,
    download: SourceDownloadResult | None = None,
) -> dict[str, Any]:
    path = str(spec.get("local_path") or spec.get("path") or "")
    if download is not None and download.path:
        path = download.path
    return {
        "source_dataset": source_name,
        "type": str(spec.get("type") or ""),
        "status": status,
        "mode": download.mode if download is not None else ("local" if path and Path(path).exists() else "skipped"),
        "reason": reason,
        "local_path": path,
        "url": str(spec.get("url") or ""),
        "hf_id": str(spec.get("hf_id") or spec.get("repo") or ""),
        "downloaded_size_bytes": download.downloaded_size_bytes if download is not None else 0,
        "total_seen": total_seen,
        "accepted": accepted,
        "atomic_train": accepted,
        "holdout": holdout,
        "stress": stress,
        "mining": mining,
        "rejected": rejected,
        "candidate_coverage": 1.0 if accepted else 0.0,
        "candidate_present_rate": 1.0 if accepted else 0.0,
        "cap_rejections": 0,
        "required_domains": ", ".join(download.required_domains) if download is not None else "",
        "required_commands": " || ".join(download.required_commands) if download is not None else "",
    }


def _real_pair_filter_rows(source_reports: list[dict[str, Any]], rejection_counts: Counter[str]) -> list[dict[str, Any]]:
    return [
        {
            "source_dataset": report["source_dataset"],
            "mode": report.get("mode", ""),
            "total_seen": report["total_seen"],
            "accepted": report["accepted"],
            "atomic_train": report.get("atomic_train", report["accepted"]),
            "holdout": report.get("holdout", 0),
            "stress": report.get("stress", 0),
            "mining": report.get("mining", 0),
            "rejected": report["rejected"],
            "rejection_reason_counts": json.dumps(dict(sorted(rejection_counts.items())), ensure_ascii=False),
            "candidate_coverage": report["candidate_coverage"],
            "candidate_present_rate": report.get("candidate_present_rate", report["candidate_coverage"]),
            "cap_rejections": report.get("cap_rejections", 0),
            "reason": report.get("reason", ""),
        }
        for report in source_reports
    ]


def _write_real_source_report(
    path: Path,
    source_reports: list[dict[str, Any]],
    rejection_counts: Counter[str],
    *,
    downloads_are_enabled: bool,
    loader_method: str,
) -> None:
    accepted_pairs = sum(int(report["accepted"]) for report in source_reports)
    holdout_pairs = sum(int(report.get("holdout", 0)) for report in source_reports)
    stress_pairs = sum(int(report.get("stress", 0)) for report in source_reports)
    mining_pairs = sum(int(report.get("mining", 0)) for report in source_reports)
    rejected_pairs = sum(int(report["rejected"]) for report in source_reports)
    candidate_present = sum(int(report["accepted"]) * float(report.get("candidate_present_rate", report.get("candidate_coverage", 0.0))) for report in source_reports)
    candidate_present_rate = candidate_present / accepted_pairs if accepted_pairs else 0.0
    verdict = "READY_FOR_REBUILD_SOURCES" if accepted_pairs >= 1000 else "BLOCKED"
    lines = [
        "# Real Error Source Report",
        "",
        f"- downloads_enabled: {'yes' if downloads_are_enabled else 'no'}",
        f"- dataset_loader_method: {loader_method}",
        f"- accepted_pairs: {accepted_pairs}",
        f"- atomic_train_rows: {accepted_pairs}",
        f"- holdout_rows: {holdout_pairs}",
        f"- stress_rows: {stress_pairs}",
        f"- mining_rows: {mining_pairs}",
        f"- rejected_pairs: {rejected_pairs}",
        "- train_real_rows_policy: only atomic single-edit known-rule rows are accepted train pairs",
        f"- rejection_reasons: {json.dumps(dict(sorted(rejection_counts.items())), ensure_ascii=False, sort_keys=True)}",
        f"- candidate_present_rate: {candidate_present_rate:.4f}",
        f"- cap_rejections: {int(rejection_counts.get('source_share_cap', 0))}",
        f"- final_verdict: {verdict}",
        "",
        "| source | status | mode | path | url/hf | bytes | seen | atomic_train | holdout | stress | mining | rejected | cap_rejections | reason | candidate_present_rate |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for report in source_reports:
        source_ref = report.get("url") or report.get("hf_id") or ""
        lines.append(
            f"| {report['source_dataset']} | {report['status']} | {report.get('mode', '')} | {report.get('local_path', '')} | "
            f"{source_ref} | {report.get('downloaded_size_bytes', 0)} | {report['total_seen']} | {report['accepted']} | "
            f"{report.get('holdout', 0)} | {report.get('stress', 0)} | {report.get('mining', 0)} | "
            f"{report['rejected']} | {report.get('cap_rejections', 0)} | {report['reason']} | "
            f"{float(report.get('candidate_present_rate', report.get('candidate_coverage', 0.0))):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
