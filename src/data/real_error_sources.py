from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
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
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type


REAL_PAIR_COLUMNS = [
    "source",
    "target",
    "error_types",
    "error_type",
    "source_dataset",
    "source_type",
    "is_clean",
    "is_hard_negative",
    "is_synthetic",
    "is_real_pair",
    "split",
    "domain",
    "rule_id",
    "rule_ids",
    "edit_operations",
    "edits",
    "metadata",
    "candidate_present",
]

SOURCE_COLUMNS = ("source", "input", "input_text", "incorrect", "erroneous", "original", "error_text")
TARGET_COLUMNS = ("target", "target_text", "correction", "correct", "corrected", "output", "correct_text")
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


@dataclass(frozen=True)
class RealErrorLoadResult:
    rows: list[dict[str, Any]]
    accepted_count: int
    rejected_count: int
    source_reports: list[dict[str, Any]]
    rejection_reason_counts: dict[str, int]
    output_path: str


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
) -> RealPairValidation:
    source = _normalize_text(source)
    target = _normalize_text(target)
    reason = _basic_pair_rejection(source, target, min_tokens=min_tokens, max_tokens=max_tokens)
    if reason:
        return RealPairValidation(False, reason, None, "", [], False)

    char_ratio = Levenshtein.distance(source, target) / max(1, max(len(source), len(target)))
    if char_ratio > max_char_edit_ratio:
        return RealPairValidation(False, "char_edit_distance_too_high", None, "", [], False)
    token_ratio = _token_edit_ratio(source, target)
    if token_ratio > max_token_edit_ratio:
        return RealPairValidation(False, "token_edit_distance_too_high", None, "", [], False)

    generator = candidate_generator or CandidateGenerator()
    candidates = generator.generate(source)
    edits = [edit for edit in DiffAnalyzer().analyze(source, target, candidates=candidates) if _is_real_edit_supported(source, edit)]
    if not edits:
        return RealPairValidation(False, "unsupported_edit_type", None, "", [], False)
    error_types = sorted({coarse_error_type(edit.edit_type) for edit in edits if coarse_error_type(edit.edit_type) != "unknown"})
    if not error_types:
        return RealPairValidation(False, "unknown_error_type", None, "", [], False)
    candidate_present = any(candidate_matches_edit(candidate, edit) for edit in edits for candidate in candidates)
    if not candidate_present:
        return RealPairValidation(False, "candidate_missing", None, _edit_summary(edits), error_types, False)

    rule_ids = _rule_ids(edits)
    row = {
        "source": source,
        "target": target,
        "error_types": json.dumps(error_types, ensure_ascii=False),
        "error_type": error_types[0],
        "source_dataset": source_dataset,
        "source_type": "real_error_pair",
        "is_clean": False,
        "is_hard_negative": False,
        "is_synthetic": False,
        "is_real_pair": True,
        "split": "train",
        "domain": domain,
        "rule_id": rule_ids[0] if rule_ids else "unknown",
        "rule_ids": json.dumps(rule_ids or ["unknown"], ensure_ascii=False),
        "edit_operations": json.dumps([asdict(edit) for edit in edits], ensure_ascii=False),
        "edits": json.dumps([asdict(edit) for edit in edits], ensure_ascii=False),
        "metadata": json.dumps({"source_type": "real_error_pair", "source_dataset": source_dataset}, ensure_ascii=False),
        "candidate_present": True,
    }
    return RealPairValidation(True, "", row, _edit_summary(edits), error_types, True)


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
    rejected_rows: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for source_name, spec in source_specs:
        if spec.get("enabled") is False:
            source_reports.append(_source_report(source_name, spec, status="skipped", reason="disabled"))
            continue
        source_type = str(spec.get("type") or "local_jsonl")
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
                )
                allowed_error_types = set(str(item) for item in spec.get("allowed_error_types", []))
                if validation.accepted and validation.row is not None and allowed_error_types and not set(validation.detected_error_types) <= allowed_error_types:
                    rejection_counts["disallowed_error_type"] += 1
                    rejected_rows.append(
                        _rejected_row(
                            source_name,
                            source,
                            target,
                            "disallowed_error_type",
                            validation.edit_summary,
                            json.dumps(validation.detected_error_types, ensure_ascii=False),
                            validation.candidate_present,
                        )
                    )
                elif validation.accepted and validation.row is not None:
                    rows.append(validation.row)
                else:
                    rejection_counts[validation.reason] += 1
                    rejected_rows.append(
                        _rejected_row(
                            source_name,
                            source,
                            target,
                            validation.reason,
                            validation.edit_summary,
                            json.dumps(validation.detected_error_types, ensure_ascii=False),
                            validation.candidate_present,
                        )
                    )
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
                status="loaded",
                reason="",
                total_seen=seen,
                accepted=len(rows) - accepted_before,
                rejected=len(rejected_rows) - rejected_before,
                download=download,
            )
        )

    rows = _deduplicate_rows(rows)
    rows, cap_rejections, cap_rejection_counts = _apply_source_caps(rows, _cap_shares(source_specs))
    if cap_rejections:
        rejected_rows.extend(cap_rejections)
        rejection_counts.update(cap_rejection_counts)
    _refresh_source_report_counts(source_reports, rows, rejected_rows)
    output = Path(output_path or "data/processed/real_error_pairs_validated.csv.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=REAL_PAIR_COLUMNS).to_csv(output, index=False)
    report_dir = Path(reports_dir or "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_real_pair_filter_rows(source_reports, rejection_counts)).to_csv(report_dir / "real_pair_filter_report.csv", index=False)
    pd.DataFrame(rejected_rows, columns=["source_dataset", "source", "target", "reason", "edit_summary", "detected_error_types", "candidate_present"]).to_csv(
        report_dir / "rejected_real_pairs.csv",
        index=False,
    )
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
    )


def _download_policy(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("download_policy", {}) or config.get("sources", {}).get("download_policy", {}) or {})


def _source_specs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw = config.get("real_sources", config.get("sources", []))
    if isinstance(raw, dict):
        return [(str(name), dict(spec or {})) for name, spec in raw.items()]
    if isinstance(raw, list):
        return [(str(spec.get("name") or f"source_{index}"), dict(spec)) for index, spec in enumerate(raw) if isinstance(spec, dict)]
    return []


def _iter_pairs(spec: dict[str, Any], path: Path) -> Iterable[tuple[str, str, dict[str, Any]]]:
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
                    "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                }
    elif source_type in {"local_csv", "csv", "tsv", "table"}:
        if path is None:
            return
        delimiter = "\t" if str(path).endswith((".tsv", ".tab")) else ","
        with _open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for item in reader:
                yield _first_value(item, SOURCE_COLUMNS), _first_value(item, TARGET_COLUMNS), {"domain": str(item.get("domain") or "")}
    elif source_type == "m2":
        if path is None:
            return
        from src.data.external_sources import _m2_pairs

        for source, target in _m2_pairs(path):
            yield source, target, {"domain": "m2"}
    elif source_type in {"hf_dataset", "huggingface_dataset"}:
        from datasets import load_dataset

        dataset_id = str(spec.get("hf_id") or spec.get("repo"))
        dataset = load_dataset(dataset_id, name=spec.get("name_in_dataset"), split=spec.get("split", "train"))
        for item in dataset:
            yield _first_value(item, SOURCE_COLUMNS), _first_value(item, TARGET_COLUMNS), {"domain": str(item.get("domain") or "")}


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
    if any(ord(char) > 0xFFFF for char in source + target):
        return "emoji"
    return ""


def _is_real_edit_supported(source_text: str, edit: Edit) -> bool:
    if not is_allowed_edit_type(edit.edit_type):
        return False
    if coarse_error_type(edit.edit_type) == "unknown":
        return False
    if edit.start < 0 or edit.end < edit.start:
        return False
    span_end = edit.end if edit.end > edit.start else edit.start + 1
    return not any(edit.start < span.end and span.start < span_end for span in find_protected_spans(source_text))


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
    result: list[str] = []
    for edit in edits:
        rule_id = edit.rule_id or "unknown"
        if rule_id not in result:
            result.append(rule_id)
    return result


def _edit_summary(edits: list[Edit]) -> str:
    return "; ".join(f"{edit.edit_type}:{edit.source}->{edit.replacement}:{edit.rule_id or 'unknown'}" for edit in edits)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\xa0", " ")).strip()


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
                str(row.get("edit_operations", "")),
                str(row.get("error_types", "[]")),
                bool(row.get("candidate_present", True)),
            )
        )
        counts["source_share_cap"] += 1
    return result, rejected, counts


def _rejected_row(
    source_dataset: str,
    source: str,
    target: str,
    reason: str,
    edit_summary: str,
    detected_error_types: str,
    candidate_present: bool,
) -> dict[str, Any]:
    return {
        "source_dataset": source_dataset,
        "source": _normalize_text(source),
        "target": _normalize_text(target),
        "reason": _normalize_text(reason),
        "edit_summary": _normalize_text(edit_summary),
        "detected_error_types": _normalize_text(detected_error_types),
        "candidate_present": bool(candidate_present),
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
) -> None:
    accepted_counts = Counter(str(row.get("source_dataset") or "") for row in rows)
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
        f"- rejected_pairs: {rejected_pairs}",
        f"- rejection_reasons: {json.dumps(dict(sorted(rejection_counts.items())), ensure_ascii=False, sort_keys=True)}",
        f"- candidate_present_rate: {candidate_present_rate:.4f}",
        f"- cap_rejections: {int(rejection_counts.get('source_share_cap', 0))}",
        f"- final_verdict: {verdict}",
        "",
        "| source | status | mode | path | url/hf | bytes | seen | accepted | rejected | cap_rejections | reason | candidate_present_rate |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for report in source_reports:
        source_ref = report.get("url") or report.get("hf_id") or ""
        lines.append(
            f"| {report['source_dataset']} | {report['status']} | {report.get('mode', '')} | {report.get('local_path', '')} | "
            f"{source_ref} | {report.get('downloaded_size_bytes', 0)} | {report['total_seen']} | {report['accepted']} | "
            f"{report['rejected']} | {report.get('cap_rejections', 0)} | {report['reason']} | "
            f"{float(report.get('candidate_present_rate', report.get('candidate_coverage', 0.0))):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
