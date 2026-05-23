from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.data.clean_corpus_sources import normalize_sentence
from src.data.open_corpora_sources import OpenCorpusSentence, load_open_corpora_sentences
from src.preprocessing.protected_spans import find_protected_spans


CLEAN_POOL_COLUMNS = [
    "text",
    "source_name",
    "source_subcorpus",
    "domain",
    "style",
    "license_status",
    "token_count",
    "char_count",
    "cyrillic_ratio",
    "source_doc_id",
    "sentence_id",
    "hash",
    "normalized_text_hash",
    "accepted_reason",
    "rejected_reason",
    "raw_source_path",
]

META_LANGUAGE_PATTERNS = (
    "правило",
    "серия",
    "серии",
    "семейство",
    "context-pairs",
    "ne-pos",
    "n-nn",
    "пример 123",
    "проверяет семейство",
    "готовит важный примере",
    "готовит итоговый примере",
    "готовит точный примере",
    "готовит рабочий примере",
)

OBSCENE_OR_SLANG_MARKERS = (
    " блин",
    " хрен",
    " фиг",
    " чувак",
    " чувиха",
    " ёп",
    " епт",
    " нах",
    " хуй",
    " пизд",
    " бляд",
    " сука",
    " говн",
    " лол",
    " кек",
    " имхо",
)

FORBIDDEN_DOMAINS = {"fiction", "proza", "poetry", "stihi", "social", "subtitles", "forum", "fanfiction"}
LATIN_CONFUSABLES = frozenset("aceopxyABCEHKMOPTXY")
DEFAULT_POOL_FILTER_CONFIG = {
    "reject_mixed_script_tokens": True,
    "reject_latin_confusable_inside_cyrillic_word": True,
    "reject_if_candidate_generator_finds_high_confidence_fix": False,
    "high_confidence_candidate_threshold": 0.95,
}


@dataclass(frozen=True)
class CleanSentencePoolResult:
    accepted_count: int
    total_seen: int
    output_path: str
    source_counts: dict[str, int]
    subcorpus_counts: dict[str, int]
    rejection_reason_counts: dict[str, int]
    source_reports: list[dict[str, Any]]
    dominance_violations: list[str]
    shortage_reason: str = ""


@dataclass(frozen=True)
class CleanSentenceDecision:
    accepted: bool
    reasons: list[str]

    def __bool__(self) -> bool:
        return self.accepted


def is_clean_sentence_acceptable(
    text: str,
    source_metadata: dict[str, Any],
    *,
    pool_config: dict[str, Any] | None = None,
    candidate_generator: Any | None = None,
) -> CleanSentenceDecision:
    reasons = clean_sentence_rejection_reasons(
        text,
        source_metadata,
        pool_config=pool_config,
        candidate_generator=candidate_generator,
    )
    return CleanSentenceDecision(accepted=not reasons, reasons=reasons)


def clean_sentence_acceptance_reason(
    text: str,
    source_metadata: dict[str, Any],
    *,
    pool_config: dict[str, Any] | None = None,
    candidate_generator: Any | None = None,
) -> tuple[bool, str, str]:
    reasons = clean_sentence_rejection_reasons(
        text,
        source_metadata,
        pool_config=pool_config,
        candidate_generator=candidate_generator,
    )
    if not reasons:
        return True, "passed_quality_filters", ""
    if len(reasons) == 1 and reasons[0] in {"url_or_email", "numeric_table"} and _looks_like_protected_hard_negative(text):
        return True, "hard_negative_candidate", ""
    reason = reasons[0]
    return False, "", reason


def clean_sentence_rejection_reason(
    text: str,
    source_metadata: dict[str, Any],
    *,
    pool_config: dict[str, Any] | None = None,
    candidate_generator: Any | None = None,
) -> str:
    reasons = clean_sentence_rejection_reasons(
        text,
        source_metadata,
        pool_config=pool_config,
        candidate_generator=candidate_generator,
    )
    return reasons[0] if reasons else ""


def clean_sentence_rejection_reasons(
    text: str,
    source_metadata: dict[str, Any],
    *,
    pool_config: dict[str, Any] | None = None,
    candidate_generator: Any | None = None,
) -> list[str]:
    text = normalize_sentence(text)
    filter_config = _effective_pool_filter_config(pool_config)
    lower = text.lower()
    domain = str(source_metadata.get("domain") or "").lower()
    style = str(source_metadata.get("style") or "").lower()
    subcorpus = str(source_metadata.get("source_subcorpus") or source_metadata.get("subcorpus") or "").lower()
    source_name = str(source_metadata.get("source_name") or "").lower()

    reasons: list[str] = []
    if _has_forbidden_domain_marker(domain, style, subcorpus, source_name):
        reasons.append("forbidden_domain_or_style")
    if bool(filter_config.get("reject_mixed_script_tokens", True)) and has_mixed_script_token(text):
        reasons.append("mixed_script_token")
    if bool(filter_config.get("reject_latin_confusable_inside_cyrillic_word", True)) and has_latin_confusable_inside_cyrillic_word(text):
        reasons.append("latin_confusable_inside_cyrillic_word")
    if len(text) < 25:
        reasons.append("too_short_chars")
    if len(text) > 220:
        reasons.append("too_long_chars")
    tokens = _word_tokens(text)
    if len(tokens) < 6:
        reasons.append("too_few_tokens")
    if len(tokens) > 35:
        reasons.append("too_many_tokens")
    if cyrillic_ratio(text) < 0.75:
        reasons.append("low_cyrillic_ratio")
    if re.search(r"<[^>]+>|\{\{|}}|\[\[|]]|#{1,6}\s|[`{}]|={2,}", text):
        reasons.append("markup")
    if re.search(r"https?://|www\.|[\w.+-]+@[\w-]+\.[\w.-]+", text, flags=re.I):
        reasons.append("url_or_email")
    if re.search(r"#[\wа-яё]+|@\w+", text, flags=re.I):
        reasons.append("social_marker")
    if _contains_emoji(text):
        reasons.append("emoji")
    if re.search(r"\b(src|tests?|docs?|github|commit|pull request|http|api)/[\w./-]+", lower):
        reasons.append("code_or_github_path")
    if re.search(r"\b(?:def|class|import|return|var|let|const)\b|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_]{1,8}", text):
        reasons.append("code_snippet")
    if re.search(r"(?:[A-Za-z]:\\|/[\w.-]+/[\w./-]+)", text):
        reasons.append("file_path")
    if len(re.findall(r"\d+(?:[,.]\d+)?", text)) >= 6:
        reasons.append("numeric_table")
    if text.startswith(("-", "—", "–")):
        reasons.append("dialogue_or_fiction_line")
    if re.search(r"\S\u2014\s|\s\u2014\S", text):
        reasons.append("malformed_dash_spacing")
    if re.search(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s", text):
        reasons.append("subtitles_format")
    if any(marker in lower for marker in OBSCENE_OR_SLANG_MARKERS):
        reasons.append("obscene_or_slang")
    if any(pattern in lower for pattern in META_LANGUAGE_PATTERNS):
        reasons.append("synthetic_meta_language")
    if re.search(r"[!?.,;:]{2,}|\.{3,}", text):
        reasons.append("broken_punctuation_artifact")
    if (
        not reasons
        and bool(filter_config.get("reject_if_candidate_generator_finds_high_confidence_fix", False))
        and _has_high_confidence_autocorrection_candidate(
            text,
            candidate_generator,
            threshold=_safe_float(filter_config.get("high_confidence_candidate_threshold", 0.95), 0.95),
        )
    ):
        reasons.append("high_confidence_autocorrection_candidate")
    return list(dict.fromkeys(reasons))


def build_clean_sentence_pool(
    config: dict[str, Any] | str | Path,
    *,
    output_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> CleanSentencePoolResult:
    config_data = _config_data(config)
    load_result = load_open_corpora_sentences(config_data)
    pool_config = _pool_config(config_data)
    min_clean_sentences = int(pool_config.get("min_clean_sentences", 150_000))
    max_source_share = float(pool_config.get("max_source_share", 0.45))
    max_subcorpus_share = float(pool_config.get("max_subcorpus_share", 0.35))
    enable_near_dedup = bool(pool_config.get("enable_near_duplicate_filter", False))
    candidate_generator = _candidate_generator_for_clean_pool(config_data, pool_config)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    near_seen: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    sample_rejections: dict[str, list[str]] = defaultdict(list)
    sample_accepted: dict[str, list[str]] = defaultdict(list)
    sample_rejected_by_source: dict[str, list[str]] = defaultdict(list)
    source_filter_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for record in load_result.records:
        metadata = {
            "source_name": record.source_name,
            "source_subcorpus": record.source_subcorpus,
            "domain": record.domain,
            "style": record.style,
        }
        source_filter_counts[record.source_name]["total_seen"] += 1
        accepted, accepted_reason, reason = clean_sentence_acceptance_reason(
            record.text,
            metadata,
            pool_config=pool_config,
            candidate_generator=candidate_generator,
        )
        normalized = normalize_for_dedup(record.text)
        near_key = normalize_template_text(record.text)
        if accepted and normalized in seen:
            reason = "duplicate_normalized_text"
            accepted = False
        if enable_near_dedup and accepted and near_key in near_seen:
            reason = "near_duplicate_normalized_text"
            accepted = False
        if reason:
            rejection_counts[reason] += 1
            source_filter_counts[record.source_name][f"rejected:{reason}"] += 1
            if len(sample_rejections[reason]) < 5:
                sample_rejections[reason].append(record.text)
            if len(sample_rejected_by_source[record.source_name]) < 5:
                sample_rejected_by_source[record.source_name].append(record.text)
            continue
        seen.add(normalized)
        near_seen.add(near_key)
        source_filter_counts[record.source_name]["accepted"] += 1
        if len(sample_accepted[record.source_name]) < 5:
            sample_accepted[record.source_name].append(record.text)
        rows.append(_pool_row(record, accepted_reason=accepted_reason))

    rows, cap_rejections = _enforce_share_caps(rows, max_source_share=max_source_share, max_subcorpus_share=max_subcorpus_share)
    rejection_counts.update(cap_rejections)
    source_counts = Counter(row["source_name"] for row in rows)
    subcorpus_counts = Counter(row["source_subcorpus"] or row["source_name"] for row in rows)
    dominance_violations = _dominance_violations(
        total=len(rows),
        source_counts=source_counts,
        subcorpus_counts=subcorpus_counts,
        max_source_share=max_source_share,
        max_subcorpus_share=max_subcorpus_share,
    )

    output = Path(output_path or "data/processed/clean_sentence_pool.csv.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CLEAN_POOL_COLUMNS).to_csv(output, index=False)

    report_dir = Path(reports_dir or "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_clean_filter_report(
        report_dir / "clean_source_filter_report.csv",
        source_filter_counts,
        sample_accepted,
        sample_rejected_by_source,
    )
    _write_clean_filter_markdown_report(
        report_dir / "clean_source_filter_report.md",
        source_filter_counts,
        sample_accepted,
        sample_rejected_by_source,
        rejection_counts,
    )
    _write_source_ingestion_report(
        report_dir / "source_ingestion_report.md",
        load_result.source_reports,
        accepted_count=len(rows),
        min_clean_sentences=min_clean_sentences,
        dominance_violations=dominance_violations,
    )

    shortage_reason = "" if len(rows) >= min_clean_sentences else f"accepted_clean_sentences_below_min:{len(rows)}<{min_clean_sentences}"
    return CleanSentencePoolResult(
        accepted_count=len(rows),
        total_seen=len(load_result.records),
        output_path=str(output),
        source_counts=dict(sorted(source_counts.items())),
        subcorpus_counts=dict(sorted(subcorpus_counts.items())),
        rejection_reason_counts=dict(sorted(rejection_counts.items())),
        source_reports=load_result.source_reports,
        dominance_violations=dominance_violations,
        shortage_reason=shortage_reason,
    )


def normalize_for_dedup(text: str) -> str:
    text = normalize_sentence(text).lower()
    text = text.replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def normalize_template_text(text: str) -> str:
    text = normalize_for_dedup(text)
    text = re.sub(r"\b[а-яё]{1,2}\.\s*", "<abbr> ", text)
    text = re.sub(r"[«»„“”\"']", "\"", text)
    text = re.sub(r"[^\w\s<>\"-]+", " ", text, flags=re.U)
    return re.sub(r"\s+", " ", text).strip()


def cyrillic_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if not letters:
        return 0.0
    cyrillic = re.findall(r"[А-Яа-яЁё]", text)
    return len(cyrillic) / len(letters)


def mixed_script_tokens(text: str) -> list[str]:
    result: list[str] = []
    for token in _letter_runs(text):
        if _has_cyrillic_letter(token) and _has_ascii_latin_letter(token):
            result.append(token)
    return result


def has_mixed_script_token(text: str) -> bool:
    return bool(mixed_script_tokens(text))


def has_latin_confusable_inside_cyrillic_word(text: str) -> bool:
    return any(
        _has_cyrillic_letter(token) and any(char in LATIN_CONFUSABLES for char in token)
        for token in _letter_runs(text)
    )


def _pool_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    config = _config_data(config)
    return dict(config.get("pool", {}) or {})


def _config_data(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, str | Path):
        import yaml

        with Path(config).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return config


def _effective_pool_filter_config(pool_config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(DEFAULT_POOL_FILTER_CONFIG)
    raw = dict(pool_config or {})
    nested = raw.pop("pool", None)
    if isinstance(nested, dict):
        config.update(nested)
    config.update(raw)
    return config


def _candidate_generator_for_clean_pool(config: dict[str, Any] | str | Path, pool_config: dict[str, Any]) -> Any | None:
    filter_config = _effective_pool_filter_config(pool_config)
    if not bool(filter_config.get("reject_if_candidate_generator_finds_high_confidence_fix", False)):
        return None
    if not isinstance(config, dict):
        return None
    candidate_generator = config.get("candidate_generator")
    if candidate_generator is not None:
        return candidate_generator
    from src.candidates.candidate_generator import CandidateGenerator

    return CandidateGenerator.from_config(config)


def _pool_row(record: OpenCorpusSentence, *, accepted_reason: str) -> dict[str, Any]:
    text = normalize_sentence(record.text)
    normalized = normalize_for_dedup(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    tokens = _word_tokens(text)
    return {
        "text": text,
        "source_name": record.source_name,
        "source_subcorpus": record.source_subcorpus,
        "domain": record.domain,
        "style": record.style,
        "license_status": record.license_status,
        "token_count": len(tokens),
        "char_count": len(text),
        "cyrillic_ratio": cyrillic_ratio(text),
        "source_doc_id": record.source_doc_id,
        "sentence_id": record.sentence_id,
        "hash": digest,
        "normalized_text_hash": digest,
        "accepted_reason": accepted_reason,
        "rejected_reason": "",
        "raw_source_path": getattr(record, "raw_source_path", ""),
    }


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?|\d+(?:[,.]\d+)?", text)


def _letter_runs(text: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalpha():
            current.append(char)
            continue
        if current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return runs


def _has_cyrillic_letter(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text))


def _has_ascii_latin_letter(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_high_confidence_autocorrection_candidate(
    text: str,
    candidate_generator: Any | None,
    *,
    threshold: float,
) -> bool:
    if not math.isfinite(threshold):
        return False
    if candidate_generator is None:
        from src.candidates.candidate_generator import CandidateGenerator

        candidate_generator = CandidateGenerator()
    protected = find_protected_spans(text)
    try:
        candidates = candidate_generator.generate(text)
    except Exception:
        return False
    for candidate in candidates:
        if _is_high_confidence_autocorrection_candidate(candidate, protected, threshold=threshold, text_length=len(text)):
            return True
    return False


def _is_high_confidence_autocorrection_candidate(
    candidate: Any,
    protected: list[Any],
    *,
    threshold: float,
    text_length: int,
) -> bool:
    if str(getattr(candidate, "edit_type", "")) == "keep":
        return False
    source = str(getattr(candidate, "source", ""))
    replacement = str(getattr(candidate, "replacement", ""))
    if replacement == source:
        return False
    if bool(getattr(candidate, "requires_scoring", False)):
        return False
    try:
        confidence = float(getattr(candidate, "confidence"))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(confidence) or confidence < threshold:
        return False
    try:
        start = int(getattr(candidate, "start"))
        end = int(getattr(candidate, "end"))
    except (TypeError, ValueError):
        return False
    if start < 0 or end < start or end > text_length:
        return False
    return not _candidate_touches_protected_span(start, end, protected)


def _candidate_touches_protected_span(start: int, end: int, protected: list[Any]) -> bool:
    for span in protected:
        span_start = int(getattr(span, "start", -1))
        span_end = int(getattr(span, "end", -1))
        if start == end:
            if span_start <= start < span_end:
                return True
            continue
        if start < span_end and span_start < end:
            return True
    return False


def _contains_emoji(text: str) -> bool:
    return any(ord(char) > 0xFFFF for char in text)


def _has_forbidden_domain_marker(*values: str) -> bool:
    for raw_value in values:
        value = str(raw_value or "").lower()
        tokens = {token for token in re.split(r"[^a-zа-яё]+", value) if token}
        if tokens & FORBIDDEN_DOMAINS:
            return True
    return False


def _looks_like_protected_hard_negative(text: str) -> bool:
    lower = text.lower()
    if re.search(r"https?://|www\.|[\w.+-]+@[\w-]+\.[\w.-]+", text, flags=re.I):
        return True
    if re.search(r"\d+(?:[,.]\d+)?\s?%|\d+[,.]\d+|\d+-[а-яё]+", lower):
        return True
    if re.search(r"\b(?:США|РФ|НББ|ООО|АО|ИП|г\.|ул\.|т\.д\.|т\.п\.)\b", text):
        return True
    return False


def _enforce_share_caps(
    rows: list[dict[str, Any]],
    *,
    max_source_share: float,
    max_subcorpus_share: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows, source_dropped = _cap_by_key(rows, key_name="source_name", max_share=max_source_share)
    rows, subcorpus_dropped = _cap_by_key(rows, key_name="source_subcorpus", max_share=max_subcorpus_share, fallback_key="source_name")
    counts: Counter[str] = Counter()
    if source_dropped:
        counts["source_share_cap"] = source_dropped
    if subcorpus_dropped:
        counts["subcorpus_share_cap"] = subcorpus_dropped
    return rows, counts


def _cap_by_key(
    rows: list[dict[str, Any]],
    *,
    key_name: str,
    max_share: float,
    fallback_key: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if not rows or max_share <= 0 or max_share >= 1:
        return rows, 0
    result = list(rows)
    dropped_total = 0
    while result:
        counts = Counter(str(row.get(key_name) or (row.get(fallback_key) if fallback_key else "") or "") for row in result)
        if len(counts) <= 1:
            return result, dropped_total
        key, count = counts.most_common(1)[0]
        share = count / max(1, len(result))
        if share <= max_share:
            return result, dropped_total
        other_count = len(result) - count
        allowed = int((max_share * other_count) // max(0.000001, 1.0 - max_share))
        allowed = max(0, allowed)
        drop_count = count - allowed
        if drop_count <= 0:
            return result, dropped_total
        kept_for_key = 0
        next_rows: list[dict[str, Any]] = []
        for row in result:
            row_key = str(row.get(key_name) or (row.get(fallback_key) if fallback_key else "") or "")
            if row_key == key:
                if kept_for_key < allowed:
                    next_rows.append(row)
                    kept_for_key += 1
                continue
            next_rows.append(row)
        dropped_total += len(result) - len(next_rows)
        result = next_rows
    return result, dropped_total


def _dominance_violations(
    *,
    total: int,
    source_counts: Counter[str],
    subcorpus_counts: Counter[str],
    max_source_share: float,
    max_subcorpus_share: float,
) -> list[str]:
    if total <= 0:
        return ["empty_clean_pool"]
    violations: list[str] = []
    for source, count in source_counts.items():
        if count / total > max_source_share:
            violations.append(f"source_dominance:{source}:{count / total:.4f}>{max_source_share:.4f}")
    for subcorpus, count in subcorpus_counts.items():
        if count / total > max_subcorpus_share:
            violations.append(f"subcorpus_dominance:{subcorpus}:{count / total:.4f}>{max_subcorpus_share:.4f}")
    return violations


def _write_clean_filter_report(
    path: Path,
    source_filter_counts: dict[str, Counter[str]],
    sample_accepted: dict[str, list[str]],
    sample_rejected_by_source: dict[str, list[str]],
) -> None:
    rows: list[dict[str, Any]] = []
    for source_name, counter in sorted(source_filter_counts.items()):
        rejected = sum(count for key, count in counter.items() if key.startswith("rejected:"))
        total_seen = int(counter.get("total_seen", 0))
        accepted = int(counter.get("accepted", 0))
        reasons = {
            key.removeprefix("rejected:"): count
            for key, count in sorted(counter.items())
            if key.startswith("rejected:")
        }
        rows.append(
            {
                "source_name": source_name,
                "total_seen": total_seen,
                "accepted": accepted,
                "rejected": int(rejected),
                "acceptance_rate": accepted / total_seen if total_seen else 0.0,
                "rejection_reason_counts": json.dumps(reasons, ensure_ascii=False, sort_keys=True),
                "sample_accepted": json.dumps(sample_accepted.get(source_name, []), ensure_ascii=False),
                "sample_rejected": json.dumps(sample_rejected_by_source.get(source_name, []), ensure_ascii=False),
            }
        )
    pd.DataFrame(
        rows,
        columns=[
            "source_name",
            "total_seen",
            "accepted",
            "rejected",
            "acceptance_rate",
            "rejection_reason_counts",
            "sample_accepted",
            "sample_rejected",
        ],
    ).to_csv(path, index=False)


def _write_clean_filter_markdown_report(
    path: Path,
    source_filter_counts: dict[str, Counter[str]],
    sample_accepted: dict[str, list[str]],
    sample_rejected_by_source: dict[str, list[str]],
    rejection_counts: Counter[str],
) -> None:
    total_seen = sum(counter.get("total_seen", 0) for counter in source_filter_counts.values())
    accepted = sum(counter.get("accepted", 0) for counter in source_filter_counts.values())
    lines = [
        "# Clean Source Filter Report",
        "",
        f"- total_seen: {int(total_seen)}",
        f"- accepted: {int(accepted)}",
        f"- rejected: {int(total_seen - accepted)}",
        f"- rejection_reason_counts: {json.dumps(dict(sorted(rejection_counts.items())), ensure_ascii=False, sort_keys=True)}",
        "",
        "| source | seen | accepted | rejected | acceptance_rate | sample_accepted | sample_rejected |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for source_name, counter in sorted(source_filter_counts.items()):
        seen = int(counter.get("total_seen", 0))
        accepted_count = int(counter.get("accepted", 0))
        rejected_count = sum(count for key, count in counter.items() if key.startswith("rejected:"))
        lines.append(
            "| {source} | {seen} | {accepted} | {rejected} | {rate:.4f} | {sample_a} | {sample_r} |".format(
                source=source_name,
                seen=seen,
                accepted=accepted_count,
                rejected=rejected_count,
                rate=accepted_count / seen if seen else 0.0,
                sample_a="<br>".join(sample_accepted.get(source_name, [])).replace("|", "\\|"),
                sample_r="<br>".join(sample_rejected_by_source.get(source_name, [])).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_source_ingestion_report(
    path: Path,
    reports: list[dict[str, Any]],
    *,
    accepted_count: int,
    min_clean_sentences: int,
    dominance_violations: list[str],
) -> None:
    lines = [
        "# Source Ingestion Report",
        "",
        f"- accepted_clean_sentences: {accepted_count}",
        f"- min_clean_sentences: {min_clean_sentences}",
        f"- dominance_violations: {', '.join(dominance_violations) if dominance_violations else ''}",
        "",
        "| source | status | mode | path | url/hf | bytes | seen | accepted | rejected | reason | license/status | used |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for report in reports:
        source_ref = report.get("url") or report.get("hf_id") or ""
        lines.append(
            "| {source_name} | {status} | {mode} | {local_path} | {source_ref} | {downloaded_size_bytes} | "
            "{total_seen} | {accepted} | {rejected} | {reason} | {license_status} | {used} |".format(
                source_ref=str(source_ref).replace("|", "\\|"),
                **{key: str(value).replace("|", "\\|") for key, value in report.items()},
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
