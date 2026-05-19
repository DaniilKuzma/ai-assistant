from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.data.clean_corpus_sources import normalize_sentence
from src.data.open_corpora_sources import OpenCorpusSentence, load_open_corpora_sentences


CLEAN_POOL_COLUMNS = [
    "text",
    "source_name",
    "source_subcorpus",
    "domain",
    "style",
    "license/status",
    "tokens_count",
    "chars_count",
    "cyrillic_ratio",
    "source_doc_id",
    "sentence_id",
    "hash",
    "accepted_reason",
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


def is_clean_sentence_acceptable(text: str, source_metadata: dict[str, Any]) -> bool:
    return clean_sentence_rejection_reason(text, source_metadata) == ""


def clean_sentence_rejection_reason(text: str, source_metadata: dict[str, Any]) -> str:
    text = normalize_sentence(text)
    lower = text.lower()
    domain = str(source_metadata.get("domain") or "").lower()
    style = str(source_metadata.get("style") or "").lower()
    subcorpus = str(source_metadata.get("source_subcorpus") or source_metadata.get("subcorpus") or "").lower()
    source_name = str(source_metadata.get("source_name") or "").lower()

    if any(marker in value for marker in FORBIDDEN_DOMAINS for value in (domain, style, subcorpus, source_name)):
        return "forbidden_domain_or_style"
    if len(text) < 25:
        return "too_short_chars"
    if len(text) > 220:
        return "too_long_chars"
    tokens = _word_tokens(text)
    if len(tokens) < 6:
        return "too_few_tokens"
    if len(tokens) > 35:
        return "too_many_tokens"
    if cyrillic_ratio(text) < 0.75:
        return "low_cyrillic_ratio"
    if re.search(r"<[^>]+>|\{\{|}}|\[\[|]]|#{1,6}\s|[`{}]|={2,}", text):
        return "markup"
    if re.search(r"https?://|www\.|[\w.+-]+@[\w-]+\.[\w.-]+", text, flags=re.I):
        return "url_or_email"
    if re.search(r"#[\wа-яё]+|@\w+", text, flags=re.I):
        return "social_marker"
    if _contains_emoji(text):
        return "emoji"
    if re.search(r"\b(src|tests?|docs?|github|commit|pull request|http|api)/[\w./-]+", lower):
        return "code_or_github_path"
    if len(re.findall(r"\d+(?:[,.]\d+)?", text)) >= 6:
        return "numeric_table"
    if text.startswith(("-", "—", "–")) and source_name in {"fiction", "proza"}:
        return "dialogue_or_fiction_line"
    if any(marker in lower for marker in OBSCENE_OR_SLANG_MARKERS):
        return "obscene_or_slang"
    if any(pattern in lower for pattern in META_LANGUAGE_PATTERNS):
        return "synthetic_meta_language"
    return ""


def build_clean_sentence_pool(
    config: dict[str, Any] | str | Path,
    *,
    output_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> CleanSentencePoolResult:
    load_result = load_open_corpora_sentences(config)
    pool_config = _pool_config(config)
    min_clean_sentences = int(pool_config.get("min_clean_sentences", 150_000))
    max_source_share = float(pool_config.get("max_source_share", 0.45))
    max_subcorpus_share = float(pool_config.get("max_subcorpus_share", 0.35))
    enable_near_dedup = bool(pool_config.get("enable_near_duplicate_filter", False))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    near_seen: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    sample_rejections: dict[str, list[str]] = defaultdict(list)
    source_filter_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for record in load_result.records:
        metadata = {
            "source_name": record.source_name,
            "source_subcorpus": record.source_subcorpus,
            "domain": record.domain,
            "style": record.style,
        }
        source_filter_counts[record.source_name]["total_seen"] += 1
        reason = clean_sentence_rejection_reason(record.text, metadata)
        normalized = normalize_for_dedup(record.text)
        near_key = normalize_template_text(record.text)
        if not reason and normalized in seen:
            reason = "duplicate_normalized_text"
        if enable_near_dedup and not reason and near_key in near_seen:
            reason = "near_duplicate_normalized_text"
        if reason:
            rejection_counts[reason] += 1
            source_filter_counts[record.source_name][f"rejected:{reason}"] += 1
            if len(sample_rejections[reason]) < 5:
                sample_rejections[reason].append(record.text)
            continue
        seen.add(normalized)
        near_seen.add(near_key)
        source_filter_counts[record.source_name]["accepted"] += 1
        rows.append(_pool_row(record))

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

    report_dir = Path(reports_dir or "reports/short_dataset_v2")
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_clean_filter_report(report_dir / "clean_source_filter_report.csv", source_filter_counts, sample_rejections)
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


def _pool_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, str | Path):
        import yaml

        with Path(config).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    return dict(config.get("pool", {}) or {})


def _pool_row(record: OpenCorpusSentence) -> dict[str, Any]:
    text = normalize_sentence(record.text)
    digest = hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()
    tokens = _word_tokens(text)
    return {
        "text": text,
        "source_name": record.source_name,
        "source_subcorpus": record.source_subcorpus,
        "domain": record.domain,
        "style": record.style,
        "license/status": record.license_status,
        "tokens_count": len(tokens),
        "chars_count": len(text),
        "cyrillic_ratio": cyrillic_ratio(text),
        "source_doc_id": record.source_doc_id,
        "sentence_id": record.sentence_id,
        "hash": digest,
        "accepted_reason": "passed_quality_filters",
    }


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?|\d+(?:[,.]\d+)?", text)


def _contains_emoji(text: str) -> bool:
    return any(ord(char) > 0xFFFF for char in text)


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
    sample_rejections: dict[str, list[str]],
) -> None:
    rows: list[dict[str, Any]] = []
    global_reasons = Counter()
    for counter in source_filter_counts.values():
        for key, count in counter.items():
            if key.startswith("rejected:"):
                global_reasons[key.removeprefix("rejected:")] += count
    for source_name, counter in sorted(source_filter_counts.items()):
        rejected = sum(count for key, count in counter.items() if key.startswith("rejected:"))
        reasons = {
            key.removeprefix("rejected:"): count
            for key, count in sorted(counter.items())
            if key.startswith("rejected:")
        }
        rows.append(
            {
                "source_name": source_name,
                "total_seen": int(counter.get("total_seen", 0)),
                "accepted": int(counter.get("accepted", 0)),
                "rejected": int(rejected),
                "rejection_reason_counts": json.dumps(reasons, ensure_ascii=False, sort_keys=True),
                "sample_rejections": json.dumps(
                    {reason: sample_rejections.get(reason, []) for reason in reasons},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    pd.DataFrame(
        rows,
        columns=["source_name", "total_seen", "accepted", "rejected", "rejection_reason_counts", "sample_rejections"],
    ).to_csv(path, index=False)


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
        "| source | status | mode | path | seen | accepted | reason | license/status |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for report in reports:
        lines.append(
            "| {source_name} | {status} | {mode} | {local_path} | {total_seen} | {accepted} | {reason} | {license_status} |".format(
                **{key: str(value).replace("|", "\\|") for key, value in report.items()}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
