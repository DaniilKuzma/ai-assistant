from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
import statistics
import time
from typing import Any


PROFILE_COLUMNS = [
    "row_index",
    "split",
    "source_length_chars",
    "source_length_tokens",
    "total_ms",
    "candidate_generation_ms",
    "dictionary_candidate_ms",
    "punctuation_candidate_ms",
    "syntax_ms",
    "diff_alignment_ms",
    "tokenizer_encode_ms",
    "replacement_encoding_ms",
    "label_build_ms",
    "candidate_count",
    "dictionary_candidate_count",
    "punctuation_candidate_count",
    "keep_candidate_count",
    "rule_ids",
    "slow_reason",
]


@dataclass
class FeatureBuildProfiler:
    enabled: bool
    output_path: Path
    summary_path: Path
    sample_size: int
    split: str
    syntax_enabled: bool
    dictionary_lexicon_size: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def should_profile(self, row_index: int) -> bool:
        return self.enabled and row_index < self.sample_size

    def record(self, row: dict[str, Any]) -> None:
        if not self.enabled or len(self.rows) >= self.sample_size:
            return
        normalized = {column: row.get(column, "") for column in PROFILE_COLUMNS}
        normalized["slow_reason"] = normalized.get("slow_reason") or _slow_reason(normalized)
        self.rows.append(normalized)

    def write_reports(
        self,
        *,
        dictionary_cache_stats: dict[str, Any],
        feature_cache_stats: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROFILE_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            self._summary(dictionary_cache_stats=dictionary_cache_stats, feature_cache_stats=feature_cache_stats),
            encoding="utf-8",
        )

    def _summary(self, *, dictionary_cache_stats: dict[str, Any], feature_cache_stats: dict[str, Any]) -> str:
        total_time = time.perf_counter() - self.started_at
        total_rows = len(self.rows)
        rows_per_sec = total_rows / total_time if total_time > 0 else 0.0
        slowest = sorted(self.rows, key=lambda row: float(row.get("total_ms") or 0.0), reverse=True)[:50]
        avg_candidates = _average(self.rows, "candidate_count")
        avg_dictionary_candidates = _average(self.rows, "dictionary_candidate_count")
        bottleneck = _bottleneck_conclusion(self.rows)
        lines = [
            "# Feature Build Profile Summary",
            "",
            f"- total rows profiled: {total_rows}",
            f"- total time: {total_time:.3f}s",
            f"- rows/sec: {rows_per_sec:.3f}",
            f"- p50/p90/p99 total_ms: {_percentile_line(self.rows, 'total_ms')}",
            f"- p50/p90/p99 candidate_generation_ms: {_percentile_line(self.rows, 'candidate_generation_ms')}",
            f"- p50/p90/p99 dictionary_candidate_ms: {_percentile_line(self.rows, 'dictionary_candidate_ms')}",
            f"- average candidates per row: {avg_candidates:.3f}",
            f"- average dictionary candidates per row: {avg_dictionary_candidates:.3f}",
            f"- syntax enabled/disabled: {'enabled' if self.syntax_enabled else 'disabled'}",
            f"- dictionary lexicon size: {self.dictionary_lexicon_size}",
            f"- dictionary cache hit/miss stats: {dictionary_cache_stats}",
            f"- feature cache hit/miss stats: {feature_cache_stats}",
            f"- clear bottleneck conclusion: {bottleneck}",
            "",
            "## Top 50 Slowest Rows",
            "",
        ]
        for row in slowest:
            lines.append(
                "- "
                f"row_index={row.get('row_index')} "
                f"total_ms={float(row.get('total_ms') or 0.0):.3f} "
                f"candidate_generation_ms={float(row.get('candidate_generation_ms') or 0.0):.3f} "
                f"dictionary_candidate_ms={float(row.get('dictionary_candidate_ms') or 0.0):.3f} "
                f"rule_ids={row.get('rule_ids')}"
            )
        return "\n".join(lines) + "\n"


def _percentile_line(rows: list[dict[str, Any]], key: str) -> str:
    values = sorted(float(row.get(key) or 0.0) for row in rows)
    if not values:
        return "0.000 / 0.000 / 0.000"
    return f"{_percentile(values, 50):.3f} / {_percentile(values, 90):.3f} / {_percentile(values, 99):.3f}"


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _average(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows]
    return statistics.fmean(values) if values else 0.0


def _slow_reason(row: dict[str, Any]) -> str:
    dictionary_ms = float(row.get("dictionary_candidate_ms") or 0.0)
    syntax_ms = float(row.get("syntax_ms") or 0.0)
    tokenizer_ms = float(row.get("tokenizer_encode_ms") or 0.0)
    diff_ms = float(row.get("diff_alignment_ms") or 0.0)
    timings = {
        "dictionary": dictionary_ms,
        "syntax": syntax_ms,
        "tokenizer": tokenizer_ms,
        "diff_alignment": diff_ms,
    }
    reason, value = max(timings.items(), key=lambda item: item[1])
    return reason if value > 0 else ""


def _bottleneck_conclusion(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no profiled rows"
    averages = {
        "dictionary": _average(rows, "dictionary_candidate_ms"),
        "syntax": _average(rows, "syntax_ms"),
        "tokenizer": _average(rows, "tokenizer_encode_ms"),
        "diff_alignment": _average(rows, "diff_alignment_ms"),
        "replacement_encoding": _average(rows, "replacement_encoding_ms"),
        "label_build": _average(rows, "label_build_ms"),
    }
    reason, value = max(averages.items(), key=lambda item: item[1])
    return f"{reason} dominates average row time ({value:.3f} ms)"
