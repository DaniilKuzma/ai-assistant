from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.candidates.matching import candidate_matches_edit
from src.validation.diff_analyzer import DiffAnalyzer, Edit


CANDIDATE_SCORE_DISTRIBUTION_COLUMNS = [
    "rule_id",
    "edit_type",
    "gold_count",
    "candidate_count",
    "positive_score_mean",
    "positive_score_p10",
    "positive_score_p50",
    "positive_score_p90",
    "negative_score_mean",
    "accepted_count",
    "rejected_by_threshold_count",
    "rejected_by_validator_count",
    "rejected_reason_counts",
    "examples_high_score_rejected",
    "examples_low_score_gold",
]


def write_candidate_score_distribution_by_rule(
    rows: list[dict[str, Any]],
    candidate_decisions: list[dict[str, Any]],
    accepted_edits: list[dict[str, Any]],
    rejected_edits: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = candidate_score_distribution_by_rule_frame(rows, candidate_decisions, accepted_edits, rejected_edits)
    frame.to_csv(output / "candidate_score_distribution_by_rule.csv", index=False)


def candidate_score_distribution_by_rule_frame(
    rows: list[dict[str, Any]],
    candidate_decisions: list[dict[str, Any]],
    accepted_edits: list[dict[str, Any]],
    rejected_edits: list[dict[str, Any]],
) -> pd.DataFrame:
    gold_by_row = _gold_edits_by_row(rows, candidate_decisions)
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(_score_bucket)
    gold_counter: Counter[tuple[str, str]] = Counter()

    for row_id, edits in gold_by_row.items():
        del row_id
        for edit in edits:
            rule_id = str(edit.rule_id or "")
            if not rule_id:
                continue
            gold_counter[(rule_id, _candidate_edit_type_from_gold(edit.edit_type))] += 1

    for record in candidate_decisions:
        candidate = record.get("candidate")
        rule_id = str(record.get("rule_id") or getattr(candidate, "rule_id", "") or "")
        edit_type = str(record.get("edit_type") or getattr(candidate, "edit_type", "") or "")
        if not rule_id or edit_type == "keep":
            continue
        key = (rule_id, edit_type)
        bucket = buckets[key]
        score = float(record.get("score", 0.0))
        bucket["candidate_count"] += 1
        row_id = int(record.get("row_id", -1))
        is_gold = any(candidate_matches_edit(candidate, edit) for edit in gold_by_row.get(row_id, []))
        if is_gold:
            bucket["positive_scores"].append(score)
            if not bool(record.get("threshold_passed", False)):
                _append_example(bucket["examples_low_score_gold"], record)
        else:
            bucket["negative_scores"].append(score)
        if not bool(record.get("threshold_passed", False)):
            bucket["rejected_by_threshold_count"] += 1
            bucket["rejected_reasons"]["threshold"] += 1
        elif bool(record.get("conflict", False)):
            bucket["rejected_reasons"]["conflict"] += 1
        elif str(record.get("validator_status", "")) == "rejected":
            bucket["rejected_by_validator_count"] += 1
            reason = str(record.get("validator_reason") or "validator")
            bucket["rejected_reasons"][reason] += 1
            _append_example(bucket["examples_high_score_rejected"], record)

    for record in accepted_edits:
        rule_id = str(record.get("rule_id") or "")
        if not rule_id:
            continue
        buckets[(rule_id, _candidate_edit_type_from_gold(str(record.get("edit_type") or "")))]["accepted_count"] += 1

    for record in rejected_edits:
        rule_id = str(record.get("rule_id") or "")
        if not rule_id:
            continue
        key = (rule_id, _candidate_edit_type_from_gold(str(record.get("edit_type") or "")))
        reason = str(record.get("reason") or "validator")
        buckets[key]["rejected_reasons"][reason] += 1

    keys = sorted(set(buckets) | set(gold_counter))
    rows_out = []
    for key in keys:
        bucket = buckets[key]
        positive_scores = bucket["positive_scores"]
        negative_scores = bucket["negative_scores"]
        rows_out.append(
            {
                "rule_id": key[0],
                "edit_type": key[1],
                "gold_count": int(gold_counter[key]),
                "candidate_count": int(bucket["candidate_count"]),
                "positive_score_mean": _mean(positive_scores),
                "positive_score_p10": _quantile(positive_scores, 0.10),
                "positive_score_p50": _quantile(positive_scores, 0.50),
                "positive_score_p90": _quantile(positive_scores, 0.90),
                "negative_score_mean": _mean(negative_scores),
                "accepted_count": int(bucket["accepted_count"]),
                "rejected_by_threshold_count": int(bucket["rejected_by_threshold_count"]),
                "rejected_by_validator_count": int(bucket["rejected_by_validator_count"]),
                "rejected_reason_counts": json.dumps(dict(bucket["rejected_reasons"]), ensure_ascii=False, sort_keys=True),
                "examples_high_score_rejected": " | ".join(bucket["examples_high_score_rejected"][:3]),
                "examples_low_score_gold": " | ".join(bucket["examples_low_score_gold"][:3]),
            }
        )
    return pd.DataFrame(rows_out, columns=CANDIDATE_SCORE_DISTRIBUTION_COLUMNS)


def _gold_edits_by_row(rows: list[dict[str, Any]], candidate_decisions: list[dict[str, Any]]) -> dict[int, list[Edit]]:
    candidates_by_row: dict[int, list[Any]] = defaultdict(list)
    for record in candidate_decisions:
        row_id = int(record.get("row_id", -1))
        candidate = record.get("candidate")
        if row_id >= 0 and candidate is not None:
            candidates_by_row[row_id].append(candidate)

    analyzer = DiffAnalyzer()
    result: dict[int, list[Edit]] = {}
    for row_id, row in enumerate(rows):
        result[row_id] = analyzer.analyze(
            str(row.get("source", "")),
            str(row.get("target", "")),
            candidates=candidates_by_row.get(row_id, []),
        )
    return result


def _score_bucket() -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "positive_scores": [],
        "negative_scores": [],
        "accepted_count": 0,
        "rejected_by_threshold_count": 0,
        "rejected_by_validator_count": 0,
        "rejected_reasons": Counter(),
        "examples_high_score_rejected": [],
        "examples_low_score_gold": [],
    }


def _candidate_edit_type_from_gold(edit_type: str) -> str:
    return {
        "spelling_replace": "spelling",
        "hyphen_change": "hyphen",
        "split_word": "split_join",
        "join_words": "split_join",
        "case_change": "case",
    }.get(edit_type, edit_type)


def _append_example(target: list[str], record: dict[str, Any]) -> None:
    if len(target) >= 3:
        return
    target.append(
        f"{record.get('source', '')}->{record.get('replacement', '')}"
        f" score={float(record.get('score', 0.0)):.4f}"
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 6)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction), 6)
