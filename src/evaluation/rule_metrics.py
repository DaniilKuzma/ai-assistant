from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.schema.edit_types import coarse_error_type


RULE_PRECISION_RECALL_COLUMNS = [
    "rule_id",
    "group",
    "gold_count",
    "predicted_count",
    "true_positive",
    "false_positive",
    "false_negative",
    "precision",
    "recall",
    "f1",
]
ERROR_BY_RULE_COLUMNS = [
    "rule_id",
    "error_type",
    "count",
    "accepted_count",
    "rejected_count",
    "false_positive_count",
    "false_negative_count",
]
RULE_WORSE_EXAMPLES_COLUMNS = [
    "rule_id",
    "source",
    "target",
    "prediction",
    "edit",
    "reason",
    "confidence",
]


def build_rule_reports(
    rows: list[dict[str, Any]],
    accepted_edits: list[dict[str, Any]],
    rejected_edits: list[dict[str, Any]],
    rules_config_path: str | Path = "configs/rules.yaml",
) -> dict[str, pd.DataFrame]:
    rule_groups = _load_rule_groups(rules_config_path)
    gold_counter, gold_by_error = _gold_counters(rows)
    predicted_counter, accepted_by_error = _edit_record_counters(accepted_edits)
    _rejected_counter, rejected_by_error = _edit_record_counters(rejected_edits)
    matched = predicted_counter & gold_counter
    false_positive_by_error = _delta_by_error(predicted_counter, matched)
    false_negative_by_error = _delta_by_error(gold_counter, matched)

    return {
        "rule_precision_recall": _rule_precision_recall_frame(
            gold_counter,
            predicted_counter,
            matched,
            rule_groups,
        ),
        "error_by_rule": _error_by_rule_frame(
            gold_by_error,
            accepted_by_error,
            rejected_by_error,
            false_positive_by_error,
            false_negative_by_error,
        ),
        "rule_worse_examples": _rule_worse_examples_frame(rows, accepted_edits, gold_counter),
    }


def write_rule_reports(
    rows: list[dict[str, Any]],
    accepted_edits: list[dict[str, Any]],
    rejected_edits: list[dict[str, Any]],
    output_dir: str | Path,
    rules_config_path: str | Path = "configs/rules.yaml",
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = build_rule_reports(rows, accepted_edits, rejected_edits, rules_config_path=rules_config_path)
    reports["rule_precision_recall"].to_csv(output / "rule_precision_recall.csv", index=False)
    reports["error_by_rule"].to_csv(output / "error_by_rule.csv", index=False)
    reports["rule_worse_examples"].to_csv(output / "rule_worse_examples.csv", index=False)


RuleEditKey = tuple[int, str, int, int, str, str, str]


def _gold_counters(rows: list[dict[str, Any]]) -> tuple[Counter[RuleEditKey], Counter[tuple[str, str]]]:
    counter: Counter[RuleEditKey] = Counter()
    by_error: Counter[tuple[str, str]] = Counter()
    analyzer = DiffAnalyzer()

    for row_id, row in enumerate(rows):
        edits = _gold_edits(row, analyzer)
        for edit in edits:
            key = _edit_key(edit, row_id)
            counter[key] += 1
            by_error[(_rule_id_from_key(key), coarse_error_type(edit.edit_type))] += 1
    return counter, by_error


def _edit_record_counters(records: list[dict[str, Any]]) -> tuple[Counter[RuleEditKey], Counter[tuple[str, str]]]:
    counter: Counter[RuleEditKey] = Counter()
    by_error: Counter[tuple[str, str]] = Counter()
    for record in records:
        edit = _edit_from_record(record)
        row_id = _record_row_id(record)
        key = _edit_key(edit, row_id)
        counter[key] += 1
        by_error[(_rule_id_from_key(key), coarse_error_type(edit.edit_type))] += 1
    return counter, by_error


def _gold_edits(row: dict[str, Any], analyzer: DiffAnalyzer) -> list[Edit]:
    if "edit_operations" in row and not _is_missing(row.get("edit_operations")):
        edits = _parse_edit_operations(row.get("edit_operations"))
    else:
        edits = analyzer.analyze(str(row.get("source", "")), str(row.get("target", "")))
    return _apply_row_rule_ids(edits, _row_rule_ids(row))


def _parse_edit_operations(value: Any) -> list[Edit]:
    operations = _parse_jsonish_list(value)
    edits: list[Edit] = []
    for operation in operations:
        if isinstance(operation, Edit):
            edits.append(operation)
        elif isinstance(operation, dict):
            edits.append(_edit_from_record(operation))
    return edits


def _apply_row_rule_ids(edits: list[Edit], rule_ids: list[str]) -> list[Edit]:
    if not rule_ids:
        return edits
    result: list[Edit] = []
    fallback_index = 0
    for edit in edits:
        if _normalize_rule_id(edit.rule_id) != UNKNOWN_RULE_ID:
            result.append(edit)
            continue
        rule_id = rule_ids[min(fallback_index, len(rule_ids) - 1)]
        fallback_index += 1
        result.append(replace(edit, rule_id=rule_id))
    return result


def _rule_precision_recall_frame(
    gold_counter: Counter[RuleEditKey],
    predicted_counter: Counter[RuleEditKey],
    matched: Counter[RuleEditKey],
    rule_groups: dict[str, str],
) -> pd.DataFrame:
    rule_ids = {
        _rule_id_from_key(key)
        for counter in (gold_counter, predicted_counter)
        for key in counter
    }
    rows: list[dict[str, Any]] = []
    for rule_id in sorted(rule_ids):
        gold_count = _count_for_rule(gold_counter, rule_id)
        predicted_count = _count_for_rule(predicted_counter, rule_id)
        true_positive = _count_for_rule(matched, rule_id)
        false_positive = max(0, predicted_count - true_positive)
        false_negative = max(0, gold_count - true_positive)
        precision, recall, f1 = _precision_recall_f1(true_positive, predicted_count, gold_count)
        rows.append(
            {
                "rule_id": rule_id,
                "group": rule_groups.get(rule_id, UNKNOWN_RULE_ID),
                "gold_count": gold_count,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows, columns=RULE_PRECISION_RECALL_COLUMNS)


def _error_by_rule_frame(
    gold_by_error: Counter[tuple[str, str]],
    accepted_by_error: Counter[tuple[str, str]],
    rejected_by_error: Counter[tuple[str, str]],
    false_positive_by_error: Counter[tuple[str, str]],
    false_negative_by_error: Counter[tuple[str, str]],
) -> pd.DataFrame:
    keys = set(gold_by_error) | set(accepted_by_error) | set(rejected_by_error) | set(false_positive_by_error) | set(false_negative_by_error)
    rows = [
        {
            "rule_id": rule_id,
            "error_type": error_type,
            "count": gold_by_error[(rule_id, error_type)],
            "accepted_count": accepted_by_error[(rule_id, error_type)],
            "rejected_count": rejected_by_error[(rule_id, error_type)],
            "false_positive_count": false_positive_by_error[(rule_id, error_type)],
            "false_negative_count": false_negative_by_error[(rule_id, error_type)],
        }
        for rule_id, error_type in sorted(keys)
    ]
    return pd.DataFrame(rows, columns=ERROR_BY_RULE_COLUMNS)


def _rule_worse_examples_frame(
    rows: list[dict[str, Any]],
    accepted_edits: list[dict[str, Any]],
    gold_counter: Counter[RuleEditKey],
) -> pd.DataFrame:
    remaining_gold = gold_counter.copy()
    examples: list[dict[str, Any]] = []
    for record in accepted_edits:
        edit = _edit_from_record(record)
        row_id = _record_row_id(record)
        key = _edit_key(edit, row_id)
        if remaining_gold[key] > 0:
            remaining_gold[key] -= 1
            continue
        row = rows[row_id] if 0 <= row_id < len(rows) else {}
        examples.append(
            {
                "rule_id": _rule_id_from_key(key),
                "source": row.get("source", ""),
                "target": row.get("target", ""),
                "prediction": row.get("prediction", ""),
                "edit": _format_edit(edit),
                "reason": str(record.get("reason") or "false_positive"),
                "confidence": _as_float(record.get("confidence"), 0.0),
            }
        )
    return pd.DataFrame(examples, columns=RULE_WORSE_EXAMPLES_COLUMNS)


def _delta_by_error(counter: Counter[RuleEditKey], matched: Counter[RuleEditKey]) -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    for key, count in counter.items():
        unmatched = count - matched.get(key, 0)
        if unmatched <= 0:
            continue
        _row_id, edit_type, _start, _end, _source, _replacement, rule_id = key
        result[(rule_id, coarse_error_type(edit_type))] += unmatched
    return result


def _edit_key(edit: Edit, row_id: int) -> RuleEditKey:
    return (
        row_id,
        str(edit.edit_type),
        _as_int(edit.start, -1),
        _as_int(edit.end, -1),
        str(edit.source or ""),
        str(edit.replacement or ""),
        _normalize_rule_id(edit.rule_id),
    )


def _edit_from_record(record: Any) -> Edit:
    if isinstance(record, Edit):
        return record
    return Edit(
        source=str(_record_value(record, "source", "")),
        replacement=str(_record_value(record, "replacement", "")),
        edit_type=str(_record_value(record, "edit_type", "unknown")),
        start=_as_int(_record_value(record, "start", -1), -1),
        end=_as_int(_record_value(record, "end", -1), -1),
        status=str(_record_value(record, "status", "proposed")),
        reason=str(_record_value(record, "reason", "")),
        confidence=_as_float(_record_value(record, "confidence", 1.0), 1.0),
        rule_id=_normalize_rule_id(_record_value(record, "rule_id", "")),
    )


def _record_value(record: Any, name: str, default: Any) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _record_row_id(record: Any) -> int:
    return _as_int(_record_value(record, "row_id", 0), 0)


def _rule_id_from_key(key: RuleEditKey) -> str:
    return key[-1]


def _count_for_rule(counter: Counter[RuleEditKey], rule_id: str) -> int:
    return sum(count for key, count in counter.items() if _rule_id_from_key(key) == rule_id)


def _precision_recall_f1(true_positive: int, predicted_count: int, gold_count: int) -> tuple[float, float, float]:
    precision = _safe_rate(true_positive, predicted_count)
    recall = _safe_rate(true_positive, gold_count)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _normalize_rule_id(value: Any) -> str:
    return normalize_rule_id(value)


def _row_rule_ids(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("rule_ids", "rule_id"):
        if key in row:
            values.extend(_parse_rule_ids(row.get(key)))
    return [rule_id for rule_id in values if rule_id != UNKNOWN_RULE_ID]


def _parse_rule_ids(value: Any) -> list[str]:
    items = _parse_jsonish_list(value)
    if items:
        return [_normalize_rule_id(item) for item in items]
    if _is_missing(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [_normalize_rule_id(item) for item in text.split(",")]


def _parse_jsonish_list(value: Any) -> list[Any]:
    if _is_missing(value):
        return []
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
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(parsed, list):
            return parsed
        return [parsed] if not _is_missing(parsed) else []
    return [value]


def _load_rule_groups(path: str | Path) -> dict[str, str]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    rule_groups: dict[str, str] = {}
    for _section, group, entry in iter_coverage_entries(load_rules_coverage(config_path)):
        rule_ids = [*(entry.get("rules", []) or []), *(entry.get("aliases", []) or [])]
        for rule_id in rule_ids:
            normalized = _normalize_rule_id(rule_id)
            if normalized != UNKNOWN_RULE_ID:
                rule_groups[normalized] = str(group)
    return rule_groups


def _format_edit(edit: Edit) -> str:
    return f"{edit.edit_type}@{edit.start}:{edit.end}: {edit.source!r} -> {edit.replacement!r}"


def _as_int(value: Any, default: int) -> int:
    if _is_missing(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False
