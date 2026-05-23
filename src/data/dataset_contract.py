from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
from typing import Any


DATASET_CONTRACT = "candidate_opportunity"
CANDIDATE_OPPORTUNITY_CONTRACT = DATASET_CONTRACT

SYNTHETIC_OPEN_CLEAN = "synthetic_augmented_from_open_clean"
REAL_ERROR_PAIR = "real_error_pair"
CLEAN_IDENTITY_OPEN = "clean_identity_from_open_clean"
HARD_NEGATIVE_OPEN = "hard_negative_from_open_clean"
CORE_SOURCE_TYPES = (SYNTHETIC_OPEN_CLEAN, REAL_ERROR_PAIR, CLEAN_IDENTITY_OPEN, HARD_NEGATIVE_OPEN)

LAYER_ATOMIC_POSITIVE = "atomic_positive"
LAYER_ATOMIC_HARD_NEGATIVE = "atomic_hard_negative"
LAYER_CLEAN_IDENTITY = "clean_identity"
LAYER_STRESS_MULTI_ERROR = "stress_multi_error"
LAYER_REAL_ATOMIC = "real_atomic"
LAYER_REAL_HOLDOUT = "real_holdout"
LAYER_REAL_MINING = "real_mining"

ATOMIC_POSITIVE = LAYER_ATOMIC_POSITIVE
ATOMIC_HARD_NEGATIVE = LAYER_ATOMIC_HARD_NEGATIVE
CLEAN_IDENTITY = LAYER_CLEAN_IDENTITY
STRESS_MULTI_ERROR = LAYER_STRESS_MULTI_ERROR
REAL_ATOMIC = LAYER_REAL_ATOMIC
REAL_HOLDOUT = LAYER_REAL_HOLDOUT
REAL_MINING = LAYER_REAL_MINING

CONTRACT_OPTIONAL_COLUMNS = [
    "dataset_contract",
    "dataset_layer",
    "is_atomic",
    "is_stress",
    "count_toward_rule_quota",
    "loss_weight",
    "gold_edit_count",
    "target_rule_id",
    "candidate_source",
    "candidate_replacement",
    "candidate_start",
    "candidate_end",
    "verification_status",
    "rejection_reason",
]

SOURCE_TYPE_ALIASES = {
    "synthetic_augmented": SYNTHETIC_OPEN_CLEAN,
    SYNTHETIC_OPEN_CLEAN: SYNTHETIC_OPEN_CLEAN,
    "real_error_pair": REAL_ERROR_PAIR,
    "clean_identity": CLEAN_IDENTITY_OPEN,
    CLEAN_IDENTITY_OPEN: CLEAN_IDENTITY_OPEN,
    "hard_negative": HARD_NEGATIVE_OPEN,
    HARD_NEGATIVE_OPEN: HARD_NEGATIVE_OPEN,
}

_POSITIVE_LAYERS = {LAYER_ATOMIC_POSITIVE, LAYER_STRESS_MULTI_ERROR, LAYER_REAL_ATOMIC}


def json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if _is_missing(value):
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def json_list(value: Any) -> list[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        if _is_missing(parsed):
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, Mapping):
            return [dict(parsed)]
        return [parsed]
    return [value]


def row_rule_ids(row: Mapping[str, Any] | Any) -> list[str]:
    result: list[str] = []
    for item in json_list(_row_get(row, "rule_ids", [])):
        rule_id = _clean_text(item)
        if rule_id and rule_id not in result:
            result.append(rule_id)
    if result:
        return result
    fallback = _clean_text(_row_get(row, "rule_id", ""))
    return [fallback] if fallback else []


def row_edits(row: Mapping[str, Any] | Any) -> list[dict[str, Any]]:
    raw = _row_get(row, "edits", "")
    if _is_blank(raw):
        raw = _row_get(row, "edit_operations", "")
    edits: list[dict[str, Any]] = []
    for edit in json_list(raw):
        if isinstance(edit, Mapping):
            edits.append(dict(edit))
    return edits


def row_gold_edit_count(row: Mapping[str, Any] | Any) -> int:
    explicit = _int_or_none(_row_get(row, "gold_edit_count", None))
    if explicit is not None:
        return max(0, explicit)
    return len(row_edits(row))


def row_dataset_layer(row: Mapping[str, Any] | Any) -> str:
    explicit = _clean_text(_row_get(row, "dataset_layer", ""))
    if explicit:
        return explicit
    source_type = _canonical_source_type(_row_get(row, "source_type", ""))
    if source_type == SYNTHETIC_OPEN_CLEAN:
        return LAYER_STRESS_MULTI_ERROR if _row_stress_flag(row) else LAYER_ATOMIC_POSITIVE
    if source_type == REAL_ERROR_PAIR:
        return LAYER_REAL_ATOMIC
    if source_type == CLEAN_IDENTITY_OPEN:
        return LAYER_CLEAN_IDENTITY
    if source_type == HARD_NEGATIVE_OPEN:
        return LAYER_ATOMIC_HARD_NEGATIVE
    if _bool(_row_get(row, "is_hard_negative", False)):
        return LAYER_ATOMIC_HARD_NEGATIVE
    if _clean_text(_row_get(row, "source", "")) and _row_get(row, "source", "") == _row_get(row, "target", ""):
        return LAYER_CLEAN_IDENTITY
    return ""


def is_positive_row(row: Mapping[str, Any] | Any) -> bool:
    return row_dataset_layer(row) in _POSITIVE_LAYERS


def is_atomic_positive_row(row: Mapping[str, Any] | Any) -> bool:
    return row_dataset_layer(row) == LAYER_ATOMIC_POSITIVE and row_gold_edit_count(row) == 1


def is_clean_identity_row(row: Mapping[str, Any] | Any) -> bool:
    return row_dataset_layer(row) == LAYER_CLEAN_IDENTITY


def is_hard_negative_row(row: Mapping[str, Any] | Any) -> bool:
    return row_dataset_layer(row) == LAYER_ATOMIC_HARD_NEGATIVE


def stable_dataset_hash(rows_or_frame: Any) -> str:
    digest = hashlib.sha256()
    for row in _iter_rows(rows_or_frame):
        payload = {
            "source": _clean_text(_row_get(row, "source", "")),
            "target": _clean_text(_row_get(row, "target", "")),
            "rule_ids": sorted(set(row_rule_ids(row))),
        }
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def ensure_contract_columns(frame_or_rows: Any) -> Any:
    if _is_frame_like(frame_or_rows):
        frame = frame_or_rows.copy()
        rows = _records_from_frame(frame)
        for column in CONTRACT_OPTIONAL_COLUMNS:
            values: list[Any] = []
            for row in rows:
                existing = row.get(column)
                value = existing if not _is_blank(existing) else _default_contract_value(row, column)
                row[column] = value
                values.append(value)
            frame[column] = values
        return frame

    if isinstance(frame_or_rows, Mapping):
        row = dict(frame_or_rows)
        _ensure_row_contract(row)
        return row

    rows = [dict(row) for row in frame_or_rows]
    for row in rows:
        _ensure_row_contract(row)
    return rows


def _ensure_row_contract(row: dict[str, Any]) -> None:
    for column in CONTRACT_OPTIONAL_COLUMNS:
        if _is_blank(row.get(column)):
            row[column] = _default_contract_value(row, column)


def _default_contract_value(row: Mapping[str, Any], column: str) -> Any:
    layer = row_dataset_layer(row)
    edit = _first_edit(row)
    if column == "dataset_contract":
        return DATASET_CONTRACT
    if column == "dataset_layer":
        return layer
    if column == "is_atomic":
        return _is_atomic_layer_compatible(row, layer)
    if column == "is_stress":
        return layer == LAYER_STRESS_MULTI_ERROR or _row_stress_flag(row)
    if column == "count_toward_rule_quota":
        return _canonical_source_type(_row_get(row, "source_type", "")) == SYNTHETIC_OPEN_CLEAN and is_atomic_positive_row(row)
    if column == "loss_weight":
        return 1.0
    if column == "gold_edit_count":
        return row_gold_edit_count(row)
    if column == "target_rule_id":
        rule_ids = row_rule_ids(row)
        return rule_ids[0] if rule_ids else _clean_text(edit.get("rule_id", ""))
    if column == "candidate_source":
        return _clean_text(edit.get("source", ""))
    if column == "candidate_replacement":
        return _clean_text(edit.get("replacement", ""))
    if column == "candidate_start":
        start = _int_or_none(edit.get("start"))
        return start if start is not None else -1
    if column == "candidate_end":
        end = _int_or_none(edit.get("end"))
        return end if end is not None else -1
    if column in {"verification_status", "rejection_reason"}:
        return ""
    return ""


def _is_atomic_layer_compatible(row: Mapping[str, Any], layer: str) -> bool:
    edit_count = row_gold_edit_count(row)
    if layer in {LAYER_ATOMIC_POSITIVE, LAYER_REAL_ATOMIC}:
        return edit_count == 1
    if layer == LAYER_ATOMIC_HARD_NEGATIVE:
        return edit_count == 0
    return False


def _first_edit(row: Mapping[str, Any]) -> dict[str, Any]:
    edits = row_edits(row)
    return edits[0] if edits else {}


def _iter_rows(rows_or_frame: Any) -> Iterable[Mapping[str, Any]]:
    if _is_frame_like(rows_or_frame):
        yield from _records_from_frame(rows_or_frame)
        return
    if isinstance(rows_or_frame, Mapping):
        yield rows_or_frame
        return
    for row in rows_or_frame:
        if isinstance(row, Mapping):
            yield row
        elif hasattr(row, "to_dict"):
            yield row.to_dict()


def _records_from_frame(frame: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in frame.to_dict("records")]
    except TypeError:
        return [dict(row) for row in frame.to_dict()]


def _is_frame_like(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "to_dict") and hasattr(value, "copy")


def _row_get(row: Mapping[str, Any] | Any, key: str, default: Any = "") -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    if hasattr(row, "get"):
        try:
            return row.get(key, default)
        except TypeError:
            return row.get(key) if key in row else default
    return default


def _canonical_source_type(value: Any) -> str:
    text = _clean_text(value)
    return SOURCE_TYPE_ALIASES.get(text, text)


def _row_stress_flag(row: Mapping[str, Any] | Any) -> bool:
    if _bool(_row_get(row, "is_stress", False)):
        return True
    return _bool(json_dict(_row_get(row, "metadata", {})).get("is_stress", False))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "off", "none", "null", "nan"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return bool(text)


def _int_or_none(value: Any) -> int | None:
    if _is_blank(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _clean_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _is_blank(value: Any) -> bool:
    if _is_missing(value):
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(value != value)
    except Exception:
        return False
    return False
