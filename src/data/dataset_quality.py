from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any, Iterable

import pandas as pd

from src.data.clean_sentence_pool import clean_sentence_rejection_reasons
from src.data.dataset_verifiers import PUNCTUATION_RULE_FAMILIES, numeric_punctuation_mismatch
from src.data.training_quality_audit import (
    artificial_marker_counts,
    clean_hard_balance_audit_frame,
    clean_hard_balance_counts,
    known_quality_bug_counts,
    quote_bracket_balance_audit_frame,
    quote_bracket_balance_counts,
    target_has_quote_bracket_balance_bug,
    text_has_quote_bracket_balance_bug,
)


SYNTAX_PUNCTUATION_RULE_IDS = frozenset(PUNCTUATION_RULE_FAMILIES) - frozenset({"final_punctuation_default"})
POSITIVE_SOURCE_TYPES = frozenset({"synthetic_augmented_from_open_clean", "real_error_pair"})
CLEAN_HARD_SOURCE_TYPES = frozenset({"clean_identity_from_open_clean", "hard_negative_from_open_clean"})
STRICT_CLEAN_OR_HARD_POOL_CONFIG = {
    "reject_mixed_script_tokens": True,
    "reject_latin_confusable_inside_cyrillic_word": True,
}


def positive_target_quality_pass(text: str) -> bool:
    return not positive_target_quality_reasons(text)


def positive_target_quality_reasons(text: str) -> list[str]:
    reasons = clean_sentence_rejection_reasons(text, {"domain": "open_clean", "style": "neutral"})
    if target_has_quote_bracket_balance_bug(text):
        reasons.append("unbalanced_quote_or_bracket")
    if _contains_known_bad_phrase(text):
        reasons.append("known_bad_phrase")
    if re.search(r"\S—\s|\s—\S", text):
        reasons.append("malformed_dash_spacing")
    return list(dict.fromkeys(reasons))


def clean_or_hard_quality_pass(text: str) -> bool:
    return not clean_or_hard_quality_reasons(text)


def clean_or_hard_quality_reasons(text: str) -> list[str]:
    reasons = clean_sentence_rejection_reasons(
        text,
        {"domain": "open_clean", "style": "neutral"},
        pool_config=STRICT_CLEAN_OR_HARD_POOL_CONFIG,
    )
    if text_has_quote_bracket_balance_bug(text):
        reasons.append("unbalanced_quote_or_bracket")
    return list(dict.fromkeys(reasons))


def row_quality_pass(row: dict[str, Any]) -> bool:
    source_type = str(row.get("source_type", ""))
    source = str(row.get("source", ""))
    target = str(row.get("target", ""))
    if source_type in POSITIVE_SOURCE_TYPES:
        return source != target and positive_target_quality_pass(target) and not target_has_quote_bracket_balance_bug(target)
    if source_type in CLEAN_HARD_SOURCE_TYPES:
        return clean_or_hard_quality_pass(source) and clean_or_hard_quality_pass(target)
    return source and target


def numeric_punctuation_mismatch_count(frame: pd.DataFrame) -> int:
    return int(len(numeric_punctuation_mismatch_audit_frame(frame)))


def numeric_punctuation_mismatch_audit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["row_index", "source_type", "rule_id", "source", "target", "reason"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        source_type = str(row.get("source_type", ""))
        if source_type not in POSITIVE_SOURCE_TYPES:
            continue
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        if not numeric_punctuation_mismatch(source, target):
            continue
        for rule_id in _row_rule_ids(row):
            if rule_id in SYNTAX_PUNCTUATION_RULE_IDS:
                rows.append(
                    {
                        "row_index": int(index) if isinstance(index, int) else str(index),
                        "source_type": source_type,
                        "rule_id": rule_id,
                        "source": source,
                        "target": target,
                        "reason": "numeric_punctuation_mismatch",
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def known_quality_bug_summary(frame: pd.DataFrame) -> dict[str, int]:
    result = dict(known_quality_bug_counts(frame))
    result.update({f"artificial_marker_{key}": int(value) for key, value in artificial_marker_counts(frame).items()})
    return result


def quote_bracket_bug_summary(frame: pd.DataFrame) -> dict[str, int]:
    return dict(quote_bracket_balance_counts(frame))


def clean_hard_bug_summary(frame: pd.DataFrame) -> dict[str, int]:
    return dict(clean_hard_balance_counts(frame))


def normalized_pair_hash(source: str, target: str) -> str:
    key = normalize_pair(source) + "\n" + normalize_pair(target)
    import hashlib

    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def normalize_pair(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def duplicate_rate(values: Iterable[str]) -> float:
    values = list(values)
    if not values:
        return 0.0
    counts = Counter(values)
    duplicates = sum(max(0, count - 1) for count in counts.values())
    return float(duplicates / len(values))


def balance_audit_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "numeric_punctuation_mismatch_audit": numeric_punctuation_mismatch_audit_frame(frame),
        "quote_bracket_balance_audit": quote_bracket_balance_audit_frame(frame),
        "clean_hard_balance_audit": clean_hard_balance_audit_frame(frame),
    }


def _row_rule_ids(row: pd.Series | dict[str, Any]) -> list[str]:
    raw = row.get("rule_ids", "[]") if isinstance(row, dict) else row.get("rule_ids", "[]")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        parsed = raw
    else:
        parsed = []
    result = [str(rule_id) for rule_id in parsed if str(rule_id)]
    if not result:
        rule_id = str(row.get("rule_id", "") if isinstance(row, dict) else row.get("rule_id", ""))
        if rule_id:
            result.append(rule_id)
    return result


def _contains_known_bad_phrase(text: str) -> bool:
    lower = text.lower()
    forbidden = (
        "несогласен с выводом",
        "ненужно комиссии",
        "по-старому плану",
        "по-новому вариант",
        "по-новому договору",
        "по-старому адресу",
        "сохранил территория",
        "записал житель",
        "читал свежая сводка",
        "в закрытая заявка",
        "по-вашему",
        "по-русски",
        "по-английски",
    )
    return any(phrase in lower for phrase in forbidden)
