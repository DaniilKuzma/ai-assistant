from __future__ import annotations

import math
from typing import Any


UNKNOWN_RULE_ID = "unknown"

RULE_ID_ALIASES = {
    "frequent_errors": "frequent_error_exact",
}


def normalize_rule_id(value: Any) -> str:
    if _is_missing(value):
        return UNKNOWN_RULE_ID
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return UNKNOWN_RULE_ID
    return RULE_ID_ALIASES.get(text, text)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False
