from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.rules.registry import rule_by_id


RULE_DEFAULT_THRESHOLDS = {"context_pair": 0.98}
FAMILY_DEFAULT_THRESHOLDS = {"tsya": 0.98}
CANDIDATE_EDIT_TYPE_DEFAULT_THRESHOLDS = {
    "split_join": 0.88,
    "hyphen": 0.88,
    "case": 0.85,
    "spelling": 0.85,
}
PUNCTUATION_DEFAULT_THRESHOLD = 0.82

RULE_FAMILY_ALIASES = {
    "dictionary_fuzzy": ("dictionary",),
    "frequent_dictionary_model_required": ("dictionary",),
    "introductory_comma": ("introductory_word",),
    "subject_predicate_dash": ("dash_subject_predicate",),
}
GROUP_FAMILY_ALIASES = {
    "dictionary_model_required": ("dictionary",),
    "introductory": ("introductory_word",),
}
PREFIX_FAMILIES = ("tsya", "ne_adjective", "ne_participle")
PUNCTUATION_LABEL_FAMILIES = {
    "COMMA": ("comma",),
    "DOT": ("final_punctuation", "final"),
    "QUESTION": ("final_punctuation", "final"),
    "EXCLAMATION": ("final_punctuation", "final"),
    "ELLIPSIS": ("final_punctuation", "final"),
    "COLON": ("colon",),
    "DASH": ("dash",),
    "SEMICOLON": ("semicolon",),
    "QUOTE_OPEN": ("quote",),
    "QUOTE_CLOSE": ("quote",),
    "BRACKET_OPEN": ("bracket",),
    "BRACKET_CLOSE": ("bracket",),
}
PUNCTUATION_EDIT_TYPES = {"punctuation_insert", "punctuation_delete", "punctuation_replace"}


def threshold_for_candidate(
    candidate: Any,
    thresholds: Mapping[str, Any],
    *,
    rule_id: str | None = None,
) -> float:
    """Resolve the score threshold for a bounded candidate."""

    resolved_rule_id = _clean(rule_id) or _clean(getattr(candidate, "rule_id", ""))
    configured = _configured_rule_threshold(resolved_rule_id, thresholds)
    if configured is not None:
        return configured
    if resolved_rule_id in RULE_DEFAULT_THRESHOLDS:
        return RULE_DEFAULT_THRESHOLDS[resolved_rule_id]

    for family in _rule_families(resolved_rule_id):
        configured = _configured_family_threshold(family, thresholds)
        if configured is not None:
            return configured
        if family in FAMILY_DEFAULT_THRESHOLDS:
            return FAMILY_DEFAULT_THRESHOLDS[family]

    edit_type = _clean(getattr(candidate, "edit_type", ""))
    configured = _configured_edit_type_threshold(edit_type, thresholds)
    if configured is not None:
        return configured

    configured = _configured_default_threshold(thresholds)
    if configured is not None:
        return configured
    return CANDIDATE_EDIT_TYPE_DEFAULT_THRESHOLDS.get(edit_type, 0.85)


def threshold_for_punctuation_prediction(prediction: Any, thresholds: Mapping[str, Any]) -> float:
    """Resolve the score threshold for a model punctuation prediction."""

    rule_id = _clean(getattr(prediction, "rule_id", ""))
    configured = _configured_rule_threshold(rule_id, thresholds)
    if configured is not None:
        return configured
    if rule_id in RULE_DEFAULT_THRESHOLDS:
        return RULE_DEFAULT_THRESHOLDS[rule_id]

    for family in _rule_families(rule_id):
        configured = _configured_family_threshold(family, thresholds)
        if configured is not None:
            return configured
        if family in FAMILY_DEFAULT_THRESHOLDS:
            return FAMILY_DEFAULT_THRESHOLDS[family]

    for family in _punctuation_families(prediction):
        configured = _configured_family_threshold(family, thresholds)
        if configured is not None:
            return configured

    configured = _threshold_value(thresholds, "punctuation_threshold")
    if configured is not None:
        return configured

    configured = _configured_default_threshold(thresholds)
    if configured is not None:
        return configured
    return PUNCTUATION_DEFAULT_THRESHOLD


def _configured_rule_threshold(rule_id: str, thresholds: Mapping[str, Any]) -> float | None:
    if not rule_id:
        return None
    for key in (f"{rule_id}_threshold", rule_id):
        value = _threshold_value(thresholds, key)
        if value is not None:
            return value
    return None


def _configured_family_threshold(family: str, thresholds: Mapping[str, Any]) -> float | None:
    if not family:
        return None
    return _threshold_value(thresholds, f"{family}_threshold")


def _configured_edit_type_threshold(edit_type: str, thresholds: Mapping[str, Any]) -> float | None:
    if not edit_type:
        return None
    value = _threshold_value(thresholds, f"{edit_type}_threshold")
    if value is not None:
        return value
    if edit_type in PUNCTUATION_EDIT_TYPES:
        return _threshold_value(thresholds, "punctuation_threshold")
    return None


def _configured_default_threshold(thresholds: Mapping[str, Any]) -> float | None:
    for key in ("default_threshold", "default_candidate_threshold"):
        value = _threshold_value(thresholds, key)
        if value is not None:
            return value
    return None


def _threshold_value(thresholds: Mapping[str, Any], key: str) -> float | None:
    if key not in thresholds:
        return None
    return float(thresholds[key])


def _rule_families(rule_id: str) -> tuple[str, ...]:
    if not rule_id:
        return ()

    families: list[str] = []
    for prefix in PREFIX_FAMILIES:
        if rule_id == prefix or rule_id.startswith(f"{prefix}_"):
            families.append(prefix)

    families.extend(RULE_FAMILY_ALIASES.get(rule_id, ()))
    rule = rule_by_id(rule_id)
    group = _clean(getattr(getattr(rule, "spec", None), "group", ""))
    if group:
        families.extend(GROUP_FAMILY_ALIASES.get(group, ()))
        families.append(group)
    return _deduplicate(families)


def _punctuation_families(prediction: Any) -> tuple[str, ...]:
    families: list[str] = []
    action = _clean(getattr(prediction, "action", "")).upper()
    label = _clean(getattr(prediction, "label", "")).upper()
    if action == "DELETE":
        families.append("punctuation_delete")
    families.extend(PUNCTUATION_LABEL_FAMILIES.get(label, ()))
    return _deduplicate(families)


def _deduplicate(items: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)


def _clean(value: Any) -> str:
    return str(value or "").strip()
