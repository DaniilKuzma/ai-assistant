from __future__ import annotations

from src.candidates.frequent_errors import CONTEXT_DEPENDENT_WHITELIST


SPELLING_TYPES = {"spelling_replace", "split_word", "join_words", "hyphen_change", "case_change"}
PUNCTUATION_TYPES = {
    "punctuation_insert",
    "punctuation_delete",
    "punctuation_replace",
    "final_punctuation",
}
ALLOWED_EDIT_TYPES = SPELLING_TYPES | PUNCTUATION_TYPES


def is_allowed_edit_type(edit_type: str) -> bool:
    return edit_type in ALLOWED_EDIT_TYPES


def is_context_dependent_pair(source: str, replacement: str) -> bool:
    return CONTEXT_DEPENDENT_WHITELIST.get(source.lower()) == replacement.lower()


def coarse_error_type(edit_type: str) -> str:
    if edit_type in {"punctuation_insert", "punctuation_delete", "punctuation_replace"}:
        return "punctuation"
    if edit_type == "final_punctuation":
        return "final_punctuation"
    if edit_type in {"split_word", "join_words"}:
        return "split_join"
    if edit_type == "hyphen_change":
        return "hyphen"
    if edit_type == "case_change":
        return "case"
    if edit_type == "spelling_replace":
        return "spelling"
    return "unknown"
