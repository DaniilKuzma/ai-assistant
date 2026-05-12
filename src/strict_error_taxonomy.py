"""Strict orthography and punctuation error taxonomy."""

from __future__ import annotations

from collections.abc import Iterable


ERROR_SCOPE_CLEAN = "clean"
ERROR_SCOPE_ORTHOGRAPHY = "orthography"
ERROR_SCOPE_PUNCTUATION = "punctuation"
ERROR_SCOPE_QUARANTINE = "quarantine"
ERROR_SCOPE_FORBIDDEN = "forbidden"


STRICT_ORTHOGRAPHY_ERROR_TYPES = frozenset(
    {
        "spelling_tsya",
        "spelling_suffix_pronunciation",
        "spelling_prefix",
        "spelling_n_nn",
        "spelling_soft_hard_sign",
        "spelling_ne_ni",
        "spelling_vowel_after_sibilant",
        "spelling_i_y_after_ts",
        "spelling_hyphen",
        "spelling_dictionary_word",
        "spelling_compound_joining",
        "spelling_capitalization",
        "spelling_abbreviation_case",
        "spelling_borrowed_word",
        "real_spelling",
    }
)

STRICT_PUNCTUATION_ERROR_TYPES = frozenset(
    {
        "punct_remove_comma_before_clause",
        "punct_remove_intro_comma",
        "punct_remove_homogeneous_comma",
        "punct_remove_final",
        "punct_extra_comma_before_single_i",
        "punct_dash_missing",
        "punct_dash_extra",
        "punct_bsp_colon_missing",
        "punct_bsp_dash_missing",
        "punct_bracket_pair_missing",
        "punct_remove_quotes",
        "punct_quote_punct_order",
        "punct_direct_speech_inner_punct",
        "punct_direct_speech_dash_missing",
        "real_punctuation",
    }
)

QUARANTINE_ERROR_TYPES = frozenset(
    {
        "spelling_replace",
        "spelling_double",
        "punct_remove_comma",
        "punct_remove_internal_comma",
        "punct_remove_all_commas",
        "punct_remove_period",
        "punct_remove_question",
        "punct_remove_exclamation",
        "punct_wrong_period_to_comma",
        "punct_wrong_comma_to_period",
        "punct_wrong_colon_to_semicolon",
        "punct_wrong_semicolon_to_colon",
        "punct_wrong_question_to_period",
        "punct_wrong_exclamation_to_period",
        "punct_extra_comma",
        "punct_bracket_extra",
        "punct_bracket_missing_close",
        "punct_ellipsis_extra",
        "punct_ellipsis_missing",
        "punct_list_missing_colon",
        "punct_list_item_punctuation",
        "punct_quote_style",
    }
)

FORBIDDEN_ERROR_TYPES = frozenset(
    {
        "spelling_delete",
        "spelling_swap",
        "spelling_extra",
        "real_other",
        "unknown",
        "keyboard",
        "spacing_only",
        "heavy_rewrite",
    }
)


def split_error_types(value: object) -> list[str]:
    """Return normalized non-empty error labels from a pipe-separated value."""
    if isinstance(value, str):
        raw_items = value.split("|")
    elif isinstance(value, Iterable):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]
    return [item.strip() for item in raw_items if item and item.strip()]


def classify_error_type(error_type: str) -> str:
    """Classify one error label for the strict scope."""
    label = str(error_type or "").strip()
    if label == ERROR_SCOPE_CLEAN:
        return ERROR_SCOPE_CLEAN
    if label in STRICT_ORTHOGRAPHY_ERROR_TYPES:
        return ERROR_SCOPE_ORTHOGRAPHY
    if label in STRICT_PUNCTUATION_ERROR_TYPES:
        return ERROR_SCOPE_PUNCTUATION
    if label in QUARANTINE_ERROR_TYPES:
        return ERROR_SCOPE_QUARANTINE
    return ERROR_SCOPE_FORBIDDEN


def error_types_are_strict(value: object) -> bool:
    """Return True when every label belongs to clean/orthography/punctuation."""
    labels = split_error_types(value)
    if not labels:
        return False
    return all(
        classify_error_type(label)
        in {ERROR_SCOPE_CLEAN, ERROR_SCOPE_ORTHOGRAPHY, ERROR_SCOPE_PUNCTUATION}
        for label in labels
    )


def forbidden_error_types(value: object) -> list[str]:
    """Return labels that are not allowed in the strict scope."""
    return [
        label
        for label in split_error_types(value)
        if classify_error_type(label) in {ERROR_SCOPE_FORBIDDEN, ERROR_SCOPE_QUARANTINE}
    ]
