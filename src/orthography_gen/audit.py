from __future__ import annotations

from typing import Any

from src.grammar_gen.safety import validate_generated_pair
from src.schema import GeneratedExample


def audit_orthographic_example(example: GeneratedExample) -> list[str]:
    reasons = list(validate_generated_pair(example))
    metadata = example.metadata
    replacement = metadata.get("replacement")
    if example.mode == "positive":
        if not isinstance(replacement, dict):
            reasons.append("missing_orthography_replacement")
        else:
            source = str(replacement.get("source") or "")
            target = str(replacement.get("target") or "")
            if not source or not target:
                reasons.append("invalid_orthography_replacement")
            if source and source not in example.source_text:
                reasons.append("orthography_source_replacement_missing")
            if target and target not in example.target_text:
                reasons.append("orthography_target_replacement_missing")
        if example.token_edit_labels.count("DICT_REPLACE") != 1:
            reasons.append("orthography_positive_requires_one_dict_replace")
    if not isinstance(metadata.get("orthography_site"), dict):
        reasons.append("missing_orthography_site")
    if not metadata.get("lexeme_card_id"):
        reasons.append("missing_lexeme_card_id")
    if not isinstance(metadata.get("orthography_rule_spec"), dict):
        reasons.append("missing_orthography_rule_spec")
    return _dedupe(reasons)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["audit_orthographic_example"]
