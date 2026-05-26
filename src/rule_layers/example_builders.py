from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from typing import Any

from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.common import gap_labels_from_text
from src.schema import GeneratedExample, WordToken
from src.schema.labels import gap_label_to_id, token_label_to_id
from src.rule_layers.base import LayerDirectCase, LayerOperation


SPAN_FIRST_LABELS = frozenset(
    {
        "MERGE_TAK_ZHE_TO_TAKZHE",
        "MERGE_TO_ZHE_TO_TOZHE",
        "MERGE_ZA_TO_TO_ZATO",
        "HYPHENATE_PARTICLE_TO",
        "HYPHENATE_PARTICLE_LIBO",
        "HYPHENATE_PARTICLE_NIBUD",
        "HYPHENATE_KOE",
        "HYPHENATE_PO_ADVERB",
    }
)


def build_token_span_replacement_example(
    case: LayerDirectCase,
    realizer: Realizer,
    *,
    layer: str,
) -> GeneratedExample:
    if not case.token_operations and not case.direct_token_labels:
        raise ValueError("Token span example requires token operations or direct token labels.")
    return build_generated_example_from_case(case, realizer, layer=layer)


def build_gap_operations_example(
    case: LayerDirectCase,
    realizer: Realizer,
    *,
    layer: str,
) -> GeneratedExample:
    if not case.gap_operations and not case.direct_gap_labels:
        raise ValueError("Gap example requires gap operations or direct gap labels.")
    return build_generated_example_from_case(case, realizer, layer=layer)


def build_generated_example_from_case(
    case: LayerDirectCase,
    realizer: Realizer,
    *,
    layer: str,
) -> GeneratedExample:
    tokens = realizer.tokenize_words_with_offsets(case.source_text)
    if not tokens:
        raise ValueError("Layer examples require at least one source token.")

    token_labels = _initial_token_labels(case, tokens)
    gap_labels = _initial_gap_labels(case, tokens)
    rule_ids = ["none"] * len(tokens)

    operation_names: list[str] = []
    source_patterns: list[str] = []
    target_patterns: list[str] = []

    for operation in case.token_operations:
        _apply_token_operation(operation, case, realizer, tokens, token_labels, rule_ids)
        operation_names.append(operation.kind)
        source_patterns.append(operation.source_pattern)
        target_patterns.append(operation.target_pattern)

    for operation in case.gap_operations:
        _apply_gap_operation(operation, case, realizer, tokens, gap_labels, rule_ids)
        operation_names.append(operation.kind)
        source_patterns.append(operation.source_pattern)
        target_patterns.append(operation.target_pattern)

    _apply_direct_rule_ids(case, token_labels, gap_labels, rule_ids)
    if case.rule_id not in rule_ids:
        rule_ids[0] = case.rule_id

    _validate_lengths(tokens, token_labels, gap_labels, rule_ids)

    metadata = _metadata(
        case,
        layer=layer,
        operation_names=operation_names,
        source_patterns=source_patterns,
        target_patterns=target_patterns,
    )
    return GeneratedExample(
        source_text=case.source_text,
        target_text=case.target_text,
        source_tokens=list(tokens),
        token_edit_labels=list(token_labels),
        gap_labels=list(gap_labels),
        rule_ids=list(rule_ids),
        primary_rule_id=case.rule_id,
        mode=case.mode,
        explanation_ids=[str(metadata.get("explanation_id") or case.rule_id)],
        metadata=metadata,
    )


def _initial_token_labels(case: LayerDirectCase, tokens: Sequence[WordToken]) -> list[str]:
    if case.direct_token_labels:
        labels = list(case.direct_token_labels)
        _require_length("direct_token_labels", labels, tokens)
    else:
        labels = ["KEEP"] * len(tokens)
    for label in labels:
        token_label_to_id(label)
    return labels


def _initial_gap_labels(case: LayerDirectCase, tokens: Sequence[WordToken]) -> list[str]:
    if case.direct_gap_labels:
        labels = list(case.direct_gap_labels)
        _require_length("direct_gap_labels", labels, tokens)
    else:
        labels = gap_labels_from_text(case.source_text, tokens)
    for label in labels:
        gap_label_to_id(label)
    return labels


def _apply_token_operation(
    operation: LayerOperation,
    case: LayerDirectCase,
    realizer: Realizer,
    tokens: Sequence[WordToken],
    labels: MutableSequence[str],
    rule_ids: MutableSequence[str],
) -> None:
    if operation.kind not in {"token_span", "token"}:
        raise ValueError(f"Unsupported token operation kind: {operation.kind!r}.")
    token_label_to_id(operation.label)
    start, end = _operation_span(operation, case.source_text, realizer, tokens)
    if start >= end:
        raise ValueError(f"Invalid token operation span: {start}:{end}.")

    labels[start] = operation.label
    rule_ids[start] = case.rule_id
    if operation.label in SPAN_FIRST_LABELS:
        for index in range(start + 1, end):
            labels[index] = "SKIP_MERGED"
            rule_ids[index] = case.rule_id


def _apply_gap_operation(
    operation: LayerOperation,
    case: LayerDirectCase,
    realizer: Realizer,
    tokens: Sequence[WordToken],
    labels: MutableSequence[str],
    rule_ids: MutableSequence[str],
) -> None:
    if operation.kind != "gap":
        raise ValueError(f"Unsupported gap operation kind: {operation.kind!r}.")
    gap_label_to_id(operation.label)
    index, _end = _operation_span(operation, case.source_text, realizer, tokens)
    if operation.label == "DELETE_PUNCTUATION" and not _punctuation_after(case.source_text, tokens[index]):
        raise ValueError(f"DELETE_PUNCTUATION source_pattern has no source punctuation: {operation.source_pattern!r}.")
    labels[index] = operation.label
    rule_ids[index] = case.rule_id


def _operation_span(
    operation: LayerOperation,
    source_text: str,
    realizer: Realizer,
    tokens: Sequence[WordToken],
) -> tuple[int, int]:
    if operation.token_index is not None:
        index = operation.token_index
        if index < 0 or index >= len(tokens):
            raise ValueError(f"token_index is outside source tokens: {index}.")
        return index, index + 1

    if operation.token_start is not None or operation.token_end is not None:
        if operation.token_start is None or operation.token_end is None:
            raise ValueError("token_start and token_end must be provided together.")
        start, end = operation.token_start, operation.token_end
        if start < 0 or end > len(tokens) or start >= end:
            raise ValueError(f"Invalid token span: {start}:{end}.")
        return start, end

    if not operation.source_pattern:
        raise ValueError("Layer operation requires source_pattern or token indexes.")
    pattern_tokens = realizer.tokenize_words_with_offsets(operation.source_pattern)
    pattern = tuple(token.text.casefold() for token in pattern_tokens)
    if not pattern:
        raise ValueError(f"source_pattern does not contain source tokens: {operation.source_pattern!r}.")
    source = tuple(token.text.casefold() for token in tokens)
    matches: list[tuple[int, int]] = []
    for start in range(0, len(source) - len(pattern) + 1):
        if source[start : start + len(pattern)] == pattern:
            matches.append((start, start + len(pattern)))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Layer operation ambiguous source_pattern without explicit token indexes: "
            f"{operation.source_pattern!r}."
        )
    raise ValueError(f"Layer operation source_pattern not found in source tokens: {operation.source_pattern!r}.")


def _apply_direct_rule_ids(
    case: LayerDirectCase,
    token_labels: Sequence[str],
    gap_labels: Sequence[str],
    rule_ids: MutableSequence[str],
) -> None:
    if case.direct_token_labels:
        for index, label in enumerate(token_labels):
            if label not in {"KEEP", "SKIP_MERGED"}:
                rule_ids[index] = case.rule_id
    if case.direct_gap_labels:
        for index, label in enumerate(gap_labels):
            if label != "NONE":
                rule_ids[index] = case.rule_id


def _metadata(
    case: LayerDirectCase,
    *,
    layer: str,
    operation_names: Sequence[str],
    source_patterns: Sequence[str],
    target_patterns: Sequence[str],
) -> dict[str, Any]:
    metadata = dict(case.metadata)
    operation = str(metadata.get("operation") or _operation_name(operation_names))
    metadata.update(
        {
            "layer": layer,
            "rule_id": case.rule_id,
            "sub_rule_id": case.sub_rule_id,
            "operation": operation,
            "expected_token_edit_count": case.expected_token_edit_count,
            "expected_gap_edit_count": case.expected_gap_edit_count,
            "expected_edit_count": case.expected_token_edit_count + case.expected_gap_edit_count,
            "source_pattern": _first_non_empty(source_patterns),
            "target_pattern": _first_non_empty(target_patterns),
            "production": False,
            "uses_construction_bank": True,
            "construction_id": str(metadata.get("construction_id") or case.sub_rule_id or case.rule_id),
            "construction_family": str(metadata.get("construction_family") or layer),
            "uses_safety_clauses": False,
            "safety_clauses": [],
        }
    )
    return metadata


def _operation_name(names: Sequence[str]) -> str:
    unique = tuple(dict.fromkeys(name for name in names if name))
    if not unique:
        return "direct_labels"
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _first_non_empty(values: Sequence[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def _require_length(name: str, labels: Sequence[str], tokens: Sequence[WordToken]) -> None:
    if len(labels) != len(tokens):
        raise ValueError(f"{name} length must match source tokens: {len(labels)} != {len(tokens)}.")


def _validate_lengths(
    tokens: Sequence[WordToken],
    token_labels: Sequence[str],
    gap_labels: Sequence[str],
    rule_ids: Sequence[str],
) -> None:
    _require_length("token_edit_labels", token_labels, tokens)
    _require_length("gap_labels", gap_labels, tokens)
    _require_length("rule_ids", rule_ids, tokens)


def _punctuation_after(text: str, token: WordToken) -> str:
    index = token.end
    while index < len(text) and text[index].isspace():
        index += 1
    if text.startswith("...", index):
        return "..."
    if index < len(text):
        return text[index]
    return ""


__all__ = [
    "build_gap_operations_example",
    "build_generated_example_from_case",
    "build_token_span_replacement_example",
]
