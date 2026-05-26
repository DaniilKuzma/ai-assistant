from __future__ import annotations

import pytest

from src.grammar_gen import Lexicon, MorphologyEngine, Realizer
from src.rule_layers.base import LayerDirectCase, LayerOperation
from src.rule_layers.example_builders import (
    build_gap_operations_example,
    build_generated_example_from_case,
    build_token_span_replacement_example,
)


def test_token_span_replacement_labels_first_span_token() -> None:
    case = LayerDirectCase(
        rule_id="ne_verb",
        family="compound_spelling",
        sub_rule_id="na_schet_merge",
        mode="positive",
        source_text="\u041e\u043d \u043f\u0438\u0441\u0430\u043b \u043d\u0430 \u0441\u0447\u0435\u0442 \u043e\u0448\u0438\u0431\u043a\u0438.",
        target_text="\u041e\u043d \u043f\u0438\u0441\u0430\u043b \u043d\u0430\u0441\u0447\u0451\u0442 \u043e\u0448\u0438\u0431\u043a\u0438.",
        token_operations=(
            LayerOperation(
                kind="token_span",
                label="SPAN_REPLACE_BY_LEXICON",
                source_pattern="\u043d\u0430 \u0441\u0447\u0435\u0442",
                target_pattern="\u043d\u0430\u0441\u0447\u0451\u0442",
            ),
        ),
        expected_token_edit_count=1,
        expected_gap_edit_count=0,
    )

    example = build_token_span_replacement_example(case, _realizer(), layer="compound_spelling")

    assert [token.text for token in example.source_tokens] == [
        "\u041e\u043d",
        "\u043f\u0438\u0441\u0430\u043b",
        "\u043d\u0430",
        "\u0441\u0447\u0435\u0442",
        "\u043e\u0448\u0438\u0431\u043a\u0438",
    ]
    assert example.token_edit_labels == ["KEEP", "KEEP", "SPAN_REPLACE_BY_LEXICON", "KEEP", "KEEP"]
    assert example.rule_ids == ["none", "none", "ne_verb", "none", "none"]
    assert example.metadata["layer"] == "compound_spelling"
    assert example.metadata["operation"] == "token_span"
    assert example.metadata["source_pattern"] == "\u043d\u0430 \u0441\u0447\u0435\u0442"
    assert example.metadata["target_pattern"] == "\u043d\u0430\u0441\u0447\u0451\u0442"
    assert example.metadata["expected_token_edit_count"] == 1
    assert example.metadata["expected_gap_edit_count"] == 0


def test_gap_insert_sets_gap_label_after_matched_source_token() -> None:
    case = LayerDirectCase(
        rule_id="comma_subordinate",
        family="syntax_punctuation",
        sub_rule_id="comma_before_chto",
        mode="positive",
        source_text="\u041e\u043d \u0437\u043d\u0430\u043b \u0447\u0442\u043e \u0434\u0435\u043b\u0430\u0442\u044c.",
        target_text="\u041e\u043d \u0437\u043d\u0430\u043b, \u0447\u0442\u043e \u0434\u0435\u043b\u0430\u0442\u044c.",
        gap_operations=(
            LayerOperation(
                kind="gap",
                label="COMMA",
                source_pattern="\u0437\u043d\u0430\u043b",
                target_pattern=",",
            ),
        ),
        expected_token_edit_count=0,
        expected_gap_edit_count=1,
    )

    example = build_gap_operations_example(case, _realizer(), layer="syntax_punctuation")

    assert example.token_edit_labels == ["KEEP", "KEEP", "KEEP", "KEEP"]
    assert example.gap_labels == ["NONE", "COMMA", "NONE", "DOT"]
    assert example.rule_ids == ["none", "comma_subordinate", "none", "none"]
    assert example.metadata["operation"] == "gap"
    assert example.metadata["expected_gap_edit_count"] == 1


def test_gap_delete_uses_delete_punctuation_for_source_gap() -> None:
    case = LayerDirectCase(
        rule_id="comma_subordinate",
        family="syntax_punctuation",
        sub_rule_id="delete_extra_comma",
        mode="positive",
        source_text="\u041e\u043d \u0437\u043d\u0430\u043b, \u0447\u0442\u043e \u0434\u0435\u043b\u0430\u0442\u044c.",
        target_text="\u041e\u043d \u0437\u043d\u0430\u043b \u0447\u0442\u043e \u0434\u0435\u043b\u0430\u0442\u044c.",
        gap_operations=(
            LayerOperation(
                kind="gap",
                label="DELETE_PUNCTUATION",
                source_pattern="\u0437\u043d\u0430\u043b",
                target_pattern="",
            ),
        ),
        expected_token_edit_count=0,
        expected_gap_edit_count=1,
    )

    example = build_generated_example_from_case(case, _realizer(), layer="syntax_punctuation")

    assert example.gap_labels == ["NONE", "DELETE_PUNCTUATION", "NONE", "DOT"]
    assert example.rule_ids == ["none", "comma_subordinate", "none", "none"]


def test_builder_rejects_unmatched_span_without_silent_label_mismatch() -> None:
    case = LayerDirectCase(
        rule_id="ne_verb",
        family="compound_spelling",
        sub_rule_id="missing_span",
        mode="positive",
        source_text="\u041e\u043d \u043f\u0438\u0441\u0430\u043b \u0442\u0435\u043a\u0441\u0442.",
        target_text="\u041e\u043d \u043f\u0438\u0441\u0430\u043b \u0442\u0435\u043a\u0441\u0442.",
        token_operations=(
            LayerOperation(
                kind="token_span",
                label="SPAN_REPLACE_BY_LEXICON",
                source_pattern="\u043d\u0430 \u0441\u0447\u0435\u0442",
                target_pattern="\u043d\u0430\u0441\u0447\u0451\u0442",
            ),
        ),
        expected_token_edit_count=1,
        expected_gap_edit_count=0,
    )

    with pytest.raises(ValueError, match="source_pattern"):
        build_token_span_replacement_example(case, _realizer(), layer="compound_spelling")


def _realizer() -> Realizer:
    return Realizer(Lexicon.default(), MorphologyEngine(use_pymorphy=False))
