from __future__ import annotations

from dataclasses import asdict

import pytest

from src.schema import (
    BOUNDARY_AFTER_LABELS,
    BOUNDARY_BEFORE_LABELS,
    GeneratedExample,
    RuntimeEdit,
    WordToken,
    boundary_after_id_to_label,
    boundary_after_label_to_id,
    boundary_before_id_to_label,
    boundary_before_label_to_id,
    gap_id_to_label,
    gap_label_to_id,
    rule_id_to_label,
    rule_tag_to_id,
    token_id_to_label,
    token_label_to_id,
)


def _valid_example() -> GeneratedExample:
    source_text = "Он незнал ответа"
    return GeneratedExample(
        source_text=source_text,
        target_text="Он не знал ответа.",
        source_tokens=[
            WordToken(text="Он", start=0, end=2, lemma="он", pos="PRON"),
            WordToken(text="незнал", start=3, end=9, lemma="знать", pos="VERB"),
            WordToken(text="ответа", start=10, end=16, lemma="ответ", pos="NOUN", feats={"Case": "Gen"}),
        ],
        token_edit_labels=["KEEP", "SPLIT_NE_VERB", "KEEP"],
        gap_labels=["NONE", "NONE", "DOT"],
        rule_ids=["clean_identity", "ne_verb", "final_punctuation"],
        primary_rule_id="ne_verb",
        mode="train",
        explanation_ids=["ne_verb_split", "final_dot"],
        metadata={"seed": 7, "generator": "unit"},
    )


def test_vocab_ids_are_stable() -> None:
    assert token_label_to_id("KEEP") == 0
    assert token_id_to_label(0) == "KEEP"
    assert token_label_to_id("DICT_REPLACE") == 19
    assert token_id_to_label(19) == "DICT_REPLACE"
    assert token_label_to_id("SPAN_REPLACE_BY_LEXICON") == 20
    assert token_id_to_label(20) == "SPAN_REPLACE_BY_LEXICON"
    assert token_label_to_id("CAPITALIZE") == 21
    assert token_id_to_label(21) == "CAPITALIZE"

    assert gap_label_to_id("NONE") == 0
    assert gap_id_to_label(0) == "NONE"
    assert gap_label_to_id("ELLIPSIS") == 8
    assert gap_id_to_label(8) == "ELLIPSIS"
    assert gap_label_to_id("DELETE_PUNCTUATION") == 9
    assert gap_id_to_label(9) == "DELETE_PUNCTUATION"
    assert gap_label_to_id("COMMA_DASH") == 10
    assert gap_id_to_label(10) == "COMMA_DASH"

    assert BOUNDARY_BEFORE_LABELS[:2] == ("NONE", "INSERT_OPEN_QUOTE")
    assert boundary_before_label_to_id("NONE") == 0
    assert boundary_before_label_to_id("NORMALIZE_OPEN_QUOTE") == 3
    assert boundary_before_id_to_label(3) == "NORMALIZE_OPEN_QUOTE"

    assert BOUNDARY_AFTER_LABELS[:2] == ("NONE", "INSERT_CLOSE_QUOTE")
    assert boundary_after_label_to_id("NONE") == 0
    assert boundary_after_label_to_id("NORMALIZE_CLOSE_QUOTE") == 3
    assert boundary_after_id_to_label(3) == "NORMALIZE_CLOSE_QUOTE"

    assert rule_tag_to_id("none") == 0
    assert rule_id_to_label(0) == "none"
    assert rule_tag_to_id("final_punctuation") == 15
    assert rule_id_to_label(15) == "final_punctuation"
    assert rule_tag_to_id("suffix_its_ets") > 15
    assert rule_tag_to_id("suffix_enn_yan") > 15
    assert rule_tag_to_id("n_nn_basic") > 15
    assert rule_tag_to_id("dictionary_normative_words") > rule_tag_to_id("morpheme_endings")
    assert rule_id_to_label(rule_tag_to_id("dictionary_borrowed_words")) == "dictionary_borrowed_words"
    assert rule_id_to_label(rule_tag_to_id("dictionary_domain_terms")) == "dictionary_domain_terms"
    assert rule_id_to_label(rule_tag_to_id("dictionary_common_misspellings")) == "dictionary_common_misspellings"
    assert rule_id_to_label(rule_tag_to_id("typo_character_noise")) == "typo_character_noise"
    assert rule_id_to_label(rule_tag_to_id("typo_keyboard_neighbor")) == "typo_keyboard_neighbor"
    assert rule_id_to_label(rule_tag_to_id("typo_space_noise")) == "typo_space_noise"
    assert rule_tag_to_id("punct_final_marks") > rule_tag_to_id("typo_space_noise")
    for syntax_rule_id in (
        "punct_final_marks",
        "punct_dash_syntax",
        "punct_homogeneous_extended",
        "punct_detached_definitions",
        "punct_detached_adverbials",
        "punct_comparative_turns",
        "punct_introductory_extended",
        "punct_address_interjection",
        "punct_complex_sentences",
        "punct_bsp",
        "punct_fixed_expression_guards",
    ):
        assert rule_id_to_label(rule_tag_to_id(syntax_rule_id)) == syntax_rule_id
    for casing_rule_id in (
        "casing_sentence_start",
        "casing_person_names",
        "casing_geo_names",
        "casing_organizations",
        "casing_documents_events",
        "casing_common_lowercase",
        "casing_formal_you_guard",
    ):
        assert rule_id_to_label(rule_tag_to_id(casing_rule_id)) == casing_rule_id
    for semantic_rule_id in (
        "semantic_service_words",
        "semantic_derived_prepositions",
        "semantic_ne_ni",
        "semantic_introductory_context",
        "semantic_comparative_context",
    ):
        assert rule_id_to_label(rule_tag_to_id(semantic_rule_id)) == semantic_rule_id


def test_unknown_labels_raise_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown token edit label"):
        token_label_to_id("UNKNOWN")
    with pytest.raises(ValueError, match="Unknown token edit id"):
        token_id_to_label(999)
    with pytest.raises(ValueError, match="Unknown token edit id"):
        token_id_to_label(-1)

    with pytest.raises(ValueError, match="Unknown gap punctuation label"):
        gap_label_to_id("QUOTE")
    with pytest.raises(ValueError, match="Unknown gap punctuation id"):
        gap_id_to_label(999)
    with pytest.raises(ValueError, match="Unknown gap punctuation id"):
        gap_id_to_label(-1)

    with pytest.raises(ValueError, match="Unknown boundary-before label"):
        boundary_before_label_to_id("QUOTE")
    with pytest.raises(ValueError, match="Unknown boundary-before id"):
        boundary_before_id_to_label(999)

    with pytest.raises(ValueError, match="Unknown boundary-after label"):
        boundary_after_label_to_id("QUOTE")
    with pytest.raises(ValueError, match="Unknown boundary-after id"):
        boundary_after_id_to_label(999)

    with pytest.raises(ValueError, match="Unknown rule label"):
        rule_tag_to_id("candidate_rule")
    with pytest.raises(ValueError, match="Unknown rule id"):
        rule_id_to_label(999)
    with pytest.raises(ValueError, match="Unknown rule id"):
        rule_id_to_label(-1)


def test_generated_example_validates_contract() -> None:
    example = _valid_example()

    assert example.source_tokens[1].text == "незнал"
    assert example.token_edit_labels[1] == "SPLIT_NE_VERB"
    assert example.gap_labels[-1] == "DOT"
    assert example.boundary_before_labels == ["NONE", "NONE", "NONE"]
    assert example.boundary_after_labels == ["NONE", "NONE", "NONE"]
    assert example.rule_ids[1] == "ne_verb"


def test_generated_example_accepts_boundary_labels() -> None:
    example = GeneratedExample(
        source_text="Проект готов.",
        target_text="«Проект готов».",
        source_tokens=[
            WordToken(text="Проект", start=0, end=6),
            WordToken(text="готов", start=7, end=12),
        ],
        token_edit_labels=["KEEP", "KEEP"],
        gap_labels=["NONE", "DOT"],
        rule_ids=["clean_identity", "clean_identity"],
        primary_rule_id="clean_identity",
        mode="positive",
        explanation_ids=[],
        metadata={},
        boundary_before_labels=["INSERT_OPEN_QUOTE", "NONE"],
        boundary_after_labels=["NONE", "INSERT_CLOSE_QUOTE"],
    )

    assert example.boundary_before_labels == ["INSERT_OPEN_QUOTE", "NONE"]
    assert example.boundary_after_labels == ["NONE", "INSERT_CLOSE_QUOTE"]


def test_generated_example_json_without_boundary_fields_loads_defaults() -> None:
    example = _valid_example()
    payload = example.to_dict()
    payload.pop("boundary_before_labels")
    payload.pop("boundary_after_labels")

    loaded = GeneratedExample.from_dict(payload)

    assert loaded.boundary_before_labels == ["NONE", "NONE", "NONE"]
    assert loaded.boundary_after_labels == ["NONE", "NONE", "NONE"]


def test_generated_example_rejects_label_length_mismatch() -> None:
    with pytest.raises(ValueError, match="token_edit_labels"):
        GeneratedExample(
            source_text="Он незнал ответа",
            target_text="Он не знал ответа.",
            source_tokens=[
                WordToken(text="Он", start=0, end=2),
                WordToken(text="незнал", start=3, end=9),
            ],
            token_edit_labels=["KEEP"],
            gap_labels=["NONE", "DOT"],
            rule_ids=["clean_identity", "ne_verb"],
            primary_rule_id="ne_verb",
            mode="train",
            explanation_ids=[],
            metadata={},
        )


def test_generated_example_json_roundtrip_preserves_stable_id() -> None:
    example = _valid_example()

    loaded = GeneratedExample.from_json(example.to_json())

    assert loaded == example
    assert loaded.stable_id() == example.stable_id()


def test_runtime_edit_is_dataclass_serializable() -> None:
    edit = RuntimeEdit(
        start=3,
        end=9,
        source="незнал",
        replacement="не знал",
        edit_type="split",
        rule_id="ne_verb",
        confidence=0.92,
        explanation="Раздельное написание не с глаголом.",
    )

    assert asdict(edit) == {
        "start": 3,
        "end": 9,
        "source": "незнал",
        "replacement": "не знал",
        "edit_type": "split",
        "rule_id": "ne_verb",
        "confidence": 0.92,
        "explanation": "Раздельное написание не с глаголом.",
    }
