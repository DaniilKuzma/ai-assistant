from __future__ import annotations

from dataclasses import asdict

import pytest

from src.schema import (
    GeneratedExample,
    RuntimeEdit,
    WordToken,
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

    assert gap_label_to_id("NONE") == 0
    assert gap_id_to_label(0) == "NONE"
    assert gap_label_to_id("ELLIPSIS") == 8
    assert gap_id_to_label(8) == "ELLIPSIS"
    assert gap_label_to_id("DELETE_PUNCTUATION") == 9
    assert gap_id_to_label(9) == "DELETE_PUNCTUATION"

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
    assert example.rule_ids[1] == "ne_verb"


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
