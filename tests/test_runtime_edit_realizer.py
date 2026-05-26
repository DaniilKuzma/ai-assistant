from __future__ import annotations

from src.runtime.edit_realizer import apply_gap_labels, apply_token_edit_labels
from src.runtime.orthographic_lexicon import CorrectionEntry, OrthographicCorrectionLexicon
from src.runtime.tokenization import tokenize_runtime_words


def _entry(
    source: str,
    target: str,
    *,
    rule_id: str = "test_span",
    operation: str = "split_join",
    confidence: float = 1.0,
) -> CorrectionEntry:
    return CorrectionEntry(
        source=source,
        target=target,
        rule_id=rule_id,
        operation=operation,
        explanation_id="fixture",
        context_class="",
        ambiguity_level="unambiguous",
        confidence=confidence,
    )


def test_split_ne_verb_applies_to_token_span() -> None:
    text = "Он незнал ответ"
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "SPLIT_NE_VERB", "KEEP"],
        [1.0, 0.95, 1.0],
        threshold=0.7,
    )

    assert corrected == "Он не знал ответ"
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("незнал", "не знал", "ne_verb")
    ]


def test_merge_tak_zhe_consumes_second_token() -> None:
    text = "Он так же пришел"
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "MERGE_TAK_ZHE_TO_TAKZHE", "SKIP_MERGED", "KEEP"],
        [1.0, 0.96, 1.0, 1.0],
        threshold=0.7,
    )

    assert corrected == "Он также пришел"
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("так же", "также", "takzhe_tak_zhe")
    ]


def test_tsya_ttsya_realizer_handles_controlled_vowel_change_to_infinitive() -> None:
    text = "\u0412\u0440\u0430\u0447 \u0445\u043e\u0442\u0435\u043b \u043e\u0448\u0438\u0431\u0430\u0435\u0442\u0441\u044f."
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "KEEP", "FIX_TSYA_TO_TTSYA"],
        [1.0, 1.0, 0.99],
        threshold=0.7,
    )

    assert corrected == "\u0412\u0440\u0430\u0447 \u0445\u043e\u0442\u0435\u043b \u043e\u0448\u0438\u0431\u0430\u0442\u044c\u0441\u044f."
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("\u043e\u0448\u0438\u0431\u0430\u0435\u0442\u0441\u044f", "\u043e\u0448\u0438\u0431\u0430\u0442\u044c\u0441\u044f", "tsya_ttsya")
    ]


def test_tsya_ttsya_realizer_handles_controlled_vowel_change_to_finite() -> None:
    text = "\u0412\u0440\u0430\u0447 \u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0442\u044c\u0441\u044f \u0443\u0442\u0440\u043e\u043c."
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "FIX_TTSYA_TO_TSYA", "KEEP"],
        [1.0, 0.99, 1.0],
        threshold=0.7,
    )

    assert corrected == "\u0412\u0440\u0430\u0447 \u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0435\u0442\u0441\u044f \u0443\u0442\u0440\u043e\u043c."
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("\u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0442\u044c\u0441\u044f", "\u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0435\u0442\u0441\u044f", "tsya_ttsya")
    ]


def test_gap_comma_insertion_preserves_spaces() -> None:
    text = "Он знал что делать"
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_gap_labels(
        text,
        tokens,
        ["NONE", "COMMA", "NONE", "NONE"],
        [1.0, 0.93, 1.0, 1.0],
        threshold=0.7,
    )

    assert corrected == "Он знал, что делать"
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("", ",", "comma_subordinate")
    ]


def test_gap_comma_insertion_does_not_duplicate_existing_comma() -> None:
    text = "Он знал, что делать"
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_gap_labels(
        text,
        tokens,
        ["NONE", "COMMA", "NONE", "NONE"],
        [1.0, 0.93, 1.0, 1.0],
        threshold=0.7,
    )

    assert corrected == text
    assert edits == []


def test_dict_replace_uses_orthographic_lexicon_replacement() -> None:
    text = "В словаре указано слово «коженный»."
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "KEEP", "KEEP", "KEEP", "DICT_REPLACE"],
        [1.0, 1.0, 1.0, 1.0, 0.96],
        threshold=0.7,
        rule_ids=["none", "none", "none", "none", "suffix_enn_yan"],
        orthographic_lexicon=OrthographicCorrectionLexicon.default(),
    )

    assert corrected == "В словаре указано слово «кожаный»."
    assert [(edit.source, edit.replacement, edit.rule_id, edit.edit_type) for edit in edits] == [
        ("коженный", "кожаный", "suffix_enn_yan", "spelling")
    ]


def test_dict_replace_skips_unknown_or_ambiguous_replacement() -> None:
    text = "В словаре указано слово «неизвестный»."
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "KEEP", "KEEP", "KEEP", "DICT_REPLACE"],
        [1.0, 1.0, 1.0, 1.0, 0.99],
        threshold=0.7,
        rule_ids=["none", "none", "none", "none", "suffix_enn_yan"],
        orthographic_lexicon=OrthographicCorrectionLexicon.default(),
    )

    assert corrected == text
    assert edits == []


def test_span_replace_by_lexicon_merges_na_schet_to_naschyot() -> None:
    text = "\u043d\u0430 \u0441\u0447\u0435\u0442"
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon(
        [_entry(text, "\u043d\u0430\u0441\u0447\u0451\u0442", rule_id="test_span")]
    )

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["SPAN_REPLACE_BY_LEXICON", "DELETE"],
        [0.99, 0.99],
        threshold=0.7,
        rule_ids=["test_span", "test_span"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == "\u043d\u0430\u0441\u0447\u0451\u0442"
    assert [(edit.source, edit.replacement, edit.rule_id, edit.edit_type) for edit in edits] == [
        ("\u043d\u0430 \u0441\u0447\u0435\u0442", "\u043d\u0430\u0441\u0447\u0451\u0442", "test_span", "split_join")
    ]


def test_span_replace_by_lexicon_splits_naschety_to_na_schyot() -> None:
    text = "\u043d\u0430\u0441\u0447\u0435\u0442"
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon(
        [_entry(text, "\u043d\u0430 \u0441\u0447\u0451\u0442", rule_id="test_span")]
    )

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["SPAN_REPLACE_BY_LEXICON"],
        [0.99],
        threshold=0.7,
        rule_ids=["test_span"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == "\u043d\u0430 \u0441\u0447\u0451\u0442"
    assert len(edits) == 1


def test_span_replace_by_lexicon_hyphenates_pol_yabloka() -> None:
    text = "\u043f\u043e\u043b \u044f\u0431\u043b\u043e\u043a\u0430"
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon(
        [
            _entry(
                text,
                "\u043f\u043e\u043b-\u044f\u0431\u043b\u043e\u043a\u0430",
                rule_id="test_span",
                operation="hyphen",
            )
        ]
    )

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["SPAN_REPLACE_BY_LEXICON", "KEEP"],
        [0.99, 1.0],
        threshold=0.7,
        rule_ids=["test_span", "none"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == "\u043f\u043e\u043b-\u044f\u0431\u043b\u043e\u043a\u0430"
    assert edits[0].edit_type == "hyphen"


def test_span_replace_by_lexicon_unhyphenates_pol_yabloka() -> None:
    text = "\u043f\u043e\u043b-\u044f\u0431\u043b\u043e\u043a\u0430"
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon(
        [
            _entry(
                text,
                "\u043f\u043e\u043b \u044f\u0431\u043b\u043e\u043a\u0430",
                rule_id="test_span",
                operation="unhyphen",
            )
        ]
    )

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["SPAN_REPLACE_BY_LEXICON"],
        [0.99],
        threshold=0.7,
        rule_ids=["test_span"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == "\u043f\u043e\u043b \u044f\u0431\u043b\u043e\u043a\u0430"
    assert edits[0].edit_type == "hyphen"


def test_span_replace_by_lexicon_skips_ambiguous_lookup() -> None:
    text = "\u043d\u0430\u0441\u0447\u0435\u0442"
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon(
        [
            _entry(text, "\u043d\u0430 \u0441\u0447\u0451\u0442", rule_id="test_span"),
            _entry(text, "\u043d\u0430\u0441\u0447\u0451\u0442", rule_id="test_span"),
        ]
    )

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["SPAN_REPLACE_BY_LEXICON"],
        [0.99],
        threshold=0.7,
        rule_ids=["test_span"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == text
    assert edits == []


def test_span_replace_by_lexicon_respects_confidence_threshold() -> None:
    text = "\u043d\u0430 \u0441\u0447\u0435\u0442"
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon(
        [_entry(text, "\u043d\u0430\u0441\u0447\u0451\u0442", rule_id="test_span")]
    )

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["SPAN_REPLACE_BY_LEXICON", "KEEP"],
        [0.2, 1.0],
        threshold=0.7,
        rule_ids=["test_span", "none"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == text
    assert edits == []


def test_delete_punctuation_removes_comma_after_word() -> None:
    text = "\u041e\u043d \u0437\u043d\u0430\u043b, \u0447\u0442\u043e \u0434\u0435\u043b\u0430\u0442\u044c"
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_gap_labels(
        text,
        tokens,
        ["NONE", "DELETE_PUNCTUATION", "NONE", "NONE"],
        [1.0, 0.95, 1.0, 1.0],
        threshold=0.7,
    )

    assert corrected == "\u041e\u043d \u0437\u043d\u0430\u043b \u0447\u0442\u043e \u0434\u0435\u043b\u0430\u0442\u044c"
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        (",", "", "punctuation")
    ]


def test_delete_punctuation_removes_dash_and_normalizes_space() -> None:
    text = "\u041e\u0442\u0447\u0451\u0442 \u2014 \u0441\u043e\u0434\u0435\u0440\u0436\u0438\u0442 \u0434\u0430\u043d\u043d\u044b\u0435."
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_gap_labels(
        text,
        tokens,
        ["DELETE_PUNCTUATION", "NONE", "NONE"],
        [0.95, 1.0, 1.0],
        threshold=0.7,
        rule_ids=["punct_dash_syntax", "none", "none"],
    )

    assert corrected == "\u041e\u0442\u0447\u0451\u0442 \u0441\u043e\u0434\u0435\u0440\u0436\u0438\u0442 \u0434\u0430\u043d\u043d\u044b\u0435."
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        (" \u2014 ", " ", "punct_dash_syntax")
    ]


def test_none_gap_label_does_not_delete_existing_comma() -> None:
    text = "\u041e\u043d \u0437\u043d\u0430\u043b, \u0447\u0442\u043e \u0434\u0435\u043b\u0430\u0442\u044c"
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_gap_labels(
        text,
        tokens,
        ["NONE", "NONE", "NONE", "NONE"],
        [1.0, 1.0, 1.0, 1.0],
        threshold=0.7,
    )

    assert corrected == text
    assert edits == []
