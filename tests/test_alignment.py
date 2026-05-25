from src.alignment.aligner import Aligner
from src.alignment.punctuation_label_builder import build_punctuation_gap_labels


def test_aligner_accepts_only_allowed_pair():
    aligner = Aligner()

    result = aligner.align("РЇ РЅРµР·РЅР°СЋ С‡С‚Рѕ РґРµР»Р°С‚СЊ", "РЇ РЅРµ Р·РЅР°СЋ, С‡С‚Рѕ РґРµР»Р°С‚СЊ.")

    assert result.is_supported
    assert {edit.edit_type for edit in result.edits} >= {"split_word", "punctuation_insert", "final_punctuation"}


def test_aligner_rejects_semantic_pair():
    aligner = Aligner()

    result = aligner.align("РЇ Р»СЋР±Р»СЋ РґРѕРј", "РЇ РѕР±РѕР¶Р°СЋ РґРѕРј")

    assert not result.is_supported


def test_label_builders_return_word_and_gap_labels():
    punctuation_labels = build_punctuation_gap_labels("РЇ РЅРµР·РЅР°СЋ С‡С‚Рѕ РґРµР»Р°С‚СЊ", "РЇ РЅРµ Р·РЅР°СЋ, С‡С‚Рѕ РґРµР»Р°С‚СЊ.")

    assert any(label.label == "COMMA" for label in punctuation_labels)
    assert punctuation_labels[-1].label == "DOT"


def test_punctuation_gap_labels_use_positioned_alignment_when_source_already_has_comma():
    labels = build_punctuation_gap_labels(
        "РЇ РґСѓРјР°СЋ С‡С‚Рѕ СЌС‚Рѕ РІР°Р¶РЅРѕ, РЅРѕ СЃР»РѕР¶РЅРѕ",
        "РЇ РґСѓРјР°СЋ, С‡С‚Рѕ СЌС‚Рѕ РІР°Р¶РЅРѕ, РЅРѕ СЃР»РѕР¶РЅРѕ.",
    )

    assert labels[1].label == "COMMA"
    assert labels[-1].label == "DOT"


def test_punctuation_gap_labels_support_colon_replacement_from_alignment():
    labels = build_punctuation_gap_labels("РћРЅ СЃРєР°Р·Р°Р» РїСЂРёРІРµС‚", "РћРЅ СЃРєР°Р·Р°Р»: РїСЂРёРІРµС‚.")

    assert labels[1].label == "COLON"
    assert labels[-1].label == "DOT"


def test_punctuation_gap_labels_mark_deleted_punctuation_as_none():
    labels = build_punctuation_gap_labels("РЇ РґСѓРјР°СЋ, С‡С‚Рѕ СЌС‚Рѕ РІР°Р¶РЅРѕ.", "РЇ РґСѓРјР°СЋ С‡С‚Рѕ СЌС‚Рѕ РІР°Р¶РЅРѕ.")

    assert labels[1].label == "NONE"
    assert labels[-1].label == "DOT"
