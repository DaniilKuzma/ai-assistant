from src.alignment.aligner import Aligner
from src.alignment.edit_label_builder import build_edit_labels
from src.alignment.punctuation_label_builder import build_punctuation_gap_labels


def test_aligner_accepts_only_allowed_pair():
    aligner = Aligner()

    result = aligner.align("Я незнаю что делать", "Я не знаю, что делать.")

    assert result.is_supported
    assert {edit.edit_type for edit in result.edits} >= {"split_word", "punctuation_insert", "final_punctuation"}


def test_aligner_rejects_semantic_pair():
    aligner = Aligner()

    result = aligner.align("Я люблю дом", "Я обожаю дом")

    assert not result.is_supported


def test_label_builders_return_word_and_gap_labels():
    edit_labels = build_edit_labels("Я незнаю что делать", "Я не знаю, что делать.")
    punctuation_labels = build_punctuation_gap_labels("Я незнаю что делать", "Я не знаю, что делать.")

    assert any(label.edit_type == "split_word" for label in edit_labels)
    assert any(label.label == "COMMA" for label in punctuation_labels)
    assert punctuation_labels[-1].label == "DOT"


def test_punctuation_gap_labels_use_positioned_alignment_when_source_already_has_comma():
    labels = build_punctuation_gap_labels(
        "Я думаю что это важно, но сложно",
        "Я думаю, что это важно, но сложно.",
    )

    assert labels[1].label == "COMMA"
    assert labels[-1].label == "DOT"


def test_punctuation_gap_labels_support_colon_replacement_from_alignment():
    labels = build_punctuation_gap_labels("Он сказал привет", "Он сказал: привет.")

    assert labels[1].label == "COLON"
    assert labels[-1].label == "DOT"
