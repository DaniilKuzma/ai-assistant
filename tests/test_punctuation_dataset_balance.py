from collections import Counter

from src.alignment.punctuation_label_builder import build_punctuation_gap_action_labels, build_punctuation_gap_labels
from src.data.full_dataset_builder import _build_targeted_punctuation_rows
from src.data.synthetic_generator import SyntheticGenerator
from src.validation.diff_analyzer import DiffAnalyzer


PUNCTUATION_GROUPS = [
    "comma_subordinate",
    "comma_conjunction",
    "introductory",
    "colon",
    "dash",
    "semicolon",
    "quotes_brackets",
    "final_punctuation",
    "delete_replace",
]


def test_targeted_punctuation_rows_cover_hundreds_for_each_group():
    diff_analyzer = DiffAnalyzer()

    for group in PUNCTUATION_GROUPS:
        rows = _build_targeted_punctuation_rows(
            group,
            required_count=220,
            diff_analyzer=diff_analyzer,
            domain="test",
            seen_pairs=set(),
        )

        assert len(rows) >= 200
        assert {row["source_dataset"] for row in rows} == {f"synthetic_balanced_punctuation_{group}"}
        assert all(row["source"] != row["target"] for row in rows)
        assert all("punctuation" in row["error_types"] or "final_punctuation" in row["error_types"] for row in rows)


def test_targeted_punctuation_rows_create_expected_mark_and_action_labels():
    diff_analyzer = DiffAnalyzer()
    rows = []
    for group in PUNCTUATION_GROUPS:
        rows.extend(
            _build_targeted_punctuation_rows(
                group,
                required_count=20,
                diff_analyzer=diff_analyzer,
                domain="test",
                seen_pairs=set(),
            )
        )

    marks = Counter(
        label.label
        for row in rows
        for label in build_punctuation_gap_labels(row["source"], row["target"])
        if label.label != "NONE"
    )
    actions = Counter(
        label.action
        for row in rows
        for label in build_punctuation_gap_action_labels(row["source"], row["target"])
    )

    assert marks["COMMA"] >= 20
    assert marks["COLON"] >= 20
    assert marks["DASH"] >= 20
    assert marks["SEMICOLON"] >= 20
    assert marks["QUOTE_OPEN"] >= 20
    assert marks["BRACKET_OPEN"] >= 20
    assert marks["QUESTION"] >= 5
    assert actions["INSERT"] >= 20
    assert actions["DELETE"] >= 20
    assert actions["REPLACE"] >= 20
    assert actions["KEEP_EXISTING"] >= 20


def test_synthetic_punctuation_does_not_modify_protected_fragments():
    generator = SyntheticGenerator(seed=5)
    target = "Напиши на test@example.com, открой https://example.com и проверь 12.05.2026, цена 40,16 руб."

    variants = generator.generate_variants_from_clean(target, max_variants=40)

    assert variants
    for variant in variants:
        assert "test@example.com" in variant.source
        assert "https://example.com" in variant.source
        assert "12.05.2026" in variant.source
        assert "40,16" in variant.source
