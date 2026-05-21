import pytest

from src.candidates.candidate_generator import Candidate
from src.validation.strict_validator import StrictValidator


def _candidate(source: str, replacement: str, rule_id: str, text: str, candidate_type: str = "spelling") -> Candidate:
    start = text.index(source)
    return Candidate(
        source,
        replacement,
        candidate_type,
        start=start,
        end=start + len(source),
        confidence=0.999,
        requires_model=True,
        rule_id=rule_id,
    )


@pytest.mark.parametrize(
    ("source", "replacement", "rule_id", "reason"),
    [
        ("ннедалеко", "недалеко", "double_consonant_candidate", "unsafe_double_consonant_candidate"),
        ("трейдерскими", "рейдерскими", "extra_letter_candidate", "known_source_lexical_guard"),
        ("блонди", "блондин", "missing_letter_candidate", "known_source_lexical_guard"),
        ("скачок", "скачек", "pattern_чо_че", "known_source_lexical_guard"),
    ],
)
def test_validator_rejects_matrix_observed_lexical_false_positives(source, replacement, rule_id, reason):
    text = f"В тексте есть {source}."
    target = text.replace(source, replacement, 1)

    result = StrictValidator().validate(text, target, trusted_edits=[_candidate(source, replacement, rule_id, text)])

    assert any(edit.status == "rejected" and edit.reason == reason and edit.rule_id == rule_id for edit in result.edits)
    assert result.apply_accepted() == text


@pytest.mark.parametrize(
    ("source", "replacement", "rule_id"),
    [
        ("апеляция", "апелляция", "double_consonant_candidate"),
        ("територия", "территория", "double_consonant_candidate"),
        ("група", "группа", "double_consonant_candidate"),
        ("сбака", "собака", "missing_letter_candidate"),
        ("молокко", "молоко", "extra_letter_candidate"),
    ],
)
def test_validator_allows_matrix_obvious_unknown_typo_repairs(source, replacement, rule_id):
    text = f"Это {source}."
    target = text.replace(source, replacement, 1)

    result = StrictValidator().validate(text, target, trusted_edits=[_candidate(source, replacement, rule_id, text)])

    assert any(edit.status == "accepted" and edit.rule_id == rule_id for edit in result.edits)
    assert result.apply_accepted() == target


@pytest.mark.parametrize(
    ("source", "replacement", "accepted"),
    [
        ("по русски", "по-русски", True),
        ("по новому", "по-новому", True),
        ("по старому", "по-старому", True),
        ("по периодически", "по-периодически", False),
        ("по официальному", "по-официальному", False),
        ("по дому", "по-дому", False),
    ],
)
def test_validator_hyphen_po_adverbs_keeps_safe_adverbs_only(source, replacement, accepted):
    text = f"Он говорил {source} о проекте."
    target = text.replace(source, replacement, 1)
    candidate = _candidate(source, replacement, "hyphen_po_adverbs", text, candidate_type="hyphen")

    result = StrictValidator().validate(text, target, trusted_edits=[candidate])

    if accepted:
        assert any(edit.status == "accepted" and edit.rule_id == "hyphen_po_adverbs" for edit in result.edits)
        assert result.apply_accepted() == target
    else:
        assert any(
            edit.status == "rejected"
            and edit.reason == "unsafe_hyphen_po_adverb"
            and edit.rule_id == "hyphen_po_adverbs"
            for edit in result.edits
        )
        assert result.apply_accepted() == text
