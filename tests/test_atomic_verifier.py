from __future__ import annotations

import json

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.data.atomic_verifier import (
    verification_to_metadata,
    verify_atomic_positive,
)


def _fast_generator() -> CandidateGenerator:
    return CandidateGenerator(dictionary_lexicon=(), dictionary_limit=0, syntax_provider=lambda _text: ())


class StaticCandidateGenerator:
    def __init__(self, candidates_by_text: dict[str, list[Candidate]]) -> None:
        self.candidates_by_text = candidates_by_text

    def generate(self, text: str):
        return list(self.candidates_by_text.get(text, []))


def _candidate(source_text: str, source: str, replacement: str, rule_id: str, edit_type: str = "spelling") -> Candidate:
    start = source_text.index(source)
    return Candidate(
        source=source,
        replacement=replacement,
        edit_type=edit_type,
        start=start,
        end=start + len(source),
        rule_id=rule_id,
    )


def test_single_candidate_backed_edit_passes_quality_and_strict_validation():
    source = "Сегодня я незнаю что делать после проверки отчета."
    target = "Сегодня я не знаю что делать после проверки отчета."

    result = verify_atomic_positive(source, target, "ne_verb", candidate_generator=_fast_generator())

    assert result.passed
    assert result.reason == "ok"
    assert result.gold_edit_count == 1
    assert result.candidate_present
    assert result.strict_validator_passed
    assert result.target_quality_passed
    assert result.extra_edit_count == 0
    assert result.matched_candidate is not None
    assert result.matched_candidate["rule_id"] == "ne_verb"


def test_multi_edit_pair_rejects_as_non_atomic():
    source = "Сегодня я незнаю что делать после проверки отчета"
    target = "Сегодня я не знаю, что делать после проверки отчета."

    result = verify_atomic_positive(source, target, "ne_verb", candidate_generator=_fast_generator())

    assert not result.passed
    assert result.reason == "non_atomic_edit_count"
    assert result.gold_edit_count > 1


def test_real_multi_edit_ne_and_spelling_pair_rejects_as_non_atomic_regression():
    source = "Гантамиров с этим решонием несогласен."
    target = "Гантамиров с этим решением не согласен."
    generator = StaticCandidateGenerator(
        {
            source: [
                _candidate(source, "решонием", "решением", "unit_spelling"),
                _candidate(source, "несогласен", "не согласен", "context_ne", edit_type="split_join"),
            ]
        }
    )

    result = verify_atomic_positive(source, target, "context_ne", candidate_generator=generator)

    assert not result.passed
    assert result.reason == "non_atomic_edit_count"
    assert result.gold_edit_count >= 2


def test_real_multi_edit_zdat_and_chto_by_pair_rejects_as_non_atomic_regression():
    source = "Я очень хотел здать экзамен, что бы заслужить доверие комиссии."
    target = "Я очень хотел сдать экзамен, чтобы заслужить доверие комиссии."
    generator = StaticCandidateGenerator(
        {
            source: [
                _candidate(source, "здать", "сдать", "unit_spelling"),
                _candidate(source, "что бы", "чтобы", "context_chto_by", edit_type="split_join"),
            ]
        }
    )

    result = verify_atomic_positive(source, target, "context_chto_by", candidate_generator=generator)

    assert not result.passed
    assert result.reason == "non_atomic_edit_count"
    assert result.gold_edit_count >= 2


def test_unknown_rule_rejects_before_candidate_lookup():
    source = "Сегодня я незнаю что делать после проверки отчета."
    target = "Сегодня я не знаю что делать после проверки отчета."

    missing = verify_atomic_positive(source, target, "", candidate_generator=_fast_generator())
    unknown = verify_atomic_positive(source, target, "unknown", candidate_generator=_fast_generator())

    assert not missing.passed
    assert missing.reason == "unknown_rule"
    assert not unknown.passed
    assert unknown.reason == "unknown_rule"


def test_candidate_missing_rejects_when_generator_has_candidate_under_other_rule():
    source = "Сегодня я незнаю что делать после проверки отчета."
    target = "Сегодня я не знаю что делать после проверки отчета."

    result = verify_atomic_positive(source, target, "frequent_error_exact", candidate_generator=_fast_generator())

    assert not result.passed
    assert result.reason == "candidate_missing"
    assert not result.candidate_present
    assert "ne_verb" in result.candidate_rule_ids


def test_strict_validator_rejection_rejects_candidate_backed_pair():
    source = "Сегодня я сделал также как редактор после проверки отчета."
    target = "Сегодня я сделал так же как редактор после проверки отчета."

    result = verify_atomic_positive(source, target, "context_tak_zhe", candidate_generator=_fast_generator())

    assert not result.passed
    assert result.reason == "strict_validator_rejected"
    assert result.candidate_present
    assert not result.strict_validator_passed


def test_verification_metadata_serializes_plain_matched_candidate_dict():
    source = "Сегодня я незнаю что делать после проверки отчета."
    target = "Сегодня я не знаю что делать после проверки отчета."

    result = verify_atomic_positive(source, target, "ne_verb", candidate_generator=_fast_generator())
    metadata = verification_to_metadata(result)

    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["atomic_verification"]["passed"] is True
    assert isinstance(decoded["atomic_verification"]["matched_candidate"], dict)
    assert decoded["atomic_verification"]["matched_candidate"]["rule_id"] == "ne_verb"
    assert "Candidate(" not in encoded
