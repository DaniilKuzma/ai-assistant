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


def _context_candidate(
    source_text: str,
    source: str,
    replacement: str,
    rule_id: str,
    *,
    confidence: float = 0.35,
) -> Candidate:
    start = source_text.index(source)
    return Candidate(
        source=source,
        replacement=replacement,
        edit_type="split_join",
        start=start,
        end=start + len(source),
        confidence=confidence,
        requires_model=True,
        rule_id=rule_id,
        mode="model_required",
        requires=("syntax", "morphology", "model"),
        group="context_split_join",
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


def test_low_confidence_exact_context_candidate_passes_positive_verification():
    source = "Редакция так же подготовила отчет для комиссии и отправила его в архив."
    target = "Редакция также подготовила отчет для комиссии и отправила его в архив."
    generator = StaticCandidateGenerator(
        {source: [_context_candidate(source, "так же", "также", "context_tak_zhe", confidence=0.35)]}
    )

    result = verify_atomic_positive(source, target, "context_tak_zhe", candidate_generator=generator)

    assert result.passed
    assert result.reason == "ok"
    assert result.gold_edit_count == 1
    assert result.candidate_present
    assert result.strict_validator_passed
    assert result.extra_edit_count == 0


def test_context_nesmotrya_known_overlap_collapses_to_one_logical_edit():
    source = "Встреча началась не смотря на дождь, поэтому участники пришли заранее."
    target = "Встреча началась несмотря на дождь, поэтому участники пришли заранее."
    generator = StaticCandidateGenerator(
        {source: [_context_candidate(source, "не смотря", "несмотря", "context_nesmotrya", confidence=0.35)]}
    )

    result = verify_atomic_positive(source, target, "context_nesmotrya", candidate_generator=generator)

    assert result.passed
    assert result.gold_edit_count == 1
    assert result.matched_candidate is not None
    assert result.matched_candidate["rule_id"] == "context_nesmotrya"


def test_arbitrary_overlapping_split_join_edits_still_reject_as_non_atomic():
    source = "Встреча началась не смотря на дождь, поэтому участники пришли заранее."
    target = "Встреча началась несмотря на дождь, поэтому участники пришли заранее."
    wrong_candidate = _context_candidate(source, "не смотря", "несмотря", "context_to_zhe", confidence=1.0)
    generator = StaticCandidateGenerator({source: [wrong_candidate]})

    result = verify_atomic_positive(source, target, "context_to_zhe", candidate_generator=generator)

    assert not result.passed
    assert result.reason in {"candidate_missing", "non_atomic_edit_count"}


def test_context_za_to_exact_candidate_backed_positive_can_pass():
    source = "Команда опоздала, за то отчет сдала вовремя и сохранила договор."
    target = "Команда опоздала, зато отчет сдала вовремя и сохранила договор."
    generator = StaticCandidateGenerator(
        {source: [_context_candidate(source, "за то", "зато", "context_za_to", confidence=0.35)]}
    )

    result = verify_atomic_positive(source, target, "context_za_to", candidate_generator=generator)

    assert result.passed
    assert result.reason == "ok"
    assert result.strict_validator_passed


def test_context_za_to_without_matching_candidate_still_rejects():
    source = "Команда опоздала, за то отчет сдала вовремя и сохранила договор."
    target = "Команда опоздала, зато отчет сдала вовремя и сохранила договор."

    result = verify_atomic_positive(source, target, "context_za_to", candidate_generator=StaticCandidateGenerator({}))

    assert not result.passed
    assert result.reason == "candidate_missing"


def test_context_candidate_extra_edit_still_rejects():
    source = "Редакция так же подготовила отчет для комиссии и отправила его в архив"
    target = "Редакция также подготовила отчет для комиссии и отправила его в архив."
    generator = StaticCandidateGenerator(
        {source: [_context_candidate(source, "так же", "также", "context_tak_zhe", confidence=0.35)]}
    )

    result = verify_atomic_positive(source, target, "context_tak_zhe", candidate_generator=generator)

    assert not result.passed
    assert result.reason == "non_atomic_edit_count"


def test_context_candidate_target_quality_failure_still_rejects():
    source = "Он так же пришел."
    target = "Он также пришел."
    generator = StaticCandidateGenerator(
        {source: [_context_candidate(source, "так же", "также", "context_tak_zhe", confidence=1.0)]}
    )

    result = verify_atomic_positive(source, target, "context_tak_zhe", candidate_generator=generator)

    assert not result.passed
    assert result.reason == "target_quality_failed"


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
