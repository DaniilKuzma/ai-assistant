from src.candidates.candidate_generator import CandidateGenerator
from src.inference.corrector import Corrector
from src.validation.strict_validator import StrictValidator


def _punctuation(candidates, rule_id: str | None = None):
    items = [
        candidate
        for candidate in candidates
        if candidate.edit_type in {"punctuation_insert", "punctuation_delete", "punctuation_replace", "final_punctuation"}
    ]
    if rule_id is None:
        return items
    return [candidate for candidate in items if candidate.rule_id == rule_id]


def test_straight_balanced_quotes_are_not_normalized_to_guillemets():
    candidates = CandidateGenerator().generate('Он сказал "Проект готов".')

    assert not _punctuation(candidates, "quote_open")
    assert not _punctuation(candidates, "quote_close")
    assert Corrector().correct('Он сказал "Проект готов".').corrected_text == 'Он сказал "Проект готов".'


def test_missing_quote_open_and_close_candidates_are_model_required():
    missing_close = CandidateGenerator().generate("Он сказал: «Проект готов.")
    missing_open = CandidateGenerator().generate("Он сказал: Проект готов».")

    close_candidate = next(candidate for candidate in _punctuation(missing_close, "quote_pair_balance") if candidate.label == "QUOTE_CLOSE")
    open_candidate = next(candidate for candidate in _punctuation(missing_open, "quote_pair_balance") if candidate.label == "QUOTE_OPEN")

    for candidate in (close_candidate, open_candidate):
        assert candidate.mode == "model_required"
        assert candidate.requires_model is True
        assert candidate.requires_scoring is True
        assert candidate.requires == ("model",)


def test_missing_bracket_open_and_close_candidates_are_model_required():
    missing_close = CandidateGenerator().generate("Проверь документ (черновик.")
    missing_open = CandidateGenerator().generate("Проверь документ черновик).")

    close_candidate = next(candidate for candidate in _punctuation(missing_close, "bracket_pair_balance") if candidate.label == "BRACKET_CLOSE")
    open_candidate = next(candidate for candidate in _punctuation(missing_open, "bracket_pair_balance") if candidate.label == "BRACKET_OPEN")

    assert close_candidate.replacement == ")"
    assert open_candidate.replacement == "("
    for candidate in (close_candidate, open_candidate):
        assert candidate.mode == "model_required"
        assert candidate.requires_model is True
        assert candidate.requires_scoring is True


def test_punctuation_noise_cleanup_is_bounded_candidate_not_plain_fallback():
    candidates = CandidateGenerator().generate("Привет,, мир.. Что!!? Правда??! Да;; нет::")
    cleanup = _punctuation(candidates, "punctuation_delete_replace")

    assert cleanup
    assert {candidate.mode for candidate in cleanup} <= {"candidate_only", "model_required"}
    assert any(candidate.source == "," and candidate.replacement == "" for candidate in cleanup)
    assert any(candidate.source == "." and candidate.replacement == "" for candidate in cleanup)
    assert any(candidate.source in {"!", "?"} and candidate.replacement == "" for candidate in cleanup)
    assert Corrector().correct("Привет,, мир").corrected_text == "Привет,, мир"


def test_sentence_start_case_is_not_plain_auto_applied_and_skips_abbreviation_contexts():
    generator = CandidateGenerator()

    assert any(candidate.rule_id == "capitalization_sentence_start" for candidate in generator.generate("сегодня хорошая погода"))
    assert Corrector().correct("сегодня хорошая погода").corrected_text == "сегодня хорошая погода"
    assert not [
        candidate
        for candidate in generator.generate("Daewoo Motor Co. распродает")
        if candidate.rule_id == "capitalization_sentence_start"
    ]
    assert not [
        candidate
        for candidate in generator.generate("Отчет за 2024 г. готов")
        if candidate.rule_id == "capitalization_sentence_start"
    ]


def test_abbreviation_case_remains_limited_to_known_uppercase_forms():
    candidates = CandidateGenerator().generate("сша и нбб открыли офис.")
    values = {(candidate.source, candidate.replacement, candidate.rule_id) for candidate in candidates}

    assert ("сша", "США", "abbreviation_case_protection") in values
    assert ("нбб", "НББ", "abbreviation_case_protection") in values
    assert Corrector().correct("сша и нбб открыли офис.").corrected_text == "сша и нбб открыли офис."


def test_validator_accepts_balance_repairs_and_rejects_created_imbalance():
    accepted = StrictValidator().validate(
        "Он сказал: «Проект готов.",
        "Он сказал: «Проект готов».",
        trusted_edits=[],
    )
    rejected = StrictValidator().validate(
        "Проверь документ черновик.",
        "Проверь документ (черновик.",
    )

    assert accepted.apply_accepted() == "Он сказал: «Проект готов»."
    assert any(edit.reason == "unbalanced_pairs" for edit in rejected.edits)
