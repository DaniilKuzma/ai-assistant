from __future__ import annotations

import pytest

from src.candidates.candidate_generator import CandidateGenerator
from src.inference.corrector import Corrector


@pytest.mark.parametrize(
    ("source", "rule_id", "label", "subtype", "syntax_family"),
    [
        ("Я думаю что проект готов.", "comma_subordinate", "COMMA", "chto", "subordinate_clause_comma"),
        ("Мы пришли но встреча закончилась.", "comma_conjunction", "COMMA", "no", "coordinating_conjunction_comma"),
        ("Вероятно проект готов.", "introductory_comma", "COMMA", "veroyatno", "introductory_words"),
        ("Друзья проверим отчёт.", "address_comma", "COMMA", "friends", "address_comma"),
        ("Проверив отчёт редактор отправил письмо.", "detached_adverbial_comma", "COMMA", "gerund", "detached_adverbial_phrases"),
        ("Отчёт подготовленный командой отправили утром.", "detached_participial_comma", "COMMA", "post_noun_participial", "detached_participial_phrases"),
        ("Команда проверила отчёты письма и заявки.", "homogeneous_comma", "COMMA", "simple_noun_series", "homogeneous_members"),
        ("Возьми следующее документы и ключи.", "enumeration_colon", "COLON", "summary_word_before_list", "homogeneous_members"),
        ("Отчёты письма заявки все готовы.", "enumeration_dash", "DASH", "summary_word_after_list", "enumeration_colon_dash"),
        ("Москва столица России.", "subject_predicate_dash", "DASH", "noun_predicate", "subject_predicate_dash"),
        ("Он замер как будто услышал шум.", "comparative_turnover_comma", "COMMA", "kak_budto", "comparative_turnovers"),
        ("Он сказал проект готов.", "direct_speech_colon", "COLON", "author_before_speech", "direct_speech_syntax"),
        ("Он сказал: «Проект готов.", "quote_pair_balance", "QUOTE_CLOSE", "quote_close_missing", "quote_bracket_balance"),
        ("Проверь документ (черновик.", "bracket_pair_balance", "BRACKET_CLOSE", "bracket_close_missing", "quote_bracket_balance"),
        ("Документ,, готов.", "punctuation_delete_replace", "NONE", "duplicate_comma", "punctuation_combinations"),
    ],
)
def test_syntax_punctuation_candidates_have_required_metadata(source, rule_id, label, subtype, syntax_family):
    candidates = CandidateGenerator().generate(source)

    candidate = _candidate(candidates, rule_id, label)

    assert candidate.edit_domain == "punctuation"
    assert candidate.syntax_family == syntax_family
    assert candidate.subtype == subtype
    assert candidate.evidence
    assert candidate.confidence_source == "syntax_rule"
    assert candidate.implementation_group.startswith("syntax_")
    assert candidate.requires_model is True
    assert candidate.mode in {"model_required", "candidate_only"}
    assert candidate.metadata["edit_domain"] == "punctuation"


def test_introductory_comma_supports_multiword_bounded_marker():
    candidates = CandidateGenerator().generate("Таким образом проект готов.")

    candidate = _candidate(candidates, "introductory_comma", "COMMA")

    assert candidate.start == candidate.end == len("Таким образом")
    assert candidate.subtype == "takim_obrazom"
    assert candidate.trigger_text == "Таким образом"


def test_address_comma_supports_respectful_multiword_opening():
    candidates = CandidateGenerator().generate("Уважаемые коллеги проверим отчёт.")

    candidate = _candidate(candidates, "address_comma", "COMMA")

    assert candidate.start == candidate.end == len("Уважаемые коллеги")
    assert candidate.subtype == "uvazhaemye_kollegi"


def test_clarification_and_apposition_candidates_are_conservative():
    clarification = CandidateGenerator().generate("Нужно проверить а именно отчёт и договор.")
    apposition = CandidateGenerator().generate("Иван опытный редактор, проверил отчёт.")

    clarification_candidate = _candidate(clarification, "clarification_comma", "COMMA")
    apposition_candidate = _candidate(apposition, "apposition_comma", "COMMA")

    assert clarification_candidate.start == clarification_candidate.end == len("Нужно проверить")
    assert clarification_candidate.subtype == "a_imenno"
    assert apposition_candidate.start == apposition_candidate.end == len("Иван")
    assert apposition_candidate.subtype == "paired_apposition_repair"


def test_syntax_punctuation_candidates_are_not_auto_applied_by_plain_corrector():
    examples = [
        "Отчёт подготовленный командой отправили утром.",
        "Команда проверила отчёты письма и заявки.",
        "Москва столица России.",
        "Нужно проверить а именно отчёт и договор.",
    ]

    for source in examples:
        assert Corrector().correct(source).corrected_text == source


@pytest.mark.parametrize(
    "source",
    [
        "Он работает как инженер.",
        "Он сказал, что проект готов.",
        "Индекс вырос на 4,71%.",
        "Сайт https://example.com работает.",
        "Он сказал: «Проект готов».",
        "Документ готов!",
        "Ты видел отчёт?",
    ],
)
def test_syntax_punctuation_hard_negatives_do_not_emit_unsafe_candidates(source):
    candidates = [
        candidate
        for candidate in CandidateGenerator().generate(source)
        if candidate.edit_domain == "punctuation" and candidate.rule_id != "final_punctuation_default"
    ]

    assert candidates == []


def _candidate(candidates, rule_id: str, label: str):
    return next(
        candidate
        for candidate in candidates
        if candidate.rule_id == rule_id and candidate.label == label
    )
