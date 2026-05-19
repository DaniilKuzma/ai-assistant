from types import SimpleNamespace

from src.candidates.candidate_generator import CandidateGenerator
from src.inference.corrector import Corrector


def test_subordinate_comma_is_model_required_gap_candidate_only():
    candidates = CandidateGenerator().generate("Я думаю что проект готов.")

    candidate = _punctuation_candidate(candidates, "comma_subordinate", "COMMA", "INSERT")

    assert candidate.replacement == ","
    assert candidate.edit_type == "punctuation_insert"
    assert candidate.start == candidate.end == len("Я думаю")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires_scoring is True
    assert candidate.requires == ("syntax", "model")
    assert candidate.gap_index == 1

    result = Corrector().correct("Я думаю что проект готов.")
    assert result.corrected_text == "Я думаю что проект готов."


def test_subordinate_comma_supports_bounded_extended_markers():
    candidates = CandidateGenerator().generate("Мы остались так как проект не готов.")

    candidate = _punctuation_candidate(candidates, "comma_subordinate", "COMMA", "INSERT")

    assert candidate.replacement == ","
    assert candidate.start == candidate.end == len("Мы остались")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires == ("syntax", "model")


def test_conjunction_comma_is_model_required_gap_candidate_only():
    candidates = CandidateGenerator().generate("Мы пришли но встреча закончилась.")

    candidate = _punctuation_candidate(candidates, "comma_conjunction", "COMMA", "INSERT")

    assert candidate.replacement == ","
    assert candidate.edit_type == "punctuation_insert"
    assert candidate.start == candidate.end == len("Мы пришли")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires_scoring is True
    assert candidate.requires == ("syntax", "model")
    assert candidate.gap_index == 1

    result = Corrector().correct("Мы пришли но встреча закончилась.")
    assert result.corrected_text == "Мы пришли но встреча закончилась."


def test_introductory_word_comma_is_model_required_gap_candidate_only():
    candidates = CandidateGenerator().generate("Конечно проект сложный.")

    candidate = _punctuation_candidate(candidates, "introductory_comma", "COMMA", "INSERT")

    assert candidate.replacement == ","
    assert candidate.start == candidate.end == len("Конечно")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires_scoring is True
    assert candidate.requires == ("syntax", "model")
    assert candidate.gap_index == 0

    result = Corrector().correct("Конечно проект сложный.")
    assert result.corrected_text == "Конечно проект сложный."


def test_introductory_comma_supports_bounded_extended_words():
    candidates = CandidateGenerator().generate("Следовательно проект сложный.")

    candidate = _punctuation_candidate(candidates, "introductory_comma", "COMMA", "INSERT")

    assert candidate.replacement == ","
    assert candidate.start == candidate.end == len("Следовательно")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True


def test_final_dot_is_available_as_gap_candidate_without_repeated_marks():
    candidates = CandidateGenerator().generate("Проект готов")

    candidate = _punctuation_candidate(candidates, "final_punctuation_default", "DOT", "INSERT")

    assert candidate.replacement == "."
    assert candidate.edit_type == "final_punctuation"
    assert candidate.start == candidate.end == len("Проект готов")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires_scoring is True
    assert candidate.requires == ("model",)
    assert candidate.gap_index == 1


def test_plain_corrector_does_not_apply_final_dot_without_scorer():
    result = Corrector().correct("Проект готов")

    assert result.corrected_text == "Проект готов"
    assert not any(edit.edit_type == "final_punctuation" and edit.status == "accepted" for edit in result.edits)


def test_punctuation_candidates_skip_numbers_percents_urls_and_ellipsis():
    examples = [
        "Индекс вырос на 4,71%.",
        "Компания поставила 1,39 млрд кубометров газа.",
        "Отчет за 2024 г.",
        "Сайт https://example.com работает.",
        "Напиши на test@example.com.",
        "Мы ждали файл…",
    ]

    for text in examples:
        candidates = _punctuation_candidates(CandidateGenerator().generate(text))
        assert not candidates


def test_subordinate_comma_candidate_is_not_duplicated_after_existing_comma():
    candidates = CandidateGenerator().generate("Он сказал, что проект готов.")

    assert not [
        candidate
        for candidate in _punctuation_candidates(candidates)
        if candidate.rule_id == "comma_subordinate" and candidate.label == "COMMA"
    ]


def test_subordinate_comma_skips_chto_esli_with_correlative_to():
    text = "Путин заметил, что если доля выросла то это важно."
    candidates = _punctuation_candidates(CandidateGenerator().generate(text))

    assert not [
        candidate
        for candidate in candidates
        if candidate.rule_id == "comma_subordinate" and candidate.start == len("Путин заметил, что")
    ]


def test_conjunction_comma_candidate_is_not_duplicated_after_existing_comma():
    candidates = CandidateGenerator().generate("Мы пришли, но встреча закончилась.")

    assert not [
        candidate
        for candidate in _punctuation_candidates(candidates)
        if candidate.rule_id == "comma_conjunction" and candidate.label == "COMMA"
    ]


def test_address_comma_candidate_uses_syntax_features_when_available(monkeypatch):
    import src.nlp.syntax as syntax

    monkeypatch.setattr(
        syntax,
        "parse_syntax",
        lambda _text: [
            SimpleNamespace(text="Даниил", pos="PROPN", feats={}, start=0, end=6, ner="PER"),
            SimpleNamespace(text="проверь", pos="VERB", feats={"Mood": "Imp"}, start=7, end=14, ner=None),
        ],
    )

    candidates = CandidateGenerator().generate("Даниил проверь текст.")

    candidate = _punctuation_candidate(candidates, "address_comma", "COMMA", "INSERT")

    assert candidate.replacement == ","
    assert candidate.start == candidate.end == len("Даниил")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires == ("syntax", "model")
    assert candidate.gap_index == 0


def test_address_comma_candidate_uses_conservative_lexical_fallback():
    candidates = CandidateGenerator().generate("Коллеги проверим отчёт.")

    candidate = _punctuation_candidate(candidates, "address_comma", "COMMA", "INSERT")

    assert candidate.replacement == ","
    assert candidate.start == candidate.end == len("Коллеги")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires == ("syntax", "model")
    assert candidate.gap_index == 0


def test_homogeneous_comma_candidate_for_repeated_conjunctions_only():
    candidates = CandidateGenerator().generate("Мы купили и чай и кофе.")

    candidate = _punctuation_candidate(candidates, "homogeneous_comma", "COMMA", "INSERT")

    assert candidate.replacement == ","
    assert candidate.start == candidate.end == len("Мы купили и чай")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires == ("syntax", "model")
    assert candidate.gap_index == 3


def test_detached_adverbial_comma_candidate_for_clear_sentence_initial_turnover():
    candidates = CandidateGenerator().generate("Закончив работу мы ушли.")

    candidate = _punctuation_candidate(candidates, "detached_adverbial_comma", "COMMA", "INSERT")

    assert candidate.replacement == ","
    assert candidate.start == candidate.end == len("Закончив работу")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires == ("syntax", "model")
    assert candidate.gap_index == 1


def test_detached_adverbial_comma_includes_prepositional_complement():
    candidates = CandidateGenerator().generate("Продолжив наступление на правительство Греф потребовал ответа.")

    candidate = _punctuation_candidate(candidates, "detached_adverbial_comma", "COMMA", "INSERT")

    assert candidate.start == candidate.end == len("Продолжив наступление на правительство")


def test_detached_adverbial_comma_not_inserted_inside_prepositional_complement():
    text = "Продолжив наступление на правительство, Греф потребовал ответа."
    candidates = _punctuation_candidates(CandidateGenerator().generate(text))

    assert not [
        candidate
        for candidate in candidates
        if candidate.rule_id == "detached_adverbial_comma"
    ]


def test_comparative_turnover_comma_candidate_for_bounded_markers():
    candidates = CandidateGenerator().generate("Он замер будто услышал шум.")

    candidate = _punctuation_candidate(candidates, "comparative_turnover_comma", "COMMA", "INSERT")

    assert candidate.replacement == ","
    assert candidate.start == candidate.end == len("Он замер")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires == ("syntax", "model")
    assert candidate.gap_index == 1


def test_bare_kak_does_not_generate_comparative_comma_candidate():
    candidates = CandidateGenerator().generate("Он работает как инженер.")

    assert not [
        candidate
        for candidate in _punctuation_candidates(candidates)
        if candidate.rule_id == "comparative_turnover_comma" and candidate.label == "COMMA"
    ]


def test_subject_predicate_dash_candidate_for_explicit_eto_pattern():
    candidates = CandidateGenerator().generate("Москва это столица России.")

    candidate = _punctuation_candidate(candidates, "subject_predicate_dash", "DASH", "INSERT")

    assert candidate.replacement == "—"
    assert candidate.start == candidate.end == len("Москва ")
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires == ("syntax", "model")
    assert candidate.gap_index == 0


def test_subject_predicate_dash_candidate_supports_multiword_subject_but_not_discourse_marker():
    positive = CandidateGenerator().generate("Главная задача это проверить пример.")
    negative = CandidateGenerator().generate("Получается это решение подходит группе альфа.")

    candidate = _punctuation_candidate(positive, "subject_predicate_dash", "DASH", "INSERT")

    assert candidate.start == candidate.end == len("Главная задача ")
    assert not [
        item
        for item in negative
        if item.rule_id == "subject_predicate_dash" and item.replacement == "—"
    ]


def test_direct_speech_candidates_are_model_required_and_not_auto_applied():
    source = "Он сказал проект готов."
    candidates = CandidateGenerator().generate(source)

    colon = _punctuation_candidate(candidates, "direct_speech_colon", "COLON", "INSERT")
    quote_open = _punctuation_candidate(candidates, "direct_speech_quotes", "QUOTE_OPEN", "INSERT")
    quote_close = _punctuation_candidate(candidates, "direct_speech_quotes", "QUOTE_CLOSE", "INSERT")

    assert colon.start == colon.end == len("Он сказал")
    assert quote_open.start == quote_open.end == len("Он сказал ")
    assert quote_close.start == quote_close.end == len("Он сказал проект готов")
    for candidate in (colon, quote_open, quote_close):
        assert candidate.mode == "model_required"
        assert candidate.requires_model is True
        assert candidate.requires_scoring is True
        assert candidate.requires == ("syntax", "model")

    result = Corrector().correct(source)
    assert result.corrected_text == source


def test_valid_direct_speech_quotes_are_not_repaired_or_rebalanced():
    candidates = _punctuation_candidates(CandidateGenerator().generate("Он сказал: «Проект готов»."))

    assert not [
        candidate
        for candidate in candidates
        if candidate.rule_id
        in {"direct_speech_colon", "direct_speech_dash", "direct_speech_quotes", "quote_pair_balance", "bracket_pair_balance"}
    ]


def test_quote_and_bracket_balance_candidates_are_model_required_only_for_one_sided_pairs():
    quote_candidates = CandidateGenerator().generate("Он сказал: «Проект готов.")
    bracket_candidates = CandidateGenerator().generate("Проверь документ (черновик.")

    quote_close = _punctuation_candidate(quote_candidates, "quote_pair_balance", "QUOTE_CLOSE", "INSERT")
    bracket_close = _punctuation_candidate(bracket_candidates, "bracket_pair_balance", "BRACKET_CLOSE", "INSERT")

    assert quote_close.start == quote_close.end == len("Он сказал: «Проект готов")
    assert bracket_close.start == bracket_close.end == len("Проверь документ (черновик")
    for candidate in (quote_close, bracket_close):
        assert candidate.mode == "model_required"
        assert candidate.requires_model is True
        assert candidate.requires_scoring is True
        assert candidate.requires == ("model",)


def test_straight_quote_candidates_use_explicit_quote_open_and_close_rule_ids():
    candidates = CandidateGenerator().generate('Он сказал "Проект готов".')

    quote_open = _punctuation_candidate(candidates, "quote_open", "QUOTE_OPEN", "REPLACE")
    quote_close = _punctuation_candidate(candidates, "quote_close", "QUOTE_CLOSE", "REPLACE")

    assert quote_open.source == '"'
    assert quote_open.replacement == "«"
    assert quote_close.source == '"'
    assert quote_close.replacement == "»"
    for candidate in (quote_open, quote_close):
        assert candidate.mode == "model_required"
        assert candidate.requires_model is True
        assert candidate.requires_scoring is True
        assert candidate.requires == ("model",)


def test_colon_dash_semicolon_candidates_are_model_required():
    examples = [
        ("Возьми следующее документы и ключи.", "enumeration_colon", "COLON"),
        ("Он понял одно проект готов.", "explanation_colon", "COLON"),
        ("Начался дождь мы остались дома.", "consequence_dash", "DASH"),
        ("Солнце село стало холодно.", "asyndetic_dash", "DASH"),
        ("Документ готов отчет отправлен.", "semicolon", "SEMICOLON"),
    ]

    for text, rule_id, label in examples:
        candidate = _punctuation_candidate(CandidateGenerator().generate(text), rule_id, label, "INSERT")
        assert candidate.mode == "model_required"
        assert candidate.requires_model is True
        assert candidate.requires_scoring is True
        assert candidate.requires == ("syntax", "model")


def test_punctuation_candidates_do_not_create_duplicate_noise():
    candidates = CandidateGenerator().generate("Мы пришли,, но встреча закончилась.")

    comma_insertions = [
        candidate
        for candidate in _punctuation_candidates(candidates)
        if candidate.action == "INSERT" and candidate.label == "COMMA"
    ]

    assert comma_insertions == []


def _punctuation_candidate(candidates, rule_id: str, label: str, action: str):
    return next(
        candidate
        for candidate in _punctuation_candidates(candidates)
        if candidate.rule_id == rule_id and candidate.label == label and candidate.action == action
    )


def _punctuation_candidates(candidates):
    return [
        candidate
        for candidate in candidates
        if candidate.edit_type in {"punctuation_insert", "punctuation_delete", "punctuation_replace", "final_punctuation"}
    ]
