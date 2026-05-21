from src.validation.strict_validator import StrictValidator
from src.candidates.candidate_generator import Candidate
import pytest


def test_validator_accepts_allowed_spelling_and_punctuation_edits():
    validator = StrictValidator()
    trusted = [
        Candidate("незнаю", "не знаю", "split_join", start=2, end=8, confidence=0.99, requires_model=True),
        Candidate("", ".", "final_punctuation", start=19, end=19, confidence=0.99, requires_model=True),
    ]

    result = validator.validate("Я незнаю что делать", "Я не знаю, что делать.", trusted_edits=trusted)

    assert [edit.status for edit in result.edits] == ["accepted", "accepted", "accepted"]
    assert result.apply_accepted() == "Я не знаю, что делать."


def test_validator_rejects_semantic_rewrite_but_keeps_allowed_edits():
    validator = StrictValidator()

    result = validator.validate("Я люблю этот дом", "Я обожаю этот дом.")

    rejected = [edit for edit in result.edits if edit.status == "rejected"]
    assert rejected
    assert result.apply_accepted() == "Я люблю этот дом"


def test_validator_accepts_final_punctuation_only_from_trusted_candidate():
    validator = StrictValidator()
    source = "Проект готов"
    target = "Проект готов."

    untrusted_result = validator.validate(source, target)
    trusted_result = validator.validate(
        source,
        target,
        trusted_edits=[
            Candidate("", ".", "final_punctuation", start=len(source), end=len(source), confidence=0.99, requires_model=True)
        ],
    )

    assert any(
        edit.edit_type == "final_punctuation"
        and edit.status == "rejected"
        and edit.reason == "requires_trusted_candidate"
        for edit in untrusted_result.edits
    )
    assert untrusted_result.apply_accepted() == source
    assert any(edit.edit_type == "final_punctuation" and edit.status == "accepted" for edit in trusted_result.edits)
    assert trusted_result.apply_accepted() == target


def test_validator_rejects_capitalization_outside_sentence_start_scope():
    validator = StrictValidator()

    date_result = validator.validate("14 августа будет встреча", "14 Августа будет встреча.")
    latin_result = validator.validate("G20 призывает к миру", "G20 Призывает к миру.")
    mid_sentence_result = validator.validate("Встреча 14 августа", "Встреча 14 Августа.")

    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in date_result.edits)
    assert not any(edit.edit_type == "final_punctuation" and edit.status == "accepted" for edit in date_result.edits)
    assert date_result.apply_accepted() == "14 августа будет встреча"
    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in latin_result.edits)
    assert not any(edit.edit_type == "final_punctuation" and edit.status == "accepted" for edit in latin_result.edits)
    assert latin_result.apply_accepted() == "G20 призывает к миру"
    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in mid_sentence_result.edits)
    assert not any(edit.edit_type == "final_punctuation" and edit.status == "accepted" for edit in mid_sentence_result.edits)
    assert mid_sentence_result.apply_accepted() == "Встреча 14 августа"


def test_validator_rejects_context_dependent_pairs_without_trusted_model_edit():
    validator = StrictValidator()

    also_result = validator.validate("Он также пришел", "Он так же пришел.")
    to_result = validator.validate("Он пришел чтобы помочь", "Он пришел что бы помочь.")

    assert any(edit.source.lower() == "также" and edit.status == "rejected" for edit in also_result.edits)
    assert also_result.apply_accepted() == "Он также пришел"
    assert any(edit.source.lower() == "чтобы" and edit.status == "rejected" for edit in to_result.edits)
    assert to_result.apply_accepted() == "Он пришел чтобы помочь"


def test_validator_accepts_context_dependent_pair_only_with_high_confidence_and_strict_context():
    validator = StrictValidator(context_pair_threshold=0.98)
    source = "Он сделал также как я"
    target = "Он сделал так же как я"
    trusted = [Candidate("также", "так же", "split_join", start=10, end=15, confidence=0.99, requires_model=True)]

    result = validator.validate(source, target, trusted_edits=trusted)
    low_confidence_result = validator.validate(
        source,
        target,
        trusted_edits=[Candidate("также", "так же", "split_join", start=10, end=15, confidence=0.97, requires_model=True)],
    )

    assert any(edit.source.lower() == "также" and edit.status == "accepted" for edit in result.edits)
    assert result.apply_accepted() == "Он сделал так же как я"
    assert any(edit.source.lower() == "также" and edit.status == "rejected" for edit in low_confidence_result.edits)
    assert low_confidence_result.apply_accepted() == "Он сделал также как я"


def test_validator_applies_partial_acceptance_using_source_spans_without_position_drift():
    validator = StrictValidator()
    trusted = [Candidate("незнаю", "не знаю", "split_join", start=2, end=8, confidence=0.99, requires_model=True)]

    result = validator.validate("Я незнаю и незнаю что делать", "Я не знаю и знаю, что делать", trusted_edits=trusted)

    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in result.edits)
    assert any(edit.edit_type == "split_word" and edit.status == "accepted" for edit in result.edits)
    assert any(edit.edit_type == "punctuation_insert" and edit.status == "accepted" for edit in result.edits)
    assert result.apply_accepted() == "Я не знаю и незнаю, что делать"


def test_prevalidator_protects_url_email_and_numbers():
    validator = StrictValidator()

    spans = validator.pre_validate("Напиши на test@example.com и открой https://example.com 12.05.2026")

    protected_texts = {span.text for span in spans}
    assert "test@example.com" in protected_texts
    assert "https://example.com" in protected_texts
    assert "12.05.2026" in protected_texts


def test_prevalidator_protects_initialisms_graphical_abbreviations_and_technical_ids():
    validator = StrictValidator()

    spans = validator.pre_validate("США РФ НББ ООО АО ИП г. Москва ул. Тверская т.д. т.п. X5 R2D2 АБ12")

    protected = {(span.text, span.kind) for span in spans}
    assert ("США", "abbreviation") in protected
    assert ("РФ", "abbreviation") in protected
    assert ("НББ", "abbreviation") in protected
    assert ("ООО", "abbreviation") in protected
    assert ("АО", "abbreviation") in protected
    assert ("ИП", "abbreviation") in protected
    assert ("г.", "abbreviation") in protected
    assert ("ул.", "abbreviation") in protected
    assert ("т.д.", "abbreviation") in protected
    assert ("т.п.", "abbreviation") in protected
    assert ("X5", "technical_id") in protected
    assert ("R2D2", "technical_id") in protected
    assert ("АБ12", "technical_id") in protected


def test_validator_rejects_punctuation_insert_inside_protected_number_without_auto_final_punctuation():
    validator = StrictValidator()

    result = validator.validate("Евро стоил 4016 руб", "Евро стоил 40,16 руб.")

    comma_edits = [edit for edit in result.edits if edit.edit_type == "punctuation_insert"]
    assert comma_edits
    assert comma_edits[0].status == "rejected"
    assert not any(edit.edit_type == "final_punctuation" and edit.status == "accepted" for edit in result.edits)
    assert result.apply_accepted() == "Евро стоил 4016 руб"


def test_validator_rejects_repeated_comma_colon_semicolon_noise():
    validator = StrictValidator()

    comma_result = validator.validate("Мы пришли, но ушли.", "Мы пришли,, но ушли.")
    colon_result = validator.validate("Он сказал: проверь.", "Он сказал:: проверь.")
    semicolon_result = validator.validate("Первая часть; вторая часть.", "Первая часть;; вторая часть.")

    for result in (comma_result, colon_result, semicolon_result):
        assert any(edit.status == "rejected" and edit.reason == "punctuation_noise" for edit in result.edits)
        assert result.apply_accepted() == result.source


def test_validator_applies_positioned_punctuation_when_unknown_word_change_is_rejected():
    validator = StrictValidator()

    result = validator.validate("Он сказал привет", "Он сказал: пока.")

    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in result.edits)
    assert any(edit.edit_type == "punctuation_insert" and edit.status == "accepted" for edit in result.edits)
    assert result.apply_accepted() == "Он сказал: привет"


def test_validator_rejects_dangerous_tsya_direction_even_with_high_confidence_model_candidate():
    validator = StrictValidator(tsya_threshold=0.98)
    source = "Они могут появиться завтра."
    target = "Они могут появится завтра."
    trusted = [
        Candidate(
            "появиться",
            "появится",
            "spelling",
            start=10,
            end=19,
            confidence=0.99,
            requires_model=True,
            rule_id="tsya_soft_delete",
        )
    ]

    result = validator.validate(source, target, trusted_edits=trusted)

    assert any(
        edit.source == "появиться"
        and edit.status == "rejected"
        and edit.reason == "dangerous_tsya"
        and edit.rule_id == "tsya_soft_delete"
        for edit in result.edits
    )
    assert result.apply_accepted() == source


def test_validator_rejects_finite_tsya_to_infinitive_even_with_high_confidence_model_candidate():
    validator = StrictValidator(tsya_threshold=0.98)
    source = "Он учится каждый день."
    target = "Он учиться каждый день."
    trusted = [
        Candidate(
            "учится",
            "учиться",
            "spelling",
            start=3,
            end=9,
            confidence=0.99,
            requires_model=True,
            rule_id="tsya_soft_insert",
        )
    ]

    result = validator.validate(source, target, trusted_edits=trusted)

    assert any(edit.source == "учится" and edit.status == "rejected" and edit.reason == "dangerous_tsya" for edit in result.edits)
    assert result.apply_accepted() == source


def test_validator_allows_useful_tsya_direction_from_high_confidence_model_candidate():
    validator = StrictValidator(tsya_threshold=0.98)
    source = "Они могут появится завтра."
    target = "Они могут появиться завтра."
    trusted = [
        Candidate(
            "появится",
            "появиться",
            "spelling",
            start=10,
            end=18,
            confidence=0.99,
            requires_model=True,
            rule_id="tsya_soft_insert",
        )
    ]

    result = validator.validate(source, target, trusted_edits=trusted)

    assert any(edit.source == "появится" and edit.status == "accepted" and edit.rule_id == "tsya_soft_insert" for edit in result.edits)
    assert result.apply_accepted() == target


def test_validator_rejects_useful_tsya_direction_without_high_confidence_model_candidate():
    validator = StrictValidator(tsya_threshold=0.98)
    source = "Они могут появится завтра."
    target = "Они могут появиться завтра."
    trusted = [
        Candidate(
            "появится",
            "появиться",
            "spelling",
            start=10,
            end=18,
            confidence=0.97,
            requires_model=True,
            rule_id="tsya_soft_insert",
        )
    ]

    result = validator.validate(source, target, trusted_edits=trusted)

    assert any(edit.source == "появится" and edit.status == "rejected" and edit.reason == "requires_trusted_candidate" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "reason"),
    [
        ("Индекс вырос на 4,71%.", "Индекс вырос на 4,7,1%.", "breaks_percent"),
        ("Компания поставила 1,39 млрд кубометров газа.", "Компания поставила 1,3,9 млрд кубометров газа.", "breaks_number"),
        ("Температура была 10.5.", "Температура была 10,5.", "breaks_number"),
    ],
)
def test_validator_rejects_number_decimal_and_percent_breakage(source, target, reason):
    validator = StrictValidator()

    result = validator.validate(source, target)

    assert any(edit.status == "rejected" and edit.reason == reason for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("Напиши на test@example.com.", "Напиши на test@example,com."),
        ("Сайт https://example.com работает.", "Сайт https://example,com работает."),
    ],
)
def test_validator_rejects_edits_inside_protected_spans_with_specific_reason(source, target):
    validator = StrictValidator()

    result = validator.validate(source, target)

    assert any(edit.status == "rejected" and edit.reason == "protected_span" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("Мы ждали файл…", "Мы ждали файл……"),
        ("Мы ждали файл…", "Мы ждали файл…."),
        ("Мы ждали файл…", "Мы ждали файл……."),
        ("Готово.", "Готово.."),
        ("Документ готов!", "Документ готов!."),
        ("Ты видел отчёт?", "Ты видел отчёт?."),
        ("Правда?", "Правда?!"),
        ("Стоп!", "Стоп!!!?"),
    ],
)
def test_validator_rejects_repeated_punctuation_noise(source, target):
    validator = StrictValidator()

    result = validator.validate(source, target)

    assert any(edit.status == "rejected" and edit.reason == "punctuation_noise" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "Он сказал: «Проект готов».",
            "Он сказал: «Проект готов.",
            Candidate("»", "", "punctuation_delete", start=24, end=25, confidence=0.99, requires_model=True),
        ),
        (
            "Проверь документ (черновик).",
            "Проверь документ черновик).",
            Candidate("(", "", "punctuation_delete", start=16, end=17, confidence=0.99, requires_model=True),
        ),
        (
            "Проверь документ черновик.",
            "Проверь документ (черновик.",
            Candidate("", "(", "punctuation_insert", start=16, end=16, confidence=0.99, requires_model=True),
        ),
    ],
)
def test_validator_rejects_edits_creating_unbalanced_quotes_or_brackets(source, target, trusted):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "unbalanced_pairs" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted", "reason"),
    [
        (
            'В тексте есть "личный кабинет".',
            "В тексте есть «личный кабинет».",
            Candidate('"', "«", "punctuation_replace", start=13, end=14, confidence=0.99999, requires_model=True, rule_id="quote_open"),
            "quote_normalization_requires_policy",
        ),
        (
            "Власти намерены добиваться компенсации.",
            "Власти намеренны добиваться компенсации.",
            Candidate(
                "намерены",
                "намеренны",
                "spelling",
                start=7,
                end=15,
                confidence=0.99999,
                requires_model=True,
                rule_id="n_nn_short_form",
            ),
            "protected_clean_word_form",
        ),
        (
            "Это не случайно важно.",
            "Это неслучайно важно.",
            Candidate(
                "не случайно",
                "неслучайно",
                "split_join",
                start=4,
                end=15,
                confidence=0.99999,
                requires_model=True,
                rule_id="ne_adverb",
            ),
            "unsafe_ne_split_join",
        ),
        (
            "Это небольшой дефицит.",
            "Это не большой дефицит.",
            Candidate(
                "небольшой",
                "не большой",
                "split_join",
                start=4,
                end=13,
                confidence=0.99999,
                requires_model=True,
                rule_id="ne_adjective",
            ),
            "unsafe_ne_split_join",
        ),
        (
            "Получается это решение подходит группе альфа.",
            "Получается — это решение подходит группе альфа.",
            Candidate("", "—", "punctuation_insert", start=10, end=10, confidence=0.99999, requires_model=True, rule_id="subject_predicate_dash"),
            "unsafe_discourse_dash",
        ),
    ],
)
def test_validator_rejects_observed_clean_overcorrection_edits(source, target, trusted, reason):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == reason for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "Он ведет отчет.",
            "Он въедет отчет.",
            Candidate("ведет", "въедет", "spelling", start=3, end=8, confidence=0.999, requires_model=True, rule_id="missing_hard_sign"),
        ),
        (
            "Они были везде.",
            "Они были въезде.",
            Candidate("везде", "въезде", "spelling", start=9, end=14, confidence=0.999, requires_model=True, rule_id="missing_hard_sign"),
        ),
        (
            "Цыгане пришли.",
            "Цигане пришли.",
            Candidate("Цыгане", "Цигане", "spelling", start=0, end=6, confidence=0.999, requires_model=True, rule_id="pattern_цы_ци"),
        ),
        (
            "Большой дом открыт.",
            "Большей дом открыт.",
            Candidate("Большой", "Большей", "spelling", start=0, end=7, confidence=0.999, requires_model=True, rule_id="pattern_шо_ше"),
        ),
        (
            "Поджог расследуют.",
            "Поджег расследуют.",
            Candidate("Поджог", "Поджег", "spelling", start=0, end=6, confidence=0.999, requires_model=True, rule_id="pattern_жо_же"),
        ),
    ],
)
def test_known_source_word_guard_rejects_observed_lexical_false_accepts(source, target, trusted):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "known_source_lexical_guard" for edit in result.edits)
    assert result.apply_accepted() == source


def test_known_source_word_guard_rejects_vedet_to_vedet_with_hard_sign():
    source = "Он ведет отчет."
    target = "Он въедет отчет."
    trusted = Candidate("ведет", "въедет", "spelling", start=3, end=8, confidence=0.999, requires_model=True, rule_id="missing_hard_sign")

    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "known_source_lexical_guard" for edit in result.edits)
    assert result.apply_accepted() == source


def test_known_source_word_guard_rejects_cygane_to_cigane():
    source = "Цыгане пришли."
    target = "Цигане пришли."
    trusted = Candidate("Цыгане", "Цигане", "spelling", start=0, end=6, confidence=0.999, requires_model=True, rule_id="pattern_цы_ци")

    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "known_source_lexical_guard" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "Мы нашли сезд.",
            "Мы нашли съезд.",
            Candidate("сезд", "съезд", "spelling", start=9, end=13, confidence=0.999, requires_model=True, rule_id="missing_hard_sign"),
        ),
        (
            "Новый обьект готов.",
            "Новый объект готов.",
            Candidate("обьект", "объект", "spelling", start=6, end=12, confidence=0.999, requires_model=True, rule_id="soft_to_hard_sign"),
        ),
        (
            "Закрыт подьезд.",
            "Закрыт подъезд.",
            Candidate("подьезд", "подъезд", "spelling", start=7, end=14, confidence=0.999, requires_model=True, rule_id="soft_to_hard_sign"),
        ),
    ],
)
def test_known_source_word_guard_allows_unknown_hard_sign_repairs(source, target, trusted):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "accepted" and edit.rule_id == trusted.rule_id for edit in result.edits)
    assert result.apply_accepted() == target


def test_validator_rejects_direct_speech_dash_inside_closing_quote():
    source = "«Команда справилась» сказала Мария."
    target = "«Команда справилась —» сказала Мария."
    trusted = Candidate("", "—", "punctuation_insert", start=19, end=19, confidence=0.999, requires_model=True, rule_id="direct_speech_dash")

    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "direct_speech_dash_inside_quotes" for edit in result.edits)
    assert result.apply_accepted() == source


def test_validator_allows_trusted_edit_that_repairs_quote_balance():
    source = "Он сказал: «Проект готов."
    target = "Он сказал: «Проект готов»."
    trusted = Candidate("", "»", "punctuation_insert", start=24, end=24, confidence=0.99, requires_model=True)

    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.edit_type == "punctuation_insert" and edit.status == "accepted" for edit in result.edits)
    assert result.apply_accepted() == target


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("Отчет за 2024 г.", "Отчет за 2024 г.."),
        ("См. приложение.", "См приложение."),
        ("Заявка № 12 готова.", "Заявка №, 12 готова."),
    ],
)
def test_validator_rejects_abbreviation_breakage(source, target):
    validator = StrictValidator()

    result = validator.validate(source, target)

    assert any(edit.status == "rejected" and edit.reason == "breaks_abbreviation" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "США согласовали документ.",
            "Сша согласовали документ.",
            Candidate("США", "Сша", "case", start=0, end=3, confidence=0.99, requires_model=True),
        ),
        (
            "НББ согласовал документ.",
            "Нбб согласовал документ.",
            Candidate("НББ", "Нбб", "case", start=0, end=3, confidence=0.99, requires_model=True),
        ),
        (
            "В г. Москва тепло.",
            "В г Москва тепло.",
            Candidate(".", "", "punctuation_delete", start=3, end=4, confidence=0.99, requires_model=True),
        ),
        (
            "См. т.д. и т.п.",
            "См. тд. и тп.",
            Candidate(".", "", "punctuation_delete", start=5, end=6, confidence=0.99, requires_model=True),
        ),
        (
            "Адрес: ул. Тверская.",
            "Адрес: ул Тверская.",
            Candidate(".", "", "punctuation_delete", start=9, end=10, confidence=0.99, requires_model=True),
        ),
    ],
)
def test_validator_rejects_mock_model_edits_that_break_abbreviations(source, target, trusted):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "breaks_abbreviation" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "Модель X5 готова.",
            "Модель x5 готова.",
            Candidate("X5", "x5", "case", start=7, end=9, confidence=0.99, requires_model=True),
        ),
        (
            "Код АБ12 готов.",
            "Код АБ,12 готов.",
            Candidate("", ",", "punctuation_insert", start=6, end=6, confidence=0.99, requires_model=True),
        ),
    ],
)
def test_validator_rejects_mock_model_edits_inside_technical_ids(source, target, trusted):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "protected_span" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "Гостелеком работает.",
            "Ростелеком работает.",
            Candidate("Гостелеком", "Ростелеком", "spelling", start=0, end=10, confidence=0.999, requires_model=True, rule_id="dictionary_fuzzy"),
        ),
        (
            "Дейли пришла.",
            "Лейли пришла.",
            Candidate("Дейли", "Лейли", "spelling", start=0, end=5, confidence=0.999, requires_model=True, rule_id="dictionary_fuzzy"),
        ),
        (
            "УФСБ согласовало документ.",
            "Фсб согласовало документ.",
            Candidate("УФСБ", "Фсб", "spelling", start=0, end=4, confidence=0.999, requires_model=True, rule_id="dictionary_fuzzy"),
        ),
        (
            "Он видел авианалет.",
            "Он видел авиабилет.",
            Candidate("авианалет", "авиабилет", "spelling", start=9, end=18, confidence=0.999, requires_model=True, rule_id="dictionary_fuzzy"),
        ),
        (
            "Тележурналистка пришла.",
            "Тележурналиста пришла.",
            Candidate("Тележурналистка", "Тележурналиста", "spelling", start=0, end=15, confidence=0.999, requires_model=True, rule_id="dictionary_fuzzy"),
        ),
        (
            "По аппеляционным жалобам.",
            "По апелляционными жалобам.",
            Candidate("аппеляционным", "апелляционными", "spelling", start=3, end=16, confidence=0.999, requires_model=True, rule_id="dictionary_fuzzy"),
        ),
    ],
)
def test_validator_rejects_required_risky_lexical_false_positives(source, target, trusted):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.rule_id == trusted.rule_id for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "replacement"),
    [
        ("мир", "миф"),
        ("замок", "звонок"),
    ],
)
def test_dictionary_fuzzy_rejects_known_clean_word_to_known_word(source, replacement):
    text = f"Это {source}."
    start = text.index(source)
    target = text[:start] + replacement + text[start + len(source) :]
    trusted = Candidate(
        source,
        replacement,
        "spelling",
        start=start,
        end=start + len(source),
        confidence=0.999,
        requires_model=True,
        rule_id="dictionary_fuzzy",
    )

    result = StrictValidator().validate(text, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "known_source_lexical_guard" for edit in result.edits)
    assert result.apply_accepted() == text


@pytest.mark.parametrize(
    ("source", "replacement"),
    [
        ("Щеголев", "Щеголяв"),
        ("Марзук", "Марчук"),
        ("Газпром", "Газром"),
    ],
)
def test_dictionary_fuzzy_rejects_name_or_org_like_token(source, replacement):
    text = f"Игорь {source} выступил."
    start = text.index(source)
    target = text[:start] + replacement + text[start + len(source) :]
    trusted = Candidate(
        source,
        replacement,
        "spelling",
        start=start,
        end=start + len(source),
        confidence=0.999,
        requires_model=True,
        rule_id="dictionary_fuzzy",
    )

    result = StrictValidator().validate(text, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "protected_lexical_guard" for edit in result.edits)
    assert result.apply_accepted() == text


@pytest.mark.parametrize(
    ("source", "replacement"),
    [
        ("карова", "корова"),
        ("територия", "территория"),
        ("апеляция", "апелляция"),
        ("група", "группа"),
        ("Обьяснил", "Объяснил"),
    ],
)
def test_dictionary_fuzzy_keeps_obvious_unknown_typpo_allowed(source, replacement):
    text = f"{source} пример." if source[:1].isupper() else f"Это {source}."
    start = text.index(source)
    target = text[:start] + replacement + text[start + len(source) :]
    trusted = Candidate(
        source,
        replacement,
        "spelling",
        start=start,
        end=start + len(source),
        confidence=0.999,
        requires_model=True,
        rule_id="dictionary_fuzzy",
    )

    result = StrictValidator().validate(text, target, trusted_edits=[trusted])

    assert any(edit.status == "accepted" and edit.rule_id == "dictionary_fuzzy" for edit in result.edits)
    assert result.apply_accepted() == target


def test_swapped_letters_rejects_known_clean_source_word():
    source = "Он держит пост."
    target = "Он держит псот."
    trusted = Candidate(
        "пост",
        "псот",
        "spelling",
        start=10,
        end=14,
        confidence=0.999,
        requires_model=True,
        rule_id="swapped_letters_candidate",
    )

    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "known_source_lexical_guard" for edit in result.edits)
    assert result.apply_accepted() == source


def test_swapped_letters_rejects_priemlet_to_primielet_variant():
    source = "Он не приемлет отмену льгот."
    target = "Он не примелет отмену льгот."
    start = source.index("приемлет")
    trusted = Candidate(
        "приемлет",
        "примелет",
        "spelling",
        start=start,
        end=start + len("приемлет"),
        confidence=0.999,
        requires_model=True,
        rule_id="swapped_letters_candidate",
    )

    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "known_source_lexical_guard" for edit in result.edits)
    assert result.apply_accepted() == source


def test_swapped_letters_allows_obvious_unknown_typo_allowed():
    source = "Во дворе стояла коорва."
    target = "Во дворе стояла корова."
    start = source.index("коорва")
    trusted = Candidate(
        "коорва",
        "корова",
        "spelling",
        start=start,
        end=start + len("коорва"),
        confidence=0.999,
        requires_model=True,
        rule_id="swapped_letters_candidate",
    )

    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "accepted" and edit.rule_id == "swapped_letters_candidate" for edit in result.edits)
    assert result.apply_accepted() == target


@pytest.mark.parametrize(
    ("source", "replacement", "reason"),
    [
        ("УФСБ", "Фсб", "breaks_abbreviation"),
        ("РИА", "Риа", "breaks_abbreviation"),
        ("Лукойл", "Ликойл", "protected_lexical_guard"),
    ],
)
def test_acronym_and_capitalized_entity_protection_for_dictionary_fuzzy(source, replacement, reason):
    text = f"{source} сообщило о проекте."
    target = text.replace(source, replacement, 1)
    trusted = Candidate(
        source,
        replacement,
        "spelling",
        start=0,
        end=len(source),
        confidence=0.999,
        requires_model=True,
        rule_id="dictionary_fuzzy",
    )

    result = StrictValidator().validate(text, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == reason for edit in result.edits)
    assert result.apply_accepted() == text


@pytest.mark.parametrize(
    ("source", "replacement", "rule_id"),
    [
        ("заминировании", "ламинировании", "dictionary_fuzzy"),
        ("отлаживанию", "отваживанию", "dictionary_fuzzy"),
        ("приемлет", "примелет", "swapped_letters_candidate"),
    ],
)
def test_observed_core_probable_clean_lexical_regressions_are_rejected(source, replacement, rule_id):
    text = f"В тексте есть {source}."
    start = text.index(source)
    target = text[:start] + replacement + text[start + len(source) :]
    trusted = Candidate(
        source,
        replacement,
        "spelling",
        start=start,
        end=start + len(source),
        confidence=0.999,
        requires_model=True,
        rule_id=rule_id,
    )

    result = StrictValidator().validate(text, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "known_source_lexical_guard" for edit in result.edits)
    assert result.apply_accepted() == text


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "Daewoo Motor Co. распродает",
            "Daewoo Motor Co. Распродает",
            Candidate("распродает", "Распродает", "case", start=17, end=27, confidence=0.999, requires_model=True, rule_id="capitalization_sentence_start"),
        ),
        (
            "А потом … возможно",
            "А потом … Возможно",
            Candidate("возможно", "Возможно", "case", start=10, end=18, confidence=0.999, requires_model=True, rule_id="capitalization_sentence_start"),
        ),
    ],
)
def test_validator_rejects_required_unsafe_sentence_start_capitalization(source, target, trusted):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(
        edit.status == "rejected"
        and edit.rule_id == "capitalization_sentence_start"
        and edit.reason == "abbreviation_sentence_start_capitalization"
        for edit in result.edits
    )
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "Я очень рада, что вам мои посты нравятся :)",
            "Я очень рада, что вам мои посты нравятся :).",
            Candidate("", ".", "final_punctuation", start=43, end=43, confidence=0.999, requires_model=True, rule_id="final_punctuation_default"),
        ),
        (
            "Слезяться глаза и плачет дождь,",
            "Слезяться глаза и плачет дождь,.",
            Candidate("", ".", "final_punctuation", start=31, end=31, confidence=0.999, requires_model=True, rule_id="final_punctuation_default"),
        ),
    ],
)
def test_validator_rejects_required_unsafe_final_dot_insertions(source, target, trusted):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(
        edit.status == "rejected"
        and edit.rule_id == "final_punctuation_default"
        and edit.reason == "unsafe_final_punctuation"
        for edit in result.edits
    )
    assert result.apply_accepted() == source
