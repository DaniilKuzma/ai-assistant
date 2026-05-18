from src.validation.strict_validator import StrictValidator
from src.candidates.candidate_generator import Candidate
import pytest


def test_validator_accepts_allowed_spelling_and_punctuation_edits():
    validator = StrictValidator()

    result = validator.validate("Я незнаю что делать", "Я не знаю, что делать.")

    assert [edit.status for edit in result.edits] == ["accepted", "accepted", "accepted"]
    assert result.apply_accepted() == "Я не знаю, что делать."


def test_validator_rejects_semantic_rewrite_but_keeps_allowed_edits():
    validator = StrictValidator()

    result = validator.validate("Я люблю этот дом", "Я обожаю этот дом.")

    rejected = [edit for edit in result.edits if edit.status == "rejected"]
    assert rejected
    assert result.apply_accepted() == "Я люблю этот дом."


def test_validator_rejects_capitalization_outside_sentence_start_scope():
    validator = StrictValidator()

    date_result = validator.validate("14 августа будет встреча", "14 Августа будет встреча.")
    latin_result = validator.validate("G20 призывает к миру", "G20 Призывает к миру.")
    mid_sentence_result = validator.validate("Встреча 14 августа", "Встреча 14 Августа.")

    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in date_result.edits)
    assert any(edit.edit_type == "final_punctuation" and edit.status == "accepted" for edit in date_result.edits)
    assert date_result.apply_accepted() == "14 августа будет встреча."
    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in latin_result.edits)
    assert any(edit.edit_type == "final_punctuation" and edit.status == "accepted" for edit in latin_result.edits)
    assert latin_result.apply_accepted() == "G20 призывает к миру."
    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in mid_sentence_result.edits)
    assert any(edit.edit_type == "final_punctuation" and edit.status == "accepted" for edit in mid_sentence_result.edits)
    assert mid_sentence_result.apply_accepted() == "Встреча 14 августа."


def test_validator_rejects_context_dependent_pairs_without_trusted_model_edit():
    validator = StrictValidator()

    also_result = validator.validate("Он также пришел", "Он так же пришел.")
    to_result = validator.validate("Он пришел чтобы помочь", "Он пришел что бы помочь.")

    assert any(edit.source.lower() == "также" and edit.status == "rejected" for edit in also_result.edits)
    assert also_result.apply_accepted() == "Он также пришел."
    assert any(edit.source.lower() == "чтобы" and edit.status == "rejected" for edit in to_result.edits)
    assert to_result.apply_accepted() == "Он пришел чтобы помочь."


def test_validator_accepts_context_dependent_pair_only_with_high_confidence_and_strict_context():
    validator = StrictValidator(context_pair_threshold=0.98)
    source = "Он сделал также как я"
    target = "Он сделал так же как я."
    trusted = [Candidate("также", "так же", "split_join", start=10, end=15, confidence=0.99, requires_model=True)]

    result = validator.validate(source, target, trusted_edits=trusted)
    low_confidence_result = validator.validate(
        source,
        target,
        trusted_edits=[Candidate("также", "так же", "split_join", start=10, end=15, confidence=0.97, requires_model=True)],
    )

    assert any(edit.source.lower() == "также" and edit.status == "accepted" for edit in result.edits)
    assert result.apply_accepted() == "Он сделал так же как я."
    assert any(edit.source.lower() == "также" and edit.status == "rejected" for edit in low_confidence_result.edits)
    assert low_confidence_result.apply_accepted() == "Он сделал также как я."


def test_validator_applies_partial_acceptance_using_source_spans_without_position_drift():
    validator = StrictValidator()

    result = validator.validate("Я незнаю и незнаю что делать", "Я не знаю и знаю, что делать.")

    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in result.edits)
    assert any(edit.edit_type == "split_word" and edit.status == "accepted" for edit in result.edits)
    assert any(edit.edit_type == "punctuation_insert" and edit.status == "accepted" for edit in result.edits)
    assert result.apply_accepted() == "Я не знаю и незнаю, что делать."


def test_prevalidator_protects_url_email_and_numbers():
    validator = StrictValidator()

    spans = validator.pre_validate("Напиши на test@example.com и открой https://example.com 12.05.2026")

    protected_texts = {span.text for span in spans}
    assert "test@example.com" in protected_texts
    assert "https://example.com" in protected_texts
    assert "12.05.2026" in protected_texts


def test_validator_rejects_punctuation_insert_inside_protected_number_but_keeps_final_punctuation():
    validator = StrictValidator()

    result = validator.validate("Евро стоил 4016 руб", "Евро стоил 40,16 руб.")

    comma_edits = [edit for edit in result.edits if edit.edit_type == "punctuation_insert"]
    assert comma_edits
    assert comma_edits[0].status == "rejected"
    assert result.apply_accepted() == "Евро стоил 4016 руб."


def test_validator_applies_positioned_punctuation_when_unknown_word_change_is_rejected():
    validator = StrictValidator()

    result = validator.validate("Он сказал привет", "Он сказал: пока.")

    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in result.edits)
    assert any(edit.edit_type == "punctuation_insert" and edit.status == "accepted" for edit in result.edits)
    assert result.apply_accepted() == "Он сказал: привет."


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

    assert any(edit.source == "появится" and edit.status == "rejected" and edit.reason == "dangerous_tsya" for edit in result.edits)
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
        ("Готово.", "Готово.."),
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
