import pytest

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.inference.corrector import Corrector
from src.validation.strict_validator import StrictValidator


RISKY_MODEL_MODES = {"candidate_only", "model_required"}

CLEAN_CASES = [
    pytest.param("Они могут появиться завтра.", id="tsya_infinitive_mogut"),
    pytest.param("Он учится каждый день.", id="tsya_finite_uchitsya"),
    pytest.param("Проект должен получиться хорошим.", id="tsya_infinitive_dolzhen"),
    pytest.param("Индекс РТС рухнул на 4,71%.", id="decimal_percent"),
    pytest.param("Компания поставила 1,39 млрд кубометров газа.", id="decimal_number"),
    pytest.param("В 2024 г. проект завершили.", id="year_abbreviation"),
    pytest.param("Мы ждали файл…", id="ellipsis_final"),
    pytest.param("Документ готов!", id="exclamation_final"),
    pytest.param("Ты видел отчёт?", id="question_final"),
    pytest.param("Сайт https://example.com работает.", id="url"),
    pytest.param("Напиши на test@example.com.", id="email"),
    pytest.param("Кто-то пришёл.", id="hyphen_particle"),
    pytest.param("Кое-где были ошибки.", id="hyphen_koe"),
    pytest.param("Он говорит по-русски.", id="hyphen_po_adverb"),
    pytest.param("Он сделал так же, как я.", id="context_tak_zhe"),
    pytest.param("Я тоже это видел.", id="context_tozhe"),
    pytest.param("Что бы ты ни сказал, решение принято.", id="context_chto_by"),
    pytest.param("Чтобы закончить, нужна проверка.", id="context_chtoby"),
    pytest.param("Он сказал, что проект готов.", id="normal_subordinate_comma"),
    pytest.param("Конечно, проект сложный.", id="normal_introductory_comma"),
    pytest.param("Мы пришли, но встреча уже закончилась.", id="normal_conjunction_comma"),
]

DANGEROUS_CANDIDATE_EXPECTATIONS = [
    pytest.param(
        "Они могут появиться завтра.",
        "появиться",
        "появится",
        "tsya_soft_delete",
        "model_required",
        id="infinitive_tsya_delete",
    ),
    pytest.param(
        "Он учится каждый день.",
        "учится",
        "учиться",
        "tsya_soft_insert",
        "model_required",
        id="finite_tsya_insert",
    ),
    pytest.param(
        "Проект должен получиться хорошим.",
        "получиться",
        "получится",
        "tsya_soft_delete",
        "model_required",
        id="modal_infinitive_tsya_delete",
    ),
    pytest.param(
        "Он сделал так же, как я.",
        "так же",
        "также",
        "context_pair",
        "candidate_only",
        id="tak_zhe_join",
    ),
    pytest.param(
        "Я тоже это видел.",
        "тоже",
        "то же",
        "context_pair",
        "candidate_only",
        id="tozhe_split",
    ),
    pytest.param(
        "Что бы ты ни сказал, решение принято.",
        "Что бы",
        "Чтобы",
        "context_pair",
        "candidate_only",
        id="chto_by_join",
    ),
    pytest.param(
        "Чтобы закончить, нужна проверка.",
        "Чтобы",
        "Что бы",
        "context_pair",
        "candidate_only",
        id="chtoby_split",
    ),
]


def _edit_candidates(text: str):
    return [
        candidate
        for candidate in CandidateGenerator().generate(text)
        if candidate.edit_type != "keep" and candidate.source != candidate.replacement
    ]


def _candidate_debug(candidate):
    return (
        candidate.source,
        candidate.replacement,
        candidate.edit_type,
        candidate.rule_id,
        candidate.mode,
        candidate.requires_model,
        candidate.requires_scoring,
    )


@pytest.mark.parametrize("source", CLEAN_CASES)
def test_clean_text_has_no_unsafe_deterministic_candidates(source):
    candidates = _edit_candidates(source)

    unsafe = [
        _candidate_debug(candidate)
        for candidate in candidates
        if (
            candidate.mode not in RISKY_MODEL_MODES
            or not candidate.requires_model
            or not candidate.requires_scoring
        )
    ]

    assert unsafe == []


@pytest.mark.parametrize(
    ("source", "candidate_source", "replacement", "rule_id", "mode"),
    DANGEROUS_CANDIDATE_EXPECTATIONS,
)
def test_dangerous_clean_candidates_are_model_gated(source, candidate_source, replacement, rule_id, mode):
    candidates = _edit_candidates(source)

    matches = [
        candidate
        for candidate in candidates
        if (
            candidate.source == candidate_source
            and candidate.replacement == replacement
            and candidate.rule_id == rule_id
        )
    ]

    assert matches
    assert all(candidate.mode == mode for candidate in matches)
    assert all(candidate.mode in RISKY_MODEL_MODES for candidate in matches)
    assert all(candidate.requires_model for candidate in matches)
    assert all(candidate.requires_scoring for candidate in matches)


@pytest.mark.parametrize("source", CLEAN_CASES)
def test_plain_corrector_keeps_clean_negative_cases_unchanged(source):
    result = Corrector().correct(source)

    assert result.corrected_text == source


def test_dangerous_candidates_can_be_generated_but_not_applied_without_model_scorer():
    source = "Они могут появиться завтра. Он сделал так же, как я."
    candidates = _edit_candidates(source)

    dangerous_pairs = {
        (candidate.source, candidate.replacement, candidate.mode, candidate.requires_scoring)
        for candidate in candidates
        if (candidate.source, candidate.replacement)
        in {("появиться", "появится"), ("так же", "также")}
    }

    assert dangerous_pairs == {
        ("появиться", "появится", "model_required", True),
        ("так же", "также", "candidate_only", True),
    }
    assert Corrector().correct(source).corrected_text == source


@pytest.mark.parametrize(
    ("source", "target", "reason"),
    [
        ("Индекс РТС рухнул на 4,71%.", "Индекс РТС рухнул на 4,7,1%.", "breaks_percent"),
        ("Компания поставила 1,39 млрд кубометров газа.", "Компания поставила 1,3,9 млрд кубометров газа.", "breaks_number"),
        ("В 2024 г. проект завершили.", "В 20,24 г. проект завершили.", "breaks_number"),
        ("Температура была 10.5.", "Температура была 10,5.", "breaks_number"),
    ],
)
def test_validator_rejects_decimal_percent_and_number_edits(source, target, reason):
    result = StrictValidator().validate(source, target)

    assert any(edit.status == "rejected" and edit.reason == reason for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted", "reason"),
    [
        (
            "Индекс РТС рухнул на 4,71%.",
            "Индекс РТС рухнул на 4,,71%.",
            Candidate("", ",", "punctuation_insert", start=22, end=22, confidence=0.99, requires_model=True),
            "breaks_percent",
        ),
        (
            "Компания поставила 1,39 млрд кубометров газа.",
            "Компания поставила 1,,39 млрд кубометров газа.",
            Candidate("", ",", "punctuation_insert", start=20, end=20, confidence=0.99, requires_model=True),
            "breaks_number",
        ),
        (
            "В 2024 г. проект завершили.",
            "В 20,24 г. проект завершили.",
            Candidate("", ",", "punctuation_insert", start=4, end=4, confidence=0.99, requires_model=True),
            "breaks_number",
        ),
    ],
)
def test_validator_rejects_mock_model_edits_inside_numbers(source, target, trusted, reason):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == reason for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("Напиши на test@example.com.", "Напиши на test@example,com."),
        ("Сайт https://example.com работает.", "Сайт https://example,com работает."),
    ],
)
def test_validator_rejects_protected_span_edits(source, target):
    result = StrictValidator().validate(source, target)

    assert any(edit.status == "rejected" and edit.reason == "protected_span" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "Сайт https://example.com работает.",
            "Сайт https://example,com работает.",
            Candidate(".", ",", "punctuation_replace", start=20, end=21, confidence=0.99, requires_model=True),
        ),
        (
            "Напиши на test@example.com.",
            "Напиши на test@example,com.",
            Candidate(".", ",", "punctuation_replace", start=22, end=23, confidence=0.99, requires_model=True),
        ),
    ],
)
def test_validator_rejects_mock_model_edits_inside_url_and_email(source, target, trusted):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "protected_span" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("Мы ждали файл…", "Мы ждали файл……"),
        ("Мы ждали файл…", "Мы ждали файл…."),
        ("Мы ждали файл…", "Мы ждали файл……."),
    ],
)
def test_validator_rejects_ellipsis_noise(source, target):
    result = StrictValidator().validate(source, target)

    assert any(edit.status == "rejected" and edit.reason == "punctuation_noise" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "Мы ждали файл…",
            "Мы ждали файл……",
            Candidate("", "…", "punctuation_insert", start=14, end=14, confidence=0.99, requires_model=True),
        ),
        (
            "Документ готов!",
            "Документ готов!!",
            Candidate("", "!", "punctuation_insert", start=15, end=15, confidence=0.99, requires_model=True),
        ),
    ],
)
def test_validator_rejects_mock_model_repeated_punctuation_noise(source, target, trusted):
    result = StrictValidator().validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "punctuation_noise" for edit in result.edits)
    assert result.apply_accepted() == source


@pytest.mark.parametrize(
    ("source", "target", "trusted"),
    [
        (
            "Они могут появиться завтра.",
            "Они могут появится завтра.",
            Candidate(
                "появиться",
                "появится",
                "spelling",
                start=10,
                end=19,
                confidence=0.99,
                requires_model=True,
                rule_id="tsya_soft_delete",
            ),
        ),
        (
            "Он учится каждый день.",
            "Он учиться каждый день.",
            Candidate(
                "учится",
                "учиться",
                "spelling",
                start=3,
                end=9,
                confidence=0.99,
                requires_model=True,
                rule_id="tsya_soft_insert",
            ),
        ),
        (
            "Проект должен получиться хорошим.",
            "Проект должен получится хорошим.",
            Candidate(
                "получиться",
                "получится",
                "spelling",
                start=14,
                end=24,
                confidence=0.99,
                requires_model=True,
                rule_id="tsya_soft_delete",
            ),
        ),
    ],
)
def test_validator_rejects_dangerous_tsya_prediction_but_keeps_rule_id(source, target, trusted):
    result = StrictValidator(tsya_threshold=0.98).validate(source, target, trusted_edits=[trusted])

    assert any(edit.status == "rejected" and edit.reason == "dangerous_tsya" and edit.rule_id == trusted.rule_id for edit in result.edits)
    assert result.apply_accepted() == source


def test_validator_does_not_blanket_ban_useful_tsya_prediction():
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

    result = StrictValidator(tsya_threshold=0.98).validate(source, target, trusted_edits=trusted)

    assert any(edit.status == "accepted" and edit.rule_id == "tsya_soft_insert" for edit in result.edits)
    assert result.apply_accepted() == target


@pytest.mark.parametrize(
    ("source", "target", "rule_id"),
    [
        ("Кто то пришел.", "Кто-то пришел.", "hyphen_particles"),
        ("Кое где это указано.", "Кое-где это указано.", "hyphen_koe_koy"),
        ("Он говорит по русски.", "Он говорит по-русски.", "hyphen_po_adverbs"),
        ("Он купил пол лимона.", "Он купил пол-лимона.", "pol_polu_compounds"),
    ],
)
def test_validator_rejects_scoring_required_hyphen_edits_without_trusted_candidate(source, target, rule_id):
    result = StrictValidator().validate(source, target)

    assert any(edit.rule_id == rule_id and edit.status == "rejected" and edit.reason == "requires_trusted_candidate" for edit in result.edits)
    assert result.apply_accepted() == source


def test_plain_corrector_does_not_apply_new_hyphen_or_context_candidates_without_scorer():
    corrector = Corrector()
    examples = [
        "Кто то пришёл.",
        "Кое кто ошибся.",
        "Он говорит по русски.",
        "Он купил пол лимона.",
        "Он тоже пришел.",
        "Он пришел чтобы помочь.",
    ]

    for source in examples:
        result = corrector.correct(source)

        assert result.corrected_text == source
