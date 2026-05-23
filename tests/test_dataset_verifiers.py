from __future__ import annotations

from src.data.dataset_verifiers import (
    semantic_alignment_for_rule,
    verify_dictionary_typo_family,
    verify_punctuation_rule,
)


def test_comma_subordinate_rejects_numeric_comma_and_wrong_family():
    numeric = verify_punctuation_rule("comma_subordinate", "Цена выросла до 12 5 процента.", "Цена выросла до 12,5 процента.")
    generic = verify_punctuation_rule("comma_subordinate", "Проект готов но требует подписи.", "Проект готов, но требует подписи.")

    assert not numeric.semantic_alignment_pass
    assert numeric.reason == "numeric_punctuation_mismatch"
    assert not generic.semantic_alignment_pass
    assert generic.actual_error_family == "comma_conjunction"


def test_comma_subordinate_accepts_safe_subordinate_clause():
    result = verify_punctuation_rule(
        "comma_subordinate",
        "Редактор заметил что документ готов.",
        "Редактор заметил, что документ готов.",
    )

    assert result.semantic_alignment_pass
    assert result.actual_error_family == "comma_subordinate"


def test_dictionary_typo_verifier_checks_exact_family():
    missing = verify_dictionary_typo_family("missing_letter_candidate", "млоко", "молоко")
    swapped = verify_dictionary_typo_family("missing_letter_candidate", "моолко", "молоко")

    assert missing.semantic_alignment_pass
    assert missing.actual_error_family == "missing_letter"
    assert not swapped.semantic_alignment_pass
    assert swapped.actual_error_family != "missing_letter"


def test_semantic_alignment_dispatches_context_pair():
    result = semantic_alignment_for_rule(
        "context_tak_zhe",
        "Он сделал также как коллега.",
        "Он сделал так же как коллега.",
    )

    assert result.semantic_alignment_pass
    assert result.actual_error_family == "context_pair"


def test_detached_participial_accepts_true_participial_phrase():
    result = verify_punctuation_rule(
        "detached_participial_comma",
        "Документ подготовленный комиссией направили в отдел.",
        "Документ, подготовленный комиссией, направили в отдел.",
    )

    assert result.semantic_alignment_pass
    assert result.actual_error_family == "detached_participial_comma"


def test_detached_participial_rejects_source_attribution_comma():
    result = verify_punctuation_rule(
        "detached_participial_comma",
        "По данным штаба отчет, подготовленный комиссией, направили в отдел.",
        "По данным штаба, отчет, подготовленный комиссией, направили в отдел.",
    )

    assert not result.semantic_alignment_pass
    assert result.actual_error_family != "detached_participial_comma"


def test_detached_participial_rejects_kotory_clause():
    result = verify_punctuation_rule(
        "detached_participial_comma",
        "Отчет который подготовленный комиссией, направили в отдел.",
        "Отчет, который подготовленный комиссией, направили в отдел.",
    )

    assert not result.semantic_alignment_pass
    assert result.actual_error_family != "detached_participial_comma"


def test_detached_participial_rejects_date_and_generic_comma():
    date_result = verify_punctuation_rule(
        "detached_participial_comma",
        "В понедельник 12 мая документ, подготовленный комиссией, направили в отдел.",
        "В понедельник, 12 мая документ, подготовленный комиссией, направили в отдел.",
    )
    generic_result = verify_punctuation_rule(
        "detached_participial_comma",
        "Команда проверила отчет и письмо, подготовленное заранее, осталось в архиве.",
        "Команда проверила отчет, и письмо, подготовленное заранее, осталось в архиве.",
    )

    assert not date_result.semantic_alignment_pass
    assert date_result.actual_error_family != "detached_participial_comma"
    assert not generic_result.semantic_alignment_pass
    assert generic_result.actual_error_family != "detached_participial_comma"
