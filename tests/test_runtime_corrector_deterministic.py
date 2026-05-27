from __future__ import annotations

from src.config.load_config import load_config
from src.runtime.corrector import Corrector
from src.schema.edits import CorrectionResult


def test_corrector_without_neural_backend_applies_safe_deterministic_edits() -> None:
    corrector = Corrector(neural_backend=None)

    result = corrector.correct("Кто то непроверил отчет по русски")

    assert isinstance(result, CorrectionResult)
    assert result.corrected_text == "Кто-то не проверил отчет по-русски"
    assert {edit.rule_id for edit in result.edits} >= {
        "hyphen_particles",
        "hyphen_po_adverb",
        "ne_verb",
    }
    assert all(edit.explanation for edit in result.edits)


def test_corrector_does_not_split_lexicalized_ne_verb() -> None:
    corrector = Corrector(neural_backend=None)

    result = corrector.correct("Он ненавидел шум")

    assert result.corrected_text == "Он ненавидел шум"
    assert result.edits == []


def test_corrector_does_not_produce_bad_vo_phrase() -> None:
    corrector = Corrector(neural_backend=None)

    result = corrector.correct("Он пошел во огород")

    assert result.corrected_text == "Он пошел во огород"
    assert "во огород" in result.corrected_text


def test_corrector_normalizes_final_spacing_without_final_dot_by_default() -> None:
    corrector = Corrector(neural_backend=None)

    result = corrector.correct("Он знал ,  что делать")

    assert result.corrected_text == "Он знал, что делать"
    assert result.corrected_text[-1] != "."
    assert any(edit.rule_id == "spacing_normalization" for edit in result.edits)


def test_corrector_can_add_safe_deterministic_final_dot_from_config() -> None:
    corrector = Corrector(
        neural_backend=None,
        config={"runtime": {"deterministic_first": True, "deterministic_final_punctuation": True}},
    )

    result = corrector.correct("Секретарь уточнил текст")

    assert result.corrected_text == "Секретарь уточнил текст."
    assert [(edit.rule_id, edit.replacement) for edit in result.edits] == [("final_punctuation", ".")]


def test_corrector_can_apply_trusted_lexicon_edits_before_neural_backend() -> None:
    config = load_config("configs/config.yaml")
    config["runtime"] = {
        **config.get("runtime", {}),
        "deterministic_lexicon": True,
        "deterministic_final_punctuation": False,
        "neural_token_edits": False,
        "neural_punctuation": False,
    }
    corrector = Corrector(neural_backend=None, config=config)

    cases = {
        "В документе встретилось «фреймворт».": "В документе встретилось «фреймворк».",
        "Пользователь открыл веб интерфейс.": "Пользователь открыл веб-интерфейс.",
        "Географ описал пол Урала в докладе.": "Географ описал пол-Урала в докладе.",
        "Отдел подвёл итоги за полу годие.": "Отдел подвёл итоги за полугодие.",
        "Комиссия слушала доклад в продолжении заседания.": "Комиссия слушала доклад в продолжение заседания.",
        "Документ вернули в следствии ошибки в расчёте.": "Документ вернули вследствие ошибки в расчёте.",
        "Учитель подчеркнул в диктанте форму «безкровный».": "Учитель подчеркнул в диктанте форму «бескровный».",
        "В журнале настройки записали слово «клеющий».": "В журнале настройки записали слово «клеящий».",
        "Мы подошли к ранний весне.": "Мы подошли к ранней весне.",
        "Методист добавил в конспект вариант «синива».": "Методист добавил в конспект вариант «синего».",
        "В учебном задании встретилось слово «смотрет».": "В учебном задании встретилось слово «смотрит».",
    }

    for source, expected in cases.items():
        result = corrector.correct(source)

        assert result.corrected_text == expected
        assert result.edits


def test_trusted_lexicon_edits_respect_forbidden_compound_contexts() -> None:
    config = load_config("configs/config.yaml")
    config["runtime"] = {
        **config.get("runtime", {}),
        "deterministic_lexicon": True,
        "deterministic_final_punctuation": False,
        "neural_token_edits": False,
        "neural_punctuation": False,
    }
    corrector = Corrector(neural_backend=None, config=config)

    result = corrector.correct("В продолжении статьи появился новый пример.")

    assert result.corrected_text == "В продолжении статьи появился новый пример."
    assert result.edits == []


def test_trusted_lexicon_edits_do_not_apply_contextual_endings_without_safe_context() -> None:
    config = load_config("configs/config.yaml")
    config["runtime"] = {
        **config.get("runtime", {}),
        "deterministic_lexicon": True,
        "deterministic_final_punctuation": False,
        "neural_token_edits": False,
        "neural_punctuation": False,
    }
    corrector = Corrector(neural_backend=None, config=config)

    result = corrector.correct("Это ранний пример.")

    assert result.corrected_text == "Это ранний пример."
    assert result.edits == []
