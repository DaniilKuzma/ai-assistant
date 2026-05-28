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


def test_trusted_lexicon_edits_keep_tak_zhe_comparison_with_manner_word() -> None:
    config = load_config("configs/config.yaml")
    config["runtime"] = {
        **config.get("runtime", {}),
        "deterministic_lexicon": True,
        "deterministic_final_punctuation": False,
        "neural_token_edits": False,
        "neural_punctuation": False,
    }
    corrector = Corrector(neural_backend=None, config=config)
    source = "\u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u043e\u0444\u043e\u0440\u043c\u0438\u043b\u0438 \u0442\u0430\u043a \u0436\u0435 \u0430\u043a\u043a\u0443\u0440\u0430\u0442\u043d\u043e, \u043a\u0430\u043a \u0441\u043f\u0440\u0430\u0432\u043a\u0443."

    result = corrector.correct(source)

    assert result.corrected_text == source
    assert result.edits == []


def test_trusted_lexicon_edits_correct_directional_compound_in_noun_phrase() -> None:
    config = load_config("configs/config.yaml")
    config["runtime"] = {
        **config.get("runtime", {}),
        "deterministic_lexicon": True,
        "deterministic_final_punctuation": False,
        "neural_token_edits": False,
        "neural_punctuation": False,
    }
    corrector = Corrector(neural_backend=None, config=config)
    source = (
        "\u0412 \u043e\u0442\u0447\u0451\u0442\u0435 \u043e\u043f\u0438\u0441\u0430\u043d "
        "\u0441\u0435\u0432\u0435\u0440\u043e \u0437\u0430\u043f\u0430\u0434\u043d\u044b\u0439 "
        "\u0440\u0430\u0439\u043e\u043d."
    )

    result = corrector.correct(source)

    assert result.corrected_text == (
        "\u0412 \u043e\u0442\u0447\u0451\u0442\u0435 \u043e\u043f\u0438\u0441\u0430\u043d "
        "\u0441\u0435\u0432\u0435\u0440\u043e-\u0437\u0430\u043f\u0430\u0434\u043d\u044b\u0439 "
        "\u0440\u0430\u0439\u043e\u043d."
    )
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in result.edits] == [
        (
            "\u0441\u0435\u0432\u0435\u0440\u043e \u0437\u0430\u043f\u0430\u0434\u043d\u044b\u0439",
            "\u0441\u0435\u0432\u0435\u0440\u043e-\u0437\u0430\u043f\u0430\u0434\u043d\u044b\u0439",
            "compound_nouns_adjectives",
        )
    ]


def test_trusted_lexicon_edits_correct_safe_n_nn_forms_without_neural_backend() -> None:
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
        "\u0421\u043b\u0435\u0441\u0430\u0440\u044c \u043f\u0440\u0438\u043d\u0451\u0441 "
        "\u043e\u043b\u043e\u0432\u044f\u043d\u044b\u0439 \u043a\u0443\u0431\u043e\u043a.": (
            "\u0421\u043b\u0435\u0441\u0430\u0440\u044c \u043f\u0440\u0438\u043d\u0451\u0441 "
            "\u043e\u043b\u043e\u0432\u044f\u043d\u043d\u044b\u0439 \u043a\u0443\u0431\u043e\u043a."
        ),
        "\u0423\u0447\u0438\u0442\u0435\u043b\u044c \u0432\u044b\u0434\u0435\u043b\u0438\u043b "
        "\u0432\u0435\u0442\u0440\u0435\u043d\u043d\u044b\u0439 \u0434\u0435\u043d\u044c.": (
            "\u0423\u0447\u0438\u0442\u0435\u043b\u044c \u0432\u044b\u0434\u0435\u043b\u0438\u043b "
            "\u0432\u0435\u0442\u0440\u0435\u043d\u044b\u0439 \u0434\u0435\u043d\u044c."
        ),
    }

    for source, expected in cases.items():
        result = corrector.correct(source)

        assert result.corrected_text == expected
        assert [edit.rule_id for edit in result.edits] == ["morpheme_n_nn"]


def test_trusted_lexicon_edits_keep_context_dependent_n_nn_without_context() -> None:
    config = load_config("configs/config.yaml")
    config["runtime"] = {
        **config.get("runtime", {}),
        "deterministic_lexicon": True,
        "deterministic_final_punctuation": False,
        "neural_token_edits": False,
        "neural_punctuation": False,
    }
    corrector = Corrector(neural_backend=None, config=config)
    source = "\u041d\u0430 \u0434\u0430\u0447\u0435 \u0441\u0442\u043e\u044f\u043b \u043a\u0440\u0430\u0448\u0435\u043d\u044b\u0439 \u0437\u0430\u0431\u043e\u0440."

    result = corrector.correct(source)

    assert result.corrected_text == source
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
