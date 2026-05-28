from __future__ import annotations

from pathlib import Path

from src.config.load_config import load_config
from src.runtime.corrector import Corrector
from src.runtime.neural_backend import DirectPrediction
from src.runtime.scope_guard import ScopeGuard
from src.runtime.tokenization import tokenize_runtime_words
from src.schema import RuntimeEdit


ROOT = Path(__file__).resolve().parents[1]


def test_fake_neural_backend_corrects_quotation_dialogue_casing_and_semantic_examples() -> None:
    corrector = Corrector(neural_backend=_FakeNewLayersBackend(), config=_runtime_config())

    assert corrector.correct("Проект готов сказал редактор.").corrected_text == (
        "«Проект готов», — сказал редактор."
    )
    assert corrector.correct("иван петров проверил отчёт.").corrected_text == (
        "Иван Петров проверил отчёт."
    )
    assert corrector.correct("Редактор спросил на счёт оплаты.").corrected_text == (
        "Редактор спросил насчёт оплаты."
    )


def test_fake_neural_backend_keeps_semantic_hard_negative_and_formal_you_guard_unchanged() -> None:
    corrector = Corrector(neural_backend=_FakeNewLayersBackend(), config=_runtime_config())

    assert corrector.correct("Студент положил деньги на счёт.").corrected_text == (
        "Студент положил деньги на счёт."
    )
    assert corrector.correct("Редактор проверил ваш документ.").corrected_text == (
        "Редактор проверил ваш документ."
    )


def test_scope_guard_accepts_punctuation_and_casing_but_rejects_semantic_word_insertion() -> None:
    guard = ScopeGuard()

    assert guard.validate_edit(
        "Проект готов сказал редактор.",
        RuntimeEdit(0, 0, "", "«", "punctuation", "quotation_dialogue", 0.99),
    )
    assert guard.validate_edit(
        "иван петров проверил отчёт.",
        RuntimeEdit(0, 4, "иван", "Иван", "casing", "casing_person_names", 0.99),
    )
    assert not guard.validate_edit(
        "Редактор спросил оплату.",
        RuntimeEdit(16, 16, "", "насчёт ", "spelling", "semantic_derived_prepositions", 0.99),
    )


class _FakeNewLayersBackend:
    def predict(self, text: str) -> DirectPrediction:
        tokens = tokenize_runtime_words(text)
        prediction = _default_prediction(len(tokens))

        if text == "Проект готов сказал редактор.":
            prediction.boundary_before_labels[0] = "INSERT_OPEN_QUOTE"
            prediction.boundary_before_confidences[0] = 0.99
            prediction.boundary_after_labels[1] = "INSERT_CLOSE_QUOTE"
            prediction.boundary_after_confidences[1] = 0.99
            prediction.rule_ids[0] = "dialogue_speech_before_author"
            prediction.rule_ids[1] = "dialogue_speech_before_author"
            return prediction

        if text == "«Проект готов» сказал редактор.":
            prediction.gap_labels[1] = "COMMA_DASH"
            prediction.gap_confidences[1] = 0.99
            prediction.rule_ids[1] = "dialogue_speech_before_author"
            return prediction

        if text == "иван петров проверил отчёт.":
            prediction.token_labels[0] = "CAPITALIZE"
            prediction.token_confidences[0] = 0.99
            prediction.rule_ids[0] = "casing_person_names"
            prediction.token_labels[1] = "CAPITALIZE"
            prediction.token_confidences[1] = 0.99
            prediction.rule_ids[1] = "casing_person_names"
            return prediction

        if text == "Редактор спросил на счёт оплаты.":
            prediction.token_labels[2] = "SPAN_REPLACE_BY_LEXICON"
            prediction.token_confidences[2] = 0.99
            prediction.rule_ids[2] = "semantic_derived_prepositions"
            return prediction

        return prediction


def _default_prediction(token_count: int) -> DirectPrediction:
    return DirectPrediction(
        token_labels=["KEEP"] * token_count,
        token_confidences=[1.0] * token_count,
        gap_labels=["NONE"] * token_count,
        gap_confidences=[1.0] * token_count,
        rule_ids=["none"] * token_count,
        token_margins=[1.0] * token_count,
        gap_margins=[1.0] * token_count,
        boundary_before_labels=["NONE"] * token_count,
        boundary_before_confidences=[1.0] * token_count,
        boundary_before_margins=[1.0] * token_count,
        boundary_after_labels=["NONE"] * token_count,
        boundary_after_confidences=[1.0] * token_count,
        boundary_after_margins=[1.0] * token_count,
    )


def _runtime_config() -> dict:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["runtime"] = {
        **config.get("runtime", {}),
        "deterministic_lexicon": True,
        "deterministic_final_punctuation": False,
        "neural_token_edits": True,
        "neural_punctuation": True,
    }
    return config
