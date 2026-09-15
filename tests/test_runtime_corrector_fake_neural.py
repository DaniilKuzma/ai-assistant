from __future__ import annotations

from dataclasses import dataclass

from src.runtime.corrector import Corrector


@dataclass(frozen=True)
class FakePrediction:
    token_labels: list[str]
    token_confidences: list[float]
    gap_labels: list[str]
    gap_confidences: list[float]
    rule_ids: list[str]
    token_margins: list[float] | None = None
    gap_margins: list[float] | None = None


class FakeDirectNeuralBackend:
    def predict(self, text: str) -> FakePrediction:
        if "незнал" in text:
            return FakePrediction(
                token_labels=["KEEP", "SPLIT_NE_VERB", "KEEP", "KEEP"],
                token_confidences=[1.0, 0.96, 1.0, 1.0],
                gap_labels=["NONE", "NONE", "NONE", "NONE"],
                gap_confidences=[1.0, 1.0, 1.0, 1.0],
                rule_ids=["none", "ne_verb", "none", "none"],
            )
        return FakePrediction(
            token_labels=["KEEP", "KEEP", "KEEP", "KEEP", "KEEP"],
            token_confidences=[1.0, 1.0, 1.0, 1.0, 1.0],
            gap_labels=["NONE", "NONE", "COMMA", "NONE", "NONE"],
            gap_confidences=[1.0, 1.0, 0.95, 1.0, 1.0],
            rule_ids=["none", "none", "comma_subordinate", "none", "none"],
        )


class LowConfidenceBackend(FakeDirectNeuralBackend):
    def predict(self, text: str) -> FakePrediction:
        prediction = super().predict(text)
        if "незнал" in text:
            return FakePrediction(
                token_labels=prediction.token_labels,
                token_confidences=[1.0, 0.2, 1.0, 1.0],
                gap_labels=prediction.gap_labels,
                gap_confidences=prediction.gap_confidences,
                rule_ids=prediction.rule_ids,
            )
        return prediction


class LowMarginBackend(FakeDirectNeuralBackend):
    def predict(self, text: str) -> FakePrediction:
        prediction = super().predict(text)
        if "незнал" in text:
            return FakePrediction(
                token_labels=prediction.token_labels,
                token_confidences=prediction.token_confidences,
                gap_labels=prediction.gap_labels,
                gap_confidences=prediction.gap_confidences,
                rule_ids=prediction.rule_ids,
                token_margins=[1.0, 0.0, 1.0, 1.0],
            )
        return prediction


def test_fake_neural_backend_applies_token_and_gap_labels() -> None:
    corrector = Corrector(
        neural_backend=FakeDirectNeuralBackend(),
        config={"runtime": {"deterministic_first": False}},
    )

    result = corrector.correct("Он незнал что делать")

    assert result.corrected_text == "Он не знал, что делать"
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in result.edits] == [
        ("незнал", "не знал", "ne_verb"),
        ("", ",", "comma_subordinate"),
    ]


def test_corrector_metadata_marks_direct_neural_backend() -> None:
    corrector = Corrector(
        neural_backend=FakeDirectNeuralBackend(),
        config={"runtime": {"deterministic_first": False}},
    )

    result = corrector.correct("РћРЅ РЅРµР·РЅР°Р» С‡С‚Рѕ РґРµР»Р°С‚СЊ")

    assert result.metadata["backend_kind"] == "direct_neural"
    assert result.metadata["model_loaded"] is True


def test_low_confidence_neural_prediction_is_skipped() -> None:
    corrector = Corrector(
        neural_backend=LowConfidenceBackend(),
        config={"runtime": {"deterministic_first": False}},
    )

    result = corrector.correct("Он незнал что делать")

    assert result.corrected_text == "Он незнал что делать"
    assert result.edits == []


def test_corrector_metadata_marks_deterministic_fallback() -> None:
    corrector = Corrector(neural_backend=None, config={"runtime": {"neural_token_edits": True}})

    result = corrector.correct("Р§РёСЃС‚С‹Р№ С‚РµРєСЃС‚.")

    assert result.metadata["backend_kind"] == "deterministic_fallback"
    assert result.metadata["model_loaded"] is False


def test_low_margin_neural_prediction_is_skipped() -> None:
    corrector = Corrector(
        neural_backend=LowMarginBackend(),
        config={"runtime": {"deterministic_first": False, "confidence_thresholds": {"min_margin": 0.1}}},
    )

    result = corrector.correct("Он незнал что делать")

    assert result.corrected_text == "Он незнал что делать"
    assert result.edits == []
