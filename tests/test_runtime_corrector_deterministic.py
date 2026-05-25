from __future__ import annotations

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
