import pytest

from src.candidates.candidate_generator import Candidate
from src.config.thresholds import threshold_for_candidate, threshold_for_punctuation_prediction
from src.inference.model_corrector import ModelPunctuationPrediction


def test_exact_rule_id_threshold_override():
    candidate = Candidate("учится", "учиться", "spelling", 0, 6, rule_id="tsya_soft_insert")

    threshold = threshold_for_candidate(
        candidate,
        {
            "tsya_soft_insert_threshold": 0.99,
            "tsya_threshold": 0.97,
            "spelling_threshold": 0.78,
        },
    )

    assert threshold == pytest.approx(0.99)


def test_family_threshold_override():
    candidate = Candidate("учиться", "учится", "spelling", 0, 7, rule_id="tsya_soft_delete")

    threshold = threshold_for_candidate(
        candidate,
        {
            "tsya_threshold": 0.97,
            "spelling_threshold": 0.78,
        },
    )

    assert threshold == pytest.approx(0.97)


def test_edit_type_threshold_fallback_when_rule_id_is_empty():
    candidate = Candidate("жызнь", "жизнь", "spelling", 0, 5, rule_id="")

    threshold = threshold_for_candidate(candidate, {"spelling_threshold": 0.78, "default_threshold": 0.85})

    assert threshold == pytest.approx(0.78)


def test_default_threshold_fallback_when_nothing_matches():
    candidate = Candidate("ошыпка", "ошибка", "unknown_edit", 0, 6)

    threshold = threshold_for_candidate(candidate, {"default_threshold": 0.83})

    assert threshold == pytest.approx(0.83)


def test_punctuation_rule_threshold():
    prediction = ModelPunctuationPrediction(1, "COMMA", 0.9, rule_id="comma_subordinate")

    threshold = threshold_for_punctuation_prediction(
        prediction,
        {
            "comma_subordinate_threshold": 0.88,
            "comma_threshold": 0.95,
            "punctuation_threshold": 0.86,
        },
    )

    assert threshold == pytest.approx(0.88)
