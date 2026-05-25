from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.model.encoder import EncoderLoadConfig, ensure_pytorch_transformers_backend, load_encoder, load_tokenizer


LEGACY_NEURAL_CORRECTOR_MESSAGE = (
    "Legacy neural corrector has been removed. Direct runtime corrector will be implemented "
    "in src/runtime in the runtime migration step."
)


@dataclass(frozen=True)
class ModelCandidatePrediction:
    candidate: Any
    score: float
    confidence: float
    rule_id: str = ""


@dataclass(frozen=True)
class ModelPunctuationPrediction:
    gap_index: int
    label: str
    confidence: float
    action: str = "INSERT"
    rule_id: str = ""

    @property
    def word_index(self) -> int:
        return self.gap_index


@dataclass(frozen=True)
class BatchedFeatureModelScores:
    candidate_scores: list[float]
    candidate_confidences: list[float]
    punctuation_label_ids: list[int]
    punctuation_confidences: list[float]
    punctuation_action_ids: list[int]


class CandidateModelBackend(Protocol):
    def score_candidates(self, text: str, candidates: list[Any]) -> list[ModelCandidatePrediction]:
        ...

    def predict_punctuation(self, text: str) -> list[ModelPunctuationPrediction]:
        ...


class TrainedModelCorrector:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(LEGACY_NEURAL_CORRECTOR_MESSAGE)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainedModelCorrector":
        raise RuntimeError(LEGACY_NEURAL_CORRECTOR_MESSAGE)


class TorchCandidateModelBackend:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(LEGACY_NEURAL_CORRECTOR_MESSAGE)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TorchCandidateModelBackend":
        raise RuntimeError(LEGACY_NEURAL_CORRECTOR_MESSAGE)


def _select_candidates(predictions: list[ModelCandidatePrediction], thresholds: dict[str, float]) -> list[Any]:
    raise RuntimeError(LEGACY_NEURAL_CORRECTOR_MESSAGE)


def _select_candidates_with_trace(
    predictions: list[ModelCandidatePrediction],
    thresholds: dict[str, float],
    **kwargs: Any,
) -> tuple[list[Any], list[dict[str, Any]]]:
    raise RuntimeError(LEGACY_NEURAL_CORRECTOR_MESSAGE)


def _apply_candidates(text: str, candidates: list[Any]) -> str:
    raise RuntimeError(LEGACY_NEURAL_CORRECTOR_MESSAGE)


def _apply_punctuation_predictions(
    text: str,
    predictions: list[ModelPunctuationPrediction],
    thresholds: dict[str, float],
    **kwargs: Any,
) -> tuple[str, list[Any]]:
    raise RuntimeError(LEGACY_NEURAL_CORRECTOR_MESSAGE)


def _annotate_decisions_with_validation(decisions: list[dict[str, Any]], edits: list[Any]) -> None:
    raise RuntimeError(LEGACY_NEURAL_CORRECTOR_MESSAGE)


def _prepare_heads_state_dict_for_module(state_dict: dict[str, Any], heads: Any) -> dict[str, Any]:
    raise RuntimeError(LEGACY_NEURAL_CORRECTOR_MESSAGE)


__all__ = [
    "BatchedFeatureModelScores",
    "CandidateModelBackend",
    "EncoderLoadConfig",
    "LEGACY_NEURAL_CORRECTOR_MESSAGE",
    "ModelCandidatePrediction",
    "ModelPunctuationPrediction",
    "TorchCandidateModelBackend",
    "TrainedModelCorrector",
    "ensure_pytorch_transformers_backend",
    "load_encoder",
    "load_tokenizer",
]
