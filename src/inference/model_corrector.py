from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.model.encoder import EncoderLoadConfig, ensure_pytorch_transformers_backend, load_encoder, load_tokenizer
from src.runtime.corrector import Corrector
from src.runtime.neural_backend import DirectNeuralBackend


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


class TrainedModelCorrector(Corrector):
    def __init__(self, neural_backend: Any | None = None, config: dict[str, Any] | None = None) -> None:
        super().__init__(neural_backend=neural_backend, config=config or {})

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainedModelCorrector":
        return cls(neural_backend=DirectNeuralBackend.from_config(config), config=config)


TorchDirectNeuralBackend = DirectNeuralBackend


__all__ = [
    "DirectNeuralBackend",
    "EncoderLoadConfig",
    "ModelPunctuationPrediction",
    "TorchDirectNeuralBackend",
    "TrainedModelCorrector",
    "ensure_pytorch_transformers_backend",
    "load_encoder",
    "load_tokenizer",
]
