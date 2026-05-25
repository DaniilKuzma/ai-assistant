from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from src.schema import GeneratedExample

if TYPE_CHECKING:
    from src.grammar_gen.builders import GrammarBuilder
    from src.grammar_gen.randomness import RandomSource
    from src.grammar_gen.realizer import Realizer


class GenerationMode(str, Enum):
    POSITIVE = "positive"
    HARD_NEGATIVE = "hard_negative"
    CLEAN_IDENTITY = "clean_identity"


@dataclass(frozen=True)
class RuleInfo:
    rule_id: str
    family: str
    description: str
    explanation: str
    deterministic: bool = False
    weight: float = 1.0


class RuleProgram(ABC):
    info: RuleInfo
    supported_modes: tuple[GenerationMode, ...]

    @abstractmethod
    def generate(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
        mode: GenerationMode,
    ) -> GeneratedExample:
        ...

    def can_generate(self, mode: GenerationMode) -> bool:
        return mode in self.supported_modes


__all__ = ["GenerationMode", "RuleInfo", "RuleProgram"]
