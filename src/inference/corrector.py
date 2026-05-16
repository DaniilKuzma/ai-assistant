from __future__ import annotations

from dataclasses import dataclass
import re

from src.candidates.candidate_generator import CandidateGenerator
from src.inference.edit_realizer import apply_candidate, ensure_final_punctuation
from src.inference.iterative_decoder import run_until_stable
from src.inference.postprocess import normalize_spacing
from src.validation.diff_analyzer import Edit
from src.validation.strict_validator import StrictValidator


@dataclass(frozen=True)
class CorrectionResult:
    source_text: str
    corrected_text: str
    edits: list[Edit]


class Corrector:
    """Safe inference facade.

    The lightweight path uses deterministic whitelist scoring. A trained model can
    be wired in later to choose among the same bounded candidates.
    """

    def __init__(self, max_passes: int = 3, mode: str = "balanced") -> None:
        self.max_passes = max_passes
        self.mode = mode
        self.candidates = CandidateGenerator()
        self.validator = StrictValidator()

    def correct(self, text: str) -> CorrectionResult:
        proposed = run_until_stable(text, self._single_pass, self.max_passes)
        validation = self.validator.validate(text, proposed)
        corrected = validation.apply_accepted()
        return CorrectionResult(text, corrected, validation.edits)

    def _single_pass(self, text: str) -> str:
        proposed = text
        offset = 0
        for candidate in self.candidates.generate(text):
            if candidate.edit_type == "keep":
                continue
            shifted = candidate.__class__(
                source=candidate.source,
                replacement=candidate.replacement,
                edit_type=candidate.edit_type,
                start=candidate.start + offset,
                end=candidate.end + offset,
                confidence=candidate.confidence,
            )
            before = proposed
            proposed = apply_candidate(proposed, shifted)
            offset += len(proposed) - len(before)

        proposed = self._punctuation_pass(proposed)
        return normalize_spacing(proposed)

    def _punctuation_pass(self, text: str) -> str:
        text = re.sub(r"\b(не знаю|думаю|считаю) что\b", r"\1, что", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(во-первых|во-вторых|в-третьих)\s+(?!,)", r"\1, ", text, flags=re.IGNORECASE)
        text = ensure_final_punctuation(text, ".")
        return text
