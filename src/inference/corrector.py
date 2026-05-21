from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.inference.edit_realizer import apply_candidate
from src.inference.postprocess import normalize_spacing
from src.rules.punctuation import apply_punctuation_rules
from src.validation.diff_analyzer import Edit
from src.validation.strict_validator import StrictValidator


@dataclass(frozen=True)
class CorrectionResult:
    source_text: str
    corrected_text: str
    edits: list[Edit]


class Corrector:
    """Safe inference facade.

    The lightweight path only applies conservative deterministic fallback edits.
    Candidate-only and model-required edits stay candidates until a scorer selects
    them.
    """

    def __init__(
        self,
        max_passes: int = 3,
        mode: str = "balanced",
        candidate_generator: CandidateGenerator | None = None,
    ) -> None:
        self.max_passes = max_passes
        self.mode = mode
        self.candidates = candidate_generator or CandidateGenerator()
        self.validator = StrictValidator()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Corrector":
        threshold_config = config.get("thresholds", {})
        return cls(
            max_passes=int(config.get("decoder", {}).get("max_passes", 3)),
            mode=str(threshold_config.get("mode", "balanced")),
            candidate_generator=CandidateGenerator.from_config(config),
        )

    def correct(self, text: str) -> CorrectionResult:
        proposed, trusted_edits = self._decode_with_trusted_edits(text)
        validation = self.validator.validate(text, proposed, trusted_edits=trusted_edits)
        corrected = validation.apply_accepted()
        return CorrectionResult(text, corrected, validation.edits)

    def _decode_with_trusted_edits(self, text: str) -> tuple[str, list[Any]]:
        current = text
        trusted_edits: list[Any] = []
        for pass_index in range(self.max_passes):
            updated, selected = self._single_pass_with_candidates(current)
            if pass_index == 0:
                trusted_edits.extend(selected)
            if updated == current:
                break
            current = updated
        return current, trusted_edits

    def _single_pass(self, text: str) -> str:
        return self._single_pass_with_candidates(text)[0]

    def _single_pass_with_candidates(self, text: str) -> tuple[str, list[Any]]:
        proposed = text
        offset = 0
        applied: list[Any] = []
        for candidate in self.candidates.generate(text):
            if not _can_apply_without_model(candidate):
                continue
            shifted = candidate.__class__(
                source=candidate.source,
                replacement=candidate.replacement,
                edit_type=candidate.edit_type,
                start=candidate.start + offset,
                end=candidate.end + offset,
                confidence=candidate.confidence,
                requires_model=candidate.requires_model,
                rule_id=candidate.rule_id,
                mode=candidate.mode,
                action=candidate.action,
                label=candidate.label,
                gap_index=candidate.gap_index,
                requires=candidate.requires,
                group=candidate.group,
            )
            before = proposed
            proposed = apply_candidate(proposed, shifted)
            offset += len(proposed) - len(before)
            applied.append(candidate)

        proposed, punctuation_edits = self._punctuation_pass_with_edits(proposed)
        applied.extend(punctuation_edits)
        return normalize_spacing(proposed), applied

    def _punctuation_pass(self, text: str) -> str:
        return self._punctuation_pass_with_edits(text)[0]

    def _punctuation_pass_with_edits(self, text: str) -> tuple[str, list[Any]]:
        return apply_punctuation_rules(text, allowed_modes={"deterministic"})


def _can_apply_without_model(candidate: Candidate) -> bool:
    if candidate.edit_type == "keep":
        return False
    return not candidate.requires_scoring
