from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.inference.edit_realizer import apply_candidate
from src.inference.postprocess import normalize_spacing
from src.memory.correction_memory import CorrectionMemory, build_memory_from_config
from src.rules.punctuation import apply_punctuation_rules
from src.validation.diff_analyzer import Edit
from src.validation.strict_validator import StrictValidator


MEMORY_SUPPRESS_DECISIONS = frozenset({"rejected", "ignored"})


@dataclass(frozen=True)
class CorrectionResult:
    source_text: str
    corrected_text: str
    edits: list[Edit]
    candidate_decisions: list[dict[str, Any]] = field(default_factory=list)


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
        correction_memory: CorrectionMemory | None = None,
        doc_id: str = "default",
        memory_enabled: bool = True,
        accepted_reuse: bool = True,
        rejected_suppress: bool = True,
    ) -> None:
        self.max_passes = max_passes
        self.mode = mode
        self.candidates = candidate_generator or CandidateGenerator()
        self.correction_memory = correction_memory if memory_enabled else None
        self.doc_id = doc_id
        self.memory_enabled = bool(memory_enabled and correction_memory is not None)
        self.memory_accepted_reuse = bool(accepted_reuse)
        self.memory_rejected_suppress = bool(rejected_suppress)
        self.validator = StrictValidator()
        self.last_candidate_decisions: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Corrector":
        threshold_config = config.get("thresholds", {})
        memory_config = config.get("correction_memory", {}) or {}
        correction_memory = build_memory_from_config(config)
        memory_enabled = correction_memory is not None
        return cls(
            max_passes=int(config.get("decoder", {}).get("max_passes", 3)),
            mode=str(threshold_config.get("mode", "balanced")),
            candidate_generator=CandidateGenerator.from_config(config),
            correction_memory=correction_memory,
            doc_id=str(memory_config.get("doc_id", "default")),
            memory_enabled=memory_enabled,
            accepted_reuse=bool(memory_config.get("accepted_reuse", True)),
            rejected_suppress=bool(memory_config.get("rejected_suppress", True)),
        )

    def correct(self, text: str) -> CorrectionResult:
        self.last_candidate_decisions = []
        proposed, trusted_edits = self._decode_with_trusted_edits(text)
        validation = self.validator.validate(text, proposed, trusted_edits=trusted_edits)
        corrected = validation.apply_accepted()
        return CorrectionResult(text, corrected, validation.edits, candidate_decisions=list(self.last_candidate_decisions))

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
            memory_decision, memory_applied, memory_reason = _memory_decision_for_candidate(
                text=text,
                candidate=candidate,
                correction_memory=self.correction_memory if self.memory_enabled else None,
                doc_id=self.doc_id,
                accepted_reuse=self.memory_accepted_reuse,
                rejected_suppress=self.memory_rejected_suppress,
            )
            suppressed_by_memory = memory_decision in MEMORY_SUPPRESS_DECISIONS and self.memory_rejected_suppress
            if suppressed_by_memory:
                self.last_candidate_decisions.append(
                    _candidate_decision(
                        candidate,
                        selected=False,
                        memory_decision=memory_decision,
                        memory_applied=memory_applied,
                        memory_reason=memory_reason,
                    )
                )
                continue
            shifted = replace(candidate, start=candidate.start + offset, end=candidate.end + offset)
            before = proposed
            proposed = apply_candidate(proposed, shifted)
            offset += len(proposed) - len(before)
            applied.append(candidate)
            self.last_candidate_decisions.append(
                _candidate_decision(
                    candidate,
                    selected=True,
                    memory_decision=memory_decision,
                    memory_applied=memory_applied,
                    memory_reason=memory_reason,
                )
            )

        proposed, punctuation_edits = self._punctuation_pass_with_edits(proposed)
        applied.extend(punctuation_edits)
        return normalize_spacing(proposed), applied

    def _punctuation_pass(self, text: str) -> str:
        return self._punctuation_pass_with_edits(text)[0]

    def _punctuation_pass_with_edits(self, text: str) -> tuple[str, list[Any]]:
        # TODO: Gate future deterministic CandidateGenerator-backed punctuation edits through CorrectionMemory.
        return apply_punctuation_rules(text, allowed_modes={"deterministic"})


def _can_apply_without_model(candidate: Candidate) -> bool:
    if candidate.edit_type == "keep":
        return False
    return not candidate.requires_scoring


def _memory_decision_for_candidate(
    *,
    text: str,
    candidate: Candidate,
    correction_memory: CorrectionMemory | None,
    doc_id: str,
    accepted_reuse: bool,
    rejected_suppress: bool,
) -> tuple[str, bool, str]:
    if correction_memory is None:
        return "", False, ""
    memory_match = correction_memory.lookup_candidate(text, candidate, doc_id=doc_id)
    if memory_match is None:
        return "", False, ""
    decision = memory_match.entry.decision
    if decision in MEMORY_SUPPRESS_DECISIONS and rejected_suppress:
        return decision, True, memory_match.reason
    if decision == "accepted" and accepted_reuse:
        return decision, True, memory_match.reason
    return decision, False, memory_match.reason


def _candidate_decision(
    candidate: Candidate,
    *,
    selected: bool,
    memory_decision: str,
    memory_applied: bool,
    memory_reason: str,
) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "rule_id": candidate.rule_id,
        "source": candidate.source,
        "replacement": candidate.replacement,
        "selected": bool(selected),
        "memory_decision": memory_decision,
        "memory_applied": bool(memory_applied),
        "memory_reason": memory_reason,
    }
