from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.candidates.candidate_generator import Candidate
from src.memory.correction_memory import CorrectionMemory, CorrectionMemoryEntry, VALID_DECISIONS
from src.validation.diff_analyzer import Edit


@dataclass(frozen=True)
class FeedbackRecord:
    doc_id: str
    source_text: str
    corrected_text: str
    edit_source: str
    edit_replacement: str
    rule_id: str
    edit_type: str
    decision: str
    memory_key: str
    created_at: str


class CorrectionFeedbackService:
    def __init__(self, memory: CorrectionMemory, doc_id: str = "default") -> None:
        self.memory = memory
        self.doc_id = doc_id

    def remember_edit_decision(
        self,
        source_text: str,
        edit: Edit,
        decision: str,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        _validate_edit(edit)
        _validate_decision(decision)
        return self.memory.remember_edit(source_text, edit, decision, doc_id=self.doc_id, metadata=metadata)

    def remember_candidate_decision(
        self,
        source_text: str,
        candidate: Candidate,
        decision: str,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        _validate_candidate(candidate)
        _validate_decision(decision)
        return self.memory.remember_candidate(source_text, candidate, decision, doc_id=self.doc_id, metadata=metadata)

    def accept_edit(
        self,
        source_text: str,
        edit: Edit,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        return self.remember_edit_decision(source_text, edit, "accepted", metadata=metadata)

    def reject_edit(
        self,
        source_text: str,
        edit: Edit,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        return self.remember_edit_decision(source_text, edit, "rejected", metadata=metadata)

    def ignore_edit(
        self,
        source_text: str,
        edit: Edit,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        return self.remember_edit_decision(source_text, edit, "ignored", metadata=metadata)


def _validate_decision(decision: str) -> None:
    if decision in VALID_DECISIONS:
        return
    allowed = ", ".join(sorted(VALID_DECISIONS))
    raise ValueError(f"Unsupported correction feedback decision: {decision!r}. Expected one of: {allowed}.")


def _validate_edit(edit: Edit) -> None:
    if not isinstance(edit, Edit):
        raise TypeError("CorrectionFeedbackService edit methods accept only Edit objects.")


def _validate_candidate(candidate: Candidate) -> None:
    if not isinstance(candidate, Candidate):
        raise TypeError("CorrectionFeedbackService candidate methods accept only Candidate objects.")
