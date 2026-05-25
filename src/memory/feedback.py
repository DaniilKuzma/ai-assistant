from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.memory.correction_memory import CorrectionMemory, CorrectionMemoryEntry, VALID_DECISIONS
from src.schema.edits import RuntimeEdit


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
        edit: RuntimeEdit,
        decision: str,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        _validate_edit(edit)
        _validate_decision(decision)
        return self.memory.remember_edit(source_text, edit, decision, doc_id=self.doc_id, metadata=metadata)

    def accept_edit(
        self,
        source_text: str,
        edit: RuntimeEdit,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        return self.remember_edit_decision(source_text, edit, "accepted", metadata=metadata)

    def reject_edit(
        self,
        source_text: str,
        edit: RuntimeEdit,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        return self.remember_edit_decision(source_text, edit, "rejected", metadata=metadata)

    def ignore_edit(
        self,
        source_text: str,
        edit: RuntimeEdit,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        return self.remember_edit_decision(source_text, edit, "ignored", metadata=metadata)


def _validate_decision(decision: str) -> None:
    if decision in VALID_DECISIONS:
        return
    allowed = ", ".join(sorted(VALID_DECISIONS))
    raise ValueError(f"Unsupported correction feedback decision: {decision!r}. Expected one of: {allowed}.")


def _validate_edit(edit: RuntimeEdit) -> None:
    if not isinstance(edit, RuntimeEdit):
        raise TypeError("CorrectionFeedbackService edit methods accept only RuntimeEdit objects.")
