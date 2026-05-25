from __future__ import annotations

import pytest

from src.memory import CorrectionFeedbackService, CorrectionMemory, FeedbackRecord
from src.schema.edits import RuntimeEdit


def _edit(text: str) -> RuntimeEdit:
    source = "сдесь"
    start = text.index(source)
    return RuntimeEdit(
        source=source,
        replacement="здесь",
        edit_type="spelling_replace",
        start=start,
        end=start + len(source),
        rule_id="frequent_error_exact",
    )


def _runtime_edit(text: str) -> RuntimeEdit:
    source = "так же"
    start = text.index(source)
    return RuntimeEdit(
        source=source,
        replacement="также",
        edit_type="split_join",
        start=start,
        end=start + len(source),
        rule_id="context_tak_zhe",
    )


def test_accept_edit_writes_accepted_decision() -> None:
    text = "Она пришла сдесь утром."
    service = CorrectionFeedbackService(CorrectionMemory(), doc_id="doc-1")

    entry = service.accept_edit(text, _edit(text))

    assert entry.decision == "accepted"
    assert entry.doc_id == "doc-1"
    assert entry.source == "сдесь"
    assert entry.replacement == "здесь"


def test_reject_edit_writes_rejected_decision() -> None:
    text = "Она пришла сдесь утром."
    service = CorrectionFeedbackService(CorrectionMemory(), doc_id="doc-1")

    entry = service.reject_edit(text, _edit(text))

    assert entry.decision == "rejected"


def test_ignore_edit_writes_ignored_decision() -> None:
    text = "Она пришла сдесь утром."
    service = CorrectionFeedbackService(CorrectionMemory(), doc_id="doc-1")

    entry = service.ignore_edit(text, _edit(text))

    assert entry.decision == "ignored"


def test_repeated_decision_for_same_key_overwrites_memory_entry() -> None:
    text = "Она пришла сдесь утром."
    memory = CorrectionMemory()
    service = CorrectionFeedbackService(memory, doc_id="doc-1")

    first = service.accept_edit(text, _edit(text), metadata={"origin": "ui"})
    second = service.reject_edit(text, _edit(text), metadata={"origin": "docx"})

    assert first.key == second.key
    assert len(memory.entries()) == 1
    assert memory.entries()[0].decision == "rejected"
    assert memory.entries()[0].metadata == {"origin": "docx"}


def test_invalid_decision_raises_value_error() -> None:
    text = "Она пришла сдесь утром."
    service = CorrectionFeedbackService(CorrectionMemory(), doc_id="doc-1")

    with pytest.raises(ValueError):
        service.remember_edit_decision(text, _edit(text), "skipped")


def test_metadata_is_saved_on_memory_entry() -> None:
    text = "Она пришла сдесь утром."
    service = CorrectionFeedbackService(CorrectionMemory(), doc_id="doc-1")

    entry = service.remember_edit_decision(text, _edit(text), "manual", metadata={"reviewer": "editor"})

    assert entry.decision == "manual"
    assert entry.metadata == {"reviewer": "editor"}


def test_feedback_service_supports_runtime_edit_generic_methods() -> None:
    edit_text = "Она пришла сдесь утром."
    runtime_text = "Он сделал так же как брат."
    memory = CorrectionMemory()
    service = CorrectionFeedbackService(memory, doc_id="doc-1")

    edit_entry = service.remember_edit_decision(edit_text, _edit(edit_text), "accepted")
    runtime_entry = service.remember_edit_decision(runtime_text, _runtime_edit(runtime_text), "rejected")

    assert edit_entry.edit_type == "spelling_replace"
    assert runtime_entry.edit_type == "split_join"
    assert runtime_entry.decision == "rejected"
    assert len(memory.entries()) == 2


def test_edit_shortcuts_reject_non_runtime_objects() -> None:
    text = "Он сделал так же как брат."
    service = CorrectionFeedbackService(CorrectionMemory(), doc_id="doc-1")

    with pytest.raises(TypeError):
        service.accept_edit(text, object())  # type: ignore[arg-type]


def test_feedback_record_is_exported_as_dataclass() -> None:
    record = FeedbackRecord(
        doc_id="doc-1",
        source_text="Она пришла сдесь утром.",
        corrected_text="Она пришла здесь утром.",
        edit_source="сдесь",
        edit_replacement="здесь",
        rule_id="frequent_error_exact",
        edit_type="spelling_replace",
        decision="accepted",
        memory_key="abc123",
        created_at="2026-05-23T00:00:00+00:00",
    )

    assert record.memory_key == "abc123"

