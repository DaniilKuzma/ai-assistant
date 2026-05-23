from __future__ import annotations

from pathlib import Path

from src.candidates.candidate_generator import Candidate
from src.memory.correction_memory import CorrectionMemory, build_memory_from_config
from src.validation.diff_analyzer import Edit


def _candidate(
    text: str,
    *,
    source: str = "так же",
    replacement: str = "также",
    rule_id: str = "context_tak_zhe",
    edit_type: str = "split_join",
) -> Candidate:
    start = text.index(source)
    return Candidate(
        source=source,
        replacement=replacement,
        edit_type=edit_type,
        start=start,
        end=start + len(source),
        rule_id=rule_id,
    )


def _edit(
    text: str,
    *,
    source: str = "сдесь",
    replacement: str = "здесь",
    rule_id: str = "frequent_error_exact",
    edit_type: str = "spelling_replace",
) -> Edit:
    start = text.index(source)
    return Edit(
        source=source,
        replacement=replacement,
        edit_type=edit_type,
        start=start,
        end=start + len(source),
        rule_id=rule_id,
    )


def test_in_memory_remember_lookup_for_candidate() -> None:
    text = "Он сделал так же как брат."
    memory = CorrectionMemory()

    entry = memory.remember_candidate(
        text,
        _candidate(text),
        "accepted",
        doc_id="doc-1",
        metadata={"origin": "ui"},
    )

    match = memory.lookup_candidate(text, _candidate(text), doc_id="doc-1")
    assert match is not None
    assert match.exact is True
    assert match.reason == "exact_context_key"
    assert match.entry == entry
    assert match.entry.decision == "accepted"
    assert match.entry.metadata == {"origin": "ui"}


def test_in_memory_remember_lookup_for_edit() -> None:
    text = "Она пришла сдесь утром."
    memory = CorrectionMemory()

    entry = memory.remember_edit(text, _edit(text), "manual", doc_id="doc-1")

    match = memory.lookup_edit(text, _edit(text), doc_id="doc-1")
    assert match is not None
    assert match.entry == entry
    assert match.entry.source == "сдесь"
    assert match.entry.replacement == "здесь"
    assert match.entry.decision == "manual"


def test_rejected_decision_is_found_in_same_context() -> None:
    text = "Он сделал так же как брат."
    memory = CorrectionMemory()

    memory.remember_candidate(text, _candidate(text), "rejected", doc_id="doc-1")

    match = memory.lookup_candidate(text, _candidate(text), doc_id="doc-1")
    assert match is not None
    assert match.entry.decision == "rejected"


def test_different_replacement_does_not_match() -> None:
    text = "Он сделал так же как брат."
    memory = CorrectionMemory()

    memory.remember_candidate(text, _candidate(text), "rejected", doc_id="doc-1")

    changed = _candidate(text, replacement="так-же")
    assert memory.lookup_candidate(text, changed, doc_id="doc-1") is None


def test_different_rule_id_does_not_match() -> None:
    text = "Он сделал так же как брат."
    memory = CorrectionMemory()

    memory.remember_candidate(text, _candidate(text), "rejected", doc_id="doc-1")

    changed = _candidate(text, rule_id="another_context_rule")
    assert memory.lookup_candidate(text, changed, doc_id="doc-1") is None


def test_different_doc_id_does_not_match() -> None:
    text = "Он сделал так же как брат."
    memory = CorrectionMemory()

    memory.remember_candidate(text, _candidate(text), "rejected", doc_id="doc-1")

    assert memory.lookup_candidate(text, _candidate(text), doc_id="doc-2") is None


def test_jsonl_save_load_round_trip(tmp_path: Path) -> None:
    text = "Он сделал так же как брат."
    storage_path = tmp_path / "nested" / "memory.jsonl"
    memory = CorrectionMemory(storage_path)
    entry = memory.remember_candidate(text, _candidate(text), "ignored", doc_id="doc-1")

    memory.save()

    loaded = CorrectionMemory(storage_path)
    loaded.load()
    match = loaded.lookup_candidate(text, _candidate(text), doc_id="doc-1")
    assert match is not None
    assert match.entry == entry
    assert storage_path.exists()


def test_build_memory_from_config_returns_none_when_disabled(tmp_path: Path) -> None:
    storage_path = tmp_path / "memory.jsonl"

    memory = build_memory_from_config(
        {
            "correction_memory": {
                "enabled": False,
                "storage_path": str(storage_path),
                "context_window_chars": 48,
            }
        }
    )

    assert memory is None
    assert not storage_path.exists()


def test_build_memory_from_config_loads_existing_jsonl_when_enabled(tmp_path: Path) -> None:
    text = "Она пришла сдесь утром."
    storage_path = tmp_path / "memory.jsonl"
    saved = CorrectionMemory(storage_path)
    saved.remember_edit(text, _edit(text), "accepted", doc_id="doc-1")
    saved.save()

    memory = build_memory_from_config(
        {
            "correction_memory": {
                "enabled": True,
                "storage_path": str(storage_path),
                "context_window_chars": 48,
            }
        }
    )

    assert memory is not None
    match = memory.lookup_edit(text, _edit(text), doc_id="doc-1")
    assert match is not None
    assert match.entry.decision == "accepted"


def test_edit_type_aliases_allow_ui_edit_decision_to_match_generated_candidate() -> None:
    text = "Она пришла сдесь утром."
    memory = CorrectionMemory()
    edit = _edit(text, edit_type="spelling_replace")
    candidate = Candidate(
        source="сдесь",
        replacement="здесь",
        edit_type="spelling",
        start=edit.start,
        end=edit.end,
        rule_id=edit.rule_id,
    )

    memory.remember_edit(text, edit, "rejected", doc_id="doc-1")

    match = memory.lookup_candidate(text, candidate, doc_id="doc-1")
    assert match is not None
    assert match.entry.decision == "rejected"


def test_corrupted_jsonl_line_does_not_break_load(tmp_path: Path) -> None:
    text = "Он сделал так же как брат."
    storage_path = tmp_path / "memory.jsonl"
    memory = CorrectionMemory(storage_path)
    memory.remember_candidate(text, _candidate(text), "rejected", doc_id="doc-1")
    memory.save()
    storage_path.write_text(
        storage_path.read_text(encoding="utf-8") + "{bad json\n",
        encoding="utf-8",
    )

    loaded = CorrectionMemory(storage_path)
    loaded.load()

    match = loaded.lookup_candidate(text, _candidate(text), doc_id="doc-1")
    assert match is not None
    assert match.entry.decision == "rejected"
    assert len(loaded.entries()) == 1


def test_normalization_matches_yo_e_and_whitespace_variants() -> None:
    remembered_text = "Он купил ёлку   на рынке."
    lookup_text = "Он купил елку на рынке."
    memory = CorrectionMemory()

    memory.remember_candidate(
        remembered_text,
        _candidate(
            remembered_text,
            source="ёлку",
            replacement="елку",
            rule_id="yo_e_candidate",
            edit_type="spelling",
        ),
        "accepted",
        doc_id="doc-1",
    )

    match = memory.lookup_candidate(
        lookup_text,
        _candidate(
            lookup_text,
            source="елку",
            replacement="ёлку",
            rule_id="yo_e_candidate",
            edit_type="spelling",
        ),
        doc_id="doc-1",
    )
    assert match is not None
    assert match.entry.decision == "accepted"
