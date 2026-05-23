from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from src.candidates.candidate_generator import Candidate
from src.validation.diff_analyzer import Edit


VALID_DECISIONS = {"accepted", "rejected", "ignored", "manual"}


@dataclass(frozen=True)
class CorrectionMemoryEntry:
    key: str
    decision: str
    doc_id: str
    rule_id: str
    edit_type: str
    source: str
    replacement: str
    left_context: str
    right_context: str
    normalized_context: str
    created_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CorrectionMemoryMatch:
    entry: CorrectionMemoryEntry
    reason: str
    exact: bool


def build_memory_key(
    *,
    doc_id: str,
    rule_id: str,
    edit_type: str,
    source: str,
    replacement: str,
    left_context: str,
    right_context: str,
) -> str:
    payload = {
        "doc_id": _normalize_text(doc_id),
        "rule_id": _normalize_text(rule_id),
        "edit_type": _normalize_text(edit_type),
        "source": _normalize_text(source),
        "replacement": _normalize_text(replacement),
        "left_context": _normalize_text(left_context),
        "right_context": _normalize_text(right_context),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CorrectionMemory:
    def __init__(
        self,
        storage_path: str | Path | None = None,
        enabled: bool = True,
        context_window_chars: int = 48,
    ) -> None:
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self.enabled = bool(enabled)
        self.context_window_chars = max(0, int(context_window_chars))
        self._entries: dict[str, CorrectionMemoryEntry] = {}

    def remember_candidate(
        self,
        text: str,
        candidate: Candidate,
        decision: str,
        doc_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        entry = self._build_entry(text, candidate, decision, doc_id, metadata)
        if self.enabled:
            self._entries[entry.key] = entry
        return entry

    def remember_edit(
        self,
        text: str,
        edit: Edit,
        decision: str,
        doc_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionMemoryEntry:
        entry = self._build_entry(text, edit, decision, doc_id, metadata)
        if self.enabled:
            self._entries[entry.key] = entry
        return entry

    def lookup_candidate(
        self,
        text: str,
        candidate: Candidate,
        doc_id: str = "default",
    ) -> CorrectionMemoryMatch | None:
        return self._lookup(text, candidate, doc_id)

    def lookup_edit(
        self,
        text: str,
        edit: Edit,
        doc_id: str = "default",
    ) -> CorrectionMemoryMatch | None:
        return self._lookup(text, edit, doc_id)

    def load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return

        loaded: dict[str, CorrectionMemoryEntry] = {}
        with self.storage_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    entry = _entry_from_mapping(raw)
                except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                    continue
                loaded[entry.key] = entry
        self._entries = loaded

    def save(self) -> None:
        if self.storage_path is None:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage_path.open("w", encoding="utf-8") as handle:
            for entry in self._entries.values():
                handle.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")

    def entries(self) -> list[CorrectionMemoryEntry]:
        return list(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    def _lookup(self, text: str, item: Candidate | Edit, doc_id: str) -> CorrectionMemoryMatch | None:
        if not self.enabled:
            return None

        context = _context_for_item(text, item, self.context_window_chars)
        key = build_memory_key(
            doc_id=doc_id,
            rule_id=item.rule_id,
            edit_type=item.edit_type,
            source=item.source,
            replacement=item.replacement,
            left_context=context.left,
            right_context=context.right,
        )
        entry = self._entries.get(key)
        if entry is None:
            return None
        return CorrectionMemoryMatch(entry=entry, reason="exact_context_key", exact=True)

    def _build_entry(
        self,
        text: str,
        item: Candidate | Edit,
        decision: str,
        doc_id: str,
        metadata: dict[str, Any] | None,
    ) -> CorrectionMemoryEntry:
        if decision not in VALID_DECISIONS:
            allowed = ", ".join(sorted(VALID_DECISIONS))
            raise ValueError(f"Unsupported correction memory decision: {decision!r}. Expected one of: {allowed}.")

        context = _context_for_item(text, item, self.context_window_chars)
        key = build_memory_key(
            doc_id=doc_id,
            rule_id=item.rule_id,
            edit_type=item.edit_type,
            source=item.source,
            replacement=item.replacement,
            left_context=context.left,
            right_context=context.right,
        )
        normalized_context = _normalize_text(" ".join([context.left, item.source, context.right]))
        return CorrectionMemoryEntry(
            key=key,
            decision=decision,
            doc_id=doc_id,
            rule_id=item.rule_id,
            edit_type=item.edit_type,
            source=item.source,
            replacement=item.replacement,
            left_context=context.left,
            right_context=context.right,
            normalized_context=normalized_context,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class _Context:
    left: str
    right: str


def _context_for_item(text: str, item: Candidate | Edit, window: int) -> _Context:
    start, end = _resolve_span(text, item)
    left_start = max(0, start - window)
    right_end = min(len(text), end + window)
    return _Context(left=text[left_start:start], right=text[end:right_end])


def _resolve_span(text: str, item: Candidate | Edit) -> tuple[int, int]:
    start = int(item.start)
    end = int(item.end)
    if 0 <= start <= end <= len(text):
        return start, end

    source = item.source
    if source:
        found = text.find(source)
        if found >= 0:
            return found, found + len(source)
    return 0, 0


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().replace("ё", "е")).strip()


def _entry_from_mapping(raw: Any) -> CorrectionMemoryEntry:
    if not isinstance(raw, dict):
        raise TypeError("Correction memory entry must be a JSON object.")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    decision = str(raw["decision"])
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Unsupported correction memory decision: {decision!r}.")
    return CorrectionMemoryEntry(
        key=str(raw["key"]),
        decision=decision,
        doc_id=str(raw["doc_id"]),
        rule_id=str(raw["rule_id"]),
        edit_type=str(raw["edit_type"]),
        source=str(raw["source"]),
        replacement=str(raw["replacement"]),
        left_context=str(raw["left_context"]),
        right_context=str(raw["right_context"]),
        normalized_context=str(raw["normalized_context"]),
        created_at=str(raw["created_at"]),
        metadata=metadata,
    )
