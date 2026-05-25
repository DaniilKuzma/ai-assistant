from __future__ import annotations

from src.runtime.edit_realizer import apply_gap_labels, apply_runtime_edits, apply_token_edit_labels


CLOSING_FINAL_WRAPPERS = frozenset("\"'»”)]}")


def ensure_final_punctuation(text: str, mark: str = ".") -> str:
    stripped = text.rstrip()
    if not stripped or _has_sentence_final_punctuation(stripped):
        return text
    return stripped + mark


def _has_sentence_final_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in CLOSING_FINAL_WRAPPERS:
        stripped = stripped[:-1].rstrip()
    return bool(stripped and stripped[-1] in ".!?…")


__all__ = ["apply_gap_labels", "apply_runtime_edits", "apply_token_edit_labels", "ensure_final_punctuation"]
