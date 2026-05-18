from __future__ import annotations

from src.candidates.candidate_generator import Candidate


CLOSING_FINAL_WRAPPERS = frozenset("\"'»”)]}")


def apply_candidate(text: str, candidate: Candidate) -> str:
    if candidate.edit_type == "keep":
        return text
    return text[: candidate.start] + candidate.replacement + text[candidate.end :]


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
