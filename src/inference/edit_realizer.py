from __future__ import annotations

from src.candidates.candidate_generator import Candidate


def apply_candidate(text: str, candidate: Candidate) -> str:
    if candidate.edit_type == "keep":
        return text
    return text[: candidate.start] + candidate.replacement + text[candidate.end :]


def ensure_final_punctuation(text: str, mark: str = ".") -> str:
    stripped = text.rstrip()
    if not stripped or stripped[-1] in ".!?…":
        return text
    return stripped + mark
