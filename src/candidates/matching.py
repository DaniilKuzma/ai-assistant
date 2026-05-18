from __future__ import annotations

from src.candidates.candidate_generator import Candidate
from src.validation.diff_analyzer import Edit


WORD_EDIT_TYPES = {"spelling_replace", "split_word", "join_words", "hyphen_change", "case_change"}


def candidate_overlaps_edit(candidate: Candidate, edit: Edit) -> bool:
    if edit.edit_type not in WORD_EDIT_TYPES or edit.start < 0 or edit.end < 0:
        return False
    return candidate.start < edit.end and edit.start < candidate.end


def candidate_matches_edit(candidate: Candidate, edit: Edit) -> bool:
    expected_type = candidate_edit_type_for_labels(candidate.edit_type)
    if expected_type == "split_word" and edit.edit_type == "join_words":
        expected_type = "join_words"
    if edit.edit_type != expected_type:
        return False
    if edit.edit_type == "case_change":
        return candidate.start <= edit.start < candidate.end and candidate.replacement.startswith(edit.replacement)
    if edit.start >= 0 and edit.end >= 0 and (candidate.start != edit.start or candidate.end != edit.end):
        return False
    return candidate.replacement.lower() == edit.replacement.lower()


def candidate_edit_type_for_labels(candidate_type: str) -> str:
    return {
        "split_join": "split_word",
        "hyphen": "hyphen_change",
        "case": "case_change",
        "spelling": "spelling_replace",
    }.get(candidate_type, candidate_type)
