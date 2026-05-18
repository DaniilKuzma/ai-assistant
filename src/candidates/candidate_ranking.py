from __future__ import annotations

from src.candidates.candidate_generator import Candidate


def rank_candidates_for_budget(candidates: list[Candidate], max_candidates: int) -> list[Candidate]:
    if max_candidates <= 0:
        return []

    edits = [candidate for candidate in candidates if candidate.edit_type != "keep"]
    keeps = [candidate for candidate in candidates if candidate.edit_type == "keep"]

    edits.sort(
        key=lambda candidate: (
            candidate.requires_scoring,
            -candidate.confidence,
            candidate.start,
            candidate.end,
            candidate.replacement.lower(),
        )
    )

    selected = edits[:max_candidates]
    remaining = max_candidates - len(selected)
    if remaining > 0:
        selected.extend(keeps[:remaining])

    return selected
