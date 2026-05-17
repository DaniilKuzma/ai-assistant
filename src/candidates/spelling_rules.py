from __future__ import annotations

from dataclasses import dataclass

from src.candidates.frequent_errors import SPLIT_JOIN_WHITELIST, WRONG_TO_CORRECT
from src.candidates.orthography_rules import OrthographyCandidate, generated_orthography_candidates


@dataclass(frozen=True)
class SpellingCandidateSpec:
    replacement: str
    edit_type: str = "spelling"
    confidence: float = 0.95
    requires_model: bool = False
    rule: str = "frequent_errors"


def spelling_candidates(word: str) -> list[str]:
    return [candidate.replacement for candidate in spelling_candidate_specs(word)]


def spelling_candidate_specs(word: str) -> list[SpellingCandidateSpec]:
    lower = word.lower()
    candidates: list[SpellingCandidateSpec] = []
    known = WRONG_TO_CORRECT.get(lower)
    if known:
        edit_type = "split_join" if lower in SPLIT_JOIN_WHITELIST else "spelling"
        candidates.append(SpellingCandidateSpec(known, edit_type=edit_type, confidence=0.95))

    for generated in generated_orthography_candidates(word):
        candidates.append(_from_generated(generated))

    return _deduplicate(candidates)


def _from_generated(candidate: OrthographyCandidate) -> SpellingCandidateSpec:
    return SpellingCandidateSpec(
        candidate.replacement,
        edit_type=candidate.edit_type,
        confidence=candidate.confidence,
        requires_model=candidate.requires_model,
        rule=candidate.rule,
    )


def _deduplicate(candidates: list[SpellingCandidateSpec]) -> list[SpellingCandidateSpec]:
    seen: set[tuple[str, str]] = set()
    result: list[SpellingCandidateSpec] = []
    for candidate in candidates:
        key = (candidate.replacement, candidate.edit_type)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result
