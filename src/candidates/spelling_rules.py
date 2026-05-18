from __future__ import annotations

from dataclasses import dataclass

from src.candidates.orthography_rules import OrthographyCandidate, generated_orthography_candidates
from src.rules.base import RuleMode


@dataclass(frozen=True)
class SpellingCandidateSpec:
    replacement: str
    edit_type: str = "spelling"
    confidence: float = 0.95
    requires_model: bool = False
    rule: str = "frequent_errors"
    mode: RuleMode = "deterministic"


def spelling_candidates(word: str) -> list[str]:
    return [candidate.replacement for candidate in spelling_candidate_specs(word)]


def spelling_candidate_specs(word: str) -> list[SpellingCandidateSpec]:
    candidates: list[SpellingCandidateSpec] = []

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
        mode=candidate.mode,
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
