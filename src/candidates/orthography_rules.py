from __future__ import annotations

from dataclasses import dataclass

from src.rules.base import RuleContext, RuleMode
from src.rules.registry import orthography_rules


@dataclass(frozen=True)
class OrthographyCandidate:
    replacement: str
    edit_type: str = "spelling"
    confidence: float = 0.92
    requires_model: bool = False
    rule: str = ""
    mode: RuleMode = "deterministic"


def generated_orthography_candidates(word: str) -> list[OrthographyCandidate]:
    candidates: list[OrthographyCandidate] = []
    context = RuleContext()
    for rule in orthography_rules():
        generator = getattr(rule, "generate_candidates", getattr(rule, "generate", None))
        if rule.spec.scope != "token" or generator is None:
            continue
        for candidate in generator(word, context):
            candidates.append(
                OrthographyCandidate(
                    candidate.replacement,
                    edit_type=candidate.edit_type,
                    confidence=candidate.confidence,
                    requires_model=candidate.requires_model,
                    rule=candidate.rule_id,
                    mode=candidate.mode,
                )
            )
    return _deduplicate(candidates, word.lower())


def _deduplicate(candidates: list[OrthographyCandidate], source: str) -> list[OrthographyCandidate]:
    seen: set[tuple[str, str]] = set()
    result: list[OrthographyCandidate] = []
    for candidate in candidates:
        key = (candidate.replacement, candidate.edit_type)
        if candidate.replacement == source or key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result
