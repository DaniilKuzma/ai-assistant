from __future__ import annotations

from src.grammar_gen.rules.orthography.hyphen_particles import (
    HyphenKoeRule,
    HyphenParticlesRule,
    HyphenPoAdverbRule,
)
from src.grammar_gen.rules.orthography.ne_verb import NeVerbRule
from src.grammar_gen.rules.orthography.takzhe import TakzheRule
from src.grammar_gen.rules.orthography.tozhe import TozheRule
from src.grammar_gen.rules.orthography.tsya_ttsya import TsyaTtsyaRule
from src.grammar_gen.rules.orthography.zato import ZatoRule


ORTHOGRAPHY_RULES = (
    NeVerbRule(),
    TakzheRule(),
    TozheRule(),
    ZatoRule(),
    HyphenParticlesRule(),
    HyphenKoeRule(),
    HyphenPoAdverbRule(),
    TsyaTtsyaRule(),
)


def register_orthography_rules(registry) -> tuple[object, ...]:
    for rule in ORTHOGRAPHY_RULES:
        if registry.get_rule(rule.info.rule_id) is None:
            registry.register_rule(rule)
    return ORTHOGRAPHY_RULES


__all__ = [
    "HyphenKoeRule",
    "HyphenParticlesRule",
    "HyphenPoAdverbRule",
    "NeVerbRule",
    "ORTHOGRAPHY_RULES",
    "TakzheRule",
    "TozheRule",
    "TsyaTtsyaRule",
    "ZatoRule",
    "register_orthography_rules",
]
