from __future__ import annotations

from src.grammar_gen.rules.punctuation.comma_adversative import CommaAdversativeRule
from src.grammar_gen.rules.punctuation.comma_homogeneous import CommaHomogeneousRule
from src.grammar_gen.rules.punctuation.comma_introductory import CommaIntroductoryRule
from src.grammar_gen.rules.punctuation.comma_subordinate import CommaSubordinateRule
from src.grammar_gen.rules.punctuation.dash_subject_predicate import DashSubjectPredicateRule
from src.grammar_gen.rules.punctuation.final_punctuation import FinalPunctuationRule


PUNCTUATION_RULES = (
    CommaSubordinateRule(),
    CommaIntroductoryRule(),
    CommaHomogeneousRule(),
    CommaAdversativeRule(),
    DashSubjectPredicateRule(),
    FinalPunctuationRule(),
)


def register_punctuation_rules(registry) -> tuple[object, ...]:
    for rule in PUNCTUATION_RULES:
        if registry.get_rule(rule.info.rule_id) is None:
            registry.register_rule(rule)
    return PUNCTUATION_RULES


__all__ = [
    "CommaAdversativeRule",
    "CommaHomogeneousRule",
    "CommaIntroductoryRule",
    "CommaSubordinateRule",
    "DashSubjectPredicateRule",
    "FinalPunctuationRule",
    "PUNCTUATION_RULES",
    "register_punctuation_rules",
]
