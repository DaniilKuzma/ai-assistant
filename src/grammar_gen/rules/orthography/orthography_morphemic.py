from __future__ import annotations

from functools import lru_cache

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.orthography_gen.compiler import OrthographicScenarioCompiler
from src.schema import GeneratedExample


class OrthographyMorphemicRule(RuleProgram):
    def __init__(
        self,
        rule_id: str,
        description: str,
        supported_modes: tuple[GenerationMode, ...],
    ) -> None:
        self.info = RuleInfo(
            rule_id=rule_id,
            family="orthography_morphemic",
            description=description,
            explanation=rule_id,
            deterministic=True,
            weight=1.0,
        )
        self.supported_modes = supported_modes

    def generate(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
        mode: GenerationMode,
    ) -> GeneratedExample:
        del builder, realizer
        return _compiler().compile_example(self.info.rule_id, mode, rng)


@lru_cache(maxsize=1)
def _compiler() -> OrthographicScenarioCompiler:
    return OrthographicScenarioCompiler.default()


def _morphemic_rules_from_specs() -> tuple[OrthographyMorphemicRule, ...]:
    compiler = _compiler()
    rules: list[OrthographyMorphemicRule] = []
    for rule_id in sorted(compiler.specs):
        modes = tuple(
            mode
            for mode in (GenerationMode.POSITIVE, GenerationMode.HARD_NEGATIVE)
            if compiler.has_mode(rule_id, mode)
        )
        if not modes:
            continue
        spec = compiler.specs[rule_id]
        rules.append(
            OrthographyMorphemicRule(
                rule_id,
                f"Controlled morphemic orthography examples for {spec.family}.",
                modes,
            )
        )
    return tuple(rules)


ORTHOGRAPHY_MORPHEMIC_RULES = _morphemic_rules_from_specs()


__all__ = ["ORTHOGRAPHY_MORPHEMIC_RULES", "OrthographyMorphemicRule"]
