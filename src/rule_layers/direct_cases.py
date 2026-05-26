from __future__ import annotations

from collections.abc import Iterable

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.rule_layers.base import LayerRuleSpec, RuleLayer
from src.rule_layers.example_builders import build_generated_example_from_case
from src.schema import GeneratedExample


class DirectCasesLayer(RuleLayer):
    def __init__(self, spec: LayerRuleSpec) -> None:
        self.spec = spec
        modes = tuple(
            GenerationMode(mode)
            for mode in dict.fromkeys(case.mode for case in spec.cases)
        )
        self._supported_modes = modes

    @property
    def supported_modes(self) -> tuple[GenerationMode, ...]:
        return self._supported_modes

    def generate(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
        mode: GenerationMode,
    ) -> GeneratedExample:
        del builder
        cases = tuple(case for case in self.spec.cases if case.mode == mode.value)
        if not cases:
            raise ValueError(f"Layer rule {self.spec.rule_id!r} has no cases for mode {mode.value!r}.")
        selected = rng.weighted_choice(tuple((case, case.weight) for case in cases))
        return build_generated_example_from_case(selected, realizer, layer=self.spec.layer)


class LayerRuleProgram(RuleProgram):
    def __init__(self, layer: RuleLayer) -> None:
        self.layer = layer
        spec = layer.spec
        self.info = RuleInfo(
            rule_id=spec.rule_id,
            family=spec.family,
            description=spec.description or f"Layered rule {spec.rule_id}",
            explanation=spec.explanation or spec.rule_id,
            deterministic=True,
            weight=spec.weight,
        )
        self.supported_modes = layer.supported_modes

    def generate(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
        mode: GenerationMode,
    ) -> GeneratedExample:
        return self.layer.generate(builder, realizer, rng, mode)


def layer_rule_programs(specs: Iterable[LayerRuleSpec]) -> tuple[LayerRuleProgram, ...]:
    return tuple(
        LayerRuleProgram(DirectCasesLayer(spec))
        for spec in specs
        if spec.enabled
    )


__all__ = ["DirectCasesLayer", "LayerRuleProgram", "layer_rule_programs"]
