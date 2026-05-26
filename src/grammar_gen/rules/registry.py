from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from src.grammar_gen.rules.base import RuleProgram

if TYPE_CHECKING:
    from src.rule_layers.base import LayerRuleSpec


class RuleRegistry:
    def __init__(self) -> None:
        self._rules: dict[str, RuleProgram] = {}

    def register_rule(self, rule: RuleProgram) -> RuleProgram:
        rule_id = rule.info.rule_id
        if not rule_id:
            raise ValueError("Rule id must not be empty.")
        if rule_id in self._rules:
            raise ValueError(f"Duplicate rule id: {rule_id}")
        self._rules[rule_id] = rule
        return rule

    def register_layered_rules(self, specs: tuple[LayerRuleSpec, ...]) -> tuple[RuleProgram, ...]:
        from src.rule_layers.direct_cases import layer_rule_programs
        from src.schema.labels import rule_tag_to_id

        for spec in specs:
            rule_tag_to_id(spec.rule_id)
        programs = layer_rule_programs(specs)
        for program in programs:
            self.register_rule(program)
        return programs

    def get_rule(self, rule_id: str) -> RuleProgram | None:
        return self._rules.get(rule_id)

    def enabled_rules(self, config: Mapping[str, Any] | None = None) -> tuple[RuleProgram, ...]:
        enabled = _enabled_rule_groups(config)
        rules = tuple(self._rules.values())
        if not enabled:
            return rules
        return tuple(
            rule
            for rule in rules
            if _is_enabled_rule(rule.info.family, rule.info.rule_id, enabled)
        )

    def all_rules(self) -> tuple[RuleProgram, ...]:
        return tuple(self._rules.values())


_DEFAULT_REGISTRY = RuleRegistry()


def default_rule_registry() -> RuleRegistry:
    return _DEFAULT_REGISTRY


def register_rule(rule: RuleProgram) -> RuleProgram:
    return default_rule_registry().register_rule(rule)


def register_layered_rules(
    registry: RuleRegistry,
    specs: tuple[LayerRuleSpec, ...],
) -> tuple[RuleProgram, ...]:
    return registry.register_layered_rules(specs)


def get_rule(rule_id: str) -> RuleProgram | None:
    return default_rule_registry().get_rule(rule_id)


def enabled_rules(config: Mapping[str, Any] | None = None) -> tuple[RuleProgram, ...]:
    return default_rule_registry().enabled_rules(config)


def _enabled_rule_groups(config: Mapping[str, Any] | None) -> frozenset[str]:
    if not isinstance(config, Mapping):
        return frozenset()
    generation = config.get("generation", {})
    if not isinstance(generation, Mapping):
        return frozenset()
    raw_groups = generation.get("enabled_rule_groups", ())
    if raw_groups is None:
        return frozenset()
    if isinstance(raw_groups, str):
        return frozenset({raw_groups})
    return frozenset(str(group) for group in raw_groups)


def _is_enabled_rule(family: str, rule_id: str, enabled: frozenset[str]) -> bool:
    if family in enabled or rule_id in enabled:
        return True
    aliases = {
        "orthography_contextual": frozenset({"contextual_orthography"}),
        "orthography_morphemic": frozenset({"morpheme"}),
        "punctuation": frozenset({"syntax_punctuation"}),
    }
    return bool(aliases.get(family, frozenset()) & enabled)


def _register_default_rules() -> None:
    from src.grammar_gen.rules.orthography import register_orthography_rules
    from src.grammar_gen.rules.punctuation import register_punctuation_rules

    register_orthography_rules(_DEFAULT_REGISTRY)
    register_punctuation_rules(_DEFAULT_REGISTRY)


__all__ = [
    "RuleRegistry",
    "default_rule_registry",
    "enabled_rules",
    "get_rule",
    "register_layered_rules",
    "register_rule",
]


_register_default_rules()
