from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.lexicon import Lexicon
from src.grammar_gen.morphology import MorphologyEngine
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleProgram
from src.grammar_gen.rules.registry import RuleRegistry
from src.grammar_gen.safety import validate_surface
from src.schema import GeneratedExample


class GenerationError(RuntimeError):
    pass


class OnlineExampleGenerator:
    def __init__(
        self,
        registry: RuleRegistry,
        lexicon: Lexicon,
        morphology: MorphologyEngine,
        config: Mapping[str, Any] | None,
        seed: int | None = None,
    ) -> None:
        self.registry = registry
        self.lexicon = lexicon
        self.morphology = morphology
        self.config = config or {}
        self.rng = RandomSource(seed=seed)
        self.builder = GrammarBuilder(lexicon, morphology, self.rng)
        self.realizer = Realizer(lexicon, morphology)

    def sample(
        self,
        mode: GenerationMode | None = None,
        rule_id: str | None = None,
    ) -> GeneratedExample:
        requested_mode = _coerce_mode(mode) if mode is not None else None
        retries = self._max_generation_retries()
        failures: list[str] = []

        if rule_id is not None and self.registry.get_rule(rule_id) is None:
            raise GenerationError(f"Rule {rule_id!r} is not registered.")

        for attempt in range(1, retries + 1):
            try:
                rule, selected_mode = self._select_rule_and_mode(requested_mode, rule_id)
                example = rule.generate(self.builder, self.realizer, self.rng, selected_mode)
                return self._validate_example(example, rule, selected_mode)
            except Exception as exc:
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")

        details = "; ".join(failures[-5:]) if failures else "no attempts recorded"
        mode_detail = requested_mode.value if requested_mode is not None else None
        raise GenerationError(
            "Failed to generate example "
            f"after {retries} retries "
            f"(mode={mode_detail!r}, rule_id={rule_id!r}). "
            f"Recent failures: {details}"
        )

    def sample_batch(self, size: int) -> list[GeneratedExample]:
        if size < 0:
            raise ValueError("size must be non-negative.")
        return [self.sample() for _ in range(size)]

    def _select_rule_and_mode(
        self,
        requested_mode: GenerationMode | None,
        rule_id: str | None,
    ) -> tuple[RuleProgram, GenerationMode]:
        if requested_mode is not None:
            return self._select_rule_for_mode(requested_mode, rule_id, family=None)

        mode, family = self._sample_mode_from_mix()
        return self._select_rule_for_mode(mode, rule_id, family=family)

    def _select_rule_for_mode(
        self,
        mode: GenerationMode,
        rule_id: str | None,
        *,
        family: str | None,
    ) -> tuple[RuleProgram, GenerationMode]:
        if rule_id is not None:
            rule = self.registry.get_rule(rule_id)
            if rule is None:
                raise GenerationError(f"Rule {rule_id!r} is not registered.")
            if not rule.can_generate(mode):
                raise GenerationError(f"Rule {rule_id!r} does not support mode {mode.value!r}.")
            if family is not None and rule.info.family != family:
                raise GenerationError(
                    f"Rule {rule_id!r} is in family {rule.info.family!r}, not sampled family {family!r}."
                )
            return rule, mode

        rules = [
            rule
            for rule in self.registry.enabled_rules(self.config)
            if rule.can_generate(mode) and (family is None or rule.info.family == family)
        ]
        if not rules:
            family_detail = f" and family {family!r}" if family is not None else ""
            raise GenerationError(f"No enabled rules support mode {mode.value!r}{family_detail}.")

        return self.rng.weighted_choice(tuple((rule, rule.info.weight) for rule in rules)), mode

    def _sample_mode_from_mix(self) -> tuple[GenerationMode, str | None]:
        mix = _generation_mix(self.config)
        key = self.rng.weighted_choice(tuple(mix.items()))
        return _mode_and_family_from_mix_key(key)

    def _validate_example(
        self,
        example: GeneratedExample,
        rule: RuleProgram,
        mode: GenerationMode,
    ) -> GeneratedExample:
        if not isinstance(example, GeneratedExample):
            raise ValueError(f"Rule {rule.info.rule_id!r} returned {type(example).__name__}, not GeneratedExample.")

        validated = GeneratedExample.from_dict(example.to_dict())
        if not validated.source_tokens:
            raise ValueError("GeneratedExample must contain at least one source token.")
        if validated.mode != mode.value:
            raise ValueError(
                f"GeneratedExample mode mismatch: {validated.mode!r} != {mode.value!r}."
            )
        if validated.primary_rule_id != rule.info.rule_id:
            raise ValueError(
                "GeneratedExample primary_rule_id mismatch: "
                f"{validated.primary_rule_id!r} != {rule.info.rule_id!r}."
            )
        if validated.primary_rule_id not in validated.rule_ids:
            raise ValueError("GeneratedExample primary_rule_id must appear in rule_ids.")

        source_reasons = validate_surface(validated.source_text)
        if source_reasons:
            raise ValueError(f"Invalid source surface: {', '.join(source_reasons)}.")
        target_reasons = validate_surface(validated.target_text)
        if target_reasons:
            raise ValueError(f"Invalid target surface: {', '.join(target_reasons)}.")

        return validated

    def _max_generation_retries(self) -> int:
        generation = self.config.get("generation", {}) if isinstance(self.config, Mapping) else {}
        grammar = generation.get("grammar", {}) if isinstance(generation, Mapping) else {}
        raw_retries = grammar.get("max_generation_retries", 20) if isinstance(grammar, Mapping) else 20
        try:
            retries = int(raw_retries)
        except (TypeError, ValueError):
            retries = 20
        return max(1, retries)


def _coerce_mode(mode: GenerationMode | str) -> GenerationMode:
    if isinstance(mode, GenerationMode):
        return mode
    return GenerationMode(str(mode))


def _generation_mix(config: Mapping[str, Any]) -> dict[str, float]:
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    raw_mix = generation.get("mix", {}) if isinstance(generation, Mapping) else {}
    if not isinstance(raw_mix, Mapping) or not raw_mix:
        return {"positive": 1.0}

    mix: dict[str, float] = {}
    for key, raw_weight in raw_mix.items():
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError(f"Generation mix weight must be non-negative for {key!r}.")
        if weight > 0:
            mix[str(key)] = weight
    if not mix:
        raise ValueError("At least one generation mix weight must be positive.")
    return mix


def _mode_and_family_from_mix_key(key: str) -> tuple[GenerationMode, str | None]:
    normalized = key.strip().lower()
    if normalized == "clean_identity":
        return GenerationMode.CLEAN_IDENTITY, None
    if normalized == "hard_negative":
        return GenerationMode.HARD_NEGATIVE, None
    if normalized == "positive":
        return GenerationMode.POSITIVE, None
    return GenerationMode.POSITIVE, key


__all__ = ["GenerationError", "OnlineExampleGenerator"]
