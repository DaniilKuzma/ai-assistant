from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config.load_config import load_config
from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.lexicon import Lexicon
from src.grammar_gen.morphology import MorphologyEngine
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleProgram
from src.grammar_gen.rules.registry import RuleRegistry
from src.grammar_gen.safety import validate_generated_pair
from src.schema import GeneratedExample


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
REMOVED_CONFIG_KEYS = frozenset({"clean_pool", "rule_lab", "correction_dataset_path"})
REMOVED_PATH_MARKERS = (
    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/test.csv",
    "data/processed/correction_dataset.csv.gz",
    "clean_sentence_pool",
    "clean_pool",
)


class GenerationError(RuntimeError):
    pass


class OnlineExampleGenerator:
    def __init__(
        self,
        registry: RuleRegistry,
        lexicon: Lexicon,
        morphology: MorphologyEngine,
        config: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.registry = registry
        self.lexicon = lexicon
        self.morphology = morphology
        self.config = config or load_config(DEFAULT_CONFIG_PATH)
        validate_generation_config(self.config)
        self.base_seed = _resolve_seed(self.config, seed)
        self.rng = RandomSource(seed=self.base_seed)
        self.builder = GrammarBuilder(lexicon, morphology, self.rng)
        self.realizer = Realizer(lexicon, morphology)

    def sample(
        self,
        mode: GenerationMode | str | None = None,
        rule_id: str | None = None,
    ) -> GeneratedExample:
        requested_mode = _coerce_mode(mode) if mode is not None else None
        return self._sample_with_rng(
            rng=self.rng,
            builder=self.builder,
            requested_mode=requested_mode,
            rule_id=rule_id,
            generation_index=None,
            generation_seed=None,
        )

    def sample_batch(self, size: int) -> list[GeneratedExample]:
        if size < 0:
            raise ValueError("size must be non-negative.")
        return [self.sample() for _ in range(size)]

    def sample_by_index(self, index: int) -> GeneratedExample:
        if not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if index < 0:
            raise ValueError("index must be non-negative.")

        generation_seed = self.base_seed + index
        rng = RandomSource(seed=generation_seed)
        builder = GrammarBuilder(self.lexicon, self.morphology, rng)
        return self._sample_with_rng(
            rng=rng,
            builder=builder,
            requested_mode=None,
            rule_id=None,
            generation_index=index,
            generation_seed=generation_seed,
        )

    def _sample_with_rng(
        self,
        *,
        rng: RandomSource,
        builder: GrammarBuilder,
        requested_mode: GenerationMode | None,
        rule_id: str | None,
        generation_index: int | None,
        generation_seed: int | None,
    ) -> GeneratedExample:
        retries = self._max_generation_retries()
        failures: list[str] = []

        if rule_id is not None and self.registry.get_rule(rule_id) is None:
            raise GenerationError(f"Rule {rule_id!r} is not registered.")

        for attempt in range(1, retries + 1):
            try:
                rule, selected_mode = self._select_rule_and_mode(requested_mode, rule_id, rng)
                example = rule.generate(builder, self.realizer, rng, selected_mode)
                validated = self._validate_example(example, rule, selected_mode)
                return _with_generation_metadata(
                    validated,
                    generation_index=generation_index,
                    generation_seed=generation_seed,
                )
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

    def _select_rule_and_mode(
        self,
        requested_mode: GenerationMode | None,
        rule_id: str | None,
        rng: RandomSource,
    ) -> tuple[RuleProgram, GenerationMode]:
        if rule_id is not None:
            rule = self.registry.get_rule(rule_id)
            if rule is None:
                raise GenerationError(f"Rule {rule_id!r} is not registered.")
            if requested_mode is not None:
                if not rule.can_generate(requested_mode):
                    raise GenerationError(
                        f"Rule {rule_id!r} does not support mode {requested_mode.value!r}."
                    )
                return rule, requested_mode
            return rule, self._sample_mode_for_rule(rule, rng)

        if requested_mode is not None:
            return self._select_rule_for_mode(requested_mode, family=None, rng=rng)

        mode, family = self._sample_mode_from_mix(rng)
        return self._select_rule_for_mode(mode, family=family, rng=rng)

    def _select_rule_for_mode(
        self,
        mode: GenerationMode,
        *,
        family: str | None,
        rng: RandomSource,
    ) -> tuple[RuleProgram, GenerationMode]:
        rules = [
            rule
            for rule in self.registry.enabled_rules(self.config)
            if rule.can_generate(mode) and (family is None or rule.info.family == family)
        ]
        if not rules:
            family_detail = f" and family {family!r}" if family is not None else ""
            raise GenerationError(f"No enabled rules support mode {mode.value!r}{family_detail}.")

        return rng.weighted_choice(tuple((rule, rule.info.weight) for rule in rules)), mode

    def _sample_mode_from_mix(self, rng: RandomSource) -> tuple[GenerationMode, str | None]:
        key = rng.weighted_choice(tuple(_generation_mix(self.config).items()))
        return _mode_and_family_from_mix_key(key)

    def _sample_mode_for_rule(self, rule: RuleProgram, rng: RandomSource) -> GenerationMode:
        mode_weights: dict[GenerationMode, float] = {}
        for key, weight in _generation_mix(self.config).items():
            mode, _family = _mode_and_family_from_mix_key(key)
            if rule.can_generate(mode):
                mode_weights[mode] = mode_weights.get(mode, 0.0) + weight

        if not mode_weights:
            raise GenerationError(f"Rule {rule.info.rule_id!r} has no supported generation mode in generation.mix.")
        return rng.weighted_choice(tuple(mode_weights.items()))

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

        pair_reasons = validate_generated_pair(validated)
        if pair_reasons:
            raise ValueError(f"Invalid generated pair: {', '.join(pair_reasons)}.")

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


def validate_generation_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        return

    generation = config.get("generation", {})
    if isinstance(generation, Mapping) and "mode" in generation:
        mode = str(generation.get("mode") or "").strip()
        if mode != "online_ast":
            raise ValueError("generation.mode must be 'online_ast' for AST-first generation.")

    data = config.get("data", {})
    if isinstance(data, Mapping) and "candidate_opportunity" in data:
        raise ValueError("Removed config key data.candidate_opportunity is not supported by AST-first generation.")

    for key_path, key, value in _walk_config_items(config):
        normalized_key = str(key).strip()
        if normalized_key in REMOVED_CONFIG_KEYS:
            raise ValueError(f"Removed config key {key_path} is not supported by AST-first generation.")
        if isinstance(value, str) and _references_removed_generation_path(value):
            raise RuntimeError(f"Removed generation path referenced at {key_path}: {value}")


def _walk_config_items(value: Any, prefix: str = "") -> list[tuple[str, str, Any]]:
    items: list[tuple[str, str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            key_path = f"{prefix}.{key_text}" if prefix else key_text
            items.append((key_path, key_text, child))
            items.extend(_walk_config_items(child, key_path))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            items.extend(_walk_config_items(child, f"{prefix}[{index}]"))
    return items


def _references_removed_generation_path(value: str) -> bool:
    normalized = value.replace("\\", "/").strip().lower()
    return any(marker in normalized for marker in REMOVED_PATH_MARKERS)


def _resolve_seed(config: Mapping[str, Any], seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    raw_seed = generation.get("seed", 0) if isinstance(generation, Mapping) else 0
    try:
        return int(raw_seed)
    except (TypeError, ValueError):
        return 0


def _generation_mix(config: Mapping[str, Any]) -> dict[str, float]:
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    raw_mix = generation.get("mix", {}) if isinstance(generation, Mapping) else {}
    if not isinstance(raw_mix, Mapping) or not raw_mix:
        return {
            "orthography_contextual": 0.35,
            "punctuation": 0.40,
            "clean_identity": 0.15,
            "hard_negative": 0.10,
        }

    mix: dict[str, float] = {}
    for key, raw_weight in raw_mix.items():
        normalized_key = _normalize_mix_key(str(key))
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError(f"Generation mix weight must be non-negative for {key!r}.")
        if weight > 0:
            mix[normalized_key] = mix.get(normalized_key, 0.0) + weight
    if not mix:
        raise ValueError("At least one generation mix weight must be positive.")
    return mix


def _normalize_mix_key(key: str) -> str:
    normalized = key.strip().lower()
    aliases = {
        "contextual_orthography": "orthography_contextual",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "orthography_contextual",
        "punctuation",
        "clean_identity",
        "hard_negative",
    }
    if normalized not in allowed:
        raise ValueError(f"Unsupported generation mix group: {key!r}.")
    return normalized


def _mode_and_family_from_mix_key(key: str) -> tuple[GenerationMode, str | None]:
    normalized = _normalize_mix_key(key)
    if normalized == "orthography_contextual":
        return GenerationMode.POSITIVE, "orthography_contextual"
    if normalized == "punctuation":
        return GenerationMode.POSITIVE, "punctuation"
    if normalized == "clean_identity":
        return GenerationMode.CLEAN_IDENTITY, None
    if normalized == "hard_negative":
        return GenerationMode.HARD_NEGATIVE, None
    raise ValueError(f"Unsupported generation mix group: {key!r}.")


def _with_generation_metadata(
    example: GeneratedExample,
    *,
    generation_index: int | None,
    generation_seed: int | None,
) -> GeneratedExample:
    if generation_index is None and generation_seed is None:
        return example

    example.metadata = dict(example.metadata)
    if generation_index is not None:
        example.metadata["generation_index"] = generation_index
    if generation_seed is not None:
        example.metadata["generation_seed"] = generation_seed
    return GeneratedExample.from_dict(example.to_dict())


__all__ = ["GenerationError", "OnlineExampleGenerator", "validate_generation_config"]
