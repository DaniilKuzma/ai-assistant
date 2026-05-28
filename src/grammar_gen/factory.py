from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.grammar_gen.generator import OnlineExampleGenerator
from src.grammar_gen.lexicon import Lexicon
from src.grammar_gen.morphology import MorphologyEngine
from src.grammar_gen.rules.registry import RuleRegistry, default_rule_registry
from src.grammar_gen.rules.registry import register_layered_rules
from src.grammar_gen.randomness import RandomSource
from src.rule_layers.casing import load_casing_specs
from src.rule_layers.compound_spelling import load_compound_spelling_specs
from src.rule_layers.dictionary_typo import load_dictionary_typo_specs
from src.rule_layers.quotation_dialogue import load_quotation_dialogue_specs
from src.rule_layers.semantic import load_semantic_specs
from src.rule_layers.spec_loader import load_layer_specs
from src.rule_layers.syntax_punctuation import load_syntax_punctuation_specs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def morphology_from_config(
    config: Mapping[str, Any],
    lexicon: Lexicon,
    *,
    critical: bool = True,
) -> MorphologyEngine:
    grammar = _grammar_config(config)
    use_pymorphy = bool(grammar.get("use_pymorphy", True))
    morphology = MorphologyEngine(use_pymorphy=use_pymorphy, lexicon=lexicon, critical=critical)
    if critical and not morphology.uses_pymorphy:
        _validate_curated_critical_coverage(lexicon)
    return morphology


def online_generator_from_config(
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
    lexicon: Lexicon | None = None,
    registry: RuleRegistry | None = None,
) -> OnlineExampleGenerator:
    resolved_lexicon = lexicon or Lexicon.default()
    morphology = morphology_from_config(config, resolved_lexicon, critical=True)
    return OnlineExampleGenerator(
        registry or _registry_from_config(config),
        resolved_lexicon,
        morphology,
        config,
        seed=seed,
    )


def _registry_from_config(config: Mapping[str, Any]) -> RuleRegistry:
    if not _rule_layers_enabled(config):
        return default_rule_registry()

    registry = RuleRegistry()
    for rule in default_rule_registry().all_rules():
        registry.register_rule(rule)

    layer_names = _configured_layer_names(config)
    if not layer_names:
        return registry

    seed = _generation_seed(config)
    rng = RandomSource(seed=seed)
    root = _layer_root(config)
    specs = tuple(
        spec
        for layer_name in layer_names
        for spec in _load_layer_specs(layer_name, root=root, rng=rng, seed=seed)
    )
    register_layered_rules(registry, specs)
    return registry


def _load_layer_specs(layer_name: str, *, root: Path, rng: RandomSource, seed: int):
    if layer_name == "compound_spelling":
        return load_compound_spelling_specs(root)
    if layer_name == "dictionary_typo":
        return load_dictionary_typo_specs(root, seed=seed)
    if layer_name == "syntax_punctuation":
        return load_syntax_punctuation_specs(root)
    if layer_name == "quotation_dialogue":
        return load_quotation_dialogue_specs(root)
    if layer_name == "casing":
        return load_casing_specs(root)
    if layer_name == "semantic":
        return load_semantic_specs(root)
    return load_layer_specs(layer_name, root=root, rng=rng)


def _grammar_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    grammar = generation.get("grammar", {}) if isinstance(generation, Mapping) else {}
    return grammar if isinstance(grammar, Mapping) else {}


def _rule_layers_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    raw = generation.get("rule_layers", {}) if isinstance(generation, Mapping) else {}
    return raw if isinstance(raw, Mapping) else {}


def _rule_layers_enabled(config: Mapping[str, Any]) -> bool:
    return bool(_rule_layers_config(config).get("enabled", False))


def _configured_layer_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    layer_config = _rule_layers_config(config)
    names: list[str] = []
    raw_layers = layer_config.get("layers", ())
    if isinstance(raw_layers, str):
        names.append(raw_layers)
    elif isinstance(raw_layers, list):
        names.extend(str(item) for item in raw_layers if str(item).strip())

    groups = layer_config.get("groups", {})
    if isinstance(groups, Mapping):
        for group_name, raw_group in groups.items():
            if not isinstance(raw_group, Mapping) or not bool(raw_group.get("enabled", False)):
                continue
            raw_group_layers = raw_group.get("layers") or (group_name,)
            if isinstance(raw_group_layers, str):
                names.append(raw_group_layers)
            elif isinstance(raw_group_layers, list):
                names.extend(str(item) for item in raw_group_layers if str(item).strip())
    return tuple(dict.fromkeys(name for name in names if name.strip()))


def _layer_root(config: Mapping[str, Any]) -> Path:
    layer_config = _rule_layers_config(config)
    configured = layer_config.get("root")
    if configured:
        path = Path(str(configured))
    else:
        paths = config.get("paths", {}) if isinstance(config, Mapping) else {}
        lexicon_dir = paths.get("lexicon_dir", "lexicon") if isinstance(paths, Mapping) else "lexicon"
        path = Path(str(lexicon_dir)) / "layers"
    return path if path.is_absolute() else PROJECT_ROOT / path


def _generation_seed(config: Mapping[str, Any]) -> int:
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    try:
        return int(generation.get("seed", 0)) if isinstance(generation, Mapping) else 0
    except (TypeError, ValueError):
        return 0


def _validate_curated_critical_coverage(lexicon: Lexicon) -> None:
    noun_columns = {"nom_sg", "gen_sg", "dat_sg", "acc_sg", "ins_sg", "loc_sg", "nom_pl", "acc_pl"}
    adjective_columns = {
        "masc_nom",
        "fem_nom",
        "neut_nom",
        "plur_nom",
        "fem_acc",
        "masc_acc_inanim",
        "neut_acc",
    }
    verb_columns = {"past_masc", "past_fem", "past_neut", "past_plur", "present_3sg", "present_3pl", "infinitive"}
    frame_verbs = {frame.verb_lemma for frame in lexicon.frames.frames}

    missing: list[str] = []
    missing.extend(
        f"noun:{noun.lemma}"
        for noun in lexicon.nouns
        if not noun_columns <= set(noun.forms)
    )
    missing.extend(
        f"adjective:{adjective.lemma}"
        for adjective in lexicon.adjectives
        if not adjective_columns <= set(adjective.forms)
    )
    missing.extend(
        f"verb:{verb.lemma}"
        for verb in lexicon.verbs
        if verb.lemma in frame_verbs and not verb_columns <= set(verb.forms)
    )
    if missing:
        preview = ", ".join(missing[:20])
        suffix = f" and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise RuntimeError(
            "pymorphy3 is unavailable and curated morphology coverage is incomplete: "
            f"{preview}{suffix}."
        )


__all__ = ["morphology_from_config", "online_generator_from_config"]
