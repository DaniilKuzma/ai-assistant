from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.grammar_gen.generator import OnlineExampleGenerator
from src.grammar_gen.lexicon import Lexicon
from src.grammar_gen.morphology import MorphologyEngine
from src.grammar_gen.rules.registry import RuleRegistry, default_rule_registry


def morphology_from_config(
    config: Mapping[str, Any],
    lexicon: Lexicon,
    *,
    critical: bool = True,
) -> MorphologyEngine:
    grammar = _grammar_config(config)
    use_pymorphy = bool(grammar.get("use_pymorphy", True))
    morphology = MorphologyEngine(use_pymorphy=use_pymorphy, lexicon=lexicon, critical=critical)
    if critical and use_pymorphy and not morphology.uses_pymorphy:
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
        registry or default_rule_registry(),
        resolved_lexicon,
        morphology,
        config,
        seed=seed,
    )


def _grammar_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    grammar = generation.get("grammar", {}) if isinstance(generation, Mapping) else {}
    return grammar if isinstance(grammar, Mapping) else {}


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
    verb_columns = {"past_masc", "past_fem", "past_neut", "past_plur", "present_3sg", "infinitive"}
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
