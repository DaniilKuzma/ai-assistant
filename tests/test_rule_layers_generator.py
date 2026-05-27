from __future__ import annotations

from src.grammar_gen import Lexicon, MorphologyEngine
from src.config.load_config import load_config
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.generator import OnlineExampleGenerator, _generation_mix, _mode_and_family_from_mix_key
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.rules.registry import RuleRegistry, default_rule_registry, register_layered_rules
from src.rule_layers.base import LayerDirectCase, LayerOperation, LayerRuleSpec


def test_new_family_mix_key_can_sample_layer_rule() -> None:
    spec = LayerRuleSpec(
        layer="compound",
        rule_id="ne_verb",
        family="compound_spelling",
        cases=(
            LayerDirectCase(
                rule_id="ne_verb",
                family="compound_spelling",
                sub_rule_id="na_schet_merge",
                mode="positive",
                source_text="\u041e\u043d \u043f\u0438\u0441\u0430\u043b \u043d\u0430 \u0441\u0447\u0435\u0442 \u043e\u0448\u0438\u0431\u043a\u0438.",
                target_text="\u041e\u043d \u043f\u0438\u0441\u0430\u043b \u043d\u0430\u0441\u0447\u0451\u0442 \u043e\u0448\u0438\u0431\u043a\u0438.",
                token_operations=(
                    LayerOperation(
                        kind="token_span",
                        label="SPAN_REPLACE_BY_LEXICON",
                        source_pattern="\u043d\u0430 \u0441\u0447\u0435\u0442",
                        target_pattern="\u043d\u0430\u0441\u0447\u0451\u0442",
                    ),
                ),
                expected_token_edit_count=1,
                expected_gap_edit_count=0,
            ),
        ),
    )
    registry = RuleRegistry()
    register_layered_rules(registry, (spec,))
    generator = OnlineExampleGenerator(
        registry,
        Lexicon.default(),
        MorphologyEngine(use_pymorphy=False),
        {
            "generation": {
                "enabled_rule_groups": ["compound_spelling"],
                "mix": {"compound_spelling": 1.0},
                "grammar": {"max_generation_retries": 3},
            }
        },
        seed=5,
    )

    example = generator.sample()

    assert example.primary_rule_id == "ne_verb"
    assert example.metadata["layer"] == "compound"
    assert example.metadata["expected_token_edit_count"] == 1


def test_generation_mix_preserves_old_config_aliases() -> None:
    mix = _generation_mix({"generation": {"mix": {"contextual_orthography": 1.0}}})
    mode, family = _mode_and_family_from_mix_key("orthography_contextual")

    assert mix == {"orthography_contextual": 1.0}
    assert mode is GenerationMode.POSITIVE
    assert family == "orthography_contextual"


def test_canonical_morpheme_mix_key_samples_morphemic_rules() -> None:
    config = load_config("configs/config.yaml")
    config["generation"]["enabled_rule_groups"] = ["morpheme"]
    config["generation"]["mix"] = {"morpheme": 1.0}
    config["generation"]["grammar"]["max_generation_retries"] = 20

    example = online_generator_from_config(config, seed=11).sample_by_index(0)

    assert example.metadata["layer"] == "morpheme"
    assert example.primary_rule_id.startswith("morpheme_") or example.primary_rule_id in {
        "suffix_its_ets",
        "suffix_enn_yan",
        "n_nn_basic",
    }


def test_default_config_still_generates_with_old_registry() -> None:
    generator = OnlineExampleGenerator(
        default_rule_registry(),
        Lexicon.default(),
        MorphologyEngine(use_pymorphy=False),
        {"generation": {"mix": {"punctuation": 1.0}, "grammar": {"max_generation_retries": 30}}},
        seed=13,
    )

    example = generator.sample()

    assert example.primary_rule_id in {"comma_subordinate", "comma_introductory", "comma_homogeneous", "comma_adversative", "dash_subject_predicate", "final_punctuation"}
