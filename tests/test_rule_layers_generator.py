from __future__ import annotations

from src.grammar_gen import Lexicon, MorphologyEngine
from src.config.load_config import load_config
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.generator import OnlineExampleGenerator, _generation_mix, _mode_and_family_from_mix_key
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.registry import RuleRegistry, default_rule_registry, register_layered_rules
from src.rule_layers.base import LayerDirectCase, LayerOperation, LayerRuleSpec
from src.schema import GeneratedExample, WordToken


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


def test_casing_mix_key_samples_casing_family() -> None:
    mix = _generation_mix({"generation": {"mix": {"casing": 1.0}}})
    mode, family = _mode_and_family_from_mix_key("casing")

    assert mix == {"casing": 1.0}
    assert mode is GenerationMode.POSITIVE
    assert family == "casing"


def test_semantic_mix_key_samples_semantic_family() -> None:
    mix = _generation_mix({"generation": {"mix": {"semantic": 1.0}}})
    mode, family = _mode_and_family_from_mix_key("semantic")

    assert mix == {"semantic": 1.0}
    assert mode is GenerationMode.POSITIVE
    assert family == "semantic"


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


def test_rule_weight_overrides_control_rule_sampling() -> None:
    registry = RuleRegistry()
    registry.register_rule(_TinyPunctuationRule("comma_subordinate"))
    registry.register_rule(_TinyPunctuationRule("comma_introductory"))
    generator = OnlineExampleGenerator(
        registry,
        Lexicon.default(),
        MorphologyEngine(use_pymorphy=False),
        {
            "generation": {
                "enabled_rule_groups": ["punctuation"],
                "mix": {"punctuation": 1.0},
                "rule_weight_overrides": {"comma_introductory": 0.0},
                "grammar": {"max_generation_retries": 3},
            }
        },
        seed=7,
    )

    sampled = {generator.sample_by_index(index).primary_rule_id for index in range(50)}

    assert sampled == {"comma_subordinate"}


def test_direct_layer_generation_varies_natural_contexts_for_small_rules() -> None:
    config = load_config("configs/config.yaml")
    config["generation"]["enabled_rule_groups"] = ["syntax_punctuation"]
    config["generation"]["mix"] = {"syntax_punctuation": 1.0}
    config["generation"]["grammar"]["max_generation_retries"] = 60
    config["generation"]["rule_layers"]["groups"]["syntax_punctuation"] = {
        "enabled": True,
        "layers": ["syntax_punctuation"],
    }
    generator = online_generator_from_config(config, seed=111)

    examples = [
        generator.sample(rule_id="punct_bsp", mode=GenerationMode.POSITIVE)
        for _ in range(120)
    ]
    unique_pairs = {(example.source_text, example.target_text) for example in examples}

    assert len(unique_pairs) >= 80
    assert not any("пример номер" in example.source_text.casefold() for example in examples)


def test_legacy_dash_subject_predicate_keeps_curated_core_shape() -> None:
    config = load_config("configs/config.yaml")
    config["generation"]["enabled_rule_groups"] = ["punctuation"]
    config["generation"]["mix"] = {"punctuation": 1.0}
    config["generation"]["rule_weight_overrides"] = {}
    generator = online_generator_from_config(config, seed=222)

    examples = [
        generator.sample(rule_id="dash_subject_predicate", mode=GenerationMode.POSITIVE)
        for _ in range(120)
    ]
    unique_pairs = {(example.source_text, example.target_text) for example in examples}

    assert len(unique_pairs) >= 15
    assert all("—" in example.target_text for example in examples)
    assert all(
        len(example.target_text.split("—", 1)[0].split()) == 1
        and len(example.target_text.split("—", 1)[1].strip(" .").split()) <= 2
        for example in examples
    )


class _TinyPunctuationRule(RuleProgram):
    def __init__(self, rule_id: str) -> None:
        self.info = RuleInfo(
            rule_id=rule_id,
            family="punctuation",
            description=rule_id,
            explanation=rule_id,
            deterministic=True,
            weight=1.0,
        )
        self.supported_modes = (GenerationMode.POSITIVE,)

    def generate(self, builder, realizer, rng, mode: GenerationMode) -> GeneratedExample:
        del builder, realizer, rng
        return GeneratedExample(
            source_text="Он знал что делать.",
            target_text="Он знал, что делать.",
            source_tokens=[
                WordToken("Он", 0, 2),
                WordToken("знал", 3, 7),
                WordToken("что", 8, 11),
                WordToken("делать", 12, 18),
            ],
            token_edit_labels=["KEEP", "KEEP", "KEEP", "KEEP"],
            gap_labels=["NONE", "COMMA", "NONE", "DOT"],
            rule_ids=["none", self.info.rule_id, "none", "none"],
            primary_rule_id=self.info.rule_id,
            mode=mode.value,
            explanation_ids=[self.info.rule_id],
            metadata={
                "expected_edit_count": 1,
                "expected_token_edit_count": 0,
                "expected_gap_edit_count": 1,
                "production": False,
                "uses_construction_bank": True,
                "construction_id": self.info.rule_id,
                "construction_family": "punctuation",
                "uses_safety_clauses": False,
                "safety_clauses": [],
            },
        )
