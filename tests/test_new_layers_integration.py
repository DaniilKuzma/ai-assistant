from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.config.load_config import load_config
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.diversity import diversity_report
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.safety import validate_generated_pair
from src.model.edit_model import DirectEditModelConfig, DirectEditTaggerModel
from src.schema.labels import (
    BOUNDARY_AFTER_LABELS,
    BOUNDARY_BEFORE_LABELS,
    GAP_PUNCTUATION_LABELS,
    RULE_LABELS,
    TOKEN_EDIT_LABELS,
)


ROOT = Path(__file__).resolve().parents[1]
QUOTATION_RULE_IDS = (
    "quote_pairing",
    "quote_normalization",
    "quote_extra_marks",
    "dialogue_author_before",
    "dialogue_speech_before_author",
    "dialogue_author_inside_speech",
    "dialogue_bracket_guards",
)
CASING_RULE_IDS = (
    "casing_sentence_start",
    "casing_person_names",
    "casing_geo_names",
    "casing_organizations",
    "casing_documents_events",
    "casing_common_lowercase",
    "casing_formal_you_guard",
)
SEMANTIC_RULE_IDS = (
    "semantic_service_words",
    "semantic_derived_prepositions",
    "semantic_ne_ni",
    "semantic_introductory_context",
    "semantic_comparative_context",
)
NEW_RULE_IDS = QUOTATION_RULE_IDS + CASING_RULE_IDS + SEMANTIC_RULE_IDS
LEGACY_SEMANTIC_OVERLAP_IDS = {
    "ne_verb",
    "takzhe_tak_zhe",
    "tozhe_to_zhe",
    "zato_za_to",
    "hyphen_particles",
    "hyphen_koe",
    "hyphen_po_adverb",
}
LEGACY_TOKEN_PREFIX = (
    "KEEP",
    "DELETE",
    "SKIP_MERGED",
    "LOWERCASE",
    "UPPERCASE",
    "SPLIT_NE_VERB",
    "MERGE_TAK_ZHE_TO_TAKZHE",
    "SPLIT_TAKZHE_TO_TAK_ZHE",
    "MERGE_TO_ZHE_TO_TOZHE",
    "SPLIT_TOZHE_TO_TO_ZHE",
    "MERGE_ZA_TO_TO_ZATO",
    "SPLIT_ZATO_TO_ZA_TO",
    "HYPHENATE_PARTICLE_TO",
    "HYPHENATE_PARTICLE_LIBO",
    "HYPHENATE_PARTICLE_NIBUD",
    "HYPHENATE_KOE",
    "HYPHENATE_PO_ADVERB",
    "FIX_TSYA_TO_TTSYA",
    "FIX_TTSYA_TO_TSYA",
    "DICT_REPLACE",
    "SPAN_REPLACE_BY_LEXICON",
)
LEGACY_GAP_PREFIX = (
    "NONE",
    "COMMA",
    "DASH",
    "COLON",
    "SEMICOLON",
    "DOT",
    "QUESTION",
    "EXCLAMATION",
    "ELLIPSIS",
    "DELETE_PUNCTUATION",
)
LEGACY_RULE_PREFIX = (
    "none",
    "clean_identity",
    "ne_verb",
    "takzhe_tak_zhe",
    "tozhe_to_zhe",
    "zato_za_to",
    "hyphen_particles",
    "hyphen_koe",
    "hyphen_po_adverb",
    "tsya_ttsya",
    "comma_subordinate",
    "comma_introductory",
    "comma_homogeneous",
    "comma_adversative",
    "dash_subject_predicate",
    "final_punctuation",
)


def test_label_schema_is_prefix_compatible_and_model_heads_use_current_sizes() -> None:
    assert TOKEN_EDIT_LABELS[: len(LEGACY_TOKEN_PREFIX)] == LEGACY_TOKEN_PREFIX
    assert TOKEN_EDIT_LABELS[len(LEGACY_TOKEN_PREFIX) :] == ("CAPITALIZE",)
    assert GAP_PUNCTUATION_LABELS[: len(LEGACY_GAP_PREFIX)] == LEGACY_GAP_PREFIX
    assert GAP_PUNCTUATION_LABELS[len(LEGACY_GAP_PREFIX) :] == ("COMMA_DASH",)
    assert BOUNDARY_BEFORE_LABELS[:4] == (
        "NONE",
        "INSERT_OPEN_QUOTE",
        "DELETE_OPEN_QUOTE",
        "NORMALIZE_OPEN_QUOTE",
    )
    assert BOUNDARY_AFTER_LABELS[:4] == (
        "NONE",
        "INSERT_CLOSE_QUOTE",
        "DELETE_CLOSE_QUOTE",
        "NORMALIZE_CLOSE_QUOTE",
    )
    assert RULE_LABELS[: len(LEGACY_RULE_PREFIX)] == LEGACY_RULE_PREFIX
    assert RULE_LABELS[-len(SEMANTIC_RULE_IDS) :] == SEMANTIC_RULE_IDS

    config = DirectEditModelConfig(lora_enabled=False)
    model = DirectEditTaggerModel.from_encoder(_FakeEncoder(), config).module

    assert model.heads["token_edit"].out_features == len(TOKEN_EDIT_LABELS)
    assert model.heads["gap_punctuation"].out_features == len(GAP_PUNCTUATION_LABELS)
    assert model.heads["boundary_before"].out_features == len(BOUNDARY_BEFORE_LABELS)
    assert model.heads["boundary_after"].out_features == len(BOUNDARY_AFTER_LABELS)
    assert model.heads["rule"].out_features == len(RULE_LABELS)


def test_full_config_registers_new_layers_without_duplicate_rule_ids() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    generator = online_generator_from_config(config, seed=13)
    registered = [rule.info.rule_id for rule in generator.registry.all_rules()]

    assert len(registered) == len(set(registered))
    assert set(NEW_RULE_IDS) <= set(registered)

    semantic_rules = {
        rule.info.rule_id
        for rule in generator.registry.all_rules()
        if rule.info.family == "semantic"
    }
    assert semantic_rules == set(SEMANTIC_RULE_IDS)
    assert not semantic_rules & LEGACY_SEMANTIC_OVERLAP_IDS


def test_rules_yaml_marks_new_layer_rules_with_grouped_layers_and_tests() -> None:
    data = yaml.safe_load((ROOT / "configs" / "rules.yaml").read_text(encoding="utf-8"))
    rows_by_rule: dict[str, list[dict]] = {rule_id: [] for rule_id in NEW_RULE_IDS}
    legacy_semantic_rows: list[dict] = []

    for section in ("orthography", "punctuation"):
        for row in data[section].values():
            implementation = row["implementation"]
            rule_ids = set(implementation["rule_ids"])
            for rule_id in rule_ids & set(NEW_RULE_IDS):
                rows_by_rule[rule_id].append(implementation)
            if implementation["layer"] == "semantic":
                legacy_semantic_rows.extend(
                    {"implementation": implementation, "legacy_rule_id": rule_id}
                    for rule_id in rule_ids & LEGACY_SEMANTIC_OVERLAP_IDS
                )

    expected_layers = {
        **{rule_id: "quotation_dialogue" for rule_id in QUOTATION_RULE_IDS},
        **{rule_id: "casing" for rule_id in CASING_RULE_IDS},
        **{rule_id: "semantic" for rule_id in SEMANTIC_RULE_IDS},
    }
    for rule_id, expected_layer in expected_layers.items():
        assert rows_by_rule[rule_id], rule_id
        assert any(row["layer"] == expected_layer for row in rows_by_rule[rule_id]), rule_id
        for row in rows_by_rule[rule_id]:
            if row["layer"] != expected_layer:
                continue
            assert row["executable"] is True
            assert row["sub_rule_ids"], rule_id
            assert row["tests"], rule_id

    assert legacy_semantic_rows == []


@pytest.mark.parametrize(
    ("family", "expected_rule_ids"),
    (
        ("quotation_dialogue", set(QUOTATION_RULE_IDS) - {"dialogue_bracket_guards"}),
        ("casing", set(CASING_RULE_IDS) - {"casing_formal_you_guard"}),
        ("semantic", {"semantic_service_words", "semantic_derived_prepositions", "semantic_comparative_context"}),
    ),
)
def test_generator_can_sample_each_new_family_by_mix(family: str, expected_rule_ids: set[str]) -> None:
    generator = online_generator_from_config(_new_family_config(family), seed=101)

    examples = [generator.sample_by_index(index) for index in range(160)]

    assert {example.metadata["family"] for example in examples} == {family}
    assert {example.metadata["layer"] for example in examples} == {family}
    assert expected_rule_ids <= {example.primary_rule_id for example in examples}
    assert all(validate_generated_pair(example) == [] for example in examples)


def test_each_new_rule_samples_every_declared_mode() -> None:
    config = _new_layers_only_config()
    generator = online_generator_from_config(config, seed=202)

    for rule_id in NEW_RULE_IDS:
        rule = generator.registry.get_rule(rule_id)
        assert rule is not None, rule_id
        for mode in (GenerationMode.POSITIVE, GenerationMode.HARD_NEGATIVE, GenerationMode.CLEAN_IDENTITY):
            if not rule.can_generate(mode):
                continue
            example = generator.sample(rule_id=rule_id, mode=mode)
            assert example.primary_rule_id == rule_id
            assert example.mode == mode.value
            assert validate_generated_pair(example) == []


def test_mixed_generation_validates_and_audits_new_layers() -> None:
    generator = online_generator_from_config(_new_layers_only_config(), seed=303)

    validation_examples = [generator.sample_by_index(index) for index in range(500)]
    audit_examples = [generator.sample_by_index(index) for index in range(1000)]

    assert all(validate_generated_pair(example) == [] for example in validation_examples)
    audit = audit_batch(audit_examples)
    assert audit["failed_examples_count"] == 0
    assert set(audit["layer_distribution"]) == {"casing", "quotation_dialogue", "semantic"}
    assert set(audit["family_distribution"]) == {"casing", "quotation_dialogue", "semantic"}


def test_new_layers_duplicate_pair_rate_stays_below_threshold() -> None:
    generator = online_generator_from_config(_new_layers_only_config(), seed=404)
    examples = [generator.sample_by_index(index) for index in range(2000)]
    report = diversity_report(examples)

    assert report["duplicate_pair_rate"] <= 0.15
    assert set(report["duplicate_rate_by_layer"]) == {"casing", "quotation_dialogue", "semantic"}
    assert set(report["layer_distribution"]) == {"casing", "quotation_dialogue", "semantic"}


@pytest.mark.parametrize(
    ("family", "max_duplicate_pair_rate"),
    (
        ("quotation_dialogue", 0.20),
        ("casing", 0.15),
        ("semantic", 0.15),
    ),
)
def test_each_new_layer_2000_example_duplicate_pair_rate_stays_below_threshold(
    family: str,
    max_duplicate_pair_rate: float,
) -> None:
    generator = online_generator_from_config(_new_family_config(family), seed=404)
    examples = [generator.sample_by_index(index) for index in range(2000)]
    report = diversity_report(examples)

    assert report["duplicate_pair_rate"] <= max_duplicate_pair_rate


def test_generation_diversity_audit_prints_layer_family_rule_and_subrule_sections() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_generation_diversity.py",
            "configs/config.yaml",
            "--count",
            "40",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for heading in (
        "layer distribution:",
        "family distribution:",
        "rule_id distribution:",
        "sub_rule_id distribution:",
        "duplicate rate per layer:",
        "top duplicates:",
    ):
        assert heading in result.stdout


def _new_family_config(family: str) -> dict:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["generation"]["enabled_rule_groups"] = [family]
    config["generation"]["mix"] = {family: 1.0}
    config["generation"]["grammar"]["max_generation_retries"] = 100
    return config


def _new_layers_only_config() -> dict:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["generation"]["enabled_rule_groups"] = ["quotation_dialogue", "casing", "semantic"]
    config["generation"]["mix"] = {
        "quotation_dialogue": 0.34,
        "casing": 0.33,
        "semantic": 0.33,
    }
    config["generation"]["grammar"]["max_generation_retries"] = 100
    return config


class _FakeEncoder(__import__("torch").nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        from types import SimpleNamespace

        import torch

        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = torch.nn.Embedding(64, hidden_size)

    def __call__(self, input_ids, attention_mask=None, **kwargs):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))
