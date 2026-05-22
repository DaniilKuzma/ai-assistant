from pathlib import Path

import yaml

from src.data.training_dataset import compute_broad_dataset_targets, resolve_broad_active_training_rules
from src.rules.syntax_synthetic import SUPPORTED_SYNTAX_RULE_IDS


def _config() -> dict:
    return yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))


def test_broad_active_training_rules_include_syntax_and_legacy_candidate_backed_rules():
    rows = resolve_broad_active_training_rules(_config())
    active = {row["rule_id"]: row for row in rows if row["include"]}

    assert len(active) == 76
    assert set(SUPPORTED_SYNTAX_RULE_IDS) <= set(active)
    assert {
        "frequent_error_exact",
        "dictionary_fuzzy",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "swapped_letters_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
        "hyphen_whitelist",
        "final_punctuation_default",
        "capitalization_sentence_start",
        "abbreviation_case_protection",
        "sdelat_prefix",
        "cy_exception",
    } <= set(active)

    excluded = {row["rule_id"]: row["reason"] for row in rows if not row["include"]}
    assert "capitalization_ner" in excluded
    assert "needs_NER" in excluded["capitalization_ner"]
    assert "quote_open" in excluded
    assert "broad_normalization" in excluded["quote_open"]
    assert "quote_close" in excluded
    assert "broad_normalization" in excluded["quote_close"]
    assert "yo_e_candidate" in excluded
    assert "disabled" in excluded["yo_e_candidate"]


def test_broad_active_training_rule_quotas_follow_dataset_plan():
    rows = resolve_broad_active_training_rules(_config())
    active = {row["rule_id"]: row for row in rows if row["include"]}

    assert active["comma_subordinate"]["target_min_examples"] == 1500
    assert active["comma_subordinate"]["target_preferred_examples"] == 2500
    assert active["ne_verb"]["target_min_examples"] == 1500
    assert active["ne_verb"]["target_preferred_examples"] == 2500
    assert active["dictionary_fuzzy"]["target_min_examples"] == 1000
    assert active["dictionary_fuzzy"]["target_preferred_examples"] == 1800
    assert active["final_punctuation_default"]["target_min_examples"] == 1000
    assert active["final_punctuation_default"]["target_preferred_examples"] == 3000


def test_dynamic_targets_scale_from_active_rule_quotas_and_real_pair_count():
    rows = resolve_broad_active_training_rules(_config())
    targets = compute_broad_dataset_targets(rows, real_pair_count=1236)

    assert targets["active_rule_count"] == 76
    assert targets["targeted_synthetic_target"] == sum(
        row["target_preferred_examples"] for row in rows if row["include"]
    )
    assert targets["total_target"] >= 200000
    assert targets["split_sizes"]["train"] == int(targets["total_target"] * 0.8)
    assert targets["split_sizes"]["val"] == int(targets["total_target"] * 0.1)
    assert targets["split_sizes"]["test"] == targets["total_target"] - targets["split_sizes"]["train"] - targets["split_sizes"]["val"]
    assert targets["clean_identity_target"] >= int(targets["total_target"] * 0.10)
    assert targets["hard_negative_target"] >= int(targets["total_target"] * 0.10)
    assert 0.03 <= targets["multi_error_stress_target"] / targets["total_target"] <= 0.05
