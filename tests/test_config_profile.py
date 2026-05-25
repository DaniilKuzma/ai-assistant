from pathlib import Path

import yaml

from src.candidates.candidate_generator import CandidateGenerator
from src.config.dictionary import dictionary_provider_from_config, load_dictionary_lexicon


def test_main_config_is_canonical_train_profile_for_candidate_opportunity_dataset():
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))

    assert config["training"]["mode"] == "train"
    assert config["training"]["run_model_training"] is True
    assert config["training"]["max_train_examples"] == 256000
    assert config["training"]["max_val_examples"] == 32000
    assert config["training"]["max_test_examples"] == 32000
    assert config["data"]["target_total_examples"] == 320000
    assert config["data"]["total_examples"] == 320000
    assert config["data"]["dataset_contract"] == "candidate_opportunity"
    assert config["data"]["dataset_build_workers"] == 8
    assert config["data"]["processed_train_path"] == "data/processed/correction_dataset.csv.gz"
    assert config["data"]["manifest_path"] == "data/processed/dataset_manifest.json"
    assert config["data"]["composition"] == {
        "atomic_positive_ratio": 0.59375,
        "atomic_hard_negative_ratio": 0.2375,
        "clean_identity_ratio": 0.10,
        "stress_multi_error_ratio": 0.05,
        "real_atomic_train_ratio": 0.01875,
        "allow_layer_target_adjustment": True,
        "fail_on_unadjusted_layer_deficit": True,
    }
    assert config["data"]["clean_pool"]["reject_mixed_script_tokens"] is True
    assert config["data"]["clean_pool"]["reject_latin_confusable_inside_cyrillic_word"] is True
    assert config["data"]["clean_pool"]["reject_if_candidate_generator_finds_high_confidence_fix"] is True
    assert config["data"]["clean_pool"]["high_confidence_candidate_threshold"] == 0.95
    assert config["data"]["synthetic"]["require_atomic_positive"] is True
    assert config["data"]["synthetic"]["max_gold_edits_for_atomic"] == 1
    assert config["data"]["synthetic"]["require_candidate_coverage"] is True
    assert config["data"]["synthetic"]["require_strict_validator_acceptance"] is True
    assert config["data"]["real_pairs"]["train_policy"] == "atomize_single_edit_known_rule_only"
    assert config["data"]["real_pairs"]["unknown_rule_policy"] == "mining_only"
    assert config["data"]["real_pairs"]["multi_edit_policy"] == "stress_or_eval_only"
    assert config["data"]["stress"]["enabled"] is True
    assert config["data"]["stress"]["count_toward_rule_quota"] is False
    assert config["data"]["stress"]["loss_weight"] == 0.4
    assert config["data"]["stress"]["min_ratio"] == 0.03
    assert config["data"]["stress"]["max_ratio"] == 0.05
    assert config["data"]["stress"]["fail_on_under_target"] is False
    assert config["data"]["rule_activation"]["expected_min_training_candidate_rule_count"] == 76
    assert config["data"]["rule_activation"]["target_training_candidate_rule_count"] == 76
    assert config["data"]["rule_activation"]["expected_min_final_active_rule_count"] == 25
    assert config["data"]["rule_activation"]["target_final_active_rule_count"] == 76
    assert config["data"]["rule_activation"]["fail_below_final_active_rule_count"] is True
    assert config["data"]["rule_activation"]["warn_below_target_final_active_rule_count"] is True
    assert config["data"]["rule_quota"]["min_atomic_positives_per_active_rule"] == 500
    assert config["data"]["rule_quota"]["preferred_atomic_positives_per_active_rule"] == 1500
    assert config["data"]["rule_quota"]["max_total_per_rule_id"] == 1500
    assert config["data"]["rule_quota"]["min_hard_negatives_per_active_rule"] == 200
    assert config["data"]["rule_quota"]["disable_rule_if_quota_not_met"] is True
    assert config["data"]["rule_data_compiler"] == {
        "enabled": True,
        "source_priority": [
            "corpus_mined",
            "syntax_mined",
            "morphology_mined",
            "real_pattern_replay",
            "rule_lab",
        ],
        "max_rule_lab_share_per_rule": 0.30,
        "fail_on_high_rule_lab_share": False,
        "min_unique_sentence_patterns_per_rule": 20,
        "min_unique_left_contexts_per_rule": 100,
        "min_unique_right_contexts_per_rule": 100,
        "max_near_duplicate_share": 0.02,
        "fail_on_low_structural_diversity": False,
    }
    assert config["data"]["audit"]["fail_on_extra_edits_in_atomic"] is True
    assert config["data"]["audit"]["fail_on_unknown_rule_in_train"] is True
    assert config["data"]["audit"]["fail_on_stale_reports"] is True
    assert config["data"]["audit"]["min_candidate_recall_for_active_rule"] == 0.95
    assert config["data"]["audit"]["fail_on_corpus_opportunity_share_below_threshold"] is False
    assert config["data"]["audit"]["destructive_diversity_pruning_enabled"] is False
    assert "training_dataset" in config["data"]
    assert "training_dataset_core" in config["data"]
    assert config["data"]["training_dataset"]["legacy_builder"] is False
    assert config["data"]["training_dataset_core"]["legacy_builder"] is False
    for section_name in ("training_dataset", "training_dataset_core"):
        section = config["data"][section_name]
        assert section["active_rule_quota"]["min_total_per_active_rule"] == 500
        assert section["active_rule_quota"]["preferred_total_per_active_rule"] == 1500
        assert section["rule_caps"]["max_total_per_rule_id"] == 1500
        assert section["rule_caps"]["max_train_per_rule_id"] == 1200
    assert config["data"]["clean_corpus"]["enabled"] is True
    assert config["data"]["external_local_files_only"] is True
    assert config["data"]["punctuation_hard_negative_clean_ratio"] > 0.0
    assert config["dictionary"]["enabled"] is True
    assert config["dictionary"]["lexicon_path"] == "data/processed/russian_lexicon.txt"
    assert config["dictionary"]["max_candidates"] == 2
    assert config["dictionary"]["min_score"] == 85
    assert config["dictionary"]["yo_e"]["enabled"] is False
    assert config["dictionary"]["yo_e"]["mode"] == "model_required"
    assert config["dictionary"]["yo_e"]["require_lexicon_support"] is True
    assert config["dictionary"]["yo_e"]["require_model_scoring"] is True
    assert "synthetic_balance" not in config["data"]
    assert {source["name"] for source in config["data"]["clean_corpus"]["sources"]} >= {
        "leipzig_news",
        "leipzig_wikipedia",
        "taiga_rest",
        "taiga_proza_filtered",
        "ud_russian_taiga",
        "opencorpora",
        "tatoeba_russian",
    }
    assert isinstance(config["model"]["local_files_only"], bool)
    assert config["model"]["max_sequence_length"] == 128
    assert config["training"]["batch_size"] == 128
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["training"]["show_progress"] is True
    assert config["training"]["progress_log_every_steps"] == 1000


def test_project_model_references_use_ruroberta_large_only():
    checked_paths = [
        Path("configs/config.yaml"),
        Path("AI_INDEX.md"),
        Path("docs/superpowers/specs/2026-05-15-russian-edit-corrector-design.md"),
        Path("src/model/encoder.py"),
        Path("src/model/edit_model.py"),
        Path("src/inference/model_corrector.py"),
        Path("src/training/train.py"),
        Path("tests/test_train_entrypoint.py"),
    ]
    stale: list[str] = []

    markers = ("Ru" + "BERT", "Ru" + "bert", "ru" + "Bert", "ru" + "bert")
    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker in text:
                stale.append(f"{path}:{marker}")

    assert stale == []


def test_threshold_profiles_include_rule_specific_thresholds():
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    profiles = config["thresholds"]
    expected = {
        "conservative": {
            "default_threshold": 0.90,
            "dictionary_threshold": 0.95,
            "tsya_threshold": 0.995,
            "ne_adjective_threshold": 0.99995,
            "ne_participle_threshold": 0.99995,
            "comma_subordinate_threshold": 0.92,
            "introductory_word_threshold": 0.94,
            "dash_subject_predicate_threshold": 0.96,
            "quote_open_threshold": 0.99999,
            "quote_close_threshold": 0.99999,
            "quote_pair_balance_threshold": 0.995,
            "bracket_pair_balance_threshold": 0.995,
            "capitalization_ner_threshold": 0.999999,
            "capitalization_sentence_start_threshold": 0.98,
            "n_nn_adjective_threshold": 0.995,
            "n_nn_participle_threshold": 0.999,
            "n_nn_short_form_threshold": 0.99995,
            "ne_adverb_threshold": 0.99995,
            "subject_predicate_dash_threshold": 0.9999,
            "direct_speech_colon_threshold": 0.995,
            "direct_speech_quotes_threshold": 0.995,
            "comma_conjunction_threshold": 0.995,
            "introductory_comma_threshold": 0.97,
            "punctuation_delete_replace_threshold": 0.9999,
        },
        "balanced": {
            "default_threshold": 0.85,
            "dictionary_threshold": 0.92,
            "tsya_threshold": 0.97,
            "ne_adjective_threshold": 0.96,
            "ne_participle_threshold": 0.98,
            "comma_subordinate_threshold": 0.88,
            "introductory_word_threshold": 0.92,
            "dash_subject_predicate_threshold": 0.93,
        },
        "aggressive": {
            "default_threshold": 0.80,
            "dictionary_threshold": 0.88,
            "tsya_threshold": 0.97,
            "ne_adjective_threshold": 0.94,
            "ne_participle_threshold": 0.96,
            "comma_subordinate_threshold": 0.82,
            "introductory_word_threshold": 0.86,
            "dash_subject_predicate_threshold": 0.90,
        },
    }

    for profile_name, thresholds in expected.items():
        profile = profiles[profile_name]
        for key, value in thresholds.items():
            assert profile[key] == value
        assert profile["tsya_threshold"] > profile["spelling_threshold"]
        assert profile["ne_participle_threshold"] > profile["spelling_threshold"]
        assert profile["punctuation_delete_threshold"] > profile["punctuation_threshold"]


def test_dictionary_config_provider_loads_normalized_cached_lexicon(tmp_path):
    lexicon_path = tmp_path / "russian_lexicon.txt"
    lexicon_path.write_text("\nКОРОВА\nмолоко\nкорова\nАпелляция\n", encoding="utf-8")

    first = load_dictionary_lexicon(lexicon_path)
    second = load_dictionary_lexicon(lexicon_path)
    provider = dictionary_provider_from_config(
        {
            "dictionary": {
                "enabled": True,
                "lexicon_path": str(lexicon_path),
                "max_candidates": 2,
                "min_score": 85,
            }
        }
    )

    assert first == ("корова", "молоко", "апелляция")
    assert second is first
    assert provider is not None
    assert provider.get_lexicon() == first


def test_dictionary_config_provider_feeds_candidate_generator(tmp_path):
    lexicon_path = tmp_path / "russian_lexicon.txt"
    lexicon_path.write_text("библиотека\nмолоко\nтерритория\nапелляция\n", encoding="utf-8")

    generator = CandidateGenerator.from_config(
        {
            "dictionary": {
                "enabled": True,
                "lexicon_path": str(lexicon_path),
                "max_candidates": 2,
                "min_score": 85,
            }
        }
    )
    candidates = generator.generate("Библеотека рядом.")

    assert any(candidate.replacement == "Библиотека" and candidate.rule_id == "dictionary_fuzzy" for candidate in candidates)


def test_dictionary_config_provider_passes_opt_in_yo_e_policy_to_candidate_generator(tmp_path):
    lexicon_path = tmp_path / "russian_lexicon.txt"
    lexicon_path.write_text("елка\nёлка\n", encoding="utf-8")

    generator = CandidateGenerator.from_config(
        {
            "dictionary": {
                "enabled": True,
                "lexicon_path": str(lexicon_path),
                "max_candidates": 2,
                "min_score": 85,
                "yo_e": {
                    "enabled": True,
                    "mode": "model_required",
                    "require_lexicon_support": True,
                    "require_model_scoring": True,
                },
            }
        }
    )
    candidates = generator.generate("Елка рядом.")

    assert any(candidate.replacement == "Ёлка" and candidate.rule_id == "yo_e_candidate" for candidate in candidates)


def test_build_russian_lexicon_script_writes_non_empty_normalized_lexicon(tmp_path):
    from scripts.build_russian_lexicon import build_lexicon

    output = tmp_path / "russian_lexicon.txt"

    stats = build_lexicon(output, max_entries=250)
    entries = output.read_text(encoding="utf-8").splitlines()

    assert stats["written"] == len(entries)
    assert stats["written"] > 100
    assert entries == sorted(set(entries))
    assert all(entry == entry.lower() for entry in entries)
    assert all(entry and all(char == "-" or "а" <= char <= "я" or char == "ё" for char in entry) for entry in entries)
