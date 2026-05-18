from pathlib import Path

import yaml

from src.candidates.candidate_generator import CandidateGenerator
from src.config.dictionary import dictionary_provider_from_config, load_dictionary_lexicon


def test_main_config_is_full_train_profile_for_450k_dataset():
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))

    assert config["training"]["mode"] == "full-train"
    assert config["training"]["run_model_training"] is True
    assert 10_000 <= config["training"]["max_train_examples"] <= 405_000
    assert 1_000 <= config["training"]["max_val_examples"] <= 22_500
    assert 1_000 <= config["training"]["max_test_examples"] <= 22_500
    assert config["data"]["target_total_examples"] == 450000
    assert config["data"]["processed_train_path"] == "data/processed/correction_dataset.csv.gz"
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
        "russian_wikipedia_dump",
    }
    assert isinstance(config["model"]["local_files_only"], bool)
    assert config["model"]["max_sequence_length"] == 128
    assert 1 <= config["training"]["batch_size"] <= 4
    assert config["training"]["gradient_accumulation_steps"] >= 4
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
            "ne_adjective_threshold": 0.98,
            "ne_participle_threshold": 0.99,
            "comma_subordinate_threshold": 0.92,
            "introductory_word_threshold": 0.94,
            "dash_subject_predicate_threshold": 0.96,
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
