from pathlib import Path

import yaml


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
    assert config["data"]["synthetic_balance"]["spelling_min_examples"] >= 30_000
    assert 10_000 <= config["data"]["synthetic_balance"]["split_join_min_examples"] <= 30_000
    assert 10_000 <= config["data"]["synthetic_balance"]["hyphen_min_examples"] <= 30_000
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
    assert config["model"]["local_files_only"] is True
    assert config["model"]["max_sequence_length"] == 128
    assert 1 <= config["training"]["batch_size"] <= 4
    assert config["training"]["gradient_accumulation_steps"] >= 4
    assert config["training"]["show_progress"] is True
    assert config["training"]["progress_log_every_steps"] == 1000
