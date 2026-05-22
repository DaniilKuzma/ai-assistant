from pathlib import Path

import yaml


def test_canonical_config_uses_broad_dataset_targets_and_no_versioned_paths():
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))

    assert config["data"]["processed_train_path"] == "data/processed/correction_dataset.csv.gz"
    assert config["data"]["manifest_path"] == "data/processed/dataset_manifest.json"
    assert config["data"]["total_examples"] >= 200000
    assert config["data"]["train_examples"] == int(config["data"]["total_examples"] * 0.8)
    assert config["data"]["val_examples"] == int(config["data"]["total_examples"] * 0.1)
    assert config["data"]["test_examples"] == (
        config["data"]["total_examples"] - config["data"]["train_examples"] - config["data"]["val_examples"]
    )
    assert config["training"]["max_train_examples"] == config["data"]["train_examples"]
    assert config["training"]["max_val_examples"] == config["data"]["val_examples"]
    assert config["training"]["max_test_examples"] == config["data"]["test_examples"]

    serialized = yaml.safe_dump(config, allow_unicode=True)
    forbidden = ("short_dataset_v2", "short_dataset_v3", "current_capability_v", "wave", "phase", "latest")
    assert not any(marker in serialized for marker in forbidden)
