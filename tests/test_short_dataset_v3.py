from pathlib import Path

import yaml


def test_short_dataset_v3_config_is_canonical_single_artifact():
    config = yaml.safe_load(Path("configs/config.short_dataset_v3.yaml").read_text(encoding="utf-8"))

    assert config["data"]["processed_train_path"] == "data/processed/short_dataset_v3/correction_dataset.csv.gz"
    assert config["data"]["manifest_path"] == "reports/short_dataset_v3/dataset_manifest.json"
    assert config["data"]["target_total_examples"] == 100000
    assert config["data"]["train_examples"] == 80000
    assert config["data"]["val_examples"] == 10000
    assert config["data"]["test_examples"] == 10000
    assert config["data"]["short_dataset_v3"]["fallback_split_sizes"] == {"train": 50000, "val": 5000, "test": 5000}


def test_train_short_v3_configs_use_versioned_outputs():
    e1 = yaml.safe_load(Path("configs/config.train_short_v3_e1.yaml").read_text(encoding="utf-8"))
    e2 = yaml.safe_load(Path("configs/config.train_short_v3_e2.yaml").read_text(encoding="utf-8"))

    for config in (e1, e2):
        assert config["data"]["processed_train_path"] == "data/processed/short_dataset_v3/correction_dataset.csv.gz"
        assert config["data"]["manifest_path"] == "reports/short_dataset_v3/dataset_manifest.json"
        assert config["thresholds"]["mode"] == "calibrated_val_guarded_v3"
        assert config["model"]["max_sequence_length"] == 128
        assert config["model"]["max_candidates"] == 16
        assert "latest" not in config["paths"]["adapter_output_dir"]
        assert "latest" not in config["paths"]["heads_output_dir"]

    assert e1["training"]["epochs"] == 1
    assert e2["training"]["epochs"] == 2
