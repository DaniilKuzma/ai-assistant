import json
from pathlib import Path

import pandas as pd

from src.config.load_config import load_config
from src.data.full_dataset_builder import build_dataset_from_config


def test_training_dataset_core_blocks_when_required_sources_missing_and_downloads_disabled(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS", raising=False)
    config = load_config("configs/config.yaml")
    config["data"]["processed_train_path"] = str(tmp_path / "data" / "correction_dataset.csv.gz")
    config["data"]["manifest_path"] = str(tmp_path / "reports" / "dataset_manifest.json")
    config["paths"]["reports_dir"] = str(tmp_path / "reports")
    config["data"]["target_total_examples"] = 30
    config["data"]["train_examples"] = 20
    config["data"]["val_examples"] = 5
    config["data"]["test_examples"] = 5
    config["data"]["training_dataset_core"]["open_corpora_sources"] = {
        "clean_sources": {
            "missing_clean": {
                "enabled": True,
                "type": "local_text",
                "local_path": str(tmp_path / "missing-clean.txt"),
            }
        },
        "download_policy": {"mode": "local_first_with_controlled_downloads"},
    }
    config["data"]["training_dataset_core"]["real_error_sources"] = {
        "real_sources": {
            "missing_real": {
                "enabled": True,
                "type": "local_jsonl",
                "local_path": str(tmp_path / "missing-real.jsonl"),
            }
        },
        "download_policy": {"mode": "local_first_with_controlled_downloads"},
    }

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads((tmp_path / "reports" / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert result["verdict"] == "BLOCKED_BY_MISSING_EXTERNAL_SOURCES"
    assert manifest["verdict"] == "BLOCKED_BY_MISSING_EXTERNAL_SOURCES"
    assert manifest["missing_external_sources"]
    assert not (tmp_path / "data" / "correction_dataset.csv.gz").exists()


def test_training_dataset_core_quota_backfill_uses_candidate_backed_rules_and_reports(tmp_path: Path):
    config = _tiny_quota_config(tmp_path)

    result = build_dataset_from_config(config, force=True)
    frame = pd.read_csv(tmp_path / "data" / "training_dataset_core" / "correction_dataset.csv.gz")
    manifest = json.loads((tmp_path / "reports" / "training_dataset_core" / "dataset_manifest.json").read_text(encoding="utf-8"))
    quota = pd.read_csv(tmp_path / "reports" / "training_dataset_core" / "active_rule_quota_report.csv")
    rejected = pd.read_csv(tmp_path / "reports" / "training_dataset_core" / "rejected_backfill_templates.csv")

    assert result["verdict"] == "READY_FOR_TRAINING_DATASET"
    assert frame["split"].value_counts().to_dict() == {"train": 60, "val": 10, "test": 10}
    assert manifest["low_count_active_rule_ids"] == []
    assert manifest["active_rule_quota_summary"]["underfilled_count"] == 0
    assert manifest["composition_by_split"]["val"]["clean_identity_from_open_clean"] >= 2
    assert manifest["composition_by_split"]["test"]["hard_negative_from_open_clean"] >= 2

    expected_rules = {
        "comma_subordinate",
        "homogeneous_comma",
        "subject_predicate_dash",
        "enumeration_colon",
        "direct_speech_colon",
        "direct_speech_quotes",
        "direct_speech_dash",
    }
    assert expected_rules <= set(quota["rule_id"])
    assert set(quota.set_index("rule_id").loc[list(expected_rules), "action"]) <= {"ok", "backfilled"}
    assert all(manifest["rule_id_counts"][rule_id] >= 3 for rule_id in expected_rules)
    assert "no_existing_candidate_backed_pattern" in set(rejected["reason"])
    assert not frame["source"].str.contains("правило|серии|семейство|context-pairs", case=False, regex=True).any()
    assert not frame["target"].str.contains("правило|серии|семейство|context-pairs", case=False, regex=True).any()


def _tiny_quota_config(tmp_path: Path) -> dict:
    clean_path = tmp_path / "clean.txt"
    clean_path.write_text("\n".join(_clean_sentences(180)) + "\n", encoding="utf-8")
    real_path = tmp_path / "real.jsonl"
    real_path.write_text(
        '{"source": "Жызнь в городе стала заметно спокойнее.", "correction": "Жизнь в городе стала заметно спокойнее.", "domain": "unit"}\n',
        encoding="utf-8",
    )
    config = load_config("configs/config.yaml")
    config["data"]["processed_train_path"] = str(tmp_path / "data" / "training_dataset_core" / "correction_dataset.csv.gz")
    config["data"]["manifest_path"] = str(tmp_path / "reports" / "training_dataset_core" / "dataset_manifest.json")
    config["paths"]["reports_dir"] = str(tmp_path / "reports" / "training_dataset_core")
    config["data"]["target_total_examples"] = 80
    config["data"]["train_examples"] = 60
    config["data"]["val_examples"] = 10
    config["data"]["test_examples"] = 10
    core = config["data"]["training_dataset_core"]
    core["source_type_targets"] = {
        "synthetic_augmented_from_open_clean": 40,
        "real_error_pair": 1,
        "clean_identity_from_open_clean": 19,
        "hard_negative_from_open_clean": 20,
    }
    core["split_source_type_targets"] = {
        "train": {
            "synthetic_augmented_from_open_clean": 30,
            "real_error_pair": 1,
            "clean_identity_from_open_clean": 14,
            "hard_negative_from_open_clean": 15,
        },
        "val": {
            "synthetic_augmented_from_open_clean": 5,
            "real_error_pair": 0,
            "clean_identity_from_open_clean": 3,
            "hard_negative_from_open_clean": 2,
        },
        "test": {
            "synthetic_augmented_from_open_clean": 5,
            "real_error_pair": 0,
            "clean_identity_from_open_clean": 2,
            "hard_negative_from_open_clean": 3,
        },
    }
    core["min_clean_pool_for_ready"] = 40
    core["min_clean_pool_hard_min"] = 40
    core["pool"]["min_clean_sentences"] = 40
    core["pool"]["max_source_share"] = 1.0
    core["pool"]["max_subcorpus_share"] = 1.0
    core["audit"]["min_active_rule_count"] = 3
    core["audit"]["require_all_source_types"] = False
    core["active_rule_quota"] = {
        "rule_ids": [
            "comma_subordinate",
            "homogeneous_comma",
            "subject_predicate_dash",
            "enumeration_colon",
            "direct_speech_colon",
            "direct_speech_quotes",
            "direct_speech_dash",
        ],
        "min_total_per_active_rule": 3,
        "preferred_total_per_active_rule": 3,
        "split_minimums": {"train": 1, "val": 1, "test": 1},
    }
    core["open_corpora_sources"] = {
        "clean_sources": {
            "unit_news": {
                "enabled": True,
                "type": "local_text",
                "local_path": str(clean_path),
                "source_subcorpus": "news",
                "domain": "news",
                "style": "neutral",
                "license_status": "unit",
                "max_sentences": 200,
            }
        },
        "download_policy": {"mode": "local_first"},
    }
    core["real_error_sources"] = {
        "real_sources": {
            "unit_pairs": {
                "enabled": True,
                "type": "local_jsonl",
                "local_path": str(real_path),
                "max_examples": 10,
            }
        },
        "download_policy": {"mode": "local_first"},
    }
    return config


def _clean_sentences(count: int) -> list[str]:
    base = [
        "Эксперты сообщили, что жизнь в городе стала спокойнее после реформы.",
        "Редакция отметила, что библиотека открылась после долгой реконструкции.",
        "Аналитики считают, что цифровой отчет помогает оценить работу региона.",
        "Компания заявила, что новый подъезд к станции будет готов осенью.",
        "Исследователи подчеркнули, что длинный период наблюдений повысил точность.",
        "Комиссия решила, что кто-то должен проверить документы перед публикацией.",
        "Авторы сообщили: «Проект готов», и эксперты приняли итоговый отчет.",
        "Когда документ будет готов, команда отправит его в городской архив.",
    ]
    return [sentence.replace(".", f" {index}.") for index in range(count) for sentence in [base[index % len(base)]]]
