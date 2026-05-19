import json
from pathlib import Path

import pandas as pd

from src.config.load_config import load_config
from src.data.full_dataset_builder import build_dataset_from_config


def test_short_dataset_v2_builds_from_open_clean_sources_without_meta_templates(tmp_path: Path):
    clean_path = tmp_path / "clean.txt"
    clean_path.write_text("\n".join(_clean_sentences(120)) + "\n", encoding="utf-8")
    real_path = tmp_path / "real.jsonl"
    real_path.write_text(
        '{"source": "Жызнь прекрасна.", "correction": "Жизнь прекрасна.", "domain": "unit"}\n',
        encoding="utf-8",
    )

    config = load_config("configs/config.short_dataset_v2.yaml")
    config["data"]["processed_train_path"] = str(tmp_path / "data" / "short_dataset_v2" / "correction_dataset.csv.gz")
    config["data"]["manifest_path"] = str(tmp_path / "reports" / "short_dataset_v2" / "dataset_manifest.json")
    config["paths"]["reports_dir"] = str(tmp_path / "reports" / "short_dataset_v2")
    config["data"]["target_total_examples"] = 80
    config["data"]["train_examples"] = 60
    config["data"]["val_examples"] = 10
    config["data"]["test_examples"] = 10
    config["data"]["short_dataset_v2"]["source_type_targets"] = {
        "synthetic_augmented": 48,
        "real_error_pair": 1,
        "clean_identity": 15,
        "hard_negative": 16,
    }
    config["data"]["short_dataset_v2"]["split_source_type_targets"] = {
        "train": {"synthetic_augmented": 36, "real_error_pair": 1, "clean_identity": 11, "hard_negative": 12},
        "val": {"synthetic_augmented": 6, "real_error_pair": 0, "clean_identity": 2, "hard_negative": 2},
        "test": {"synthetic_augmented": 6, "real_error_pair": 0, "clean_identity": 2, "hard_negative": 2},
    }
    config["data"]["short_dataset_v2"]["min_clean_pool_for_ready"] = 40
    config["data"]["short_dataset_v2"]["pool"]["min_clean_sentences"] = 40
    config["data"]["short_dataset_v2"]["pool"]["max_source_share"] = 1.0
    config["data"]["short_dataset_v2"]["pool"]["max_subcorpus_share"] = 1.0
    config["data"]["short_dataset_v2"]["audit"]["min_active_rule_count"] = 0
    config["data"]["short_dataset_v2"]["audit"]["require_all_source_types"] = False
    config["data"]["short_dataset_v2"]["open_corpora_sources"] = {
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
    config["data"]["short_dataset_v2"]["real_error_sources"] = {
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

    result = build_dataset_from_config(config, force=True)
    frame = pd.read_csv(tmp_path / "data" / "short_dataset_v2" / "correction_dataset.csv.gz")
    manifest = json.loads((tmp_path / "reports" / "short_dataset_v2" / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert result["total"] == 80
    assert frame["split"].value_counts().to_dict() == {"train": 60, "val": 10, "test": 10}
    assert set(frame["source_type"]) == {"synthetic_augmented", "real_error_pair", "clean_identity", "hard_negative"}
    assert not frame["source"].str.contains("правило|серии|семейство|context-pairs", case=False, regex=True).any()
    assert not frame["target"].str.contains("правило|серии|семейство|context-pairs", case=False, regex=True).any()
    assert manifest["verdict"] == "READY_FOR_SHORT_TRAINING_DATASET_V2"
    assert manifest["composition"]["synthetic_augmented"] == 48
    assert manifest["composition"]["real_error_pair"] == 1
    assert manifest["clean_source_counts"]["unit_news"] >= 40


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
