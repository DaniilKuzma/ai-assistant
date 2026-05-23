import json
from pathlib import Path

import pandas as pd

from src.config.load_config import load_config
from src.data._training_dataset_builder import _effective_active_rule_ids, _rule_counts_from_rows, _rule_id_counts
from src.data.full_dataset_builder import build_dataset_from_config
from src.data.operator_dataset_builder import _rule_counts as _operator_rule_counts
from src.rules.capabilities import RuleCapability


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
    blocked = pd.read_csv(tmp_path / "reports" / "training_dataset_core" / "blocked_rules_report.csv")
    matrix = pd.read_csv(tmp_path / "reports" / "training_dataset_core" / "rule_capability_matrix.csv")

    assert result["verdict"] == "READY_FOR_TRAINING_DATASET"
    assert frame["split"].value_counts().to_dict() == {"train": 60, "val": 10, "test": 10}
    assert manifest["low_count_active_rule_ids"] == []
    assert manifest["active_rule_quota_summary"]["underfilled_count"] == 0
    assert manifest["composition_by_split"]["val"]["clean_identity_from_open_clean"] >= 2
    assert manifest["composition_by_split"]["test"]["hard_negative_from_open_clean"] >= 2

    expected_rules = {
        "comma_subordinate",
        "subject_predicate_dash",
        "homogeneous_comma",
        "direct_speech_dash",
        "address_comma",
        "comma_conjunction",
        "hyphen_whitelist",
    }
    blocked_config_rules = {"capitalization_ner", "yo_e_candidate"}
    assert expected_rules <= set(manifest["active_rule_ids"])
    assert blocked_config_rules.isdisjoint(set(manifest["active_rule_ids"]))
    assert blocked_config_rules.isdisjoint(set(frame["rule_ids"].astype(str)))
    assert expected_rules <= set(quota["rule_id"])
    assert blocked_config_rules.isdisjoint(set(quota["rule_id"]))
    assert set(quota.set_index("rule_id").loc[list(expected_rules), "action"]) <= {"ok", "backfilled"}
    assert all(manifest["rule_id_counts"][rule_id] >= 3 for rule_id in expected_rules)
    assert "capitalization_ner" in " ".join(blocked["project_rule_ids"].astype(str).tolist())
    assert "yo_e_candidate" in " ".join(blocked["project_rule_ids"].astype(str).tolist())
    assert len(matrix) >= len(blocked)
    assert (tmp_path / "reports" / "training_dataset_core" / "rule_eligibility_report.csv").exists()
    assert (tmp_path / "reports" / "training_dataset_core" / "missing_module_rules_report.csv").exists()
    assert {"rule_id", "reason"} <= set(rejected.columns)
    assert not frame["source"].str.contains("правило|серии|семейство|context-pairs", case=False, regex=True).any()
    assert not frame["target"].str.contains("правило|серии|семейство|context-pairs", case=False, regex=True).any()


def test_effective_active_rule_ids_filter_blocked_eval_and_mining_rules():
    config = load_config("configs/config.yaml")
    core_config = dict(config["data"]["training_dataset_core"])
    quota_config = {
        "rule_ids": [
            "dictionary_fuzzy",
            "comma_subordinate",
            "hyphen_whitelist",
            "final_punctuation_default",
            "frequent_error_exact",
        ]
    }
    capabilities = [
        _capability("dictionary_fuzzy", "BLOCK_NEEDS_DICTIONARY", requires=["dictionary"]),
        _capability("comma_subordinate", "BLOCK_NEEDS_SYNTAX", requires=["syntax"]),
        _capability("hyphen_whitelist", "EVAL_ONLY"),
        _capability("final_punctuation_default", "MINING_ONLY"),
        _capability("frequent_error_exact", "INCLUDE_NOW", eligible=True),
    ]

    assert _effective_active_rule_ids(
        config,
        core_config,
        quota_config=quota_config,
        capabilities=capabilities,
    ) == ["frequent_error_exact"]


def test_rule_quota_counts_only_atomic_positive_contract_rows():
    rows = [
        {
            "rule_ids": json.dumps(["unit_atomic"]),
            "dataset_layer": "atomic_positive",
            "count_toward_rule_quota": True,
            "gold_edit_count": 1,
            "edits": json.dumps([{"source": "млоко", "replacement": "молоко"}], ensure_ascii=False),
        },
        {
            "rule_ids": json.dumps(["unit_atomic"]),
            "dataset_layer": "stress_multi_error",
            "count_toward_rule_quota": True,
            "gold_edit_count": 2,
            "edits": json.dumps([{}, {}]),
        },
        {
            "rule_ids": json.dumps(["unit_atomic"]),
            "dataset_layer": "atomic_positive",
            "count_toward_rule_quota": False,
            "gold_edit_count": 1,
            "edits": json.dumps([{}]),
        },
        {
            "rule_ids": json.dumps(["real_rule"]),
            "dataset_layer": "real_atomic",
            "count_toward_rule_quota": True,
            "gold_edit_count": 1,
            "edits": json.dumps([{}]),
        },
    ]
    frame = pd.DataFrame(rows)

    assert _operator_rule_counts(frame) == {"unit_atomic": 1}
    assert _rule_counts_from_rows(rows) == {"unit_atomic": 1}
    assert _rule_id_counts(frame) == {"unit_atomic": 1}


def _capability(
    rule_id: str,
    decision: str,
    *,
    eligible: bool = False,
    requires: list[str] | None = None,
) -> RuleCapability:
    return RuleCapability(
        taxonomy_key=f"test_{rule_id}",
        domain="test",
        entry_type="rule",
        title=rule_id,
        orfogrammka_id="",
        project_rule_ids=[rule_id],
        implementation_status="current",
        requires=requires or [],
        executable=eligible,
        training_eligible=eligible,
        training_decision=decision,
        training_reason=decision,
        has_candidate_path=eligible,
        has_synthetic_support=eligible,
        has_hard_negative_support=eligible,
        has_validator_support=eligible,
        has_dictionary_support=False,
        has_syntax_support=False,
        has_morphology_support=False,
        has_ner_support=False,
        risk_level="low",
    )


def _tiny_quota_config(tmp_path: Path) -> dict:
    clean_sentences = _clean_sentences(240)
    clean_path = tmp_path / "clean-a.txt"
    clean_path.write_text("\n".join(clean_sentences[:120]) + "\n", encoding="utf-8")
    clean_path_b = tmp_path / "clean-b.txt"
    clean_path_b.write_text("\n".join(clean_sentences[120:]) + "\n", encoding="utf-8")
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
        "real_error_pair": 0,
        "clean_identity_from_open_clean": 20,
        "hard_negative_from_open_clean": 20,
    }
    core["split_source_type_targets"] = {
        "train": {
            "synthetic_augmented_from_open_clean": 30,
            "real_error_pair": 0,
            "clean_identity_from_open_clean": 15,
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
    core["multi_error_stress_target"] = 0
    core["pool"]["min_clean_sentences"] = 40
    core["pool"]["max_source_share"] = 1.0
    core["pool"]["max_subcorpus_share"] = 1.0
    core["audit"]["min_active_rule_count"] = 3
    core["audit"]["require_all_source_types"] = False
    core["audit"]["corpus_opportunity_share_min"] = 0.0
    core["audit"]["fallback_template_share_max"] = 1.0
    core["active_rule_quota"] = {
        "rule_ids": [
            "comma_subordinate",
            "subject_predicate_dash",
            "homogeneous_comma",
            "direct_speech_dash",
            "address_comma",
            "comma_conjunction",
            "hyphen_whitelist",
            "capitalization_ner",
            "yo_e_candidate",
        ],
        "min_total_per_active_rule": 3,
        "preferred_total_per_active_rule": 3,
        "split_minimums": {"train": 1, "val": 1, "test": 1},
    }
    core["open_corpora_sources"] = {
        "clean_sources": {
            "unit_news_a": {
                "enabled": True,
                "type": "local_text",
                "local_path": str(clean_path),
                "source_subcorpus": "news",
                "domain": "news",
                "style": "neutral",
                "license_status": "unit",
                "max_sentences": 200,
            },
            "unit_news_b": {
                "enabled": True,
                "type": "local_text",
                "local_path": str(clean_path_b),
                "source_subcorpus": "analysis",
                "domain": "analysis",
                "style": "neutral",
                "license_status": "unit",
                "max_sentences": 200,
            }
        },
        "download_policy": {"mode": "local_first"},
    }
    core["real_error_sources"] = {
        "real_sources": {},
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
