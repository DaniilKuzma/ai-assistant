import json
from pathlib import Path

import pandas as pd

from src.config.candidate_dataset_config import candidate_dataset_core_compat_config, candidate_dataset_paths
from src.config.load_config import load_config
from src.data._training_dataset_builder import _effective_active_rule_ids, _rule_counts_from_rows, _rule_id_counts
from src.data.full_dataset_builder import build_dataset_from_config
from src.data.operator_dataset_builder import _rule_counts as _operator_rule_counts
from src.rules.capabilities import RuleCapability


def test_candidate_dataset_blocks_when_clean_pool_missing(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS", raising=False)
    config = load_config("configs/config.yaml")
    candidate = config["data"]["candidate_opportunity"]
    candidate["paths"]["correction_dataset_path"] = str(tmp_path / "data" / "correction_dataset.csv.gz")
    candidate["paths"]["manifest_path"] = str(tmp_path / "reports" / "dataset_manifest.json")
    candidate["paths"]["reports_dir"] = str(tmp_path / "reports")
    candidate["paths"]["clean_pool_path"] = str(tmp_path / "missing-clean-pool.csv.gz")
    config["paths"]["reports_dir"] = str(tmp_path / "reports")
    candidate["totals"] = {"total_examples": 30, "train_examples": 20, "val_examples": 5, "test_examples": 5}

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads((tmp_path / "reports" / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert result["verdict"] == "DATASET_BLOCKED"
    assert manifest["verdict"] == "DATASET_BLOCKED"
    assert manifest["audit_errors"] == ["missing_clean_sentence_pool"]
    assert not (tmp_path / "data" / "correction_dataset.csv.gz").exists()


def test_candidate_dataset_quota_compat_view_uses_canonical_block(tmp_path: Path):
    config = _tiny_quota_config(tmp_path)
    paths = candidate_dataset_paths(config)
    core = candidate_dataset_core_compat_config(config)

    expected_rules = [
        "comma_subordinate",
        "subject_predicate_dash",
        "homogeneous_comma",
        "direct_speech_dash",
        "address_comma",
        "comma_conjunction",
        "hyphen_whitelist",
        "capitalization_ner",
        "yo_e_candidate",
    ]
    assert paths["correction_dataset_path"] == str(tmp_path / "data" / "training_dataset_core" / "correction_dataset.csv.gz")
    assert core["legacy_builder"] is False
    assert core["requested_total"] == 80
    assert core["active_rule_quota"]["rule_ids"] == expected_rules
    assert core["active_rule_quota"]["min_total_per_active_rule"] == 3
    assert core["active_rule_quota"]["preferred_total_per_active_rule"] == 3
    assert core["rule_caps"]["max_total_per_rule_id"] == 3
    assert core["source_type_targets"]["synthetic_augmented_from_open_clean"] == 22
    assert core["source_type_targets"]["clean_identity_from_open_clean"] == 29
    assert core["source_type_targets"]["hard_negative_from_open_clean"] == 29


def test_effective_active_rule_ids_filter_blocked_eval_and_mining_rules():
    config = load_config("configs/config.yaml")
    core_config = candidate_dataset_core_compat_config(config)
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
            "edits": json.dumps([{"source": "РјР»РѕРєРѕ", "replacement": "РјРѕР»РѕРєРѕ"}], ensure_ascii=False),
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
    config = load_config("configs/config.yaml")
    candidate = config["data"]["candidate_opportunity"]
    candidate["paths"]["correction_dataset_path"] = str(tmp_path / "data" / "training_dataset_core" / "correction_dataset.csv.gz")
    candidate["paths"]["manifest_path"] = str(tmp_path / "reports" / "training_dataset_core" / "dataset_manifest.json")
    candidate["paths"]["reports_dir"] = str(tmp_path / "reports" / "training_dataset_core")
    candidate["paths"]["clean_pool_path"] = str(tmp_path / "data" / "training_dataset_core" / "clean_sentence_pool.csv.gz")
    config["paths"]["reports_dir"] = str(tmp_path / "reports" / "training_dataset_core")
    candidate["totals"] = {"total_examples": 80, "train_examples": 60, "val_examples": 10, "test_examples": 10}
    candidate["composition"] = {
        "atomic_positive_target": 22,
        "real_atomic_train_target": 0,
        "clean_identity_target": 29,
        "atomic_hard_negative_target": 29,
        "stress_multi_error_target": 0,
    }
    candidate["min_clean_pool_for_ready"] = 40
    candidate["min_clean_pool_hard_min"] = 40
    candidate["max_source_share"] = 1.0
    candidate["max_subcorpus_share"] = 1.0
    candidate["audit"]["min_active_rule_count"] = 3
    candidate["audit"]["require_all_source_types"] = False
    candidate["audit"]["corpus_opportunity_share_min"] = 0.0
    candidate["audit"]["fallback_template_share_max"] = 1.0
    candidate["rule_quota"] = {
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
        "min_atomic_positives_per_active_rule": 3,
        "preferred_atomic_positives_per_active_rule": 3,
        "max_total_per_rule_id": 3,
        "min_hard_negatives_per_active_rule": 0,
        "disable_rule_if_quota_not_met": True,
        "split_minimums": {"train": 1, "val": 1, "test": 1},
    }
    return config


def _clean_sentences(count: int) -> list[str]:
    base = [
        "Р­РєСЃРїРµСЂС‚С‹ СЃРѕРѕР±С‰РёР»Рё, С‡С‚Рѕ Р¶РёР·РЅСЊ РІ РіРѕСЂРѕРґРµ СЃС‚Р°Р»Р° СЃРїРѕРєРѕР№РЅРµРµ РїРѕСЃР»Рµ СЂРµС„РѕСЂРјС‹.",
        "Р РµРґР°РєС†РёСЏ РѕС‚РјРµС‚РёР»Р°, С‡С‚Рѕ Р±РёР±Р»РёРѕС‚РµРєР° РѕС‚РєСЂС‹Р»Р°СЃСЊ РїРѕСЃР»Рµ РґРѕР»РіРѕР№ СЂРµРєРѕРЅСЃС‚СЂСѓРєС†РёРё.",
        "РђРЅР°Р»РёС‚РёРєРё СЃС‡РёС‚Р°СЋС‚, С‡С‚Рѕ С†РёС„СЂРѕРІРѕР№ РѕС‚С‡РµС‚ РїРѕРјРѕРіР°РµС‚ РѕС†РµРЅРёС‚СЊ СЂР°Р±РѕС‚Сѓ СЂРµРіРёРѕРЅР°.",
        "РљРѕРјРїР°РЅРёСЏ Р·Р°СЏРІРёР»Р°, С‡С‚Рѕ РЅРѕРІС‹Р№ РїРѕРґСЉРµР·Рґ Рє СЃС‚Р°РЅС†РёРё Р±СѓРґРµС‚ РіРѕС‚РѕРІ РѕСЃРµРЅСЊСЋ.",
        "РСЃСЃР»РµРґРѕРІР°С‚РµР»Рё РїРѕРґС‡РµСЂРєРЅСѓР»Рё, С‡С‚Рѕ РґР»РёРЅРЅС‹Р№ РїРµСЂРёРѕРґ РЅР°Р±Р»СЋРґРµРЅРёР№ РїРѕРІС‹СЃРёР» С‚РѕС‡РЅРѕСЃС‚СЊ.",
        "РљРѕРјРёСЃСЃРёСЏ СЂРµС€РёР»Р°, С‡С‚Рѕ РєС‚Рѕ-С‚Рѕ РґРѕР»Р¶РµРЅ РїСЂРѕРІРµСЂРёС‚СЊ РґРѕРєСѓРјРµРЅС‚С‹ РїРµСЂРµРґ РїСѓР±Р»РёРєР°С†РёРµР№.",
        "РђРІС‚РѕСЂС‹ СЃРѕРѕР±С‰РёР»Рё: В«РџСЂРѕРµРєС‚ РіРѕС‚РѕРІВ», Рё СЌРєСЃРїРµСЂС‚С‹ РїСЂРёРЅСЏР»Рё РёС‚РѕРіРѕРІС‹Р№ РѕС‚С‡РµС‚.",
        "РљРѕРіРґР° РґРѕРєСѓРјРµРЅС‚ Р±СѓРґРµС‚ РіРѕС‚РѕРІ, РєРѕРјР°РЅРґР° РѕС‚РїСЂР°РІРёС‚ РµРіРѕ РІ РіРѕСЂРѕРґСЃРєРѕР№ Р°СЂС…РёРІ.",
    ]
    return [sentence.replace(".", f" {index}.") for index in range(count) for sentence in [base[index % len(base)]]]
