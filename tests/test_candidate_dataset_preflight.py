from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.rules.capabilities import RuleCapability


def _capability(rule_id: str, decision: str = "INCLUDE_NOW") -> RuleCapability:
    return RuleCapability(
        taxonomy_key=f"fixture_{rule_id}",
        domain="test",
        entry_type="leaf_rule",
        title=rule_id,
        orfogrammka_id="",
        project_rule_ids=[rule_id],
        implementation_status="model_required",
        requires=[],
        executable=True,
        training_eligible=decision == "INCLUDE_NOW",
        training_decision=decision,
        training_reason=decision,
        has_candidate_path=True,
        has_synthetic_support=True,
        has_hard_negative_support=True,
        has_validator_support=True,
        has_dictionary_support=True,
        has_syntax_support=True,
        has_morphology_support=True,
        has_ner_support=True,
        risk_level="low",
    )


def _config(tmp_path: Path) -> dict:
    return {
        "data": {
            "dataset_contract": "candidate_opportunity",
            "clean_pool": {
                "reject_mixed_script_tokens": True,
                "reject_latin_confusable_inside_cyrillic_word": True,
                "reject_if_candidate_generator_finds_high_confidence_fix": True,
                "high_confidence_candidate_threshold": 0.95,
            },
            "rule_activation": {
                "mode": "expanded_safe",
                "include_decisions": [
                    "INCLUDE_NOW",
                    "INCLUDE_AFTER_THRESHOLD_CALIBRATION",
                    "INCLUDE_AFTER_TRAINING",
                ],
                "expected_min_production_ready_rule_count": 12,
                "expected_min_training_candidate_rule_count": 76,
                "expected_min_final_active_rule_count": 25,
                "target_training_candidate_rule_count": 76,
                "target_final_active_rule_count": 76,
                "fail_below_min_training_candidate_rule_count": True,
                "warn_below_target_training_candidate_rule_count": True,
                "fail_below_final_active_rule_count": True,
                "warn_below_target_final_active_rule_count": True,
            },
            "audit": {"fail_on_clean_pool_contamination": False},
            "processed_train_path": str(tmp_path / "processed" / "correction_dataset.csv.gz"),
            "manifest_path": str(tmp_path / "processed" / "dataset_manifest.json"),
        },
        "nlp": {"syntax": {"enabled": True}},
        "dictionary": {"enabled": False},
    }


def _write_config(tmp_path: Path, config: dict | None = None) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config or _config(tmp_path), allow_unicode=True), encoding="utf-8")
    return path


def _patch_dependencies(monkeypatch, *, morphology: bool = True, syntax: bool = True) -> None:
    import scripts.preflight_candidate_dataset as preflight

    def fake_find_spec(name: str):
        if name in {"pymorphy3", "pymorphy2"}:
            return object() if morphology else None
        if name == "natasha":
            return object() if syntax else None
        return object()

    monkeypatch.setattr(preflight.importlib.util, "find_spec", fake_find_spec)


def _patch_capabilities(monkeypatch, *, production: int = 12, training: int = 76) -> None:
    import scripts.preflight_candidate_dataset as preflight

    capabilities = [
        *[_capability(f"prod_{index}", "INCLUDE_NOW") for index in range(production)],
        *[
            _capability(f"candidate_{index}", "INCLUDE_AFTER_TRAINING")
            for index in range(max(0, training - production))
        ],
    ]
    monkeypatch.setattr(preflight, "load_rule_capabilities", lambda _path, config=None: capabilities)


def _patch_activation_counts(monkeypatch, *, production: int = 12, training: int = 76, final: int = 76) -> None:
    import scripts.preflight_candidate_dataset as preflight

    monkeypatch.setattr(preflight, "load_rule_capabilities", lambda _path, config=None: [])
    monkeypatch.setattr(preflight, "expanded_activation_blocked_rows", lambda _capabilities, policy=None, config=None: [])

    def fake_capability_manifest_fields(_capabilities, policy=None):
        return {
            "production_ready_rule_count": production,
            "training_candidate_rule_count": training,
            "active_rule_count": final,
            "final_active_rule_count": final,
            "production_ready_rule_ids": [f"prod_{index}" for index in range(production)],
            "training_candidate_rule_ids": [f"train_{index}" for index in range(training)],
            "active_rule_ids": [f"active_{index}" for index in range(final)],
            "activation_policy": {"errors": [], "warnings": []},
        }

    monkeypatch.setattr(preflight, "capability_manifest_fields", fake_capability_manifest_fields)


def test_stale_dataset_without_contract_columns_is_detected(tmp_path: Path):
    from scripts.preflight_candidate_dataset import check_stale_artifacts

    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    processed.mkdir()
    reports.mkdir()
    pd.DataFrame([{"source": "ошыпка", "target": "ошибка"}]).to_csv(
        processed / "correction_dataset.csv.gz",
        index=False,
    )
    (processed / "dataset_manifest.json").write_text(
        json.dumps({"source_type_counts": {"synthetic": 1}}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = check_stale_artifacts(processed_dir=processed, reports_dir=reports)

    assert result["stale"] is True
    assert "stale_dataset_missing_contract_columns" in result["reasons"]
    assert "stale_manifest_missing_layer_counts" in result["reasons"]
    assert "stale_reports_manifest_missing" in result["reasons"]


def test_exact_active_rule_count_twelve_is_not_required(tmp_path: Path, monkeypatch):
    from scripts.preflight_candidate_dataset import PreflightOptions, run_preflight

    _patch_dependencies(monkeypatch)
    _patch_capabilities(monkeypatch, production=12, training=76)

    result = run_preflight(
        PreflightOptions(
            config_path=_write_config(tmp_path),
            processed_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
        )
    )

    assert result["ok"] is True
    assert result["activation_summary"]["production_ready_rule_count"] == 12
    assert result["activation_summary"]["training_candidate_rule_count"] == 76
    assert result["activation_summary"]["active_rule_count"] == 76
    assert not any("active_rule_count" in error and "12" in error for error in result["errors"])


def test_production_ready_and_training_candidate_minimums_pass(tmp_path: Path, monkeypatch):
    from scripts.preflight_candidate_dataset import PreflightOptions, run_preflight

    _patch_dependencies(monkeypatch)
    _patch_capabilities(monkeypatch, production=13, training=76)

    result = run_preflight(
        PreflightOptions(
            config_path=_write_config(tmp_path),
            processed_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
        )
    )

    assert result["ok"] is True
    assert result["errors"] == []


def test_training_candidate_below_minimum_fails_with_diagnostics(tmp_path: Path, monkeypatch):
    from scripts.preflight_candidate_dataset import PreflightOptions, run_preflight

    _patch_dependencies(monkeypatch)
    _patch_capabilities(monkeypatch, production=12, training=75)

    result = run_preflight(
        PreflightOptions(
            config_path=_write_config(tmp_path),
            processed_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
        )
    )

    assert result["ok"] is False
    assert "training_candidate_rule_count_below_min:75<76" in result["errors"]
    assert len(result["activation_summary"]["training_candidate_rule_ids"]) == 75
    assert "blockers" in result["activation_summary"]
    assert "missing_module_counts" in result["activation_summary"]


def test_missing_dependencies_report_readable_diagnostics(tmp_path: Path, monkeypatch):
    from scripts.preflight_candidate_dataset import PreflightOptions, run_preflight

    _patch_dependencies(monkeypatch, morphology=False, syntax=False)
    _patch_capabilities(monkeypatch, production=12, training=76)

    result = run_preflight(
        PreflightOptions(
            config_path=_write_config(tmp_path),
            processed_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
        )
    )

    assert result["ok"] is False
    assert "missing_morphology_dependency" in result["errors"]
    assert "missing_syntax_dependency" in result["errors"]
    assert "install project requirements" in " ".join(result["dependency_summary"]["messages"])


def test_clean_stale_artifacts_keeps_raw_data(tmp_path: Path):
    from scripts.preflight_candidate_dataset import clean_stale_artifacts

    processed = tmp_path / "data" / "processed"
    raw = tmp_path / "data" / "raw" / "external"
    reports = tmp_path / "reports" / "dataset_build"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    reports.mkdir(parents=True)
    generated = processed / "correction_dataset.csv.gz"
    generated.write_text("source,target\n", encoding="utf-8")
    cache_file = processed / "features_cache" / "cache.bin"
    cache_file.parent.mkdir()
    cache_file.write_text("cache", encoding="utf-8")
    report = reports / "old_report.csv"
    report.write_text("x\n", encoding="utf-8")
    raw_file = raw / "corpus.csv"
    raw_file.write_text("manual", encoding="utf-8")

    cleaned = clean_stale_artifacts(processed_dir=processed, reports_dir=reports)

    assert str(generated) in cleaned
    assert str(cache_file) in cleaned
    assert str(report) in cleaned
    assert not generated.exists()
    assert not cache_file.exists()
    assert not report.exists()
    assert raw_file.exists()


def test_config_contains_explicit_high_confidence_threshold():
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))

    assert config["data"]["clean_pool"]["high_confidence_candidate_threshold"] == 0.95


def test_final_active_below_target_warns_without_blocking(tmp_path: Path, monkeypatch):
    from scripts.preflight_candidate_dataset import PreflightOptions, run_preflight

    _patch_dependencies(monkeypatch)
    _patch_activation_counts(monkeypatch, production=12, training=76, final=50)

    result = run_preflight(
        PreflightOptions(
            config_path=_write_config(tmp_path),
            processed_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
        )
    )

    assert result["ok"] is True
    assert "final_active_rule_count_below_target:50<76" in result["warnings"]
    assert not any(error.startswith("final_active_rule_count_below_min:") for error in result["errors"])


def test_final_active_below_configured_minimum_blocks(tmp_path: Path, monkeypatch):
    from scripts.preflight_candidate_dataset import PreflightOptions, run_preflight

    _patch_dependencies(monkeypatch)
    _patch_activation_counts(monkeypatch, production=12, training=76, final=24)

    result = run_preflight(
        PreflightOptions(
            config_path=_write_config(tmp_path),
            processed_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
        )
    )

    assert result["ok"] is False
    assert "final_active_rule_count_below_min:24<25" in result["errors"]


def test_operator_audit_soft_gates_warn_when_configured():
    from src.data import operator_dataset_builder as builder

    audit_config = builder._operator_audit_config(
        {
            "data": {
                "stress": {"min_ratio": 0.03, "max_ratio": 0.05},
                "audit": {},
            }
        }
    )
    kwargs = _operator_audit_kwargs(stress_count=1000, corpus_share=0.50, audit_config=audit_config)

    errors = builder._audit_errors(**kwargs)
    warnings = builder._audit_warnings(
        total=kwargs["total"],
        composition=kwargs["composition"],
        corpus_share=kwargs["corpus_share"],
        audit_config=kwargs["audit_config"],
        production_gates=True,
    )

    assert "stress_ratio_outside_3_5_percent" not in errors
    assert "corpus_opportunity_share_below_threshold" not in errors
    assert "stress_ratio_below_target" in warnings
    assert "corpus_opportunity_share_below_threshold" in warnings


def test_operator_audit_stress_above_max_stays_blocking():
    from src.data import operator_dataset_builder as builder

    audit_config = builder._operator_audit_config(
        {
            "data": {
                "stress": {"min_ratio": 0.03, "max_ratio": 0.05, "fail_on_under_target": False},
                "audit": {"fail_on_corpus_opportunity_share_below_threshold": False},
            }
        }
    )
    kwargs = _operator_audit_kwargs(stress_count=12000, corpus_share=0.80, audit_config=audit_config)

    errors = builder._audit_errors(**kwargs)

    assert "stress_ratio_outside_3_5_percent" in errors


def _operator_audit_kwargs(*, stress_count: int, corpus_share: float, audit_config: dict) -> dict:
    total = 200000
    return {
        "total": total,
        "requested_total": total,
        "split_sizes": {"train": 160000, "val": 20000, "test": 20000},
        "composition": {
            "clean_identity_from_open_clean": 20000,
            "hard_negative_from_open_clean": 20000,
            "real_error_pair": 1,
            "multi_error_stress": stress_count,
        },
        "active_rule_ids": [],
        "rule_counts": {},
        "known": {},
        "quote": {},
        "clean_hard": {},
        "numeric_mismatch": 0,
        "corpus_share": corpus_share,
        "fallback_share": 0.0,
        "diversity": {"failed_rule_count": 0},
        "extended_summary": {"blocking_issue_count": 0},
        "semantic_summary": {"failed_rows": 0},
        "atomic_purity_summary": {"failed_rows": 0},
        "extra_edit_summary": {"failed_rows": 0},
        "unknown_rule_train_summary": {"failed_rows": 0},
        "mixed_script_clean_summary": {"failed_rows": 0},
        "real_pair_atomization_summary": {"failed_rows": 0},
        "recall_summary": {"active_min_excluding_unknown": 1.0},
        "audit_config": audit_config,
        "production_gates": True,
    }


def test_clean_pool_mixed_script_sample_reports_contamination(tmp_path: Path):
    from scripts.preflight_candidate_dataset import scan_clean_pool

    path = tmp_path / "clean_sentence_pool.csv.gz"
    pd.DataFrame(
        [
            {"text": "Новая cистема обработки данных заработала утром после проверки."},
            {"text": "Новая система обработки данных заработала утром после проверки."},
        ]
    ).to_csv(path, index=False)

    result = scan_clean_pool(
        clean_pool_path=path,
        config=_config(tmp_path),
        row_limit=50000,
        full_scan=False,
    )

    assert result["scanned_rows"] == 2
    assert result["mixed_script_token_count"] == 1
    assert result["latin_confusable_inside_cyrillic_word_count"] == 1
    assert "clean_pool_contamination_detected" in result["warnings"]
    assert "setup_data_sources" in result["message"]


def test_build_dataset_stops_before_generation_when_preflight_fails(tmp_path: Path, monkeypatch):
    import scripts.build_dataset as build_script

    config = _config(tmp_path)
    config_path = _write_config(tmp_path, config)
    preflight_result = {
        "ok": False,
        "errors": ["missing_syntax_dependency"],
        "warnings": [],
        "dependency_summary": {"ok": False, "messages": ["missing_syntax_dependency: install project requirements"]},
        "activation_summary": {},
        "stale_artifacts": {"stale": False, "reasons": []},
        "clean_pool_summary": {},
    }
    printed: list[dict] = []

    monkeypatch.setattr(build_script, "run_preflight", lambda _options: preflight_result, raising=False)
    monkeypatch.setattr(build_script, "print_preflight_summary", lambda result: printed.append(result), raising=False)
    monkeypatch.setattr(build_script, "_write_activation_verified_reports", lambda: None)

    def fail_if_build_starts(*_args, **_kwargs):
        raise AssertionError("dataset generation should not run after preflight failure")

    monkeypatch.setattr(build_script, "build_training_dataset_from_config", fail_if_build_starts)

    exit_code = build_script.main(["--config", str(config_path)])

    assert exit_code == 1
    assert printed == [preflight_result]
