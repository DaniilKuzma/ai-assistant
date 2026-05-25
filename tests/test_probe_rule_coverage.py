from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import yaml

from src.candidates.candidate_generator import Candidate
from src.rules.capabilities import RuleCapability


class ProbeCandidateGenerator:
    def generate(self, text: str, **_kwargs):
        candidates: list[Candidate] = []
        lower = text.lower()
        pairs = {
            "context_tak_zhe": ("так же", "также"),
            "context_chto_by": ("что бы", "чтобы"),
            "tsya_soft_insert": ("появится", "появиться"),
            "keyboard_typo_candidate": ("компьюьерные", "компьютерные"),
        }
        for rule_id, (source, replacement) in pairs.items():
            start = lower.find(source)
            if start >= 0:
                original = text[start : start + len(source)]
                candidates.append(
                    Candidate(
                        source=original,
                        replacement=replacement.capitalize() if original[:1].isupper() else replacement,
                        edit_type="split_join" if "context" in rule_id else "spelling",
                        start=start,
                        end=start + len(source),
                        confidence=1.0,
                        requires_model=True,
                        rule_id=rule_id,
                        mode="model_required",
                    )
                )
        marker = " что "
        start = lower.find(marker)
        if start >= 0:
            candidates.append(
                Candidate(
                    source="",
                    replacement=",",
                    edit_type="punctuation_insert",
                    start=start,
                    end=start,
                    confidence=1.0,
                    requires_model=True,
                    rule_id="comma_subordinate",
                    mode="model_required",
                    action="INSERT",
                    label="COMMA",
                    syntax_family="subordinate_clause_comma",
                )
            )
        return candidates


class EmptyCandidateGenerator:
    def generate(self, text: str, **_kwargs):
        del text
        return []


class CandidateGeneratorFactory:
    generator_cls = ProbeCandidateGenerator

    @classmethod
    def from_config(cls, config: dict, **_kwargs):
        del config
        return cls.generator_cls()


def _load_probe():
    return importlib.import_module("scripts.probe_rule_coverage")


def _capability(rule_id: str) -> RuleCapability:
    return RuleCapability(
        taxonomy_key=rule_id,
        domain="unit",
        entry_type="rule",
        title=rule_id,
        orfogrammka_id="",
        project_rule_ids=[rule_id],
        implementation_status="model_required",
        requires=["validator"],
        executable=True,
        training_eligible=True,
        training_decision="INCLUDE_NOW",
        training_reason="unit",
        has_candidate_path=True,
        has_synthetic_support=True,
        has_hard_negative_support=True,
        has_validator_support=True,
        has_dictionary_support=False,
        has_syntax_support=rule_id == "comma_subordinate",
        has_morphology_support=rule_id.startswith("tsya_"),
        has_ner_support=False,
        risk_level="low",
    )


def _patch_probe(monkeypatch, probe, rule_ids: list[str], generator_cls=ProbeCandidateGenerator) -> None:
    CandidateGeneratorFactory.generator_cls = generator_cls
    monkeypatch.setattr(probe, "CandidateGenerator", CandidateGeneratorFactory)
    monkeypatch.setattr(probe, "load_rule_capabilities", lambda _path, **_kwargs: [_capability(rule_id) for rule_id in rule_ids])


def _write_clean_pool(path: Path, texts: list[str]) -> Path:
    rows = [
        {
            "text": text,
            "source_name": "unit",
            "source_subcorpus": "unit",
            "domain": "open_clean",
            "sentence_id": f"s{index}",
            "hash": f"h{index}",
        }
        for index, text in enumerate(texts)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_config(
    tmp_path: Path,
    *,
    clean_pool_path: Path,
    rule_ids: list[str],
    reports_dir: Path,
    processed_dir: Path,
    min_atomic: int = 1,
    preferred_atomic: int = 1,
    min_hard: int = 1,
    expected_ready: int = 1,
    min_recall: float = 0.95,
    rule_lab_recipe_path: Path | None = None,
    real_pattern_paths: list[Path] | None = None,
) -> Path:
    config = {
        "model": {"max_candidates": 16},
        "data": {
            "candidate_opportunity": {
                "dataset_build_workers": 1,
                "paths": {
                    "processed_dir": str(processed_dir),
                    "reports_dir": str(reports_dir),
                    "clean_pool_path": str(clean_pool_path),
                    "correction_dataset_path": str(processed_dir / "correction_dataset.csv.gz"),
                    "manifest_path": str(processed_dir / "dataset_manifest.json"),
                    "rule_lab_recipes_config": str(rule_lab_recipe_path or tmp_path / "missing_rule_lab_recipes.yaml"),
                },
                "rule_activation": {
                    "mode": "expanded_safe",
                    "expected_min_final_active_rule_count": expected_ready,
                    "target_final_active_rule_count": max(expected_ready, len(rule_ids)),
                    "fail_below_final_active_rule_count": True,
                },
                "rule_quota": {
                    "rule_ids": rule_ids,
                    "min_atomic_positives_per_active_rule": min_atomic,
                    "preferred_atomic_positives_per_active_rule": preferred_atomic,
                    "max_total_per_rule_id": preferred_atomic,
                    "min_hard_negatives_per_active_rule": min_hard,
                    "disable_rule_if_quota_not_met": True,
                },
                "rule_data_compiler": {
                    "enabled": True,
                    "max_scan_rows_per_rule": 100,
                    "source_priority": [
                        "corpus_mined",
                        "syntax_mined",
                        "morphology_mined",
                        "real_pattern_replay",
                        "rule_lab",
                    ],
                    "real_pattern_paths": [str(path) for path in real_pattern_paths or []],
                },
                "rule_lab": {
                    "enabled": True,
                    "fill_underfilled_rules": True,
                    "max_generated_per_rule": preferred_atomic,
                    "max_hard_negatives_per_rule": max(1, min_hard),
                    "fail_on_low_diversity": False,
                    "max_per_template_share": 1.0,
                    "max_per_slot_value_share": 1.0,
                },
                "audit": {"min_candidate_recall_for_active_rule": min_recall},
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _run_probe(probe, config_path: Path, processed_dir: Path, reports_dir: Path, *extra: str) -> int:
    return probe.main(
        [
            "--config",
            str(config_path),
            "--processed-dir",
            str(processed_dir),
            "--reports-dir",
            str(reports_dir),
            "--no-full-build",
            "--json",
            *extra,
        ]
    )


def _summary(reports_dir: Path) -> dict:
    return json.loads((reports_dir / "coverage_probe_summary.json").read_text(encoding="utf-8"))


def test_probe_passes_ready_fixture_and_does_not_write_forbidden_outputs(tmp_path, monkeypatch, capsys):
    probe = _load_probe()
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    clean_pool = _write_clean_pool(
        processed_dir / "clean_sentence_pool.csv.gz",
        [
            "Редакция также подготовила отчет для комиссии и отправила его в архив.",
            "Так же, как раньше, редакция проверила отчет утром после заседания.",
        ],
    )
    config_path = _write_config(
        tmp_path,
        clean_pool_path=clean_pool,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        rule_ids=["context_tak_zhe"],
    )
    _patch_probe(monkeypatch, probe, ["context_tak_zhe"])

    exit_code = _run_probe(probe, config_path, processed_dir, reports_dir)
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "READY_FOR_FULL_BUILD_PROBE_PASSED" in stdout
    assert not (processed_dir / "correction_dataset.csv.gz").exists()
    assert not (processed_dir / "train.csv").exists()
    assert not (processed_dir / "val.csv").exists()
    assert not (processed_dir / "test.csv").exists()
    summary = _summary(reports_dir)
    assert summary["status"] == "READY_FOR_FULL_BUILD_PROBE_PASSED"
    assert summary["metrics"]["rules_with_atomic_ge_500"] == 0
    assert (reports_dir / "coverage_probe_report.csv").exists()
    assert (reports_dir / "coverage_probe_ready_rules.csv").exists()


def test_probe_reads_config_thresholds_and_fails_when_ready_rules_below_expected(tmp_path, monkeypatch, capsys):
    probe = _load_probe()
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    clean_pool = _write_clean_pool(
        processed_dir / "clean_sentence_pool.csv.gz",
        [
            "Редакция также подготовила отчет для комиссии и отправила его в архив.",
            "Так же, как раньше, редакция проверила отчет утром после заседания.",
        ],
    )
    config_path = _write_config(
        tmp_path,
        clean_pool_path=clean_pool,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        rule_ids=["context_tak_zhe"],
        min_atomic=2,
        preferred_atomic=2,
        expected_ready=1,
    )
    _patch_probe(monkeypatch, probe, ["context_tak_zhe"])

    exit_code = _run_probe(probe, config_path, processed_dir, reports_dir)
    stdout = capsys.readouterr().out

    assert exit_code != 0
    assert "NOT_READY_FOR_FULL_BUILD" in stdout
    summary = _summary(reports_dir)
    assert summary["thresholds"]["min_atomic"] == 2
    assert summary["thresholds"]["min_hard"] == 1
    backlog = pd.read_csv(reports_dir / "coverage_probe_underfilled_rules.csv").set_index("rule_id")
    assert int(backlog.loc["context_tak_zhe", "min_atomic_required"]) == 2


def test_underfilled_backlog_recommended_next_action_for_candidate_missing(tmp_path, monkeypatch):
    probe = _load_probe()
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    clean_pool = _write_clean_pool(
        processed_dir / "clean_sentence_pool.csv.gz",
        ["Редакция также подготовила отчет для комиссии и отправила его в архив."],
    )
    config_path = _write_config(
        tmp_path,
        clean_pool_path=clean_pool,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        rule_ids=["context_tak_zhe"],
    )
    _patch_probe(monkeypatch, probe, ["context_tak_zhe"], generator_cls=EmptyCandidateGenerator)

    assert _run_probe(probe, config_path, processed_dir, reports_dir) != 0

    backlog = pd.read_csv(reports_dir / "coverage_probe_underfilled_rules.csv").set_index("rule_id")
    assert backlog.loc["context_tak_zhe", "top_rejection_reason"] == "candidate_missing"
    assert backlog.loc["context_tak_zhe", "recommended_next_action"] == "fix_candidate_generator_or_rule_mapping"
    assert probe.recommended_next_action("recall_under_min") == "inspect_candidate_generator_recall"
    assert probe.recommended_next_action("low_structural_diversity") == "add_corpus_contexts_or_templates"


def test_probe_reports_source_mix_and_rule_lab_fallback(tmp_path, monkeypatch):
    probe = _load_probe()
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    clean_pool = _write_clean_pool(
        processed_dir / "clean_sentence_pool.csv.gz",
        [
            "Редакция также подготовила отчет для комиссии и отправила его в архив.",
            "Редактор заметил, что документ готов утром после заседания комиссии.",
            "Команда может появиться завтра после заседания редакции.",
            "Компьютерные мониторы показали новый отчет районной комиссии.",
        ],
    )
    real_path = processed_dir / "real_error_pairs_atomic.csv.gz"
    pd.DataFrame(
        [
            {
                "source": "Компьюьерные мониторы показали отчет.",
                "target": "Компьютерные мониторы показали отчет.",
                "rule_id": "keyboard_typo_candidate",
                "rule_ids": json.dumps(["keyboard_typo_candidate"], ensure_ascii=False),
                "edit_count": 1,
                "candidate_present": True,
                "strict_validator_passed": True,
            }
        ]
    ).to_csv(real_path, index=False)
    recipe_path = tmp_path / "rule_lab_recipes.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "rules": {
                    "context_tak_zhe": {
                        "enabled": True,
                        "family": "split_join",
                        "mutation": {"target_fragment": "также", "source_fragment": "так же"},
                        "positive_templates": [{"template_id": "unused", "target": "Редактор также проверил отчет утром."}],
                        "hard_negative_templates": [],
                    },
                    "context_chto_by": {
                        "enabled": True,
                        "family": "split_join",
                        "mutation": {"target_fragment": "чтобы", "source_fragment": "что бы"},
                        "positive_templates": [{"template_id": "fallback", "target": "Редактор пришёл, чтобы проверить документ утром."}],
                        "hard_negative_templates": [],
                    },
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rule_ids = [
        "context_tak_zhe",
        "comma_subordinate",
        "tsya_soft_insert",
        "keyboard_typo_candidate",
        "context_chto_by",
    ]
    config_path = _write_config(
        tmp_path,
        clean_pool_path=clean_pool,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        rule_ids=rule_ids,
        min_hard=0,
        expected_ready=0,
        rule_lab_recipe_path=recipe_path,
        real_pattern_paths=[real_path],
    )
    _patch_probe(monkeypatch, probe, rule_ids)
    monkeypatch.setattr(probe.rule_data_compiler, "_morphology_available", lambda: True)

    _run_probe(probe, config_path, processed_dir, reports_dir)

    report = pd.read_csv(reports_dir / "coverage_probe_report.csv").set_index("rule_id")
    assert int(report.loc["context_tak_zhe", "corpus_mined_positive_count"]) == 1
    assert int(report.loc["context_tak_zhe", "rule_lab_positive_count"]) == 0
    assert int(report.loc["comma_subordinate", "syntax_mined_positive_count"]) == 1
    assert int(report.loc["tsya_soft_insert", "morphology_mined_positive_count"]) == 1
    assert int(report.loc["keyboard_typo_candidate", "real_pattern_replay_positive_count"]) == 1
    assert int(report.loc["context_chto_by", "rule_lab_positive_count"]) == 1
    assert "median_rule_lab_share" in _summary(reports_dir)["metrics"]


def test_probe_rejects_bad_hard_negatives(tmp_path, monkeypatch):
    probe = _load_probe()
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    clean_pool = _write_clean_pool(
        processed_dir / "clean_sentence_pool.csv.gz",
        ["Редакция также подготовила отчет для комиссии и отправила его в архив."],
    )
    recipe_path = tmp_path / "rule_lab_recipes.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "rules": {
                    "context_tak_zhe": {
                        "enabled": True,
                        "family": "split_join",
                        "mutation": {"target_fragment": "также", "source_fragment": "так же"},
                        "positive_templates": [],
                        "hard_negative_templates": [
                            {"template_id": "bad_quote", "text": '"Так же, как раньше, редакция проверила отчет утром.'}
                        ],
                    }
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        clean_pool_path=clean_pool,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        rule_ids=["context_tak_zhe"],
        rule_lab_recipe_path=recipe_path,
    )
    _patch_probe(monkeypatch, probe, ["context_tak_zhe"])

    assert _run_probe(probe, config_path, processed_dir, reports_dir) != 0

    rejection = pd.read_csv(reports_dir / "coverage_probe_rejection_report.csv")
    assert "hard_negative_quality_failed:unbalanced_ascii_quotes" in set(rejection["reason"])


def test_probe_reports_median_rule_lab_share(tmp_path, monkeypatch):
    probe = _load_probe()
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    clean_pool = _write_clean_pool(processed_dir / "clean_sentence_pool.csv.gz", ["Редактор проверил документ утром."])
    recipe_path = tmp_path / "rule_lab_recipes.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "rules": {
                    "context_chto_by": {
                        "enabled": True,
                        "family": "split_join",
                        "mutation": {"target_fragment": "чтобы", "source_fragment": "что бы"},
                        "positive_templates": [{"template_id": "fallback", "target": "Редактор пришёл, чтобы проверить документ утром."}],
                        "hard_negative_templates": [],
                    }
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        clean_pool_path=clean_pool,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        rule_ids=["context_chto_by"],
        min_hard=0,
        expected_ready=1,
        rule_lab_recipe_path=recipe_path,
    )
    _patch_probe(monkeypatch, probe, ["context_chto_by"])

    assert _run_probe(probe, config_path, processed_dir, reports_dir) == 0

    assert _summary(reports_dir)["metrics"]["median_rule_lab_share"] == 1.0
