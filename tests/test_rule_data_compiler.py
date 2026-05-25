from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd

from src.candidates.candidate_generator import Candidate
from src.config.load_config import load_config
from src.data.dataset_contract import DATASET_CONTRACT, LAYER_ATOMIC_HARD_NEGATIVE, LAYER_ATOMIC_POSITIVE
from src.rules.capabilities import RuleCapability


class ContextCandidateGenerator:
    PAIRS = {
        "context_chto_by": ("что бы", "чтобы"),
        "context_tak_zhe": ("так же", "также"),
        "context_to_zhe": ("то же", "тоже"),
        "context_za_to": ("за то", "зато"),
        "context_nesmotrya": ("не смотря", "несмотря"),
        "context_vsledstvie": ("в следствие", "вследствие"),
    }

    def generate(self, text: str, **_kwargs):
        candidates: list[Candidate] = []
        lower = text.lower()
        for rule_id, (source, replacement) in self.PAIRS.items():
            start = lower.find(source)
            if start >= 0:
                candidates.append(
                    Candidate(
                        source=text[start : start + len(source)],
                        replacement=replacement,
                        edit_type="split_join",
                        start=start,
                        end=start + len(source),
                        confidence=0.35,
                        requires_model=True,
                        rule_id=rule_id,
                        mode="model_required",
                        requires=("syntax", "morphology", "model"),
                        group="context_split_join",
                    )
                )
        return candidates


class ContextCandidateGeneratorFactory:
    @classmethod
    def from_config(cls, _config: dict):
        return ContextCandidateGenerator()


def _clean_row(text: str, row_id: str = "row-1") -> dict:
    return {
        "text": text,
        "source_name": "unit_clean",
        "source_subcorpus": "context",
        "domain": "open_clean",
        "sentence_id": row_id,
        "hash": row_id,
    }


def _context_capability() -> RuleCapability:
    return RuleCapability(
        taxonomy_key="context_tak_zhe",
        domain="orthography",
        entry_type="rule",
        title="context tak zhe",
        orfogrammka_id="",
        project_rule_ids=["context_tak_zhe"],
        implementation_status="model_required",
        requires=["syntax", "morphology", "model", "validator"],
        executable=True,
        training_eligible=True,
        training_decision="INCLUDE_NOW",
        training_reason="unit",
        has_candidate_path=True,
        has_synthetic_support=True,
        has_hard_negative_support=True,
        has_validator_support=True,
        has_dictionary_support=False,
        has_syntax_support=False,
        has_morphology_support=True,
        has_ner_support=False,
        risk_level="medium",
    )


def _load_compiler():
    return importlib.import_module("src.data.rule_data_compiler")


def test_corpus_backed_miner_creates_positive_from_real_clean_sentence():
    compiler = _load_compiler()
    clean_rows = [
        _clean_row("Редакция также подготовила отчет для комиссии и отправила его в архив."),
    ]

    result = compiler.compile_rule_data(
        clean_rows,
        ["context_tak_zhe"],
        ContextCandidateGenerator(),
        {"data": {"rule_quota": {"preferred_atomic_positives_per_active_rule": 2}}},
    )

    assert len(result.atomic_positive_rows) == 1
    row = result.atomic_positive_rows[0]
    metadata = json.loads(row["metadata"])
    assert row["source"] == "Редакция так же подготовила отчет для комиссии и отправила его в архив."
    assert row["target"] == clean_rows[0]["text"]
    assert row["dataset_contract"] == DATASET_CONTRACT
    assert row["dataset_layer"] == LAYER_ATOMIC_POSITIVE
    assert row["source_type"] == "synthetic_augmented_from_open_clean"
    assert row["activation_source"] == "corpus_mined"
    assert row["count_toward_rule_quota"] is True
    assert row["gold_edit_count"] == 1
    assert row["verification_status"] == "passed"
    assert metadata["source_name"] == "unit_clean"
    assert metadata["miner_name"] == "corpus_backed_split_join"


def test_split_join_miner_rejects_duplicate_source_target_rule():
    compiler = _load_compiler()
    clean_text = "Редакция также подготовила отчет для комиссии и отправила его в архив."

    result = compiler.compile_rule_data(
        [_clean_row(clean_text, "row-1"), _clean_row(clean_text, "row-2")],
        ["context_tak_zhe"],
        ContextCandidateGenerator(),
        {},
    )

    assert len(result.atomic_positive_rows) == 1
    assert any(row["reason"] == "duplicate_pair" for row in result.rejection_rows)


def test_hard_negative_source_equals_target_and_bad_unbalanced_row_rejects():
    compiler = _load_compiler()
    clean_rows = [
        _clean_row("Так же, как раньше, редакция проверила отчет утром после заседания.", "good"),
        _clean_row('"Так же, как раньше, редакция проверила отчет утром после заседания.', "bad"),
    ]

    result = compiler.compile_rule_data(clean_rows, ["context_tak_zhe"], ContextCandidateGenerator(), {})

    assert len(result.hard_negative_rows) == 1
    row = result.hard_negative_rows[0]
    assert row["source"] == row["target"]
    assert row["dataset_layer"] == LAYER_ATOMIC_HARD_NEGATIVE
    assert row["target_rule_id"] == "context_tak_zhe"
    assert row["count_toward_rule_quota"] is False
    reasons = {item["reason"] for item in result.rejection_rows}
    assert "hard_negative_quality_failed:unbalanced_ascii_quotes" in reasons


def test_compiler_writes_reports(tmp_path: Path):
    compiler = _load_compiler()
    result = compiler.compile_rule_data(
        [_clean_row("Редакция также подготовила отчет для комиссии и отправила его в архив.")],
        ["context_tak_zhe"],
        ContextCandidateGenerator(),
        {},
    )

    compiler.write_rule_data_compiler_reports(result, tmp_path)

    assert (tmp_path / "rule_data_source_report.csv").exists()
    assert (tmp_path / "rule_structural_diversity_report.csv").exists()
    assert (tmp_path / "rule_miner_rejection_report.csv").exists()
    assert (tmp_path / "rule_underfilled_backlog.csv").exists()
    diversity = pd.read_csv(tmp_path / "rule_structural_diversity_report.csv")
    assert diversity.loc[0, "rule_lab_share"] == 0


def test_compiler_does_not_depend_on_rule_lab_module():
    compiler = _load_compiler()
    source = Path(compiler.__file__).read_text(encoding="utf-8")

    assert "rule_lab_generation" not in source
    assert all(row.rule_lab_positive_count == 0 for row in compiler.compile_rule_data([], [], ContextCandidateGenerator(), {}).source_stats_rows)


def test_compiler_handles_missing_clean_rows_gracefully():
    compiler = _load_compiler()

    result = compiler.compile_rule_data([], ["context_tak_zhe"], ContextCandidateGenerator(), {})

    assert result.atomic_positive_rows == []
    assert result.hard_negative_rows == []
    assert result.source_stats_rows[0].rule_id == "context_tak_zhe"
    assert result.underfilled_rows


def test_compiler_rows_count_toward_quota_in_tiny_builder_fixture(tmp_path: Path, monkeypatch):
    import src.data.operator_dataset_builder as operator_builder
    from src.data.full_dataset_builder import build_dataset_from_config

    clean_pool_path = tmp_path / "clean_sentence_pool.csv.gz"
    pd.DataFrame(
        [
            _clean_row("Редакция также подготовила отчет для комиссии и отправила его в архив.", "p1"),
            _clean_row("Так же, как раньше, редакция проверила отчет утром после заседания.", "h1"),
            _clean_row("Команда проверила короткую заметку утром после заседания редакции.", "c1"),
        ]
    ).to_csv(clean_pool_path, index=False)

    config = load_config("configs/config.yaml")
    config["paths"]["reports_dir"] = str(tmp_path / "reports")
    config["data"]["processed_train_path"] = str(tmp_path / "data" / "operator_dataset.csv.gz")
    config["data"]["manifest_path"] = str(tmp_path / "reports" / "dataset_manifest.json")
    config["data"]["clean_pool_path"] = str(clean_pool_path)
    config["data"]["clean_pool_chunksize"] = 2
    config["data"]["dataset_build_workers"] = 1
    config["data"]["total_examples"] = 3
    config["data"]["target_total_examples"] = 3
    config["data"]["train_examples"] = 3
    config["data"]["val_examples"] = 0
    config["data"]["test_examples"] = 0
    config["data"]["composition"] = {
        "atomic_positive_target": 1,
        "atomic_hard_negative_target": 1,
        "clean_identity_target": 1,
        "stress_multi_error_target": 0,
        "real_atomic_train_target": 0,
    }
    config["data"]["rule_quota"] = {
        "rule_ids": ["context_tak_zhe"],
        "min_atomic_positives_per_active_rule": 1,
        "preferred_atomic_positives_per_active_rule": 1,
        "max_total_per_rule_id": 1,
        "min_hard_negatives_per_active_rule": 1,
        "disable_rule_if_quota_not_met": True,
    }
    config["data"]["rule_activation"]["expected_min_final_active_rule_count"] = 0
    config["data"]["rule_activation"]["target_final_active_rule_count"] = 0
    config["data"]["rule_activation"]["fail_below_final_active_rule_count"] = False
    config["data"]["rule_activation"]["warn_below_target_final_active_rule_count"] = False
    config["data"]["rule_data_compiler"] = {"enabled": True}

    def _empty_operator_atomic_result(*_args, **_kwargs):
        return operator_builder.AtomicPositiveGenerationResult(
            rows=[],
            generation_rows=[],
            rejection_rows=[],
            rules_without_atomic_positive=[
                {
                    "rule_id": "context_tak_zhe",
                    "status": "not_generated",
                    "reason": "unit_fixture_uses_compiler",
                }
            ],
        )

    def _empty_syntax_atomic_result(*_args, **_kwargs):
        return operator_builder.SyntaxAtomicPositiveGenerationResult(
            rows=[],
            generation_rows=[],
            rejection_rows=[],
            supported_rule_ids=["context_tak_zhe"],
            selected_rule_ids=[],
        )

    monkeypatch.setattr(operator_builder, "CandidateGenerator", ContextCandidateGeneratorFactory, raising=False)
    monkeypatch.setattr(operator_builder, "load_rule_capabilities", lambda _path, **_kwargs: [_context_capability()])
    monkeypatch.setattr(
        operator_builder,
        "generate_atomic_positive_rows_from_clean_pool",
        _empty_operator_atomic_result,
    )
    monkeypatch.setattr(
        operator_builder,
        "_generate_atomic_positive_rows_from_syntax_synthetic_result",
        _empty_syntax_atomic_result,
    )

    result = build_dataset_from_config(config, force=True)
    frame = pd.read_csv(config["data"]["processed_train_path"])
    quota = pd.read_csv(Path(config["paths"]["reports_dir"]) / "dataset_build" / "active_rule_quota_report.csv")

    assert result["total"] == 3
    assert not frame[frame["activation_source"].astype(str).eq("corpus_mined")].empty
    assert int(quota.set_index("rule_id").loc["context_tak_zhe", "pre_gate_atomic_positive_count"]) >= 1
    assert (Path(config["paths"]["reports_dir"]) / "dataset_build" / "rule_data_source_report.csv").exists()
