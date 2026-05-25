from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd

from src.candidates.candidate_generator import Candidate
from src.config.load_config import load_config
from src.data.dataset_contract import DATASET_CONTRACT, LAYER_ATOMIC_HARD_NEGATIVE, LAYER_ATOMIC_POSITIVE
from src.rules.capabilities import RuleCapability


class StaticCandidateGenerator:
    def __init__(self, candidates_by_text: dict[str, list[Candidate]] | None = None) -> None:
        self.candidates_by_text = candidates_by_text or {}

    def generate(self, text: str, **_kwargs):
        return list(self.candidates_by_text.get(text, []))


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


class MorphologyCandidateGenerator(ContextCandidateGenerator):
    TSYA_PAIRS = {
        "tsya_soft_insert": ("появится", "появиться"),
        "tsya_soft_delete": ("готовиться", "готовится"),
    }

    def generate(self, text: str, **kwargs):
        candidates: list[Candidate] = list(super().generate(text, **kwargs))
        lower = text.lower()
        for rule_id, (source, replacement) in self.TSYA_PAIRS.items():
            start = lower.find(source)
            if start >= 0:
                candidates.append(
                    Candidate(
                        source=text[start : start + len(source)],
                        replacement=replacement,
                        edit_type="spelling",
                        start=start,
                        end=start + len(source),
                        confidence=1.0,
                        requires_model=True,
                        rule_id=rule_id,
                        mode="model_required",
                        requires=("morphology", "syntax", "model"),
                        group="tsya",
                    )
                )
        start = lower.find("длиный")
        if start >= 0:
            candidates.append(
                Candidate(
                    source=text[start : start + len("длиный")],
                    replacement="длинный",
                    edit_type="spelling",
                    start=start,
                    end=start + len("длиный"),
                    confidence=1.0,
                    requires_model=True,
                    rule_id="n_nn_adjective",
                    mode="model_required",
                    requires=("morphology", "syntax", "dictionary", "model"),
                    group="n_nn",
                )
            )
        return candidates


class PunctuationCandidateGenerator:
    def generate(self, text: str, **_kwargs):
        candidates: list[Candidate] = []
        marker = " что "
        start = text.lower().find(marker)
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
                    requires=("syntax", "model"),
                    group="punctuation",
                    syntax_family="subordinate_clause_comma",
                )
            )
        return candidates


class ReplayCandidateGenerator(ContextCandidateGenerator):
    def generate(self, text: str, **kwargs):
        candidates: list[Candidate] = list(super().generate(text, **kwargs))
        lower = text.lower()
        start = lower.find("компьюьерные")
        if start >= 0:
            candidates.append(
                Candidate(
                    source=text[start : start + len("компьюьерные")],
                    replacement="Компьютерные" if text[start : start + 1].isupper() else "компьютерные",
                    edit_type="spelling",
                    start=start,
                    end=start + len("компьюьерные"),
                    confidence=1.0,
                    requires_model=True,
                    rule_id="keyboard_typo_candidate",
                    mode="model_required",
                    requires=("dictionary", "model"),
                    group="typos",
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
    source_report = pd.read_csv(tmp_path / "rule_data_source_report.csv")
    assert list(source_report.columns) == [
        "rule_id",
        "corpus_mined_positive_count",
        "syntax_mined_positive_count",
        "morphology_mined_positive_count",
        "real_pattern_replay_positive_count",
        "rule_lab_positive_count",
        "total_atomic_positive_count",
        "corpus_mined_hard_negative_count",
        "rule_lab_hard_negative_count",
        "total_hard_negative_count",
    ]
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


def test_morphology_miner_disabled_cleanly_if_analyzer_unavailable(monkeypatch):
    compiler = _load_compiler()
    monkeypatch.setattr(compiler, "_morphology_available", lambda: False)

    result = compiler.compile_rule_data(
        [_clean_row("Команда может появиться завтра после заседания редакции.")],
        ["tsya_soft_insert"],
        MorphologyCandidateGenerator(),
        {},
        target_counts={"atomic_positive": 1, "atomic_hard_negative": 0},
    )

    assert result.atomic_positive_rows == []
    assert any(row["reason"] == "morphology_unavailable" for row in result.rejection_rows)


def test_tsya_morphology_miner_produces_verified_one_edit_examples():
    compiler = _load_compiler()

    result = compiler.compile_rule_data(
        [
            _clean_row("Команда может появиться завтра после заседания редакции.", "insert"),
            _clean_row("Команда готовится к запуску проекта после заседания редакции.", "delete"),
        ],
        ["tsya_soft_insert", "tsya_soft_delete"],
        MorphologyCandidateGenerator(),
        {},
        target_counts={"atomic_positive": 1, "atomic_hard_negative": 0},
    )

    rows_by_rule = {row["rule_id"]: row for row in result.atomic_positive_rows}
    assert rows_by_rule["tsya_soft_insert"]["source"] == "Команда может появится завтра после заседания редакции."
    assert rows_by_rule["tsya_soft_delete"]["source"] == "Команда готовиться к запуску проекта после заседания редакции."
    assert all(row["activation_source"] == "morphology_mined" for row in rows_by_rule.values())
    assert all(row["gold_edit_count"] == 1 for row in rows_by_rule.values())


def test_n_nn_morphology_miner_requires_candidate_and_validator_support():
    compiler = _load_compiler()

    result = compiler.compile_rule_data(
        [_clean_row("Редактор подготовил длинный отчет для районной комиссии утром.")],
        ["n_nn_adjective"],
        MorphologyCandidateGenerator(),
        {},
        target_counts={"atomic_positive": 1, "atomic_hard_negative": 0},
    )

    assert len(result.atomic_positive_rows) == 1
    row = result.atomic_positive_rows[0]
    assert row["source"] == "Редактор подготовил длиный отчет для районной комиссии утром."
    assert row["rule_id"] == "n_nn_adjective"
    assert row["activation_source"] == "morphology_mined"


def test_syntax_miner_uses_real_clean_corpus_contexts_when_available():
    compiler = _load_compiler()

    result = compiler.compile_rule_data(
        [_clean_row("Редактор заметил, что документ готов утром после заседания комиссии.")],
        ["comma_subordinate"],
        PunctuationCandidateGenerator(),
        {},
        target_counts={"atomic_positive": 1, "atomic_hard_negative": 0},
    )

    assert len(result.atomic_positive_rows) == 1
    row = result.atomic_positive_rows[0]
    assert row["source"] == "Редактор заметил что документ готов утром после заседания комиссии."
    assert row["activation_source"] == "syntax_mined"


def test_syntax_miner_reports_opportunities_seen_0_when_no_opportunities():
    compiler = _load_compiler()

    result = compiler.compile_rule_data(
        [_clean_row("Редактор проверил документ утром после заседания комиссии.")],
        ["comma_subordinate"],
        PunctuationCandidateGenerator(),
        {},
        target_counts={"atomic_positive": 1, "atomic_hard_negative": 0},
    )

    assert result.atomic_positive_rows == []
    assert any(row["reason"] == "opportunities_seen_0" for row in result.rejection_rows)


def test_real_pattern_replay_does_not_put_raw_dirty_pairs_into_train(tmp_path: Path):
    compiler = _load_compiler()
    real_path = tmp_path / "real_error_pairs_atomic.csv.gz"
    raw_source = "Компьюьерные мониторы для людей с плохим зрением."
    raw_target = "Компьютерные мониторы для людей с плохим зрением."
    pd.DataFrame(
        [
            {
                "source": raw_source,
                "target": raw_target,
                "rule_id": "keyboard_typo_candidate",
                "rule_ids": json.dumps(["keyboard_typo_candidate"], ensure_ascii=False),
                "edit_count": 1,
                "candidate_present": True,
                "strict_validator_passed": True,
            }
        ]
    ).to_csv(real_path, index=False)

    result = compiler.compile_rule_data(
        [_clean_row("Компьютерные мониторы показали новый отчет районной комиссии.")],
        ["keyboard_typo_candidate"],
        ReplayCandidateGenerator(),
        {"data": {"rule_data_compiler": {"real_pattern_paths": [str(real_path)]}}},
        target_counts={"atomic_positive": 1, "atomic_hard_negative": 0},
    )

    assert len(result.atomic_positive_rows) == 1
    row = result.atomic_positive_rows[0]
    assert row["source"] != raw_source
    assert row["target"] != raw_target
    assert row["source"] == "Компьюьерные мониторы показали новый отчет районной комиссии."
    assert row["activation_source"] == "real_pattern_replay"


def test_real_pattern_replay_rejects_unsafe_pattern(tmp_path: Path):
    compiler = _load_compiler()
    real_path = tmp_path / "real_error_pairs_mining.csv.gz"
    pd.DataFrame(
        [
            {
                "source": "Он проверил документ и здал отчет.",
                "target": "Он проверил документ и сдал итоговый отчет.",
                "rule_id": "dictionary_fuzzy",
                "rule_ids": json.dumps(["dictionary_fuzzy"], ensure_ascii=False),
                "edit_count": 2,
                "candidate_present": False,
                "strict_validator_passed": False,
            }
        ]
    ).to_csv(real_path, index=False)

    result = compiler.compile_rule_data(
        [_clean_row("Редактор проверил итоговый отчет районной комиссии.")],
        ["dictionary_fuzzy"],
        StaticCandidateGenerator(),
        {"data": {"rule_data_compiler": {"real_pattern_paths": [str(real_path)]}}},
        target_counts={"atomic_positive": 1, "atomic_hard_negative": 0},
    )

    assert result.atomic_positive_rows == []
    assert any(row["reason"] == "pattern_replay_unsafe" for row in result.rejection_rows)


def test_source_priority_fills_from_corpus_before_fallback_sources(tmp_path: Path):
    compiler = _load_compiler()
    real_path = tmp_path / "real_error_pairs_atomic.csv.gz"
    pd.DataFrame(
        [
            {
                "source": "Команда так же подготовила отчет для комиссии.",
                "target": "Команда также подготовила отчет для комиссии.",
                "rule_id": "context_tak_zhe",
                "rule_ids": json.dumps(["context_tak_zhe"], ensure_ascii=False),
                "edit_count": 1,
                "candidate_present": True,
                "strict_validator_passed": True,
            }
        ]
    ).to_csv(real_path, index=False)

    result = compiler.compile_rule_data(
        [_clean_row("Редакция также подготовила отчет для комиссии и отправила его в архив.")],
        ["context_tak_zhe"],
        ContextCandidateGenerator(),
        {
            "data": {
                "rule_data_compiler": {
                    "source_priority": ["corpus_mined", "real_pattern_replay"],
                    "real_pattern_paths": [str(real_path)],
                }
            }
        },
        target_counts={"atomic_positive": 1, "atomic_hard_negative": 0},
    )

    assert len(result.atomic_positive_rows) == 1
    assert result.atomic_positive_rows[0]["activation_source"] == "corpus_mined"


def test_rejection_report_aggregates_miner_name_and_reason(tmp_path: Path):
    compiler = _load_compiler()
    result = compiler.compile_rule_data(
        [_clean_row("Редактор проверил документ утром после заседания комиссии.")],
        ["comma_subordinate"],
        PunctuationCandidateGenerator(),
        {},
        target_counts={"atomic_positive": 1, "atomic_hard_negative": 0},
    )

    compiler.write_rule_data_compiler_reports(result, tmp_path)

    report = pd.read_csv(tmp_path / "rule_miner_rejection_report.csv")
    assert list(report.columns) == ["rule_id", "miner_name", "reason", "count", "example_source", "example_target"]
    assert report.loc[0, "miner_name"] == "syntax_mined"
    assert report.loc[0, "reason"] == "opportunities_seen_0"
    assert int(report.loc[0, "count"]) == 1


def test_underfilled_backlog_recommended_next_action_for_hard_negative_gap(tmp_path: Path):
    compiler = _load_compiler()
    result = compiler.compile_rule_data(
        [_clean_row("Редакция также подготовила отчет для комиссии и отправила его в архив.")],
        ["context_tak_zhe"],
        ContextCandidateGenerator(),
        {},
        target_counts={"atomic_positive": 1, "atomic_hard_negative": 1},
    )

    compiler.write_rule_data_compiler_reports(result, tmp_path)

    backlog = pd.read_csv(tmp_path / "rule_underfilled_backlog.csv")
    row = backlog.set_index("rule_id").loc["context_tak_zhe"]
    assert row["top_rejection_reason"] == "hard_negative_count_under_min"
    assert row["recommended_next_action"] == "add_hard_negative_miner_or_templates"
