from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.evaluate import evaluate_rows, evaluate_rows_detailed
from src.evaluation.threshold_sweep import threshold_sweep
from src.inference.corrector import CorrectionResult
from src.inference.model_corrector import ModelCandidatePrediction, TrainedModelCorrector
from src.validation.diff_analyzer import Edit


def test_evaluate_rows_writes_required_reports(tmp_path: Path):
    rows = [
        {
            "source": "Я незнаю что делать",
            "target": "Я не знаю, что делать.",
            "error_types": ["split_join", "punctuation", "final_punctuation"],
            "source_dataset": "unit",
            "is_clean": False,
        },
        {
            "source": "Чистый текст.",
            "target": "Чистый текст.",
            "error_types": [],
            "source_dataset": "unit",
            "is_clean": True,
            "is_synthetic": False,
        },
        {
            "source": "Во первых это важно",
            "target": "Во-первых, это важно.",
            "error_types": ["hyphen", "punctuation", "final_punctuation"],
            "source_dataset": "synthetic_open_corpus_hyphen",
            "is_clean": False,
            "is_synthetic": True,
        },
    ]

    metrics = evaluate_rows(rows, corrector=RequiredReportsCorrector(), output_dir=tmp_path)

    assert metrics["exact_match"] == 1.0
    assert "combined_score" in metrics
    for name in [
        "evaluation_summary.csv",
        "error_by_type.csv",
        "rule_precision_recall.csv",
        "error_by_rule.csv",
        "rule_worse_examples.csv",
        "candidate_recall_by_rule.csv",
        "gap_label_coverage_by_rule.csv",
        "clean_overcorrection_examples.csv",
        "dirty_worse_examples.csv",
        "accepted_edits.csv",
        "rejected_edits.csv",
    ]:
        assert (tmp_path / name).exists()
    assert not pd.read_csv(tmp_path / "accepted_edits.csv").empty
    assert list(pd.read_csv(tmp_path / "dirty_worse_examples.csv").columns)
    rule_summary = pd.read_csv(tmp_path / "rule_precision_recall.csv").set_index("rule_id")
    assert "frequent_errors" not in rule_summary.index
    assert rule_summary.loc["frequent_error_exact", "group"] == "typical_dictionary_words"
    assert rule_summary.loc["frequent_error_exact", "true_positive"] == 1
    summary = pd.read_csv(tmp_path / "evaluation_summary.csv")
    for column in ["combined_score", "real_exact_match", "synthetic_exact_match", "clean_exact_match"]:
        assert column in summary.columns


def test_error_by_type_report_parses_json_encoded_error_type_lists(tmp_path: Path):
    rows = [
        {
            "source": "врядли это важно",
            "target": "Вряд ли это важно.",
            "error_types": '["case", "split_join", "final_punctuation"]',
            "source_dataset": "unit",
            "is_clean": False,
        }
    ]

    evaluate_rows(rows, output_dir=tmp_path)

    report = pd.read_csv(tmp_path / "error_by_type.csv")
    assert set(report["error_type"]) == {"case", "split_join", "final_punctuation"}


def test_threshold_scores_mark_individual_partial_edits_correct():
    rows = [
        {
            "source": "Я незнаю что делать",
            "target": "Я не знаю, что делать.",
            "error_types": ["split_join", "punctuation", "final_punctuation"],
            "source_dataset": "unit",
            "is_clean": False,
        }
    ]

    result = evaluate_rows_detailed(rows, corrector=PartialCorrector())
    sweep = threshold_sweep(result.edit_scores, [0.9])

    assert result.metrics["exact_match"] == 0.0
    assert result.metrics["dirty_improved_rate"] == 1.0
    assert result.metrics["dirty_worse_rate"] == 0.0
    assert result.edit_scores[0]["is_correct"] is True
    assert sweep.iloc[0]["precision"] == 1.0
    assert sweep.iloc[0]["recall"] == pytest.approx(1 / 3)


def test_evaluation_reports_include_rule_id_in_edit_outputs(tmp_path: Path):
    rows = [
        {
            "source": "Я недумаю",
            "target": "Я не думаю",
            "error_types": ["split_join"],
            "source_dataset": "unit",
            "is_clean": False,
        }
    ]

    result = evaluate_rows_detailed(rows, corrector=RuleIdCorrector(), output_dir=tmp_path)
    accepted_report = pd.read_csv(tmp_path / "accepted_edits.csv")

    assert "rule_id" in accepted_report.columns
    assert accepted_report.loc[0, "rule_id"] == "ne_verb"
    assert result.accepted_edits[0]["rule_id"] == "ne_verb"
    assert result.edit_scores[0]["rule_id"] == "ne_verb"


def test_score_distribution_report_written(tmp_path: Path):
    rows = [
        {
            "source": "Я незнаю что делать",
            "target": "Я не знаю что делать",
            "error_types": ["split_join"],
            "source_dataset": "unit",
            "is_clean": False,
        }
    ]
    corrector = TrainedModelCorrector(
        TraceableBackend({"не знаю": 0.99}),
        thresholds={"split_join_threshold": 0.9},
    )

    evaluate_rows_detailed(rows, corrector=corrector, output_dir=tmp_path)

    report = pd.read_csv(tmp_path / "candidate_score_distribution_by_rule.csv")
    assert list(report.columns) == [
        "rule_id",
        "edit_type",
        "gold_count",
        "candidate_count",
        "positive_score_mean",
        "positive_score_p10",
        "positive_score_p50",
        "positive_score_p90",
        "negative_score_mean",
        "accepted_count",
        "rejected_by_threshold_count",
        "rejected_by_validator_count",
        "rejected_reason_counts",
        "examples_high_score_rejected",
        "examples_low_score_gold",
    ]
    row = report.loc[report["rule_id"] == "frequent_error_exact"].iloc[0]
    assert row["candidate_count"] > 0
    assert row["accepted_count"] > 0


class PartialCorrector:
    def correct(self, text: str) -> CorrectionResult:
        return CorrectionResult(
            source_text=text,
            corrected_text="Я не знаю что делать",
            edits=[
                Edit(
                    "незнаю",
                    "не знаю",
                    "split_word",
                    start=2,
                    end=8,
                    status="accepted",
                    confidence=0.95,
                )
            ],
        )


class RequiredReportsCorrector:
    def correct(self, text: str) -> CorrectionResult:
        if text == "Я незнаю что делать":
            return CorrectionResult(
                source_text=text,
                corrected_text="Я не знаю, что делать.",
                edits=[
                    Edit(
                        "незнаю",
                        "не знаю",
                        "split_word",
                        start=2,
                        end=8,
                        status="accepted",
                        confidence=0.95,
                        rule_id="frequent_errors",
                    ),
                    Edit(
                        "",
                        ",",
                        "punctuation_insert",
                        start=10,
                        end=10,
                        status="accepted",
                        confidence=0.86,
                        rule_id="comma_subordinate",
                    ),
                    Edit(
                        "",
                        ".",
                        "final_punctuation",
                        start=19,
                        end=19,
                        status="accepted",
                        confidence=0.9,
                        rule_id="final_punctuation_default",
                    ),
                ],
            )
        if text == "Во первых это важно":
            return CorrectionResult(
                source_text=text,
                corrected_text="Во-первых, это важно.",
                edits=[
                    Edit(
                        "Во первых",
                        "Во-первых",
                        "hyphen_change",
                        start=0,
                        end=9,
                        status="accepted",
                        confidence=0.88,
                        rule_id="hyphen_whitelist",
                    ),
                    Edit(
                        "",
                        ",",
                        "punctuation_insert",
                        start=9,
                        end=9,
                        status="accepted",
                        confidence=0.86,
                        rule_id="introductory_comma",
                    ),
                    Edit(
                        "",
                        ".",
                        "final_punctuation",
                        start=19,
                        end=19,
                        status="accepted",
                        confidence=0.9,
                        rule_id="final_punctuation_default",
                    ),
                ],
            )
        return CorrectionResult(source_text=text, corrected_text=text, edits=[])


class RuleIdCorrector:
    def correct(self, text: str) -> CorrectionResult:
        return CorrectionResult(
            source_text=text,
            corrected_text="Я не думаю",
            edits=[
                Edit(
                    "недумаю",
                    "не думаю",
                    "split_word",
                    start=2,
                    end=9,
                    status="accepted",
                    confidence=0.97,
                    rule_id="ne_verb",
                )
            ],
        )


class TraceableBackend:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score_candidates(self, text: str, candidates: list) -> list[ModelCandidatePrediction]:
        return [
            ModelCandidatePrediction(
                candidate=candidate,
                score=self.scores.get(candidate.replacement, 0.0),
                confidence=self.scores.get(candidate.replacement, 0.0),
            )
            for candidate in candidates
        ]

    def predict_punctuation(self, text: str):
        return []
