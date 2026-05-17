from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.evaluate import evaluate_rows, evaluate_rows_detailed
from src.evaluation.threshold_sweep import threshold_sweep
from src.inference.corrector import CorrectionResult
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
            "source_dataset": "synthetic_rules",
            "is_clean": False,
            "is_synthetic": True,
        },
    ]

    metrics = evaluate_rows(rows, output_dir=tmp_path)

    assert metrics["exact_match"] == 1.0
    assert "combined_score" in metrics
    for name in [
        "evaluation_summary.csv",
        "error_by_type.csv",
        "clean_overcorrection_examples.csv",
        "dirty_worse_examples.csv",
        "accepted_edits.csv",
        "rejected_edits.csv",
    ]:
        assert (tmp_path / name).exists()
    assert not pd.read_csv(tmp_path / "accepted_edits.csv").empty
    assert list(pd.read_csv(tmp_path / "dirty_worse_examples.csv").columns)
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
