from pathlib import Path

import pandas as pd

from scripts.threshold_safety import build_safety_rows, build_test_regression_reports


def test_build_safety_rows_excludes_test_and_marks_train_real_rows():
    frame = pd.DataFrame(
        [
            {"source": "val clean", "target": "val clean", "split": "val", "source_type": "clean_identity_from_open_clean", "is_clean": True, "is_hard_negative": False, "is_synthetic": False},
            {"source": "val hard", "target": "val hard", "split": "val", "source_type": "hard_negative_from_open_clean", "is_clean": True, "is_hard_negative": True, "is_synthetic": False},
            {"source": "val real", "target": "val real.", "split": "val", "source_type": "real_error_pair", "is_clean": False, "is_hard_negative": False, "is_synthetic": False},
            {"source": "train clean 1", "target": "train clean 1", "split": "train", "source_type": "clean_identity_from_open_clean", "is_clean": True, "is_hard_negative": False, "is_synthetic": False},
            {"source": "train clean 2", "target": "train clean 2", "split": "train", "source_type": "clean_identity_from_open_clean", "is_clean": True, "is_hard_negative": False, "is_synthetic": False},
            {"source": "train hard", "target": "train hard", "split": "train", "source_type": "hard_negative_from_open_clean", "is_clean": True, "is_hard_negative": True, "is_synthetic": False},
            {"source": "train real", "target": "train real.", "split": "train", "source_type": "real_error_pair", "is_clean": False, "is_hard_negative": False, "is_synthetic": False},
            {"source": "test clean", "target": "test clean", "split": "test", "source_type": "clean_identity_from_open_clean", "is_clean": True, "is_hard_negative": False, "is_synthetic": False},
        ]
    )

    safety = build_safety_rows(frame, train_clean_sample=1, train_hard_negative_sample=1)

    assert set(safety["split"]) == {"val", "train"}
    assert "test clean" not in set(safety["source"])
    assert (safety["safety_bucket"] == "val_clean_identity").sum() == 1
    assert (safety["safety_bucket"] == "val_hard_negative").sum() == 1
    assert (safety["safety_bucket"] == "val_real").sum() == 1
    assert (safety["safety_bucket"] == "calibration_safety_train_clean").sum() == 1
    assert (safety["safety_bucket"] == "calibration_safety_train_hard_negative").sum() == 1
    assert (safety["source_type"] == "calibration_safety_train_real").sum() == 1


def test_build_test_regression_reports_counts_rules_and_writes_expected_columns(tmp_path: Path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    dataset = tmp_path / "dataset.csv.gz"
    pd.DataFrame(
        [
            {
                "source": "чистый миллонов",
                "target": "чистый миллонов",
                "prediction": "чистый милонов",
                "split": "test",
                "source_type": "clean_identity_from_open_clean",
                "is_clean": True,
                "is_synthetic": False,
            },
            {
                "source": "грязный билет",
                "target": "грязный билет",
                "prediction": "грязный билет.",
                "split": "test",
                "source_type": "real_error_pair",
                "is_clean": False,
                "is_synthetic": False,
            },
        ]
    ).drop(columns=["prediction"]).to_csv(dataset, index=False)
    pd.DataFrame(
        [
            {
                "source": "чистый миллонов",
                "target": "чистый миллонов",
                "prediction": "чистый милонов",
                "error_types": "[]",
                "source_dataset": "fixture",
                "is_clean": True,
                "is_synthetic": False,
                "split": "test",
                "domain": "fixture",
            }
        ]
    ).to_csv(report_dir / "clean_overcorrection_examples.csv", index=False)
    pd.DataFrame(
        [
            {
                "source": "грязный билет",
                "target": "грязный билет",
                "prediction": "грязный билет.",
                "error_types": "[]",
                "source_dataset": "fixture",
                "is_clean": False,
                "is_synthetic": False,
                "split": "test",
                "domain": "fixture",
            }
        ]
    ).to_csv(report_dir / "dirty_worse_examples.csv", index=False)
    pd.DataFrame(
        [
            {"row_id": 0, "source": "миллонов", "replacement": "милонов", "edit_type": "spelling_replace", "rule_id": "double_consonant_candidate", "status": "accepted", "reason": "trusted", "confidence": 0.65},
            {"row_id": 1, "source": "", "replacement": ".", "edit_type": "final_punctuation", "rule_id": "final_punctuation_default", "status": "accepted", "reason": "trusted", "confidence": 0.88},
        ]
    ).to_csv(report_dir / "accepted_edits.csv", index=False)
    pd.DataFrame(
        [
            {"row_id": 1, "source": "билет", "replacement": "биллет", "edit_type": "spelling_replace", "rule_id": "dictionary_fuzzy", "status": "rejected", "reason": "threshold", "confidence": 0.7}
        ]
    ).to_csv(report_dir / "rejected_edits.csv", index=False)
    pd.DataFrame(
        [
            {"rule_id": "double_consonant_candidate", "edit_type": "spelling", "gold_count": 0, "candidate_count": 1, "positive_score_mean": 0.0, "positive_score_p10": 0.0, "positive_score_p50": 0.0, "positive_score_p90": 0.0, "negative_score_mean": 0.65, "accepted_count": 1, "rejected_by_threshold_count": 0, "rejected_by_validator_count": 0, "rejected_reason_counts": "{}"},
        ]
    ).to_csv(report_dir / "candidate_score_distribution_by_rule.csv", index=False)

    by_rule, examples = build_test_regression_reports(
        report_dir=report_dir,
        dataset_path=dataset,
        profile_thresholds={"double_consonant_candidate_threshold": 0.76, "final_punctuation_threshold": 0.88},
    )

    assert list(by_rule.columns) == [
        "rule_id",
        "edit_type",
        "clean_overcorrection_count",
        "dirty_worse_count",
        "real_dirty_worse_count",
        "accepted_count",
        "rejected_count",
        "examples",
        "suspected_threshold_key",
        "recommendation",
    ]
    clean_row = by_rule[by_rule["rule_id"] == "double_consonant_candidate"].iloc[0]
    dirty_row = by_rule[by_rule["rule_id"] == "final_punctuation_default"].iloc[0]
    assert clean_row["clean_overcorrection_count"] == 1
    assert dirty_row["real_dirty_worse_count"] == 1
    assert (report_dir / "test_regression_by_rule.csv").exists()
    assert set(examples.columns) >= {"rule_id", "source", "target", "prediction", "score", "threshold", "reason"}
