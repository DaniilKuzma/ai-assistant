from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.calibrate_thresholds import (
    SafetyCaps,
    default_output_dir,
    passes_safety_caps,
    select_best_safe_threshold,
    write_calibration_reports,
)


def test_default_output_dir_uses_timestamped_subdir_inside_config_reports_dir(tmp_path: Path):
    config = {"paths": {"reports_dir": str(tmp_path / "reports" / "short_dataset_v2")}}

    output_dir = default_output_dir(config, timestamp="20260520_153012")

    assert output_dir == tmp_path / "reports" / "short_dataset_v2" / "threshold_calibration" / "20260520_153012"


def test_threshold_calibration_writes_reports_and_latest_pointers(tmp_path: Path):
    output_dir = tmp_path / "reports" / "short_dataset_v2" / "threshold_calibration" / "20260520_153012"
    latest_dir = output_dir.parent

    write_calibration_reports(
        output_dir=output_dir,
        latest_dir=latest_dir,
        sweep_rows=[
            {
                "rule_id": "dictionary_fuzzy",
                "candidate_threshold": 0.80,
                "edit_precision": 0.95,
                "clean_overcorrection_rate": 0.0,
            }
        ],
        decision_rows=[
            {
                "rule_id": "dictionary_fuzzy",
                "current_threshold": 0.95,
                "recommended_threshold": 0.80,
                "recommendation": "lower",
                "reason": "safe recall gain",
            }
        ],
        summary="# Threshold Sweep Summary\n\n- verdict: READY_FOR_THRESHOLD_UPDATE\n",
        recommended_thresholds={"dictionary_fuzzy_threshold": 0.80},
        manifest={"timestamp": "20260520_153012", "verdict": "READY_FOR_THRESHOLD_UPDATE"},
    )

    for name in [
        "threshold_sweep_by_rule.csv",
        "threshold_calibration_decision_by_rule.csv",
        "threshold_sweep_summary.md",
        "recommended_thresholds.yaml",
        "calibration_manifest.json",
    ]:
        assert (output_dir / name).exists()

    assert (latest_dir / "latest_manifest.json").exists()
    assert (latest_dir / "latest_recommended_thresholds.yaml").exists()
    assert yaml.safe_load((output_dir / "recommended_thresholds.yaml").read_text(encoding="utf-8")) == {
        "dictionary_fuzzy_threshold": 0.80
    }
    assert pd.read_csv(output_dir / "threshold_sweep_by_rule.csv").loc[0, "rule_id"] == "dictionary_fuzzy"


def test_select_best_safe_threshold_rejects_clean_overcorrection_cap_violation():
    current_metrics = {
        "spelling_f1": 0.003,
        "edit_recall": 0.18,
        "clean_overcorrection_rate": 0.0,
        "dirty_worse_rate": 0.0,
        "real_dirty_worse_rate": 0.0,
        "edit_precision": 0.95,
        "punctuation_f1": 0.72,
    }
    candidates = [
        {
            "candidate_threshold": 0.75,
            "spelling_f1": 0.20,
            "edit_recall": 0.30,
            "clean_overcorrection_rate": 0.02,
            "dirty_worse_rate": 0.0,
            "real_dirty_worse_rate": 0.0,
            "edit_precision": 0.95,
            "punctuation_f1": 0.72,
        },
        {
            "candidate_threshold": 0.80,
            "spelling_f1": 0.04,
            "edit_recall": 0.20,
            "clean_overcorrection_rate": 0.0,
            "dirty_worse_rate": 0.0,
            "real_dirty_worse_rate": 0.0,
            "edit_precision": 0.95,
            "punctuation_f1": 0.72,
        },
    ]

    selected = select_best_safe_threshold(
        current_threshold=0.95,
        current_metrics=current_metrics,
        candidate_rows=candidates,
        caps=SafetyCaps(clean_overcorrection_rate=0.005),
    )

    assert selected["candidate_threshold"] == pytest.approx(0.80)


def test_recommended_thresholds_never_violate_clean_overcorrection_cap_on_small_fixture():
    unsafe_metrics = {
        "clean_overcorrection_rate": 0.006,
        "dirty_worse_rate": 0.0,
        "real_dirty_worse_rate": 0.0,
        "edit_precision": 0.95,
        "punctuation_f1": 0.72,
    }

    assert not passes_safety_caps(unsafe_metrics, SafetyCaps(clean_overcorrection_rate=0.005))
