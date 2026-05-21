import json
from pathlib import Path

import pandas as pd

from src.data.matrix_eval_dataset import (
    MATRIX_EVAL_COLUMNS,
    build_matrix_eval_dataset,
    write_matrix_eval_dataset,
)


def test_matrix_eval_dataset_builds_candidate_backed_rows_for_selected_rules(tmp_path: Path):
    result = build_matrix_eval_dataset(
        output_dir=tmp_path,
        reports_dir=tmp_path,
        selected_rule_ids=["final_punctuation_default", "comma_subordinate"],
        min_examples_per_rule=1,
        preferred_examples_per_rule=2,
        max_examples_per_rule=3,
        hard_negative_count=2,
        current_dataset_path=tmp_path / "missing_current.csv.gz",
        clean_pool_path=tmp_path / "missing_clean_pool.csv.gz",
    )

    frame = result.frame
    assert list(frame.columns) == MATRIX_EVAL_COLUMNS
    assert set(frame["split"]) == {"matrix_eval"}
    assert {"final_punctuation_default", "comma_subordinate"} <= set(frame["rule_id"])

    positives = frame[frame["source_type"] != "matrix_hard_negative"]
    assert positives["candidate_present"].all()
    for row in positives.to_dict("records"):
        assert row["rule_id"] in json.loads(row["candidate_rule_ids"])

    assert (frame["source_type"] == "matrix_hard_negative").any()
    assert result.manifest["verdict"] in {"MATRIX_EVAL_DATASET_READY", "MATRIX_EVAL_DATASET_PARTIAL"}


def test_write_matrix_eval_dataset_creates_expected_artifacts(tmp_path: Path):
    result = write_matrix_eval_dataset(
        output_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        selected_rule_ids=["final_punctuation_default"],
        min_examples_per_rule=1,
        preferred_examples_per_rule=1,
        max_examples_per_rule=2,
        hard_negative_count=1,
        current_dataset_path=tmp_path / "missing_current.csv.gz",
        clean_pool_path=tmp_path / "missing_clean_pool.csv.gz",
    )

    assert Path(result["dataset_path"]).exists()
    assert Path(result["by_rule_path"]).exists()
    assert Path(result["manifest_path"]).exists()
    assert (tmp_path / "reports" / "matrix_eval_dataset_report.md").exists()
    assert (tmp_path / "reports" / "matrix_eval_candidate_recall_by_rule.csv").exists()
    assert (tmp_path / "reports" / "matrix_eval_gap_coverage_by_rule.csv").exists()

    frame = pd.read_csv(result["dataset_path"])
    assert len(frame) >= 2
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["total_rows"] == len(frame)
