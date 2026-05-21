from pathlib import Path

import pandas as pd

from src.evaluation.activation_plan import write_verified_activation_plan


def test_verified_activation_plan_resolves_ready_count_mismatch(tmp_path: Path):
    output_csv = tmp_path / "next_dataset_activation_plan_verified.csv"
    output_md = tmp_path / "next_dataset_activation_plan_verified.md"

    frame = write_verified_activation_plan(
        matrix_activation_path="reports/matrix_eval/next_dataset_activation_plan.csv",
        matrix_summary_path="reports/matrix_eval/matrix_rule_eval_summary.csv",
        matrix_under_quota_path="reports/matrix_eval/under_quota_rule_audit.csv",
        output_csv_path=output_csv,
        output_md_path=output_md,
    )

    includes = frame[frame["activation_decision"] == "INCLUDE"]["rule_id"].tolist()
    hyphen = frame[frame["rule_id"] == "hyphen_whitelist"].iloc[0]

    assert len(includes) == 11
    assert "hyphen_whitelist" not in includes
    assert hyphen["activation_decision"] == "EXCLUDE"
    assert "under quota" in hyphen["reason"]
    assert output_csv.exists()
    assert "Matrix READY_NEXT_DATASET count: 12" in output_md.read_text(encoding="utf-8")
    assert pd.read_csv(output_csv)["rule_id"].nunique() == len(frame)
