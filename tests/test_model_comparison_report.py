from pathlib import Path

import pandas as pd

from src.evaluation.activation_plan import write_model_comparison_report


def test_working_vs_canonical_comparison_keeps_v1_when_training_not_run(tmp_path: Path):
    v1 = tmp_path / "v1.csv"
    canonical = tmp_path / "canonical.csv"
    output_csv = tmp_path / "comparison.csv"
    output_md = tmp_path / "comparison.md"
    pd.DataFrame(
        [{"edit_f1": 0.40, "spelling_f1": 0.20, "punctuation_f1": 0.70, "edit_precision": 0.99, "clean_overcorrection_rate": 0.0, "dirty_worse_rate": 0.0}]
    ).to_csv(v1, index=False)
    pd.DataFrame(
        [{"edit_f1": 0.42, "spelling_f1": 0.22, "punctuation_f1": 0.71, "edit_precision": 0.99, "clean_overcorrection_rate": 0.0, "dirty_worse_rate": 0.0}]
    ).to_csv(canonical, index=False)

    decision = write_model_comparison_report(
        working_summary_path=v1,
        canonical_summary_path=canonical,
        output_csv_path=output_csv,
        output_md_path=output_md,
        training_ran=False,
    )

    assert decision == "READY_FOR_COLAB_TRAINING"
    assert output_csv.exists()
    assert "READY_FOR_COLAB_TRAINING" in output_md.read_text(encoding="utf-8")
