from pathlib import Path

import pandas as pd

from src.evaluation.reports import write_loss_curve, write_threshold_precision_recall_plot


def test_report_plot_helpers_write_png_files(tmp_path: Path):
    loss_path = tmp_path / "loss_curves.png"
    threshold_path = tmp_path / "threshold_precision_recall.png"

    write_loss_curve([1.0, 0.8, 0.6], loss_path)
    write_threshold_precision_recall_plot(
        pd.DataFrame(
            [
                {"threshold": 0.5, "precision": 0.7, "recall": 0.9},
                {"threshold": 0.8, "precision": 0.9, "recall": 0.6},
            ]
        ),
        threshold_path,
    )

    assert loss_path.exists()
    assert threshold_path.exists()
