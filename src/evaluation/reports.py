from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import is_dirty_worse_row


def write_dataset_report(stats: dict[str, int], path: str | Path) -> None:
    lines = ["# Dataset Report", "", *[f"- {key}: {value}" for key, value in stats.items()]]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evaluation_summary(metrics: dict[str, float], path: str | Path) -> None:
    pd.DataFrame([metrics]).to_csv(path, index=False)


def write_edit_logs(accepted: list[dict], rejected: list[dict], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    columns = ["row_id", "source", "replacement", "edit_type", "rule_id", "status", "reason", "confidence"]
    pd.DataFrame(accepted, columns=columns).to_csv(output / "accepted_edits.csv", index=False)
    pd.DataFrame(rejected, columns=columns).to_csv(output / "rejected_edits.csv", index=False)


def write_training_report(metrics: dict[str, Any], path: str | Path) -> None:
    lines = ["# Training Report", "", *[f"- {key}: {value}" for key, value in metrics.items()]]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_loss_curve(losses: list[float], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(list(range(1, len(losses) + 1)), losses, marker="o")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title("Training loss")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def write_threshold_precision_recall_plot(frame: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(frame["threshold"], frame["precision"], marker="o", label="precision")
    axis.plot(frame["threshold"], frame["recall"], marker="o", label="recall")
    axis.set_xlabel("threshold")
    axis.set_ylabel("score")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def write_required_evaluation_reports(
    rows: list[dict],
    metrics: dict[str, float],
    output_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    example_columns = ["source", "target", "prediction", "error_types", "source_dataset", "is_clean", "is_synthetic", "split", "domain"]

    pd.DataFrame([{**metrics, **(metadata or {})}]).to_csv(output / "evaluation_summary.csv", index=False)
    pd.DataFrame(_error_by_type(rows)).to_csv(output / "error_by_type.csv", index=False)
    pd.DataFrame(
        [row for row in rows if row.get("is_clean") and row.get("prediction") != row.get("target")],
        columns=example_columns,
    ).to_csv(
        output / "clean_overcorrection_examples.csv", index=False
    )
    pd.DataFrame(
        [
            row
            for row in rows
            if not row.get("is_clean") and is_dirty_worse_row(row)
        ],
        columns=example_columns,
    ).to_csv(output / "dirty_worse_examples.csv", index=False)


def _error_by_type(rows: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for row in rows:
        for error_type in _iter_error_types(row.get("error_types", []) or []):
            counts[error_type] = counts.get(error_type, 0) + 1
    return [{"error_type": key, "count": value} for key, value in sorted(counts.items())]


def _iter_error_types(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
        return [str(parsed)] if str(parsed) else []
    return [str(value)] if str(value) else []
