from pathlib import Path
import os
import subprocess
import sys

import pandas as pd
import yaml


REQUIRED_REPORTS = [
    "dataset_report.md",
    "training_report.md",
    "evaluation_summary.csv",
    "error_by_type.csv",
    "accepted_edits.csv",
    "rejected_edits.csv",
]


def test_pipeline_runs_with_model_training_disabled(tmp_path: Path):
    processed_path = tmp_path / "dataset.csv.gz"
    pd.DataFrame(
        [
            {
                "source": "Я незнаю что делать",
                "target": "Я не знаю, что делать.",
                "split": "train",
                "is_clean": False,
                "is_synthetic": True,
                "error_types": '["split_join", "punctuation", "final_punctuation"]',
                "source_dataset": "unit",
                "domain": "unit",
            },
            {
                "source": "Чистый текст.",
                "target": "Чистый текст.",
                "split": "train",
                "is_clean": True,
                "is_synthetic": False,
                "error_types": "[]",
                "source_dataset": "unit",
                "domain": "unit",
            },
            {
                "source": "Во первых это важно",
                "target": "Во-первых, это важно.",
                "split": "test",
                "is_clean": False,
                "is_synthetic": True,
                "error_types": '["hyphen", "punctuation", "final_punctuation"]',
                "source_dataset": "unit",
                "domain": "unit",
            },
            {
                "source": "Проверочный текст.",
                "target": "Проверочный текст.",
                "split": "test",
                "is_clean": True,
                "is_synthetic": False,
                "error_types": "[]",
                "source_dataset": "unit",
                "domain": "unit",
            },
        ]
    ).to_csv(processed_path, index=False)

    config = {
        "model": {"max_sequence_length": 32, "max_candidates": 8},
        "training": {
            "run_model_training": True,
            "evaluation_split": "test",
            "max_train_examples": 2,
            "max_test_examples": 2,
            "show_progress": False,
        },
        "data": {"processed_train_path": str(processed_path)},
        "labels": {
            "punctuation": {"NONE": 0, "COMMA": 1, "DOT": 2},
            "punctuation_actions": {"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2},
            "error_types": {"keep": 0, "split_join": 1, "hyphen": 2, "punctuation": 3, "final_punctuation": 4},
        },
        "metrics": {"combined_score_weights": {"exact_match": 1.0}},
        "thresholds": {"mode": "balanced", "balanced": {"spelling_threshold": 0.85}},
        "paths": {
            "adapter_output_dir": str(tmp_path / "models" / "adapters"),
            "heads_output_dir": str(tmp_path / "models" / "heads"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    env = {**os.environ, "RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING": "1"}
    completed = subprocess.run(
        [sys.executable, "-m", "src.training.train", str(config_path)],
        cwd=Path.cwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for report_name in REQUIRED_REPORTS:
        report_path = tmp_path / "reports" / report_name
        assert report_path.exists(), f"{report_name} does not exist"
        assert report_path.stat().st_size > 0, f"{report_name} is empty"

    training_report = (tmp_path / "reports" / "training_report.md").read_text(encoding="utf-8")
    assert "- model_training_disabled: True" in training_report
    assert "- model_training_disabled_source: RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING" in training_report
    assert "- evaluation_backend: no_model" in training_report

    summary = pd.read_csv(tmp_path / "reports" / "evaluation_summary.csv")
    assert summary.loc[0, "evaluation_backend"] == "no_model"
    assert bool(summary.loc[0, "model_training_disabled"]) is True
