from pathlib import Path

import yaml

from src.training.train import train


def test_train_entrypoint_prepares_features_and_all_reports(tmp_path: Path):
    config = {
        "model": {
            "primary_encoder": "ai-forever/ruRoberta-large",
            "fallback_encoder": "ai-forever/ruBert-base",
            "local_files_only": False,
            "max_sequence_length": 32,
            "max_candidates": 8,
            "lora": {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.05},
        },
        "training": {
            "run_model_training": False,
            "max_train_examples": 4,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
            "mixed_precision": False,
        },
        "data": {"debug_clean_texts": ["Я не знаю, что делать.", "Чистый текст."]},
        "labels": {
            "punctuation": {"NONE": 0, "COMMA": 1, "DOT": 2},
            "error_types": {"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        },
        "thresholds": {"mode": "balanced", "balanced": {"spelling_threshold": 0.85}},
        "paths": {
            "adapter_output_dir": str(tmp_path / "models" / "adapters"),
            "heads_output_dir": str(tmp_path / "models" / "heads"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    result = train(config_path)

    assert result["status"] == "features_prepared"
    assert result["feature_count"] == 4
    assert result["evaluation_count"] == 4
    assert result["reports_dir"] == str(tmp_path / "reports")
    assert "evaluation_summary.csv" in result["report_paths"]
    for name in [
        "dataset_report.md",
        "training_report.md",
        "evaluation_summary.csv",
        "accepted_edits.csv",
        "rejected_edits.csv",
        "loss_curves.png",
        "threshold_precision_recall.png",
    ]:
        assert (tmp_path / "reports" / name).exists()

    dataset_report = (tmp_path / "reports" / "dataset_report.md").read_text(encoding="utf-8")
    assert "- total: 4" in dataset_report
    training_report = (tmp_path / "reports" / "training_report.md").read_text(encoding="utf-8")
    assert "- checkpoint_metric: combined_score" in training_report
    assert "- checkpoint_metric_value:" in training_report
    assert "- combined_score:" in training_report


def test_train_uses_trained_corrector_for_reports_when_model_training_runs(monkeypatch, tmp_path: Path):
    used = {"trained": False}

    class FakeTrainedCorrector:
        @classmethod
        def from_config(cls, config):
            used["trained"] = True
            return cls()

        def correct(self, text):
            from src.inference.corrector import CorrectionResult

            return CorrectionResult(text, text, [])

    monkeypatch.setattr("src.training.train._run_model_training", lambda config, features: {"status": "trained", "model_training_ran": True, "train_loss": 0.1})
    monkeypatch.setattr("src.training.train._build_features", lambda config, rows: [])
    monkeypatch.setattr("src.training.train.TrainedModelCorrector", FakeTrainedCorrector)

    config = {
        "model": {"max_sequence_length": 32, "max_candidates": 8},
        "training": {
            "run_model_training": True,
            "max_train_examples": 2,
            "max_val_examples": 1,
            "max_test_examples": 1,
            "show_progress": False,
        },
        "data": {"debug_clean_texts": ["Я не знаю, что делать.", "Чистый текст."]},
        "labels": {
            "punctuation": {"NONE": 0, "COMMA": 1, "DOT": 2},
            "error_types": {"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        },
        "thresholds": {"mode": "balanced", "balanced": {"spelling_threshold": 0.85}},
        "paths": {
            "adapter_output_dir": str(tmp_path / "models" / "adapters"),
            "heads_output_dir": str(tmp_path / "models" / "heads"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    train(config_path)

    assert used["trained"] is True
