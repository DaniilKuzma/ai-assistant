from pathlib import Path

import pandas as pd
import yaml

from src.training.train import _build_features, _load_evaluation_rows, evaluate_trained_model, train


def test_train_entrypoint_prepares_features_and_all_reports(tmp_path: Path):
    config = {
        "model": {
            "primary_encoder": "ai-forever/ruRoberta-large",
            "fallback_encoder": "ai-forever/ruRoberta-large",
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
    assert "rule_precision_recall.csv" in result["report_paths"]
    for name in [
        "dataset_report.md",
        "training_report.md",
        "evaluation_summary.csv",
        "rule_precision_recall.csv",
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


def test_build_features_uses_dictionary_config(tmp_path: Path):
    lexicon_path = tmp_path / "russian_lexicon.txt"
    lexicon_path.write_text("библиотека\n", encoding="utf-8")
    config = {
        "model": {
            "primary_encoder": "ai-forever/ruRoberta-large",
            "fallback_encoder": "ai-forever/ruRoberta-large",
            "local_files_only": False,
            "max_sequence_length": 32,
            "max_candidates": 8,
        },
        "training": {"run_model_training": False, "show_progress": False},
        "dictionary": {
            "enabled": True,
            "lexicon_path": str(lexicon_path),
            "max_candidates": 2,
            "min_score": 85,
        },
        "labels": {
            "punctuation": {"NONE": 0, "DOT": 1},
            "error_types": {"keep": 0, "spelling": 1},
        },
    }

    features = _build_features(config, [{"source": "Библеотека открыта.", "target": "Библиотека открыта."}])

    assert "dictionary_fuzzy" in features[0].candidate_rule_ids


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


def test_training_evaluation_uses_validation_split_only(tmp_path: Path):
    processed_path = tmp_path / "dataset.csv.gz"
    rows = [
        {"source": "val 1", "target": "val 1", "split": "val", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
        {"source": "val 2", "target": "val 2", "split": "val", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
        {"source": "test 1", "target": "test 1", "split": "test", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
        {"source": "test 2", "target": "test 2", "split": "test", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
    ]
    pd.DataFrame(rows).to_csv(processed_path, index=False)

    evaluation_rows = _load_evaluation_rows(
        {
            "data": {"processed_train_path": str(processed_path)},
            "training": {"max_val_examples": 2, "max_test_examples": 2},
        },
        fallback_rows=[],
    )

    assert [row["split"] for row in evaluation_rows] == ["val", "val"]


def test_training_evaluation_split_can_be_configured_to_test(tmp_path: Path):
    processed_path = tmp_path / "dataset.csv.gz"
    rows = [
        {"source": "val 1", "target": "val 1", "split": "val", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
        {"source": "val 2", "target": "val 2", "split": "val", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
        {"source": "test 1", "target": "test 1", "split": "test", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
        {"source": "test 2", "target": "test 2", "split": "test", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
    ]
    pd.DataFrame(rows).to_csv(processed_path, index=False)

    evaluation_rows = _load_evaluation_rows(
        {
            "data": {"processed_train_path": str(processed_path)},
            "training": {"evaluation_split": "test", "max_val_examples": 2, "max_test_examples": 2},
        },
        fallback_rows=[],
    )

    assert [row["split"] for row in evaluation_rows] == ["test", "test"]


def test_training_reports_do_not_run_hidden_second_evaluation(monkeypatch, tmp_path: Path):
    calls = {"correct": 0}

    class CountingCorrector:
        def correct(self, text):
            from src.inference.corrector import CorrectionResult
            from src.validation.diff_analyzer import Edit

            calls["correct"] += 1
            return CorrectionResult(
                text,
                text,
                [Edit("", ".", "final_punctuation", status="accepted", confidence=0.9)],
            )

    monkeypatch.setattr("src.training.train._build_features", lambda config, rows: [])
    monkeypatch.setattr("src.training.train._build_evaluation_corrector", lambda config, model_training_ran: CountingCorrector())

    config = {
        "model": {"max_sequence_length": 32, "max_candidates": 8},
        "training": {
            "run_model_training": False,
            "max_train_examples": 2,
            "max_val_examples": 4,
            "max_test_examples": 4,
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

    result = train(config_path)

    assert calls["correct"] == result["evaluation_count"]


def test_evaluate_trained_model_uses_test_split_only(tmp_path: Path):
    processed_path = tmp_path / "dataset.csv.gz"
    rows = [
        {"source": "val 1", "target": "val 1", "split": "val", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
        {"source": "val 2", "target": "val 2", "split": "val", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
        {"source": "test 1", "target": "test 1", "split": "test", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
        {"source": "test 2", "target": "test 2", "split": "test", "is_clean": True, "is_synthetic": False, "error_types": "[]"},
    ]
    pd.DataFrame(rows).to_csv(processed_path, index=False)

    seen_sources = []

    class RecordingCorrector:
        def correct(self, text):
            from src.inference.corrector import CorrectionResult

            seen_sources.append(text)
            return CorrectionResult(text, text, [])

    config = {
        "training": {"max_val_examples": 2, "max_test_examples": 2},
        "data": {"processed_train_path": str(processed_path)},
        "paths": {"reports_dir": str(tmp_path / "reports")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    result = evaluate_trained_model(config_path, split="test", corrector=RecordingCorrector())

    assert seen_sources == ["test 1", "test 2"]
    assert result["evaluation_split"] == "test"
    assert result["evaluation_count"] == 2
