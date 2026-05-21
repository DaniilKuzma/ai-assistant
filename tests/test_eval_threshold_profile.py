from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.training.train import _select_evaluation_corrector, train


class _NoopCorrector:
    def correct(self, text):
        from src.inference.corrector import CorrectionResult

        return CorrectionResult(text, text, [])


def _checkpoint_paths(tmp_path: Path) -> dict[str, str]:
    adapter_dir = tmp_path / "models" / "adapters"
    heads_dir = tmp_path / "models" / "heads"
    adapter_dir.mkdir(parents=True)
    heads_dir.mkdir(parents=True)
    (heads_dir / "heads.pt").write_bytes(b"placeholder")
    return {
        "adapter_output_dir": str(adapter_dir),
        "heads_output_dir": str(heads_dir),
    }


def test_eval_only_existing_checkpoint_uses_configured_threshold_profile(monkeypatch, tmp_path: Path):
    seen_modes = []

    class FakeTrainedCorrector:
        @classmethod
        def from_config(cls, config):
            seen_modes.append(config["thresholds"]["mode"])
            return _NoopCorrector()

    monkeypatch.setattr("src.training.train.TrainedModelCorrector", FakeTrainedCorrector)

    _corrector, metadata = _select_evaluation_corrector(
        {
            "paths": _checkpoint_paths(tmp_path),
            "thresholds": {
                "mode": "calibrated_guarded",
                "calibrated_guarded": {"spelling_threshold": 0.75},
                "conservative": {"spelling_threshold": 0.95},
            },
        },
        model_training_ran=False,
        model_training_disabled_source="RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING",
    )

    assert seen_modes == ["calibrated_guarded"]
    assert metadata["evaluation_backend"] == "existing_checkpoint"
    assert metadata["threshold_profile_used"] == "calibrated_guarded"
    assert metadata["threshold_profile_source"] == "config"
    assert metadata["threshold_mode_from_config"] == "calibrated_guarded"
    assert metadata["thresholds_mode_fallback_used"] is False


def test_eval_only_missing_threshold_mode_falls_back_and_reports_it(monkeypatch, tmp_path: Path):
    seen_modes = []

    class FakeTrainedCorrector:
        @classmethod
        def from_config(cls, config):
            seen_modes.append(config["thresholds"]["mode"])
            return _NoopCorrector()

    monkeypatch.setattr("src.training.train.TrainedModelCorrector", FakeTrainedCorrector)

    _corrector, metadata = _select_evaluation_corrector(
        {
            "paths": _checkpoint_paths(tmp_path),
            "thresholds": {
                "conservative": {"spelling_threshold": 0.95},
                "default": {"spelling_threshold": 0.85},
            },
        },
        model_training_ran=False,
        model_training_disabled_source="RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING",
    )

    assert seen_modes == ["conservative"]
    assert metadata["threshold_profile_used"] == "conservative"
    assert metadata["threshold_profile_source"] == "fallback:conservative"
    assert metadata["threshold_mode_from_config"] == ""
    assert metadata["thresholds_mode_fallback_used"] is True


def test_eval_only_missing_named_threshold_profile_fails_clearly(tmp_path: Path):
    with pytest.raises(ValueError, match="missing_threshold_profile"):
        _select_evaluation_corrector(
            {
                "paths": _checkpoint_paths(tmp_path),
                "thresholds": {
                    "mode": "calibrated_guarded",
                    "conservative": {"spelling_threshold": 0.95},
                },
            },
            model_training_ran=False,
            model_training_disabled_source="RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING",
        )


def test_eval_only_report_includes_threshold_profile_metadata_and_does_not_overwrite_artifacts(
    monkeypatch,
    tmp_path: Path,
):
    class FakeTrainedCorrector:
        @classmethod
        def from_config(cls, config):
            return _NoopCorrector()

    monkeypatch.setenv("RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING", "1")
    monkeypatch.setattr("src.training.train.TrainedModelCorrector", FakeTrainedCorrector)

    paths = _checkpoint_paths(tmp_path)
    artifact_thresholds = Path(paths["adapter_output_dir"]) / "thresholds.json"
    artifact_thresholds.write_text('{"mode": "conservative"}\n', encoding="utf-8")

    config = {
        "model": {"max_sequence_length": 32, "max_candidates": 8},
        "training": {
            "run_model_training": True,
            "max_train_examples": 2,
            "max_val_examples": 2,
            "show_progress": False,
        },
        "data": {"debug_clean_texts": ["Чистый текст.", "Проверочный текст."]},
        "labels": {
            "punctuation": {"NONE": 0, "COMMA": 1, "DOT": 2},
            "error_types": {"keep": 0, "punctuation": 1},
        },
        "thresholds": {
            "mode": "calibrated_guarded",
            "calibrated_guarded": {"spelling_threshold": 0.75},
            "conservative": {"spelling_threshold": 0.95},
        },
        "paths": {
            **paths,
            "reports_dir": str(tmp_path / "reports"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    result = train(config_path)

    training_report = (tmp_path / "reports" / "training_report.md").read_text(encoding="utf-8")
    summary = pd.read_csv(tmp_path / "reports" / "evaluation_summary.csv")
    assert result["threshold_profile_used"] == "calibrated_guarded"
    assert "- threshold_profile_used: calibrated_guarded" in training_report
    assert "- threshold_profile_source: config" in training_report
    assert bool(summary.loc[0, "thresholds_mode_fallback_used"]) is False
    assert summary.loc[0, "threshold_profile_used"] == "calibrated_guarded"
    assert artifact_thresholds.read_text(encoding="utf-8") == '{"mode": "conservative"}\n'
