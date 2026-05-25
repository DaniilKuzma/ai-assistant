from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.config.load_config import load_config
import src.training.train as train_module
from src.training.train import main, train


ROOT = Path(__file__).resolve().parents[1]


def _debug_training_config(tmp_path: Path) -> Path:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["model"] = {
        **config.get("model", {}),
        "max_sequence_length": 24,
        "local_files_only": True,
        "lora": {"enabled": False},
    }
    config["generation"] = {
        **config.get("generation", {}),
        "samples_per_epoch": 4,
    }
    config["training"] = {
        **config.get("training", {}),
        "epochs": 1,
        "batch_size": 2,
        "eval_batch_size": 2,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "mixed_precision": False,
        "num_workers": 0,
        "save_each_epoch": False,
        "validate_each_epoch": True,
        "smoke_steps": 2,
    }
    config["paths"] = {
        **config.get("paths", {}),
        "adapter_output_dir": str(tmp_path / "models" / "adapters"),
        "heads_output_dir": str(tmp_path / "models" / "heads"),
        "generated_eval_dir": str(tmp_path / "generated_eval"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config_path


def test_train_entrypoint_runs_direct_debug_smoke_without_ruroberta(monkeypatch, tmp_path: Path) -> None:
    def fail_load_tokenizer(*args, **kwargs):
        raise AssertionError("debug-model training must not load ruRoberta tokenizer")

    monkeypatch.setattr("src.model.encoder.load_tokenizer", fail_load_tokenizer)
    config_path = _debug_training_config(tmp_path)

    result = train(
        config_path,
        overrides={"smoke": True, "steps": 2},
        debug_model=True,
    )

    heads_dir = tmp_path / "models" / "heads"
    summary = json.loads((heads_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert Path(result["heads_path"]) == heads_dir / "heads.pt"
    assert Path(result["labels_path"]) == heads_dir / "labels.json"
    assert Path(result["summary_path"]) == heads_dir / "training_summary.json"
    assert summary["debug_model"] is True
    assert summary["smoke"] is True
    assert summary["epochs"] == 1
    assert summary["samples_per_epoch"] == 4
    assert summary["total_train_steps"] == 2
    assert summary["validation_history"][0]["status"] == "skipped"


def test_cli_parses_direct_training_overrides(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_train_model(config_path, *, overrides=None, debug_model=False):
        seen["config_path"] = config_path
        seen["overrides"] = dict(overrides or {})
        seen["debug_model"] = debug_model
        return {
            "summary_path": "models/heads/latest/training_summary.json",
            "heads_path": "models/heads/latest/heads.pt",
        }

    monkeypatch.setattr(train_module, "train_model", fake_train_model)

    exit_code = main(
        [
            "configs/config.yaml",
            "--smoke",
            "--debug-model",
            "--steps",
            "2",
            "--epochs",
            "3",
            "--batch-size",
            "4",
            "--samples-per-epoch",
            "8",
            "--learning-rate",
            "0.001",
            "--gradient-accumulation-steps",
            "2",
            "--no-mixed-precision",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert seen == {
        "config_path": "configs/config.yaml",
        "debug_model": True,
        "overrides": {
            "steps": 2,
            "epochs": 3,
            "batch_size": 4,
            "samples_per_epoch": 8,
            "learning_rate": 0.001,
            "gradient_accumulation_steps": 2,
            "smoke": True,
            "mixed_precision": False,
        },
    }
    assert output == {
        "summary_path": "models/heads/latest/training_summary.json",
        "heads_path": "models/heads/latest/heads.pt",
    }
