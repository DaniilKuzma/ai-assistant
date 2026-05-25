from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.config.load_config import load_config
from src.training.trainer import train_model


ROOT = Path(__file__).resolve().parents[1]


def test_debug_model_smoke_training_saves_heads_and_summary(monkeypatch, tmp_path: Path) -> None:
    def fail_load_tokenizer(*args, **kwargs):
        raise AssertionError("debug-model training must not load ruRoberta tokenizer")

    monkeypatch.setattr("src.model.encoder.load_tokenizer", fail_load_tokenizer)
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
        "save_each_epoch": True,
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

    result = train_model(
        config_path,
        overrides={"smoke": True, "steps": 2},
        debug_model=True,
    )

    heads_dir = tmp_path / "models" / "heads"
    summary = json.loads((heads_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert (heads_dir / "heads.pt").exists()
    assert (heads_dir / "labels.json").exists()
    assert summary["total_train_steps"] == 2
    assert summary["epochs"] == 1
    assert summary["samples_per_epoch"] == 4
    assert summary["batch_size"] == 2
    assert summary["validation_history"][0]["status"] == "skipped"
    assert result["summary_path"] == str(heads_dir / "training_summary.json")
