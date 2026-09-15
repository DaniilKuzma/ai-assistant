from __future__ import annotations

from pathlib import Path

from src.config.load_config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _join(*parts: str) -> str:
    return "".join(parts)


def test_training_config_uses_100k_online_ast_morphemic_generation() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    generation = config["generation"]
    training = config["training"]

    assert generation["samples_per_epoch"] == 100_000
    assert training["epochs"] == 4
    assert training["batch_size"] == 32
    assert generation["samples_per_epoch"] * training["epochs"] == 400_000
    assert generation["mode"] == "online_ast"
    assert "morpheme" in generation["enabled_rule_groups"]
    assert "orthography_morphemic" in generation["enabled_rule_groups"]
    assert "morpheme" in generation["mix"]
    assert "orthography_morphemic" not in generation["mix"]
    assert generation["orthography_morphemic"]["enabled"] is True


def test_config_does_not_reference_legacy_candidate_dataset_pipeline() -> None:
    raw_config = (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")

    for legacy_key in (
        _join("clean", "_sentence", "_pool"),
        _join("correction", "_dataset"),
        _join("candidate", "_opportunity"),
        _join("rule", "_lab"),
    ):
        assert legacy_key not in raw_config
