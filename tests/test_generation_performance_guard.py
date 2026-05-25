from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.config.load_config import load_config
from src.grammar_gen import Lexicon, MorphologyEngine
from src.grammar_gen.generator import OnlineExampleGenerator
from src.grammar_gen.rules.registry import default_rule_registry


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_generation_500_examples_under_threshold() -> None:
    from src.grammar_gen.performance import benchmark_generation

    config = load_config(ROOT / "configs" / "config.yaml")

    result = benchmark_generation(config, count=500, seed=13)

    assert result["total_seconds"] < 20
    assert result["examples_per_second"] > 0
    assert result["avg_ms_per_example"] > 0
    assert result["audit_failure_count"] == 0
    assert sum(result["rule_distribution"].values()) == 500
    assert sum(result["mode_distribution"].values()) == 500


def test_benchmark_cli_writes_json_report_to_requested_output(tmp_path: Path) -> None:
    output_path = tmp_path / "generation_benchmark.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_generation.py",
            "configs/config.yaml",
            "--count",
            "25",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "examples/sec" in result.stdout
    assert output_path.exists()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["count"] == 25
    assert report["seed"] == 13
    assert report["audit_failure_count"] == 0
    assert sum(report["rule_distribution"].values()) == 25
    assert sum(report["mode_distribution"].values()) == 25


def test_candidate_opportunity_config_is_rejected() -> None:
    config = _config_with({"data": {"candidate_opportunity": "data/processed/train.csv"}})

    with pytest.raises(ValueError, match="candidate_opportunity"):
        _make_generator(config)


def test_removed_generation_path_is_rejected() -> None:
    config = _config_with({"paths": {"generated_eval_dir": "data/processed/train.csv"}})

    with pytest.raises(RuntimeError, match="data/processed/train.csv"):
        _make_generator(config)


def test_grammar_gen_does_not_import_pandas() -> None:
    offenders: list[str] = []
    for path in (ROOT / "src" / "grammar_gen").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "pandas" or alias.name.startswith("pandas.") for alias in node.names):
                    offenders.append(str(path.relative_to(ROOT)))
            elif isinstance(node, ast.ImportFrom) and (node.module == "pandas" or str(node.module).startswith("pandas.")):
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_benchmark_generation_does_not_create_train_csv() -> None:
    from src.grammar_gen.performance import benchmark_generation

    train_csv = ROOT / "data" / "processed" / "train.csv"
    assert not train_csv.exists()

    benchmark_generation(load_config(ROOT / "configs" / "config.yaml"), count=50, seed=13)

    assert not train_csv.exists()


def _config_with(overrides: dict) -> dict:
    config = load_config(ROOT / "configs" / "config.yaml")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def _make_generator(config: dict) -> OnlineExampleGenerator:
    return OnlineExampleGenerator(
        default_rule_registry(),
        Lexicon.default(),
        MorphologyEngine(use_pymorphy=False),
        config,
        seed=13,
    )
