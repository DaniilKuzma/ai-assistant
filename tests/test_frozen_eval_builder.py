from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.grammar_gen.audit import audit_batch
from src.schema import GeneratedExample, WordToken
from src.schema.serialization import read_jsonl_examples, write_jsonl_examples
from scripts import build_frozen_eval as frozen_builder
from scripts.build_frozen_eval import SPLIT_SEED_OFFSETS, build_frozen_eval


ROOT = Path(__file__).resolve().parents[1]


def test_build_frozen_eval_writes_valid_val_jsonl_and_manifest(tmp_path: Path) -> None:
    config_path = _small_config(tmp_path)
    output = tmp_path / "val.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_frozen_eval.py",
            str(config_path),
            "--split",
            "val",
            "--count",
            "50",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()

    manifest_path = output.with_name("val.manifest.json")
    assert manifest_path.exists()

    examples = read_jsonl_examples(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(examples) == 50
    assert manifest["count"] == 50
    assert manifest["audit_failures_count"] == 0
    assert audit_batch(examples)["failed_examples_count"] == 0


def test_frozen_eval_split_seeds_do_not_overlap_training_range(tmp_path: Path) -> None:
    config_path = _small_config(tmp_path)
    for split in ("val", "test", "regression"):
        manifest, examples = build_frozen_eval(
            config_path=config_path,
            split=split,
            count=20,
            output_path=tmp_path / f"{split}.jsonl",
        )

        train_start = int(manifest["train_seed_range_start"])
        train_end = int(manifest["train_seed_range_end"])
        seeds = [int(example.metadata["generation_seed"]) for example in examples]

        assert manifest["split_seed_overlap_with_train"] is False
        assert all(seed < train_start or seed > train_end for seed in seeds)


def test_build_frozen_eval_val_seed_offset_is_not_zero() -> None:
    assert SPLIT_SEED_OFFSETS["val"] != 0


def test_build_frozen_eval_filters_train_existing_split_and_current_text_duplicates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _small_config(tmp_path, samples_per_epoch=2, epochs=1, seed=0)
    write_jsonl_examples(tmp_path / "val.jsonl", [_example("Соседняя фраза.")])

    class FakeGenerator:
        def __init__(self, seed: int | None) -> None:
            self.seed = seed

        def sample_by_index(self, index: int) -> GeneratedExample:
            if self.seed == 0:
                return [
                    _example("Тренировочная фраза."),
                    _example("Тренировочная цель."),
                ][index]
            return [
                _example("Тренировочная фраза."),
                _example("Соседняя фраза."),
                _example("Новая первая фраза."),
                _example("Новая первая фраза."),
                _example("Новая вторая фраза."),
            ][index]

    def fake_factory(config: dict, seed: int | None = None) -> FakeGenerator:
        return FakeGenerator(seed)

    monkeypatch.setattr(frozen_builder, "online_generator_from_config", fake_factory)

    manifest, examples = build_frozen_eval(
        config_path=config_path,
        split="test",
        count=2,
        output_path=tmp_path / "test.jsonl",
    )

    assert [example.source_text for example in examples] == [
        "Новая первая фраза.",
        "Новая вторая фраза.",
    ]
    assert [example.target_text for example in examples] == [
        "Новая первая фраза.",
        "Новая вторая фраза.",
    ]
    assert manifest["candidate_scan_count"] == 5
    assert manifest["dedupe_skipped_count"] == 3
    assert manifest["dedupe_skipped_reasons"] == {
        "source_text": 3,
        "target_text": 3,
    }


def _small_config(
    tmp_path: Path,
    *,
    samples_per_epoch: int = 4,
    epochs: int = 1,
    seed: int = 13,
) -> Path:
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    config["generation"]["seed"] = seed
    config["generation"]["samples_per_epoch"] = samples_per_epoch
    config["training"]["epochs"] = epochs
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _example(text: str) -> GeneratedExample:
    token_text = text[:-1]
    return GeneratedExample(
        source_text=text,
        target_text=text,
        source_tokens=[WordToken(text=token_text, start=0, end=len(token_text))],
        token_edit_labels=["KEEP"],
        gap_labels=["NONE"],
        rule_ids=["clean_identity"],
        primary_rule_id="clean_identity",
        mode="clean_identity",
        explanation_ids=["clean_identity"],
        metadata={
            "uses_safety_clauses": False,
            "expected_edit_count": 0,
            "expected_token_edit_count": 0,
            "expected_gap_edit_count": 0,
        },
    )
