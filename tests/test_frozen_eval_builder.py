from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.grammar_gen.audit import audit_batch
from src.schema.serialization import read_jsonl_examples
from scripts.build_frozen_eval import SPLIT_SEED_OFFSETS, build_frozen_eval


ROOT = Path(__file__).resolve().parents[1]


def test_build_frozen_eval_writes_valid_val_jsonl_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "val.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_frozen_eval.py",
            "configs/config.yaml",
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
    for split in ("val", "test", "regression"):
        manifest, examples = build_frozen_eval(
            config_path=ROOT / "configs" / "config.yaml",
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
