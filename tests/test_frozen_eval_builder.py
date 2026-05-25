from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.grammar_gen.audit import audit_batch
from src.schema.serialization import read_jsonl_examples


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
