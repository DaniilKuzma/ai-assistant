from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE_KEYS = (
    "generator_audit",
    "frozen_eval",
    "training_smoke",
    "evaluation",
    "runtime_corrector",
    "streamlit_builder",
)


def test_end_to_end_smoke_cli_runs_all_stages_without_train_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "e2e"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_end_to_end_smoke.py",
            "configs/config.yaml",
            "--output",
            str(output_dir),
            "--count",
            "10",
            "--steps",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    summary_path = output_dir / "e2e_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {key: summary[key]["status"] for key in STAGE_KEYS} == {
        key: "ok" for key in STAGE_KEYS
    }
    assert not (ROOT / "data" / "processed" / "train.csv").exists()
