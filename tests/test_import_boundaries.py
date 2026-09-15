from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_and_orthography_modules_import_in_fresh_subprocesses() -> None:
    modules = (
        "src.runtime.corrector",
        "src.runtime.edit_realizer",
        "src.runtime.orthographic_lexicon",
        "src.orthography_gen.lexeme_cards",
        "src.orthography_gen.compiler",
        "src.evaluation.evaluate",
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    failures: list[str] = []
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            failures.append(f"{module}: {result.stderr.strip()}")

    assert failures == []
