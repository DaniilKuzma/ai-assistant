from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_training_artifacts(output_dir: str | Path, config: dict[str, Any], label_mappings: dict[str, Any], thresholds: dict[str, float]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "label_mappings.json").write_text(json.dumps(label_mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")


def load_training_artifacts(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    return {
        "config": json.loads((output / "config.json").read_text(encoding="utf-8")),
        "label_mappings": json.loads((output / "label_mappings.json").read_text(encoding="utf-8")),
        "thresholds": json.loads((output / "thresholds.json").read_text(encoding="utf-8")),
    }
