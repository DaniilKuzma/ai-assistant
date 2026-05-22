from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.data.full_dataset_builder import build_dataset_from_config


def main() -> None:
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    result = build_dataset_from_config(config, force=True)
    print(
        "[dataset-build] "
        + json.dumps(
            {"stage": "entrypoint_result", "result": result},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
