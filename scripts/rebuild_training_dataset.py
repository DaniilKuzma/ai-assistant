from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_dataset import _dataset_summary_payload
from src.config.candidate_dataset_config import candidate_dataset_paths, validate_candidate_dataset_config
from src.config.load_config import load_config
from src.data.full_dataset_builder import build_dataset_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the configured training dataset without running model training.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", default=True)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config_errors = validate_candidate_dataset_config(config)
    if config_errors:
        print(
            "[dataset-build] "
            + json.dumps(
                {"stage": "config_invalid", "errors": config_errors},
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    result = build_dataset_from_config(config, force=args.force)
    summary = _dataset_summary_payload(result, manifest_path=Path(str(candidate_dataset_paths(config)["manifest_path"])))
    print(
        "[dataset-build] "
        + json.dumps(
            {"stage": "entrypoint_result", "result": result},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        "[dataset-build] " + json.dumps({"stage": "final_contract_summary", **summary}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    if args.fail_on_blocked and str(summary.get("verdict", "")).endswith("BLOCKED"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
