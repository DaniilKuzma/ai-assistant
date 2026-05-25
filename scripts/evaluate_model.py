from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import evaluate_corrector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate runtime Corrector on frozen GeneratedExample JSONL.")
    parser.add_argument("config", help="Path to YAML config.")
    parser.add_argument("--dataset", required=True, help="Frozen eval JSONL, for example data/generated_eval/val.jsonl.")
    parser.add_argument("--output", required=True, help="Directory for direct evaluation reports.")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Frozen eval dataset not found: {dataset_path}", file=sys.stderr)
        print(
            "Run scripts/build_frozen_eval.py first, for example: "
            "python scripts/build_frozen_eval.py configs/config.yaml --split val --count 5000 "
            "--output data/generated_eval/val.jsonl",
            file=sys.stderr,
        )
        return 2

    summary = evaluate_corrector(args.config, dataset_path, args.output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
