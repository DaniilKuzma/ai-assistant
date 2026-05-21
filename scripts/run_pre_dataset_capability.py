from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pre_dataset_capability import write_pre_dataset_capability_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-dataset capability audit without building data or training.")
    parser.add_argument("--rules", default="configs/rules.yaml")
    parser.add_argument("--output-dir", default="reports/pre_dataset_capability")
    parser.add_argument("--matrix-reports-dir", default="reports/matrix_eval")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--clean-pool", default="data/processed/clean_sentence_pool.csv.gz")
    parser.add_argument("--no-update-rules-yaml", action="store_true")
    args = parser.parse_args()

    outputs = write_pre_dataset_capability_outputs(
        rules_config_path=Path(args.rules),
        output_dir=Path(args.output_dir),
        matrix_reports_dir=Path(args.matrix_reports_dir),
        config_path=Path(args.config),
        clean_pool_path=Path(args.clean_pool),
        update_rules_yaml=not args.no_update_rules_yaml,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
