from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.load_config import load_config
from src.data.short_dataset_v3 import build_short_dataset_v3_from_config
from src.evaluation.wave1_expansion import write_verified_wave1_activation_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical short_dataset_v3 for Wave 1 expansion.")
    parser.add_argument("--config", default="configs/config.short_dataset_v3.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    _write_wave1_verified_reports()
    result = build_short_dataset_v3_from_config(config, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _write_wave1_verified_reports() -> None:
    wave_path = Path("reports/matrix_eval_phase2/wave1_activation_plan.csv")
    summary_path = Path("reports/matrix_eval_phase2/working_v1_eval/matrix_rule_eval_summary.csv")
    if not summary_path.exists():
        summary_path = Path("reports/matrix_eval_phase2/matrix_rule_eval_summary.csv")
    quota_path = Path("reports/matrix_eval_phase2/under_quota_rule_audit.csv")
    if not (wave_path.exists() and summary_path.exists() and quota_path.exists()):
        return
    write_verified_wave1_activation_plan(
        phase2_wave1_path=wave_path,
        phase2_summary_path=summary_path,
        under_quota_path=quota_path,
        output_csv_path="reports/wave1_expansion/wave1_activation_plan_verified.csv",
        output_md_path="reports/wave1_expansion/wave1_activation_plan_verified.md",
    )


if __name__ == "__main__":
    main()
