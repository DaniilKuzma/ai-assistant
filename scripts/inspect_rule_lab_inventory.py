from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.load_config import load_config
from src.data.rule_data_compiler import build_rule_lab_inventory_rows, write_rule_lab_inventory_reports
from src.data.rule_lab_generation import (
    all_rule_lab_recipe_rule_ids,
    disabled_rule_lab_recipe_rule_ids,
    enabled_rule_lab_recipe_rule_ids,
)
from src.rules.capabilities import active_rule_ids_for_training, activation_policy_from_config, load_rule_capabilities


def build_inventory_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    training_rows = [row for row in rows if bool(row.get("in_training_candidates"))]
    disabled_reasons = Counter(
        str(row.get("disabled_reason") or "disabled")
        for row in training_rows
        if bool(row.get("has_any_recipe")) and not bool(row.get("recipe_enabled"))
    )
    no_support = [
        str(row["rule_id"])
        for row in training_rows
        if not str(row.get("generation_support_sources") or "")
    ]
    missing = [
        str(row["rule_id"])
        for row in training_rows
        if not bool(row.get("has_any_recipe"))
    ]
    return {
        "training_candidate_rule_count": len(training_rows),
        "recipes_total": len(all_rule_lab_recipe_rule_ids(config)),
        "recipes_enabled": len(enabled_rule_lab_recipe_rule_ids(config)),
        "recipes_disabled": len(disabled_rule_lab_recipe_rule_ids(config)),
        "candidate_rules_with_enabled_recipe": sum(1 for row in training_rows if bool(row.get("recipe_enabled"))),
        "candidate_rules_with_disabled_recipe": sum(
            1 for row in training_rows if bool(row.get("has_any_recipe")) and not bool(row.get("recipe_enabled"))
        ),
        "candidate_rules_missing_recipe": len(missing),
        "candidate_rules_with_no_generation_support": len(no_support),
        "disabled_by_reason": dict(sorted(disabled_reasons.items())),
        "top_missing_rules": missing[:20],
        "top_no_support_rules": no_support[:20],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Rule Lab inventory without building the training dataset.")
    parser.add_argument("--config", default="configs/config.yaml", help="Candidate dataset config path.")
    parser.add_argument("--reports-dir", default="reports/dataset_build", help="Directory for Rule Lab inventory reports.")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    capabilities = load_rule_capabilities("configs/rules.yaml", config=config)
    policy = activation_policy_from_config(config)
    training_candidate_rule_ids = active_rule_ids_for_training(capabilities, policy=policy)
    rows = build_rule_lab_inventory_rows(training_candidate_rule_ids, config, capabilities=capabilities)
    write_rule_lab_inventory_reports(rows, args.reports_dir)
    summary = build_inventory_summary(rows, config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
