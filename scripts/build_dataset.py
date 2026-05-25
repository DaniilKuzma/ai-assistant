from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.candidate_dataset_config import candidate_dataset_paths, validate_candidate_dataset_config
from src.config.load_config import load_config
from src.data.training_dataset import build_training_dataset_from_config
from src.evaluation.activation_plan import write_verified_activation_plan
from scripts.preflight_candidate_dataset import PreflightOptions, print_preflight_summary, run_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical training_dataset for Activation expansion.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true")
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
    preflight_result = run_preflight(_preflight_options_for_build(config, args))
    print_preflight_summary(preflight_result)
    if not preflight_result["ok"]:
        return 1
    _write_activation_verified_reports()
    result = build_training_dataset_from_config(config, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    paths = candidate_dataset_paths(config)
    manifest_path = Path(str(paths["manifest_path"]))
    summary = _dataset_summary_payload(result, manifest_path=manifest_path)
    print("[dataset-build] " + json.dumps({"stage": "final_contract_summary", **summary}, ensure_ascii=False, sort_keys=True), flush=True)
    _print_final_audit_summary(manifest_path)
    if args.fail_on_blocked and str(summary.get("verdict", "")).endswith("BLOCKED"):
        return 2
    return 0


def _preflight_options_for_build(config: dict[str, Any], args: argparse.Namespace) -> PreflightOptions:
    paths = candidate_dataset_paths(config)
    return PreflightOptions(
        config_path=args.config,
        processed_dir=Path(str(paths["processed_dir"])),
        reports_dir=Path(str(paths["reports_dir"])),
        block_stale_artifacts=not bool(args.force),
    )


def _dataset_summary_payload(result: dict[str, Any], manifest_path: Path | None) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    if manifest_path is not None and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    return {
        "dataset_contract": str(manifest.get("dataset_contract") or result.get("dataset_contract") or ""),
        "dataset_hash": str(manifest.get("dataset_hash") or result.get("dataset_hash") or ""),
        "verdict": str(manifest.get("verdict") or result.get("verdict") or ""),
        "audit_errors": list(manifest.get("audit_errors", result.get("audit_errors", [])) or []),
        "layer_counts": dict(manifest.get("layer_counts", result.get("layer_counts", {})) or {}),
    }


def _write_activation_verified_reports() -> None:
    activation_path = Path("reports/matrix_eval/next_dataset_activation_plan.csv")
    summary_path = Path("reports/matrix_eval/working_eval/matrix_rule_eval_summary.csv")
    if not summary_path.exists():
        summary_path = Path("reports/matrix_eval/matrix_rule_eval_summary.csv")
    quota_path = Path("reports/matrix_eval/under_quota_rule_audit.csv")
    if not (activation_path.exists() and summary_path.exists() and quota_path.exists()):
        return
    write_verified_activation_plan(
        matrix_activation_path=activation_path,
        matrix_summary_path=summary_path,
        under_quota_path=quota_path,
        output_csv_path="reports/activation_expansion/next_dataset_activation_plan_verified.csv",
        output_md_path="reports/activation_expansion/next_dataset_activation_plan_verified.md",
    )


def _print_final_audit_summary(manifest_path: Path) -> None:
    if not manifest_path.exists():
        print("training_dataset manifest missing; final audit summary unavailable")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    composition = manifest.get("composition", {})
    recall = manifest.get("candidate_recall_summary", {})
    gap = manifest.get("gap_label_coverage_summary", {})
    rule_counts = sorted(
        ((rule_id, count) for rule_id, count in manifest.get("rule_id_counts", {}).items()),
        key=lambda item: int(item[1]),
        reverse=True,
    )[:10]
    quality = {
        "template_leakage": manifest.get("template_leakage_summary", {}),
        "synthetic_normalized_pair_duplicate_rate": manifest.get("synthetic_normalized_pair_duplicate_rate", 0),
        "top_normalized_pair_count": manifest.get("top_normalized_pair_count", 0),
        "meta_language_counts": manifest.get("meta_language_counts", {}),
        "suspicious_template_counts": manifest.get("suspicious_template_counts", {}),
    }
    summary = {
        "target_total": manifest.get("requested_total"),
        "actual_total": manifest.get("total"),
        "split_sizes": manifest.get("split_sizes"),
        "composition": composition,
        "active_target_rule_count": manifest.get("active_target_rule_count"),
        "stable_core_rule_count": manifest.get("stable_core_rule_count"),
        "activation_rule_count": manifest.get("activation_rule_count"),
        "bounded_activation_rule_count": manifest.get("bounded_activation_rule_count"),
        "excluded_rule_count": manifest.get("excluded_rule_count"),
        "synthetic_count": composition.get("synthetic_augmented_from_open_clean", 0),
        "real_pair_count": composition.get("real_error_pair", 0),
        "clean_identity_count": composition.get("clean_identity_from_open_clean", 0),
        "hard_negative_count": composition.get("hard_negative_from_open_clean", 0),
        "candidate_recall_min": recall.get("active_min_excluding_unknown"),
        "candidate_recall_mean": recall.get("active_mean_excluding_unknown"),
        "gap_coverage_min": gap.get("active_min_excluding_unknown"),
        "gap_coverage_mean": gap.get("active_mean_excluding_unknown"),
        "underfilled_active_rules": manifest.get("low_count_active_rule_ids", []),
        "top_rule_counts": rule_counts,
        "quality_metrics": quality,
        "config_paths": {
            "dataset": "configs/config.yaml",
            "train_e1": "configs/config.yaml",
            "train_e2": "configs/config.yaml",
        },
        "final_verdict": manifest.get("verdict"),
    }
    print("TRAINING DATASET FINAL AUDIT SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
