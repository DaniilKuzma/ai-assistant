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
    _print_final_audit_summary(Path(config["data"]["manifest_path"]))


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


def _print_final_audit_summary(manifest_path: Path) -> None:
    if not manifest_path.exists():
        print("short_dataset_v3 manifest missing; final audit summary unavailable")
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
        "stable_v2_core_rule_count": manifest.get("stable_v2_core_rule_count"),
        "wave1_rule_count": manifest.get("wave1_rule_count"),
        "bounded_phase3_rule_count": manifest.get("bounded_phase3_rule_count"),
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
            "dataset": "configs/config.short_dataset_v3.yaml",
            "train_e1": "configs/config.train_short_v3_e1.yaml",
            "train_e2": "configs/config.train_short_v3_e2.yaml",
        },
        "final_verdict": manifest.get("verdict"),
    }
    print("SHORT DATASET V3 FINAL AUDIT SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
