from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.wave1_expansion import normalize_phase3_decision, write_phase3_final_report


REPORTS_DIR = Path("reports/matrix_eval_phase3")
DATA_DIR = Path("data/processed/matrix_eval_phase3")
PHASE2_COMPATIBLE_OUTPUTS = {
    "matrix_eval_candidate_recall_by_rule.csv": "candidate_recall_by_rule.csv",
    "matrix_eval_gap_coverage_by_rule.csv": "gap_label_coverage_by_rule.csv",
    "rule_expansion_backlog_v2.csv": "rule_expansion_backlog_v3.csv",
    "wave1_activation_plan.csv": "next_dataset_activation_plan_v3.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Matrix Eval Phase 3 without mutating WORKING_V1 artifacts.")
    parser.add_argument("--stage", choices=["inventory", "dataset", "eval-working-v1", "post-eval", "audit", "all"], default="all")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--skip-existing-run", action="store_true")
    args, extra = parser.parse_known_args()

    reports_dir = Path(args.reports_dir)
    data_dir = Path(args.data_dir)
    if not args.skip_existing_run:
        _run_existing_matrix_pipeline(args, extra, reports_dir=reports_dir, data_dir=data_dir)
    _normalize_outputs(reports_dir=reports_dir)
    verdict = write_phase3_final_report(reports_dir=reports_dir, data_dir=data_dir)
    print(json.dumps({"verdict": verdict, "reports_dir": str(reports_dir), "data_dir": str(data_dir)}, ensure_ascii=False, indent=2))


def _run_existing_matrix_pipeline(args: argparse.Namespace, extra: list[str], *, reports_dir: Path, data_dir: Path) -> None:
    command = [
        sys.executable,
        "scripts/run_matrix_eval.py",
        "--phase2",
        "--stage",
        args.stage,
        "--config",
        args.config,
        "--reports-dir",
        str(reports_dir),
        "--data-dir",
        str(data_dir),
        "--tmp-config",
        "/tmp/config.matrix_eval_phase3_working_v1.yaml",
        "--feature-cache-dir",
        str(data_dir / "features_cache"),
        *extra,
    ]
    env = os.environ.copy()
    env["RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING"] = "1"
    env["PYTHONPATH"] = str(ROOT_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(command, check=True, env=env)


def _normalize_outputs(*, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in PHASE2_COMPATIBLE_OUTPUTS.items():
        source = reports_dir / source_name
        target = reports_dir / target_name
        if source.exists():
            shutil.copyfile(source, target)

    summary_path = reports_dir / "working_v1_eval" / "matrix_rule_eval_summary.csv"
    target_summary = reports_dir / "matrix_rule_eval_summary.csv"
    if summary_path.exists():
        shutil.copyfile(summary_path, target_summary)
    if target_summary.exists():
        _rewrite_phase3_summary(target_summary)

    activation_md = reports_dir / "next_dataset_activation_plan_v3.md"
    csv_path = reports_dir / "next_dataset_activation_plan_v3.csv"
    if csv_path.exists() and not activation_md.exists():
        activation_md.write_text("# Next Dataset Activation Plan v3\n\nSee CSV artifact for normalized Phase 3 decisions.\n", encoding="utf-8")


def _rewrite_phase3_summary(path: Path) -> None:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
        fieldnames = list(rows[0].keys()) if rows else []
    if "phase3_decision" not in fieldnames:
        fieldnames.append("phase3_decision")
    for row in rows:
        row["phase3_decision"] = normalize_phase3_decision(row.get("decision", ""), row.get("blockers") or row.get("reason") or "")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
