from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.matrix_eval_dataset import write_matrix_eval_dataset
from src.evaluation.matrix_eval_reports import (
    assert_matrix_eval_audit,
    write_expansion_backlog,
    write_hard_negative_matrix_coverage,
    write_matrix_rule_eval_summary,
    write_next_dataset_activation_plan,
)
from src.evaluation.matrix_eval import (
    assert_matrix_eval_audit,
    write_matrix_final_report,
    write_rule_expansion_backlog,
    write_under_quota_rule_audit,
    write_validator_fix_report,
    write_next_dataset_activation_plan,
)
from src.evaluation.matrix_inventory import write_rule_matrix_inventory_outputs
from src.training.train import train


REPORTS_DIR = Path("reports/matrix_eval")
DATA_DIR = Path("data/processed/matrix_eval")
MATRIX_REPORTS_DIR = Path("reports/matrix_eval")
MATRIX_DATA_DIR = Path("data/processed/matrix_eval")
WORKING_EVAL_DIR = REPORTS_DIR / "working_eval"
TMP_CONFIG_PATH = Path("/tmp/config.matrix_eval_working.yaml")
MATRIX_TMP_CONFIG_PATH = Path("/tmp/config.matrix_eval_working.yaml")
BASELINE_MANIFEST_PATH = Path("data/processed/matrix_eval/matrix_eval_manifest.json")
PRODUCTION_GUARD_PATHS = ("configs/config.yaml", "models/current/adapters", "models/current/heads")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full matrix coverage evaluation/audit.")
    parser.add_argument("--stage", choices=["inventory", "dataset", "eval-working-v1", "post-eval", "audit", "all"], default="all")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--rules", default="configs/rules.yaml")
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--matrix", action="store_true", help="Write Matrix Eval Matrix artifacts only.")
    parser.add_argument("--tmp-config", default="")
    parser.add_argument("--feature-cache-dir", default="")
    parser.add_argument("--baseline-manifest", default=str(BASELINE_MANIFEST_PATH))
    parser.add_argument("--current-eval-dir", default="reports/eval_test_calibrated_canonical_short_core")
    parser.add_argument("--dataset-manifest", default="reports/dataset_manifest.json")
    parser.add_argument("--current-dataset", default="data/processed/correction_dataset.csv.gz")
    parser.add_argument("--clean-pool", default="data/processed/clean_sentence_pool.csv.gz")
    parser.add_argument("--min-examples", type=int, default=50)
    parser.add_argument("--preferred-examples", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--hard-negatives", type=int, default=300)
    args = parser.parse_args()
    if args.matrix and args.preferred_examples == 100:
        args.preferred_examples = args.min_examples

    reports_dir = Path(args.reports_dir)
    data_dir = Path(args.data_dir)
    if args.matrix:
        if reports_dir == REPORTS_DIR:
            reports_dir = MATRIX_REPORTS_DIR
        if data_dir == DATA_DIR:
            data_dir = MATRIX_DATA_DIR
    working_eval_dir = reports_dir / "working_eval"
    tmp_config_path = Path(args.tmp_config) if args.tmp_config else (
        MATRIX_TMP_CONFIG_PATH if args.matrix else TMP_CONFIG_PATH
    )
    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else data_dir / "features_cache"
    stages = ["inventory", "dataset", "eval-working-v1", "post-eval", "audit"] if args.stage == "all" else [args.stage]
    result: dict[str, Any] = {}
    protected_status_before = _protected_git_status()
    for stage in stages:
        if stage == "inventory":
            result["inventory"] = write_rule_matrix_inventory_outputs(
                output_dir=reports_dir,
                rules_config_path=args.rules,
                reports_dir=args.current_eval_dir,
                dataset_manifest_path=args.dataset_manifest,
                current_dataset_path=args.current_dataset,
            )
        elif stage == "dataset":
            result["dataset"] = write_matrix_eval_dataset(
                output_dir=data_dir,
                reports_dir=reports_dir,
                rules_config_path=args.rules,
                config_path=args.config,
                min_examples_per_rule=args.min_examples,
                preferred_examples_per_rule=args.preferred_examples,
                max_examples_per_rule=args.max_examples,
                hard_negative_count=args.hard_negatives,
                current_dataset_path=args.current_dataset,
                clean_pool_path=args.clean_pool,
                extended_backfill_rule_ids=_baseline_underfilled_rule_ids(Path(args.baseline_manifest)) if args.matrix else None,
            )
            if args.matrix:
                _copy_dataset_manifest_to_reports(data_dir=data_dir, reports_dir=reports_dir)
        elif stage == "eval-working-v1":
            result["eval"] = _run_working_eval(
                args.config,
                data_dir=data_dir,
                working_eval_dir=working_eval_dir,
                tmp_config_path=tmp_config_path,
                feature_cache_dir=feature_cache_dir,
            )
        elif stage == "post-eval":
            result["post_eval"] = _write_post_eval_reports(
                reports_dir=reports_dir,
                data_dir=data_dir,
                working_eval_dir=working_eval_dir,
                matrix=args.matrix,
                baseline_manifest_path=Path(args.baseline_manifest),
            )
        elif stage == "audit":
            if args.matrix:
                result["audit"] = assert_matrix_eval_audit(root_dir=reports_dir, data_dir=data_dir, eval_dir=working_eval_dir)
                result["summary"] = write_matrix_final_report(
                    root_dir=reports_dir,
                    data_dir=data_dir,
                    eval_dir=working_eval_dir,
                    audit=result["audit"],
                )
            else:
                result["audit"] = assert_matrix_eval_audit(root_dir=reports_dir, data_dir=data_dir, eval_dir=working_eval_dir)
                result["summary"] = _audit_summary(reports_dir=reports_dir, data_dir=data_dir, working_eval_dir=working_eval_dir, audit=result["audit"])
    if args.matrix:
        protected_status_after = _protected_git_status()
        result["production_state_guard"] = {
            "before": protected_status_before,
            "after": protected_status_after,
            "unchanged": protected_status_before == protected_status_after,
        }
        if protected_status_before != protected_status_after:
            raise RuntimeError("production config/checkpoint git status changed during matrix run")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _run_working_eval(
    config_path: str | Path,
    *,
    data_dir: Path,
    working_eval_dir: Path,
    tmp_config_path: Path,
    feature_cache_dir: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config.setdefault("training", {})["run_model_training"] = False
    config["training"]["evaluation_split"] = "matrix_eval"
    config["training"]["max_matrix_eval_examples"] = 10_000_000
    config.setdefault("thresholds", {})["mode"] = "calibrated_guarded"
    config.setdefault("paths", {})["reports_dir"] = str(working_eval_dir)
    config.setdefault("data", {})["processed_train_path"] = str(data_dir / "matrix_eval.csv.gz")
    config["data"]["manifest_path"] = str(data_dir / "matrix_eval_manifest.json")
    config.setdefault("evaluation", {})["full_eval_examples"] = 10_000_000
    feature_cache = config.setdefault("training", {}).setdefault("feature_cache", {})
    feature_cache["cache_dir"] = str(feature_cache_dir)
    tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    previous = os.environ.get("RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING")
    os.environ["RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING"] = "1"
    try:
        result = train(tmp_config_path)
        if int(float(result.get("model_training_ran", 0) or 0)) != 0:
            raise RuntimeError("eval-only safety guard failed: model_training_ran is nonzero")
        return result
    finally:
        if previous is None:
            os.environ.pop("RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING", None)
        else:
            os.environ["RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING"] = previous


def _write_post_eval_reports(
    *,
    reports_dir: Path,
    data_dir: Path,
    working_eval_dir: Path,
    matrix: bool = False,
    baseline_manifest_path: Path = BASELINE_MANIFEST_PATH,
) -> dict[str, str]:
    summary = write_matrix_rule_eval_summary(
        dataset_path=data_dir / "matrix_eval.csv.gz",
        inventory_path=reports_dir / "rule_matrix_inventory.csv",
        eval_dir=working_eval_dir,
        output_path=working_eval_dir / "matrix_rule_eval_summary.csv",
    )
    write_hard_negative_matrix_coverage(
        dataset_path=data_dir / "matrix_eval.csv.gz",
        output_path=reports_dir / "hard_negative_matrix_coverage.csv",
        accepted_edits_path=working_eval_dir / "accepted_edits.csv",
        rejected_edits_path=working_eval_dir / "rejected_edits.csv",
        clean_overcorrection_path=working_eval_dir / "clean_overcorrection_examples.csv",
    )
    if matrix:
        write_under_quota_rule_audit(
            baseline_manifest_path=baseline_manifest_path,
            inventory_path=reports_dir / "rule_matrix_inventory.csv",
            matrix_summary_path=working_eval_dir / "matrix_rule_eval_summary.csv",
            matrix_dataset_path=data_dir / "matrix_eval.csv.gz",
            output_path=reports_dir / "under_quota_rule_audit.csv",
        )
        write_validator_fix_report(
            summary_path=working_eval_dir / "matrix_rule_eval_summary.csv",
            accepted_edits_path=working_eval_dir / "accepted_edits.csv",
            rejected_edits_path=working_eval_dir / "rejected_edits.csv",
            output_path=reports_dir / "validator_fix_report.md",
        )
        write_rule_expansion_backlog(
            inventory_path=reports_dir / "rule_matrix_inventory.csv",
            summary_path=working_eval_dir / "matrix_rule_eval_summary.csv",
            output_csv_path=reports_dir / "rule_expansion_backlog_core.csv",
            output_md_path=reports_dir / "rule_expansion_backlog_core.md",
        )
        write_next_dataset_activation_plan(
            summary_path=working_eval_dir / "matrix_rule_eval_summary.csv",
            under_quota_audit_path=reports_dir / "under_quota_rule_audit.csv",
            backlog_core_path=reports_dir / "rule_expansion_backlog_core.csv",
            output_csv_path=reports_dir / "activation_activation_plan.csv",
            output_md_path=reports_dir / "activation_activation_plan.md",
        )
        _append_matrix_rule_expansion_pointer()
        return {"matrix_rule_eval_summary_rows": str(len(summary)), "matrix_reports_dir": str(reports_dir)}
    write_expansion_backlog(
        inventory_path=reports_dir / "rule_matrix_inventory.csv",
        summary_path=working_eval_dir / "matrix_rule_eval_summary.csv",
        output_path=reports_dir / "rule_expansion_backlog.csv",
    )
    write_next_dataset_activation_plan(
        backlog_path=reports_dir / "rule_expansion_backlog.csv",
        summary_path=working_eval_dir / "matrix_rule_eval_summary.csv",
        output_path=reports_dir / "next_dataset_activation_plan.md",
    )
    _append_rule_expansion_pointer()
    return {"matrix_rule_eval_summary_rows": str(len(summary))}


def _copy_dataset_manifest_to_reports(*, data_dir: Path, reports_dir: Path) -> None:
    source = data_dir / "matrix_eval_manifest.json"
    if not source.exists():
        return
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, reports_dir / "matrix_eval_manifest.json")


def _baseline_underfilled_rule_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(rule_id) for rule_id in manifest.get("underfilled_rule_ids", [])]


def _append_rule_expansion_pointer() -> None:
    path = Path("docs/rule_expansion_plan.md")
    text = path.read_text(encoding="utf-8")
    marker = "## Matrix Eval Findings"
    block = (
        "\n## Matrix Eval Findings\n\n"
        "- Full matrix audit artifacts are under `reports/matrix_eval/`.\n"
        "- Dedicated eval corpus is under `data/processed/matrix_eval/`.\n"
        "- Use `reports/matrix_eval/next_dataset_activation_plan.md` for the next dataset cycle.\n"
    )
    if marker not in text:
        path.write_text(text.rstrip() + block, encoding="utf-8")


def _append_matrix_rule_expansion_pointer() -> None:
    path = Path("docs/rule_expansion_plan.md")
    text = path.read_text(encoding="utf-8")
    marker = "## Matrix Eval Matrix Findings"
    block = (
        "\n## Matrix Eval Matrix Findings\n\n"
        "- Matrix artifacts are under `reports/matrix_eval/`.\n"
        "- Matrix eval corpus is under `data/processed/matrix_eval/`.\n"
        "- Use `reports/matrix_eval/activation_activation_plan.md` and "
        "`reports/matrix_eval/rule_expansion_backlog_core.md` for `training_dataset` planning.\n"
        "- Planned and metadata-only matrix entries remain backlog items until executable support exists.\n"
    )
    if marker not in text:
        path.write_text(text.rstrip() + block, encoding="utf-8")


def _audit_summary(*, reports_dir: Path, data_dir: Path, working_eval_dir: Path, audit: dict[str, Any]) -> dict[str, Any]:
    inventory = _read_csv(reports_dir / "rule_matrix_inventory.csv")
    matrix_summary = _read_csv(working_eval_dir / "matrix_rule_eval_summary.csv")
    dataset = _read_csv(data_dir / "matrix_eval.csv.gz")
    result: dict[str, Any] = {
        "total_matrix_entries": int(len(inventory)),
        "final_verdict": audit.get("verdict", "MATRIX_EVAL_BLOCKED"),
        "matrix_eval_summary_path": str(working_eval_dir / "matrix_rule_eval_summary.csv"),
        "expansion_backlog_path": str(reports_dir / "rule_expansion_backlog.csv"),
        "next_activation_plan_path": str(reports_dir / "next_dataset_activation_plan.md"),
    }
    if not inventory.empty:
        statuses = inventory["executable_status"].astype(str).value_counts().to_dict()
        result.update(
            {
                "executable_active_count": int(statuses.get("EXECUTABLE_ACTIVE", 0)),
                "executable_inactive_count": int(statuses.get("EXECUTABLE_INACTIVE", 0)),
                "metadata_only_count": int(statuses.get("METADATA_ONLY", 0)),
                "planned_only_count": int(statuses.get("PLANNED_ONLY", 0)),
                "rules_with_no_candidate_path": int(
                    (~inventory["candidate_generator_support"].astype(str).str.lower().isin(["true", "1"])).sum()
                ),
                "rules_needing_syntax": int(inventory["executable_status"].astype(str).str.contains("SYNTAX").sum()),
                "rules_needing_dictionary": int(inventory["executable_status"].astype(str).str.contains("DICTIONARY").sum()),
                "rules_needing_ner": int(inventory["executable_status"].astype(str).str.contains("NER").sum()),
            }
        )
    if not matrix_summary.empty:
        result.update(
            {
                "evaluated_rule_count": int(matrix_summary["rule_id"].nunique()),
                "rules_with_candidate_recall_ge_0_85": int((matrix_summary["candidate_recall"].astype(float) >= 0.85).sum()),
                "rules_with_candidate_recall_lt_0_85": int((matrix_summary["candidate_recall"].astype(float) < 0.85).sum()),
                "rules_needing_validator": int((matrix_summary["decision"].astype(str) == "NEEDS_VALIDATOR").sum()),
                "rules_ready_for_next_dataset_cycle": int((matrix_summary["decision"].astype(str) == "READY_NEXT_DATASET").sum()),
                "rules_to_keep_inactive": int(matrix_summary["decision"].astype(str).isin(["KEEP_INACTIVE", "UNSAFE"]).sum()),
                "top_weak_rules_by_f1": matrix_summary.sort_values("f1", ascending=True).head(10)["rule_id"].tolist(),
                "top_unsafe_rules": matrix_summary[
                    (matrix_summary["clean_overcorrection_count"].astype(int) > 0)
                    | (matrix_summary["dirty_worse_count"].astype(int) > 0)
                ]["rule_id"].head(10).tolist(),
            }
        )
    if not dataset.empty:
        result["matrix_eval_rows"] = int(len(dataset))
    return result


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _protected_git_status() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in PRODUCTION_GUARD_PATHS:
        status = _git_status_porcelain(path)
        if status:
            result[path] = status
    return result


def _git_status_porcelain(path: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--", path],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return ""
    return completed.stdout.strip()


if __name__ == "__main__":
    main()
