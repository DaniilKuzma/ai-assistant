from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id


VALIDATOR_NEEDED_RULE_IDS = (
    "double_consonant_candidate",
    "extra_letter_candidate",
    "hyphen_po_adverbs",
    "missing_letter_candidate",
    "pattern_чо_че",
)

UNDER_QUOTA_RULE_AUDIT_COLUMNS = [
    "rule_id",
    "matrix_group",
    "current_eval_count",
    "target_eval_count",
    "candidate_path_exists",
    "synthetic_generator_exists",
    "hard_negative_exists",
    "can_backfill_with_existing_generator",
    "backfill_attempted",
    "backfill_success_count",
    "candidate_recall_after_backfill",
    "decision",
    "reason",
]

BACKLOG_CORE_COLUMNS = [
    "priority",
    "rule_id",
    "matrix_id",
    "group",
    "blocker",
    "required_work",
    "estimated_risk",
    "recommended_next_prompt",
    "examples_needed",
    "notes",
]

ACTIVATION_ACTIVATION_COLUMNS = [
    "rule_id",
    "matrix_group",
    "eval_count",
    "candidate_recall",
    "precision",
    "recall",
    "f1",
    "clean_overcorrection_count",
    "dirty_worse_count",
    "under_quota_decision",
    "blockers",
    "activation_decision",
    "reason",
]

MATRIX_REQUIRED_REPORTS = [
    "under_quota_rule_audit.csv",
    "validator_fix_report.md",
    "next_dataset_activation_plan.csv",
    "rule_expansion_backlog_core.csv",
]

MATRIX_REQUIRED_EVAL_REPORTS = [
    "evaluation_summary.csv",
    "rule_precision_recall.csv",
    "error_by_rule.csv",
    "accepted_edits.csv",
    "rejected_edits.csv",
    "matrix_rule_eval_summary.csv",
]

NO_CANDIDATE_DECISIONS = {"NO_CANDIDATE_PATH", "NEEDS_IMPLEMENTATION"}
BLOCKER_ORDER = {
    "NEEDS_VALIDATOR": 1,
    "NEEDS_SYNTHETIC_GENERATOR": 2,
    "NEEDS_THRESHOLD_CALIBRATION": 3,
    "NEEDS_REAL_DATA": 3,
    "NEEDS_MORE_TRAINING": 3,
    "NEEDS_CANDIDATE_GENERATOR": 4,
    "NEEDS_SYNTAX_FEATURES": 5,
    "NEEDS_DICTIONARY_SUPPORT": 5,
    "NEEDS_NER_SUPPORT": 5,
    "KEEP_INACTIVE_UNSAFE": 6,
}


def write_under_quota_rule_audit(
    *,
    baseline_manifest_path: str | Path,
    inventory_path: str | Path,
    matrix_summary_path: str | Path,
    matrix_dataset_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    manifest = _read_json(baseline_manifest_path)
    inventory = _read_csv(inventory_path)
    summary = _read_csv(matrix_summary_path)
    dataset = _read_csv(matrix_dataset_path)
    target = int(manifest.get("quota", {}).get("min_eval_examples_per_executable_rule", 50))
    baseline_counts = {str(key): int(value) for key, value in (manifest.get("rule_id_counts") or {}).items()}
    no_candidate = {normalize_rule_id(value) for value in manifest.get("no_candidate_path_rule_ids", [])}
    underfilled = [normalize_rule_id(value) for value in manifest.get("underfilled_rule_ids", [])]

    inventory_by_rule = _inventory_by_rule(inventory)
    summary_by_rule = _summary_by_rule(summary)
    matrix_counts = _dataset_rule_counts(dataset)
    rows: list[dict[str, Any]] = []
    for rule_id in sorted(rule_id for rule_id in underfilled if rule_id != UNKNOWN_RULE_ID):
        inv = inventory_by_rule.get(rule_id, {})
        item = summary_by_rule.get(rule_id, {})
        current_count = int(baseline_counts.get(rule_id, _number(inv.get("current_eval_count"), 0)))
        matrix_count = int(matrix_counts.get(rule_id, _number(item.get("eval_examples"), 0)))
        candidate_path = _boolish(inv.get("candidate_generator_support")) and rule_id not in no_candidate
        synthetic_generator = _boolish(inv.get("synthetic_corruption_support"))
        backfill_success = max(0, matrix_count - current_count)
        backfill_attempted = candidate_path and current_count < target
        recall_after = float(_number(item.get("candidate_recall"), 0.0))
        can_backfill = candidate_path and (synthetic_generator or backfill_success > 0)
        decision, reason = _under_quota_decision(
            rule_id=rule_id,
            target=target,
            matrix_count=matrix_count,
            current_count=current_count,
            candidate_path=candidate_path,
            synthetic_generator=synthetic_generator,
            backfill_success=backfill_success,
            recall_after=recall_after,
        )
        rows.append(
            {
                "rule_id": rule_id,
                "matrix_group": str(item.get("matrix_group") or inv.get("group") or inv.get("matrix_id") or ""),
                "current_eval_count": current_count,
                "target_eval_count": target,
                "candidate_path_exists": candidate_path,
                "synthetic_generator_exists": synthetic_generator,
                "hard_negative_exists": False,
                "can_backfill_with_existing_generator": can_backfill,
                "backfill_attempted": backfill_attempted,
                "backfill_success_count": backfill_success,
                "candidate_recall_after_backfill": recall_after,
                "decision": decision,
                "reason": reason,
            }
        )
    frame = pd.DataFrame(rows, columns=UNDER_QUOTA_RULE_AUDIT_COLUMNS)
    _write_csv(frame, output_path)
    return frame


def write_validator_fix_report(
    *,
    summary_path: str | Path,
    accepted_edits_path: str | Path,
    rejected_edits_path: str | Path,
    output_path: str | Path,
) -> str:
    summary = _read_csv(summary_path)
    accepted = _read_csv(accepted_edits_path)
    rejected = _read_csv(rejected_edits_path)
    summary_by_rule = _summary_by_rule(summary)
    lines = [
        "# Validator Fix Report",
        "",
        "Matrix validator fixes are bounded safety guards only; thresholds and checkpoints are unchanged.",
        "",
        "| rule_id | decision | accepted | rejected | top_rejections | note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for rule_id in VALIDATOR_NEEDED_RULE_IDS:
        item = summary_by_rule.get(rule_id, {})
        decision = str(item.get("decision", "NOT_EVALUATED") or "NOT_EVALUATED")
        accepted_count = _edit_count(accepted, rule_id)
        rejected_count = _edit_count(rejected, rule_id)
        note = _validator_note(rule_id, decision, accepted_count, rejected_count)
        lines.append(
            f"| {rule_id} | {decision} | {accepted_count} | {rejected_count} | "
            f"{_top_rejection_reasons(rejected, rule_id)} | {note} |"
        )
    text = "\n".join(lines) + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


def write_rule_expansion_backlog(
    *,
    inventory_path: str | Path,
    summary_path: str | Path,
    output_csv_path: str | Path,
    output_md_path: str | Path,
) -> pd.DataFrame:
    inventory = _read_csv(inventory_path)
    summary = _read_csv(summary_path)
    inventory_by_rule = _inventory_by_rule(inventory)
    summary_by_rule = _summary_by_rule(summary)
    rule_ids = sorted(set(inventory_by_rule) | set(summary_by_rule))
    rows: list[dict[str, Any]] = []
    for rule_id in rule_ids:
        inv = inventory_by_rule.get(rule_id, {})
        item = summary_by_rule.get(rule_id, {})
        blocker = _blocker_for(rule_id, inv, item)
        if not blocker:
            continue
        group = str(item.get("matrix_group") or inv.get("group") or inv.get("matrix_id") or "")
        rows.append(
            {
                "priority": BLOCKER_ORDER[blocker],
                "rule_id": rule_id,
                "matrix_id": str(inv.get("matrix_id") or group),
                "group": group,
                "blocker": blocker,
                "required_work": _required_work_for(blocker, rule_id),
                "estimated_risk": _risk_for(blocker),
                "recommended_next_prompt": _prompt_for(blocker, rule_id),
                "examples_needed": _examples_needed_for(blocker),
                "notes": str(item.get("reason") or inv.get("decision_reason") or ""),
            }
        )
    for inv in _matrix_only_inventory_rows(inventory):
        matrix_id = str(inv.get("matrix_id") or inv.get("group") or "")
        blocker = _blocker_for(matrix_id, inv, {})
        if not blocker:
            continue
        group = str(inv.get("group") or matrix_id)
        rows.append(
            {
                "priority": BLOCKER_ORDER[blocker],
                "rule_id": "",
                "matrix_id": matrix_id,
                "group": group,
                "blocker": blocker,
                "required_work": _required_work_for(blocker, matrix_id),
                "estimated_risk": _risk_for(blocker),
                "recommended_next_prompt": _prompt_for(blocker, matrix_id),
                "examples_needed": _examples_needed_for(blocker),
                "notes": str(inv.get("decision_reason") or inv.get("executable_status") or ""),
            }
        )
    frame = pd.DataFrame(rows, columns=BACKLOG_CORE_COLUMNS).sort_values(["priority", "rule_id"], kind="stable")
    _write_csv(frame, output_csv_path)
    _write_backlog_md(frame, output_md_path)
    return frame


def write_next_dataset_activation_plan(
    *,
    summary_path: str | Path,
    under_quota_audit_path: str | Path,
    backlog_core_path: str | Path,
    output_csv_path: str | Path,
    output_md_path: str | Path,
) -> pd.DataFrame:
    summary = _read_csv(summary_path)
    under_quota = _read_csv(under_quota_audit_path)
    backlog = _read_csv(backlog_core_path)
    under_by_rule = under_quota.set_index("rule_id").to_dict("index") if "rule_id" in under_quota.columns else {}
    blockers_by_rule = _blockers_by_rule(backlog)
    rows: list[dict[str, Any]] = []
    for item in summary.to_dict("records"):
        rule_id = normalize_rule_id(item.get("rule_id", ""))
        if not rule_id or rule_id == UNKNOWN_RULE_ID:
            continue
        blockers = blockers_by_rule.get(rule_id, [])
        under_decision = str(under_by_rule.get(rule_id, {}).get("decision", ""))
        include, reason = _activation_decision(item, under_decision, blockers)
        rows.append(
            {
                "rule_id": rule_id,
                "matrix_group": str(item.get("matrix_group", "")),
                "eval_count": int(_number(item.get("eval_examples"), 0)),
                "candidate_recall": float(_number(item.get("candidate_recall"), 0.0)),
                "precision": float(_number(item.get("precision"), 0.0)),
                "recall": float(_number(item.get("recall"), 0.0)),
                "f1": float(_number(item.get("f1"), 0.0)),
                "clean_overcorrection_count": int(_number(item.get("clean_overcorrection_count"), 0)),
                "dirty_worse_count": int(_number(item.get("dirty_worse_count"), 0)),
                "under_quota_decision": under_decision,
                "blockers": ";".join(blockers),
                "activation_decision": "INCLUDE" if include else "EXCLUDE",
                "reason": reason,
            }
        )
    frame = pd.DataFrame(rows, columns=ACTIVATION_ACTIVATION_COLUMNS).sort_values(["activation_decision", "rule_id"], kind="stable")
    _write_csv(frame, output_csv_path)
    _write_activation_md(frame, output_md_path)
    return frame


def assert_matrix_eval_audit(
    *,
    root_dir: str | Path = "reports/matrix_eval",
    data_dir: str | Path = "data/processed/matrix_eval",
    eval_dir: str | Path | None = None,
    config_path: str | Path = "configs/config.yaml",
) -> dict[str, Any]:
    root = Path(root_dir)
    data = Path(data_dir)
    eval_base = Path(eval_dir) if eval_dir is not None else root / "working_eval"
    missing: list[str] = []
    errors: list[str] = []
    for name in MATRIX_REQUIRED_REPORTS:
        if not (root / name).exists():
            missing.append(name)
    for name in ("matrix_eval.csv.gz", "matrix_eval_manifest.json"):
        if not (data / name).exists():
            missing.append(name)
    for name in MATRIX_REQUIRED_EVAL_REPORTS:
        if not (eval_base / name).exists():
            missing.append(f"working_eval/{name}")

    dataset = _read_csv(data / "matrix_eval.csv.gz")
    if not dataset.empty and {"source_type", "candidate_present"}.issubset(dataset.columns):
        positive = dataset[dataset["source_type"].astype(str) != "matrix_hard_negative"]
        bad = positive[~positive["candidate_present"].astype(str).str.lower().isin(["true", "1"])]
        if not bad.empty:
            errors.append("positive matrix eval rows without candidate_present=true")
        synthetic_bad = positive[
            positive["is_synthetic"].astype(str).str.lower().isin(["true", "1"])
            & ~positive["candidate_present"].astype(str).str.lower().isin(["true", "1"])
        ]
        if not synthetic_bad.empty:
            errors.append("synthetic matrix eval rows without candidate_present=true")
        meta_like = _meta_template_rows(dataset)
        if not meta_like.empty:
            errors.append("matrix eval contains meta-template rows")

    if Path(config_path).exists():
        text = Path(config_path).read_text(encoding="utf-8")
        if "mode: calibrated_guarded" not in text:
            errors.append("working v1 threshold profile is not calibrated_guarded")

    training_report = eval_base / "training_report.md"
    if training_report.exists():
        model_training_ran = _training_report_model_training_ran(training_report)
        if model_training_ran is not None and model_training_ran != 0.0:
            errors.append("eval run appears to have trained a model")

    if errors:
        verdict = "MATRIX_EVAL_MATRIX_BLOCKED"
    elif missing:
        verdict = "MATRIX_EVAL_MATRIX_PARTIAL"
    else:
        verdict = "MATRIX_EVAL_MATRIX_COMPLETE"
    result = {"verdict": verdict, "missing_outputs": missing, "errors": errors}
    summary_path = root / "matrix_audit_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _training_report_model_training_ran(path: Path) -> float | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "model_training_ran" not in line:
            continue
        _prefix, _sep, value = line.partition(":")
        if not value:
            continue
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def write_matrix_final_report(
    *,
    root_dir: str | Path,
    data_dir: str | Path,
    eval_dir: str | Path,
    audit: dict[str, Any],
) -> str:
    root = Path(root_dir)
    summary = _read_csv(Path(eval_dir) / "matrix_rule_eval_summary.csv")
    under = _read_csv(root / "under_quota_rule_audit.csv")
    backlog = _read_csv(root / "rule_expansion_backlog_core.csv")
    activation = _read_csv(root / "next_dataset_activation_plan.csv")
    manifest = _read_json(Path(data_dir) / "matrix_eval_manifest.json")
    lines = [
        "# Matrix Eval Matrix Final Report",
        "",
        f"- final verdict: {audit.get('verdict', '')}",
        f"- under-quota rules backfilled: {_count_value(under, 'decision', 'BACKFILLED')}",
        f"- under-quota rules still under quota: {_count_not_value(under, 'decision', 'BACKFILLED')}",
        f"- validator-needed rules fixed: {_validator_fixed_count(summary)}",
        f"- validator-needed rules still blocked: {_validator_blocked_count(summary)}",
        f"- matrix evaluated rule count: {summary['rule_id'].nunique() if 'rule_id' in summary.columns else 0}",
        f"- matrix candidate recall >=0.85 count: {_count_numeric_ge(summary, 'candidate_recall', 0.85)}",
        f"- no-candidate-path count: {len(manifest.get('no_candidate_path_rule_ids', []))}",
        f"- activation rule_ids: {', '.join(activation[activation['activation_decision'] == 'INCLUDE']['rule_id'].astype(str).tolist()) if 'activation_decision' in activation.columns else ''}",
        f"- rules still needing syntax: {', '.join(_rules_for_blocker(backlog, 'NEEDS_SYNTAX_FEATURES'))}",
        f"- rules still needing dictionary: {', '.join(_rules_for_blocker(backlog, 'NEEDS_DICTIONARY_SUPPORT'))}",
        f"- rules still needing NER: {', '.join(_rules_for_blocker(backlog, 'NEEDS_NER_SUPPORT'))}",
        f"- rules still needing candidate generator: {', '.join(_rules_for_blocker(backlog, 'NEEDS_CANDIDATE_GENERATOR'))}",
        f"- rules still needing validator: {', '.join(_rules_for_blocker(backlog, 'NEEDS_VALIDATOR'))}",
        f"- top weak rules by F1: {', '.join(_top_weak_rules(summary))}",
        f"- unsafe/keep inactive rules: {', '.join(_rules_for_blocker(backlog, 'KEEP_INACTIVE_UNSAFE'))}",
    ]
    text = "\n".join(lines) + "\n"
    (root / "matrix_final_report.md").write_text(text, encoding="utf-8")
    return text


def _under_quota_decision(
    *,
    rule_id: str,
    target: int,
    matrix_count: int,
    current_count: int,
    candidate_path: bool,
    synthetic_generator: bool,
    backfill_success: int,
    recall_after: float,
) -> tuple[str, str]:
    if not candidate_path:
        return "NO_CANDIDATE_PATH", "No verified candidate-backed eval path exists."
    if matrix_count >= target and recall_after >= 0.85:
        return "BACKFILLED", f"Candidate-backed examples reached target {target}."
    if not synthetic_generator and matrix_count == 0:
        return "NEEDS_IMPLEMENTATION", "Candidate metadata exists, but no safe generator/data path produced rows."
    if backfill_success > 0:
        return "KEEP_UNDER_QUOTA", f"Backfill added {backfill_success} rows but did not reach target {target}."
    if current_count > 0:
        return "KEEP_UNDER_QUOTA", "Existing examples remain below quota; no additional safe rows found."
    if rule_id.startswith(("quote_", "bracket_", "punctuation_")):
        return "UNSAFE_TO_BACKFILL", "Punctuation/balance backfill needs dedicated safety policy."
    return "NEEDS_IMPLEMENTATION", "Existing generators did not produce candidate-backed rows."


def _blocker_for(rule_id: str, inv: dict[str, Any], item: dict[str, Any]) -> str:
    decision = str(item.get("decision", ""))
    requires = {str(value).lower() for value in _parse_jsonish_list(inv.get("requires"))}
    candidate = _boolish(inv.get("candidate_generator_support"))
    if decision == "READY_NEXT_DATASET":
        return ""
    if decision == "NEEDS_VALIDATOR":
        return "NEEDS_VALIDATOR"
    if decision == "NEEDS_THRESHOLD":
        return "NEEDS_THRESHOLD_CALIBRATION"
    if decision == "NEEDS_MORE_TRAINING":
        return "NEEDS_MORE_TRAINING"
    if _number(item.get("clean_overcorrection_count"), 0) > 0 or _number(item.get("dirty_worse_count"), 0) > 0:
        return "KEEP_INACTIVE_UNSAFE"
    if "ner" in requires:
        return "NEEDS_NER_SUPPORT"
    if "syntax" in requires:
        return "NEEDS_SYNTAX_FEATURES"
    if "dictionary" in requires and not candidate:
        return "NEEDS_DICTIONARY_SUPPORT"
    if not candidate or _number(item.get("candidate_recall"), 0.0) < 0.85:
        return "NEEDS_CANDIDATE_GENERATOR"
    if _number(item.get("eval_examples"), 0) <= 0:
        return "NEEDS_SYNTHETIC_GENERATOR"
    if decision == "KEEP_INACTIVE":
        return "KEEP_INACTIVE_UNSAFE"
    return "NEEDS_REAL_DATA"


def _activation_decision(item: dict[str, Any], under_decision: str, blockers: list[str]) -> tuple[bool, str]:
    unsafe_blockers = {
        "NEEDS_VALIDATOR",
        "NEEDS_SYNTAX_FEATURES",
        "NEEDS_DICTIONARY_SUPPORT",
        "NEEDS_NER_SUPPORT",
        "NEEDS_CANDIDATE_GENERATOR",
        "NEEDS_SYNTHETIC_GENERATOR",
        "NEEDS_THRESHOLD_CALIBRATION",
        "KEEP_INACTIVE_UNSAFE",
    }
    if any(blocker in unsafe_blockers for blocker in blockers):
        return False, "blocked by " + ",".join(blockers)
    if _number(item.get("candidate_recall"), 0.0) < 0.85:
        return False, "candidate recall below 0.85"
    if _number(item.get("clean_overcorrection_count"), 0) > 0 or _number(item.get("dirty_worse_count"), 0) > 0:
        return False, "clean overcorrection or dirty-worse risk"
    if int(_number(item.get("eval_examples"), 0)) < 50 and under_decision != "BACKFILLED":
        return False, "under quota"
    if str(item.get("decision", "")) == "READY_NEXT_DATASET":
        return True, "passes matrix gates"
    precision = float(_number(item.get("precision"), 0.0))
    recall = float(_number(item.get("recall"), 0.0))
    if precision >= 0.80 and recall >= 0.50:
        return True, "passes metric gates"
    return False, "model metrics not ready"


def _inventory_by_rule(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        rule_ids = _parse_rule_ids(row.get("listed_rule_ids")) or [normalize_rule_id(row.get("rule_id", ""))]
        for rule_id in rule_ids:
            if rule_id and rule_id != UNKNOWN_RULE_ID:
                result.setdefault(rule_id, row)
    return result


def _matrix_only_inventory_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        if _parse_rule_ids(row.get("listed_rule_ids")):
            continue
        matrix_id = str(row.get("matrix_id") or "")
        if matrix_id:
            rows.append(row)
    return rows


def _summary_by_rule(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "rule_id" not in frame.columns:
        return {}
    return {normalize_rule_id(row.get("rule_id", "")): row for row in frame.to_dict("records")}


def _dataset_rule_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "rule_id" not in frame.columns:
        return {}
    positives = frame[frame.get("source_type", pd.Series(dtype=str)).astype(str) != "matrix_hard_negative"]
    return {normalize_rule_id(key): int(value) for key, value in Counter(positives["rule_id"].map(normalize_rule_id)).items()}


def _blockers_by_rule(frame: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if frame.empty or not {"rule_id", "blocker"}.issubset(frame.columns):
        return result
    for row in frame.to_dict("records"):
        result.setdefault(normalize_rule_id(row.get("rule_id", "")), []).append(str(row.get("blocker", "")))
    return {key: sorted(set(value)) for key, value in result.items()}


def _required_work_for(blocker: str, rule_id: str) -> str:
    return {
        "NEEDS_VALIDATOR": f"Add bounded StrictValidator guard for {rule_id} and preserve known positives.",
        "NEEDS_SYNTHETIC_GENERATOR": f"Add candidate-backed synthetic/eval generator coverage for {rule_id}.",
        "NEEDS_SYNTAX_FEATURES": f"Add syntax-backed features and validators before activating {rule_id}.",
        "NEEDS_DICTIONARY_SUPPORT": f"Add dictionary support needed for {rule_id} candidate generation.",
        "NEEDS_NER_SUPPORT": f"Add NER-backed spans and negative tests for {rule_id}.",
        "NEEDS_CANDIDATE_GENERATOR": f"Implement bounded candidate generator support for {rule_id}.",
        "NEEDS_REAL_DATA": f"Collect real candidate-backed examples for {rule_id}.",
        "NEEDS_THRESHOLD_CALIBRATION": f"Calibrate {rule_id} on validation data without tuning on test.",
        "NEEDS_MORE_TRAINING": f"Include safe {rule_id} positives in a future training cycle.",
        "KEEP_INACTIVE_UNSAFE": f"Keep {rule_id} inactive until safety policy is redesigned.",
    }[blocker]


def _risk_for(blocker: str) -> str:
    if blocker in {"NEEDS_VALIDATOR", "NEEDS_SYNTHETIC_GENERATOR", "NEEDS_REAL_DATA"}:
        return "low"
    if blocker in {"NEEDS_THRESHOLD_CALIBRATION", "NEEDS_MORE_TRAINING", "NEEDS_CANDIDATE_GENERATOR"}:
        return "medium"
    return "high"


def _prompt_for(blocker: str, rule_id: str) -> str:
    return f"Resolve {blocker} for {rule_id} with candidate-backed tests and no production threshold/checkpoint mutation."


def _examples_needed_for(blocker: str) -> str:
    return {
        "NEEDS_VALIDATOR": "FP hard negatives and preserved positive repairs",
        "NEEDS_SYNTHETIC_GENERATOR": "bounded candidate-backed synthetic positives",
        "NEEDS_SYNTAX_FEATURES": "syntax-labeled positives and hard negatives",
        "NEEDS_DICTIONARY_SUPPORT": "dictionary-backed positives and clean negatives",
        "NEEDS_NER_SUPPORT": "NER-labeled entity positives and ordinary-word negatives",
        "NEEDS_CANDIDATE_GENERATOR": "candidate-backed positives for each edit shape",
        "NEEDS_REAL_DATA": "real dirty/clean pairs",
        "NEEDS_THRESHOLD_CALIBRATION": "validation distribution with accepted/rejected edits",
        "NEEDS_MORE_TRAINING": "safe training positives plus hard negatives",
        "KEEP_INACTIVE_UNSAFE": "safety design examples before data generation",
    }[blocker]


def _validator_note(rule_id: str, decision: str, accepted_count: int, rejected_count: int) -> str:
    if decision != "NEEDS_VALIDATOR" and accepted_count == 0:
        return "fixed or safely blocked after guard update"
    if rejected_count > 0:
        return "guard is active; verify positive recall before activation"
    return f"{rule_id} still needs validator review"


def _edit_count(frame: pd.DataFrame, rule_id: str) -> int:
    if frame.empty or "rule_id" not in frame.columns:
        return 0
    return int((frame["rule_id"].map(normalize_rule_id) == rule_id).sum())


def _top_rejection_reasons(frame: pd.DataFrame, rule_id: str) -> str:
    if frame.empty or not {"rule_id", "reason"}.issubset(frame.columns):
        return ""
    rows = frame[frame["rule_id"].map(normalize_rule_id) == rule_id]
    counts = Counter(str(reason) for reason in rows["reason"] if str(reason))
    return "; ".join(f"{reason}:{count}" for reason, count in counts.most_common(3))


def _write_backlog_md(frame: pd.DataFrame, output_path: str | Path) -> None:
    lines = ["# Rule Expansion Backlog", "", "Blocked matrix rules by required next work.", ""]
    for blocker, rows in frame.groupby("blocker", sort=False) if not frame.empty else []:
        lines.extend([f"## {blocker}", ""])
        lines.extend(f"- {row.rule_id}: {row.required_work}" for row in rows.itertuples())
        lines.append("")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _write_activation_md(frame: pd.DataFrame, output_path: str | Path) -> None:
    include = frame[frame["activation_decision"] == "INCLUDE"]["rule_id"].astype(str).tolist() if not frame.empty else []
    exclude = frame[frame["activation_decision"] != "INCLUDE"]["rule_id"].astype(str).tolist() if not frame.empty else []
    lines = [
        "# Next Dataset Activation Plan",
        "",
        "This matrix plan is for `training_dataset`; it does not mutate production thresholds or checkpoints.",
        "",
        "## Include",
        "",
        *_md_list(include),
        "## Exclude",
        "",
        *_md_list(exclude),
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def _parse_rule_ids(value: Any) -> list[str]:
    return [normalize_rule_id(item) for item in _parse_jsonish_list(value) if normalize_rule_id(item) != UNKNOWN_RULE_ID]


def _parse_jsonish_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return parsed if isinstance(parsed, list) else [parsed]
    if pd.isna(value):
        return []
    return [value]


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _number(value: Any, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _meta_template_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    patterns = ("правило серии", "проверяет семейство", "metadata_only", "готовит точный примере")
    mask = pd.Series(False, index=dataset.index)
    for column in ("source", "target", "template_id", "metadata"):
        if column not in dataset.columns:
            continue
        text = dataset[column].astype(str).str.lower()
        for pattern in patterns:
            mask = mask | text.str.contains(pattern, regex=False, na=False)
    return dataset[mask]


def _md_list(values: list[str]) -> list[str]:
    return [*[f"- {value}" for value in values], ""] if values else ["- none", ""]


def _count_value(frame: pd.DataFrame, column: str, value: str) -> int:
    return int((frame[column].astype(str) == value).sum()) if column in frame.columns else 0


def _count_not_value(frame: pd.DataFrame, column: str, value: str) -> int:
    return int((frame[column].astype(str) != value).sum()) if column in frame.columns else 0


def _count_numeric_ge(frame: pd.DataFrame, column: str, value: float) -> int:
    return int((pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce").fillna(0.0) >= value).sum())


def _rules_for_blocker(frame: pd.DataFrame, blocker: str) -> list[str]:
    if frame.empty or not {"rule_id", "matrix_id", "blocker"}.issubset(frame.columns):
        return []
    rows = frame[frame["blocker"] == blocker]
    ids = [_display_backlog_identifier(row) for row in rows.to_dict("records")]
    return sorted(identifier for identifier in ids if identifier)


def _display_backlog_identifier(row: dict[str, Any]) -> str:
    for key in ("rule_id", "matrix_id"):
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _top_weak_rules(frame: pd.DataFrame) -> list[str]:
    if frame.empty or not {"rule_id", "f1"}.issubset(frame.columns):
        return []
    sortable = frame.copy()
    sortable["f1"] = pd.to_numeric(sortable["f1"], errors="coerce").fillna(0.0)
    return sortable.sort_values(["f1", "rule_id"], kind="stable").head(10)["rule_id"].astype(str).tolist()


def _validator_fixed_count(frame: pd.DataFrame) -> int:
    if frame.empty or "rule_id" not in frame.columns:
        return 0
    subset = frame[frame["rule_id"].isin(VALIDATOR_NEEDED_RULE_IDS)]
    if subset.empty:
        return 0
    return int((subset["decision"].astype(str) != "NEEDS_VALIDATOR").sum())


def _validator_blocked_count(frame: pd.DataFrame) -> int:
    if frame.empty or "rule_id" not in frame.columns:
        return len(VALIDATOR_NEEDED_RULE_IDS)
    subset = frame[frame["rule_id"].isin(VALIDATOR_NEEDED_RULE_IDS)]
    return int((subset["decision"].astype(str) == "NEEDS_VALIDATOR").sum())
