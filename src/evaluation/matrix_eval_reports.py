from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id


MATRIX_RULE_EVAL_SUMMARY_COLUMNS = [
    "rule_id",
    "matrix_group",
    "eval_examples",
    "candidate_recall",
    "gap_coverage",
    "predicted_count",
    "true_positive",
    "false_positive",
    "false_negative",
    "precision",
    "recall",
    "f1",
    "clean_overcorrection_count",
    "dirty_worse_count",
    "accepted_count",
    "rejected_count",
    "top_rejection_reasons",
    "decision",
    "reason",
]

HARD_NEGATIVE_COLUMNS = [
    "trap_type",
    "related_rule_id",
    "examples_count",
    "accepted_bad_edits_count",
    "rejected_bad_edits_count",
    "clean_overcorrection_count",
    "decision",
]

BACKLOG_COLUMNS = [
    "priority",
    "rule_id",
    "matrix_id",
    "group",
    "status",
    "blocker",
    "required_work",
    "estimated_risk",
    "recommended_next_prompt",
    "notes",
]

REQUIRED_MATRIX_EVAL_OUTPUTS = [
    "rule_matrix_inventory.csv",
    "rule_matrix_inventory_summary.md",
    "rule_id_alias_audit.csv",
    "matrix_eval.csv.gz",
    "matrix_eval_by_rule.csv",
    "matrix_eval_manifest.json",
    "matrix_eval_dataset_report.md",
    "matrix_eval_candidate_recall_by_rule.csv",
    "matrix_eval_gap_coverage_by_rule.csv",
    "hard_negative_matrix_coverage.csv",
    "rule_expansion_backlog.csv",
    "next_dataset_activation_plan.md",
]

REQUIRED_WORKING_EVAL_OUTPUTS = [
    "evaluation_summary.csv",
    "rule_precision_recall.csv",
    "error_by_rule.csv",
    "accepted_edits.csv",
    "rejected_edits.csv",
    "clean_overcorrection_examples.csv",
    "dirty_worse_examples.csv",
    "matrix_rule_eval_summary.csv",
]


def write_matrix_rule_eval_summary(
    *,
    dataset_path: str | Path,
    inventory_path: str | Path,
    eval_dir: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    dataset = _read_csv(dataset_path)
    inventory = _read_csv(inventory_path)
    eval_base = Path(eval_dir)
    rule_metrics = _read_csv(eval_base / "rule_precision_recall.csv")
    candidate_recall = _read_csv(eval_base / "candidate_recall_by_rule.csv")
    gap_coverage = _read_csv(eval_base / "gap_label_coverage_by_rule.csv")
    accepted = _read_csv(eval_base / "accepted_edits.csv")
    rejected = _read_csv(eval_base / "rejected_edits.csv")
    clean_over = _read_csv(eval_base / "clean_overcorrection_examples.csv")
    dirty_worse = _read_csv(eval_base / "dirty_worse_examples.csv")

    matrix_groups = _inventory_groups(inventory)
    dataset_counts = _dataset_rule_counts(dataset)
    rule_ids = sorted(
        {
            *dataset_counts,
            *_frame_rule_ids(rule_metrics),
            *_frame_rule_ids(candidate_recall),
            *_inventory_rule_ids(inventory),
        }
        - {"clean_identity_hard_negative", UNKNOWN_RULE_ID, ""}
    )
    rows: list[dict[str, Any]] = []
    for rule_id in rule_ids:
        metrics = _rule_metric(rule_metrics, rule_id)
        candidate = _candidate_recall(candidate_recall, dataset, rule_id)
        gap = _gap_coverage(gap_coverage, rule_id)
        accepted_count = _edit_count(accepted, rule_id)
        rejected_count = _edit_count(rejected, rule_id)
        clean_count = _clean_overcorrection_count(clean_over, accepted, dataset, rule_id)
        dirty_count = _dirty_worse_count(dirty_worse, accepted, dataset, rule_id)
        top_reasons = _top_rejection_reasons(rejected, rule_id)
        decision, reason = _decision_for_rule(
            eval_examples=dataset_counts.get(rule_id, 0),
            candidate_recall=candidate,
            precision=metrics["precision"],
            recall=metrics["recall"],
            predicted_count=metrics["predicted_count"],
            false_positive=metrics["false_positive"],
            false_negative=metrics["false_negative"],
            clean_overcorrection_count=clean_count,
            dirty_worse_count=dirty_count,
            rejected_count=rejected_count,
        )
        rows.append(
            {
                "rule_id": rule_id,
                "matrix_group": matrix_groups.get(rule_id, ""),
                "eval_examples": dataset_counts.get(rule_id, 0),
                "candidate_recall": candidate,
                "gap_coverage": gap,
                "predicted_count": metrics["predicted_count"],
                "true_positive": metrics["true_positive"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "clean_overcorrection_count": clean_count,
                "dirty_worse_count": dirty_count,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "top_rejection_reasons": top_reasons,
                "decision": decision,
                "reason": reason,
            }
        )
    frame = pd.DataFrame(rows, columns=MATRIX_RULE_EVAL_SUMMARY_COLUMNS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def write_hard_negative_matrix_coverage(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    accepted_edits_path: str | Path | None = None,
    rejected_edits_path: str | Path | None = None,
    clean_overcorrection_path: str | Path | None = None,
) -> pd.DataFrame:
    dataset = _read_csv(dataset_path)
    accepted = _read_csv(accepted_edits_path) if accepted_edits_path else pd.DataFrame()
    rejected = _read_csv(rejected_edits_path) if rejected_edits_path else pd.DataFrame()
    clean_over = _read_csv(clean_overcorrection_path) if clean_overcorrection_path else pd.DataFrame()
    hard = dataset[dataset.get("source_type", pd.Series(dtype=str)).astype(str) == "matrix_hard_negative"] if not dataset.empty else pd.DataFrame()
    trap_counts: Counter[str] = Counter()
    for row in hard.to_dict("records"):
        traps = _metadata_traps(row.get("metadata"))
        for trap in traps or ["clean_identity"]:
            trap_counts[trap] += 1
    if not trap_counts:
        trap_counts["no_hard_negatives"] = 0

    hard_row_ids = set(_hard_negative_row_ids(dataset))
    accepted_bad_frame = _filter_edit_rows_by_row_id(accepted, hard_row_ids)
    rejected_bad_frame = _filter_edit_rows_by_row_id(rejected, hard_row_ids)
    accepted_by_rule = Counter(
        normalize_rule_id(value)
        for value in accepted_bad_frame.get("rule_id", [])
        if not pd.isna(value)
    )
    rejected_by_rule = Counter(
        normalize_rule_id(value)
        for value in rejected_bad_frame.get("rule_id", [])
        if not pd.isna(value)
    )
    clean_over_count = len(clean_over)
    rows: list[dict[str, Any]] = []
    for trap_type, examples_count in sorted(trap_counts.items()):
        related = _related_rule_for_trap(trap_type)
        accepted_bad = accepted_by_rule.get(related, 0) if related else int(sum(accepted_by_rule.values()))
        rejected_bad = rejected_by_rule.get(related, 0) if related else int(sum(rejected_by_rule.values()))
        decision = "UNSAFE" if accepted_bad or clean_over_count else ("NO_EXAMPLES" if examples_count == 0 else "SAFE")
        rows.append(
            {
                "trap_type": trap_type,
                "related_rule_id": related or "",
                "examples_count": int(examples_count),
                "accepted_bad_edits_count": int(accepted_bad),
                "rejected_bad_edits_count": int(rejected_bad),
                "clean_overcorrection_count": int(clean_over_count),
                "decision": decision,
            }
        )
    frame = pd.DataFrame(rows, columns=HARD_NEGATIVE_COLUMNS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def write_expansion_backlog(
    *,
    inventory_path: str | Path,
    summary_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    inventory = _read_csv(inventory_path)
    summary = _read_csv(summary_path)
    summary_by_rule = summary.set_index("rule_id").to_dict("index") if "rule_id" in summary.columns else {}
    rows: list[dict[str, Any]] = []
    for row in inventory.to_dict("records"):
        rule_ids = _parse_rule_ids(row.get("listed_rule_ids")) or [str(row.get("matrix_id", ""))]
        for rule_id in rule_ids:
            eval_decision = str(summary_by_rule.get(rule_id, {}).get("decision", ""))
            required_work, blocker, risk, priority = _required_work(row, eval_decision)
            rows.append(
                {
                    "priority": priority,
                    "rule_id": rule_id,
                    "matrix_id": str(row.get("matrix_id", "")),
                    "group": str(row.get("group", "")),
                    "status": str(row.get("executable_status", row.get("status_from_yaml", ""))),
                    "blocker": blocker,
                    "required_work": required_work,
                    "estimated_risk": risk,
                    "recommended_next_prompt": _recommended_prompt(rule_id, required_work),
                    "notes": str(row.get("decision_reason", "")),
                }
            )
    if not rows and not summary.empty:
        for row in summary.to_dict("records"):
            required_work, blocker, risk, priority = _required_work(row, str(row.get("decision", "")))
            rows.append(
                {
                    "priority": priority,
                    "rule_id": str(row.get("rule_id", "")),
                    "matrix_id": str(row.get("matrix_group", "")),
                    "group": str(row.get("matrix_group", "")),
                    "status": str(row.get("decision", "")),
                    "blocker": blocker,
                    "required_work": required_work,
                    "estimated_risk": risk,
                    "recommended_next_prompt": _recommended_prompt(str(row.get("rule_id", "")), required_work),
                    "notes": str(row.get("reason", "")),
                }
            )
    frame = pd.DataFrame(rows, columns=BACKLOG_COLUMNS).sort_values(["priority", "rule_id"], kind="stable")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def write_next_dataset_activation_plan(
    *,
    backlog_path: str | Path,
    summary_path: str | Path,
    output_path: str | Path,
) -> str:
    backlog = _read_csv(backlog_path)
    summary = _read_csv(summary_path)
    ready = _rule_list(summary, "decision", "READY_NEXT_DATASET")
    keep = sorted(set(_rule_list(summary, "decision", "KEEP_INACTIVE")) | set(_rule_list(summary, "decision", "UNSAFE")))
    threshold = _rule_list(summary, "decision", "NEEDS_THRESHOLD")
    validator = _rule_list(summary, "decision", "NEEDS_VALIDATOR")
    training = _rule_list(summary, "decision", "NEEDS_MORE_TRAINING")
    implementation = sorted(
        set(_rule_list(summary, "decision", "NEEDS_RULE_IMPLEMENTATION"))
        | set(_rule_list(backlog, "required_work", "add_candidate_generator"))
    )
    syntax_ner = sorted(
        set(_rule_list(backlog, "required_work", "add_syntax_features"))
        | set(_rule_list(backlog, "required_work", "add_NER_support"))
    )
    real_data = sorted(set(training) | set(_rule_list(backlog, "required_work", "add_real_data")))
    lines = [
        "# Next Dataset Activation Plan",
        "",
        "This plan is based on matrix eval artifacts only. It does not change production thresholds or checkpoints.",
        "",
        "## Rules to Include in Next Dataset Cycle",
        "",
        *_md_list(ready),
        "## Rules to Keep Inactive",
        "",
        *_md_list(keep),
        "## Rules to Implement Next",
        "",
        *_md_list(implementation),
        "## Rules Needing New Real Data",
        "",
        *_md_list(real_data),
        "## Rules Needing Syntax or NER",
        "",
        *_md_list(syntax_ner),
        "## Rules Needing Threshold Sweep",
        "",
        *_md_list(threshold),
        "## Rules Unsafe for Now",
        "",
        *_md_list(sorted(set(keep) | set(validator))),
        "",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    output.write_text(text, encoding="utf-8")
    return text


def assert_matrix_eval_audit(
    *,
    root_dir: str | Path = "reports/matrix_eval",
    data_dir: str | Path | None = None,
    eval_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root_dir)
    data = Path(data_dir) if data_dir is not None else root
    eval_base = Path(eval_dir) if eval_dir is not None else root / "working_eval"
    missing: list[str] = []
    errors: list[str] = []

    for name in REQUIRED_MATRIX_EVAL_OUTPUTS:
        candidates = [root / name, data / name]
        if not any(path.exists() for path in candidates):
            missing.append(name)
    for name in REQUIRED_WORKING_EVAL_OUTPUTS:
        if not (eval_base / name).exists():
            missing.append(f"working_eval/{name}")

    inventory_path = _first_existing(root / "rule_matrix_inventory.csv", data / "rule_matrix_inventory.csv")
    if inventory_path:
        inventory = _read_csv(inventory_path)
        if "decision" in inventory.columns and inventory["decision"].astype(str).str.strip().eq("").any():
            errors.append("inventory contains empty decisions")
    else:
        errors.append("rule_matrix_inventory.csv is missing")

    dataset_path = _first_existing(root / "matrix_eval.csv.gz", data / "matrix_eval.csv.gz")
    if dataset_path:
        dataset = _read_csv(dataset_path)
        if not dataset.empty and {"source_type", "candidate_present"}.issubset(dataset.columns):
            positive = dataset[~dataset["source_type"].astype(str).isin(["matrix_hard_negative", "metadata"])]
            bad = positive[~positive["candidate_present"].astype(str).str.lower().isin(["true", "1"])]
            if not bad.empty:
                errors.append("positive matrix eval rows without candidate_present=true")
            meta_like = _meta_template_rows(dataset)
            if len(dataset) and len(meta_like) / len(dataset) > 0.05:
                errors.append("fake/meta templates dominate matrix eval rows")
    else:
        errors.append("matrix_eval.csv.gz is missing")

    manifest_path = _first_existing(root / "matrix_eval_manifest.json", data / "matrix_eval_manifest.json")
    manifest_verdict = ""
    if manifest_path:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            manifest_verdict = str(manifest.get("verdict", ""))
        except Exception:
            errors.append("matrix_eval_manifest.json is not valid JSON")

    if errors or any("matrix_eval.csv.gz" in item or "rule_matrix_inventory.csv" in item for item in missing):
        verdict = "MATRIX_EVAL_BLOCKED"
    elif missing or manifest_verdict == "MATRIX_EVAL_DATASET_PARTIAL":
        verdict = "MATRIX_EVAL_PARTIAL"
    else:
        verdict = "MATRIX_EVAL_COMPLETE"
    return {
        "verdict": verdict,
        "missing_outputs": missing,
        "errors": errors,
        "manifest_verdict": manifest_verdict,
    }


def _decision_for_rule(
    *,
    eval_examples: int,
    candidate_recall: float,
    precision: float,
    recall: float,
    predicted_count: int,
    false_positive: int,
    false_negative: int,
    clean_overcorrection_count: int,
    dirty_worse_count: int,
    rejected_count: int,
) -> tuple[str, str]:
    if candidate_recall < 0.85:
        return "NEEDS_RULE_IMPLEMENTATION", "candidate recall below 0.85"
    if clean_overcorrection_count > 0 or dirty_worse_count > 0:
        return "NEEDS_VALIDATOR", "clean overcorrection or dirty-worse examples found"
    if false_positive > 0 and precision < 0.80:
        return "NEEDS_VALIDATOR", "false positives indicate bounded validator guard is needed"
    if eval_examples <= 0:
        return "KEEP_INACTIVE", "no candidate-backed matrix eval examples"
    if predicted_count == 0 and false_negative > 0:
        return "NEEDS_THRESHOLD", "candidate path exists but model accepted no edits"
    if recall < 0.50 and precision >= 0.80 and rejected_count > 0:
        return "NEEDS_THRESHOLD", "accepted count is low with otherwise low false positive risk"
    if precision >= 0.80 and (recall >= 0.50 or false_negative == 0):
        return "READY_NEXT_DATASET", "passes matrix eval safety gates"
    return "NEEDS_MORE_TRAINING", "candidate path exists but current model distribution is weak"


def _rule_metric(frame: pd.DataFrame, rule_id: str) -> dict[str, Any]:
    defaults = {
        "predicted_count": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    if frame.empty or "rule_id" not in frame.columns:
        return defaults
    rows = frame[frame["rule_id"].map(normalize_rule_id) == rule_id]
    if rows.empty:
        return defaults
    result = dict(defaults)
    for column in ("predicted_count", "true_positive", "false_positive", "false_negative"):
        result[column] = int(pd.to_numeric(rows[column], errors="coerce").fillna(0).sum()) if column in rows.columns else 0
    tp = result["true_positive"]
    predicted = result["predicted_count"]
    gold = tp + result["false_negative"]
    result["precision"] = _safe_rate(tp, predicted)
    result["recall"] = _safe_rate(tp, gold)
    result["f1"] = _f1(result["precision"], result["recall"])
    if "precision" in rows.columns and predicted == 0:
        result["precision"] = float(pd.to_numeric(rows["precision"], errors="coerce").fillna(0).max())
    if "recall" in rows.columns and gold == 0:
        result["recall"] = float(pd.to_numeric(rows["recall"], errors="coerce").fillna(0).max())
    if "f1" in rows.columns and result["f1"] == 0:
        result["f1"] = float(pd.to_numeric(rows["f1"], errors="coerce").fillna(0).max())
    return result


def _candidate_recall(frame: pd.DataFrame, dataset: pd.DataFrame, rule_id: str) -> float:
    if not frame.empty and "rule_id" in frame.columns and "candidate_recall" in frame.columns:
        rows = frame[frame["rule_id"].map(normalize_rule_id) == rule_id]
        if not rows.empty:
            return float(pd.to_numeric(rows["candidate_recall"], errors="coerce").fillna(0).max())
    if not dataset.empty and "rule_id" in dataset.columns:
        rows = dataset[dataset["rule_id"].map(normalize_rule_id) == rule_id]
        if not rows.empty and "candidate_present" in rows.columns:
            return float(rows["candidate_present"].astype(str).str.lower().isin(["true", "1"]).mean())
    return 0.0


def _gap_coverage(frame: pd.DataFrame, rule_id: str) -> float:
    if frame.empty or "rule_id" not in frame.columns:
        return 0.0
    rows = frame[frame["rule_id"].map(normalize_rule_id) == rule_id]
    if rows.empty:
        return 0.0
    for column in ("gap_candidate_recall", "current_gap_coverage"):
        if column in rows.columns:
            return float(pd.to_numeric(rows[column], errors="coerce").fillna(0).max())
    return 0.0


def _dataset_rule_counts(dataset: pd.DataFrame) -> dict[str, int]:
    if dataset.empty or "rule_id" not in dataset.columns:
        return {}
    positive = dataset[dataset.get("source_type", pd.Series(dtype=str)).astype(str) != "matrix_hard_negative"]
    return {normalize_rule_id(key): int(value) for key, value in Counter(positive["rule_id"].map(normalize_rule_id)).items()}


def _inventory_groups(inventory: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in inventory.to_dict("records"):
        group = str(row.get("group", row.get("matrix_id", "")))
        for rule_id in _parse_rule_ids(row.get("listed_rule_ids")):
            result.setdefault(rule_id, group)
    return result


def _inventory_rule_ids(inventory: pd.DataFrame) -> set[str]:
    values: set[str] = set()
    for value in inventory.get("listed_rule_ids", []):
        values.update(_parse_rule_ids(value))
    return values


def _frame_rule_ids(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "rule_id" not in frame.columns:
        return set()
    return {normalize_rule_id(value) for value in frame["rule_id"] if not pd.isna(value)}


def _edit_count(frame: pd.DataFrame, rule_id: str) -> int:
    if frame.empty or "rule_id" not in frame.columns:
        return 0
    return int((frame["rule_id"].map(normalize_rule_id) == rule_id).sum())


def _clean_overcorrection_count(clean_over: pd.DataFrame, accepted: pd.DataFrame, dataset: pd.DataFrame, rule_id: str) -> int:
    if clean_over.empty:
        return 0
    if "row_id" not in accepted.columns or "rule_id" not in accepted.columns:
        return 0
    clean_row_ids = set(_hard_negative_row_ids(dataset))
    accepted_rule_rows = set(
        int(value)
        for value in accepted[accepted["rule_id"].map(normalize_rule_id) == rule_id].get("row_id", [])
        if not pd.isna(value)
    )
    return len(clean_row_ids & accepted_rule_rows) if clean_row_ids else 0


def _dirty_worse_count(dirty_worse: pd.DataFrame, accepted: pd.DataFrame, dataset: pd.DataFrame, rule_id: str) -> int:
    del accepted, dataset
    if dirty_worse.empty:
        return 0
    if "rule_id" in dirty_worse.columns:
        return int((dirty_worse["rule_id"].map(normalize_rule_id) == rule_id).sum())
    return 0


def _hard_negative_row_ids(dataset: pd.DataFrame) -> list[int]:
    if dataset.empty or "source_type" not in dataset.columns:
        return []
    return [
        int(index)
        for index, row in dataset.iterrows()
        if str(row.get("source_type", "")) == "matrix_hard_negative"
    ]


def _filter_edit_rows_by_row_id(frame: pd.DataFrame, row_ids: set[int]) -> pd.DataFrame:
    if frame.empty or "row_id" not in frame.columns or not row_ids:
        return pd.DataFrame(columns=frame.columns)
    numeric_row_ids = pd.to_numeric(frame["row_id"], errors="coerce")
    return frame[numeric_row_ids.isin(row_ids)]


def _top_rejection_reasons(frame: pd.DataFrame, rule_id: str) -> str:
    if frame.empty or "rule_id" not in frame.columns or "reason" not in frame.columns:
        return ""
    rows = frame[frame["rule_id"].map(normalize_rule_id) == rule_id]
    counts = Counter(str(value) for value in rows["reason"] if not pd.isna(value))
    return "; ".join(f"{reason}:{count}" for reason, count in counts.most_common(3))


def _required_work(row: dict[str, Any], eval_decision: str) -> tuple[str, str, str, int]:
    executable_status = str(row.get("executable_status", row.get("status", "")))
    requires = " ".join(_parse_jsonish_list(row.get("requires")))
    support = str(row.get("candidate_generator_support", "")).lower()
    if eval_decision == "READY_NEXT_DATASET":
        return "add_real_data", "ready for dataset expansion", "low", 1
    if eval_decision == "NEEDS_THRESHOLD":
        return "threshold_calibration", "threshold suppresses accepted edits", "medium", 2
    if eval_decision == "NEEDS_MORE_TRAINING":
        return "training", "model score distribution is weak", "medium", 2
    if eval_decision == "NEEDS_VALIDATOR":
        return "add_validator_guard", "clean FP or dirty worse risk", "medium", 3
    if "NEEDS_NER" in executable_status or "ner" in requires:
        return "add_NER_support", "NER-backed spans required", "high", 5
    if "NEEDS_SYNTAX" in executable_status or "syntax" in requires:
        return "add_syntax_features", "syntax-backed validation required", "high", 5
    if "NEEDS_DICTIONARY" in executable_status or "dictionary" in requires:
        if support != "true":
            return "add_dictionary_support", "dictionary-backed candidate path missing", "high", 5
    if executable_status in {"NO_CANDIDATE_PATH", "NEEDS_RULE_IMPLEMENTATION", "PLANNED_ONLY"} or support != "true":
        return "add_candidate_generator", "no executable candidate path", "medium", 4
    if executable_status == "EXECUTABLE_INACTIVE":
        return "add_synthetic_corruption", "candidate path exists but eval/data backfill is missing", "medium", 1
    return "add_real_data", "needs matrix data expansion", "low", 1


def _recommended_prompt(rule_id: str, required_work: str) -> str:
    return f"Implement {required_work} for {rule_id} with candidate-backed tests and matrix eval audit only."


def _rule_list(frame: pd.DataFrame, column: str, value: str) -> list[str]:
    if frame.empty or column not in frame.columns:
        return []
    if column == "required_work":
        selected = frame[frame[column].astype(str).str.contains(value, regex=False, na=False)]
    else:
        selected = frame[frame[column].astype(str) == value]
    rule_column = "rule_id" if "rule_id" in selected.columns else "matrix_id"
    return sorted(str(item) for item in selected.get(rule_column, []) if str(item))


def _md_list(values: list[str]) -> list[str]:
    if not values:
        return ["- none", ""]
    return [*[f"- {value}" for value in values], ""]


def _metadata_traps(value: Any) -> list[str]:
    metadata = _parse_jsonish_dict(value)
    traps = metadata.get("trap_types") or metadata.get("trap_type") or []
    if isinstance(traps, str):
        return [traps]
    if isinstance(traps, list):
        return [str(item) for item in traps if str(item)]
    return []


def _related_rule_for_trap(trap_type: str) -> str:
    if "tsya" in trap_type:
        return "tsya_soft_insert"
    if "n_nn" in trap_type:
        return "n_nn_adjective"
    if "ne_ni" in trap_type:
        return "ne_verb"
    if "hyphen" in trap_type:
        return "hyphen_whitelist"
    if "abbreviation" in trap_type:
        return "abbreviation_case_protection"
    if "punctuation" in trap_type or "quotes" in trap_type:
        return "punctuation_delete_replace"
    return ""


def _meta_template_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return dataset
    patterns = (
        "правило серии",
        "проверяет семейство",
        "готовит важный примере",
        "готовит итоговый примере",
        "готовит точный примере",
        "готовит рабочий примере",
        "metadata_only",
    )
    mask = pd.Series(False, index=dataset.index)
    for column in ("source", "target", "template_id", "metadata"):
        if column not in dataset.columns:
            continue
        text = dataset[column].astype(str).str.lower()
        for pattern in patterns:
            mask = mask | text.str.contains(pattern, regex=False, na=False)
    return dataset[mask]


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
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    if pd.isna(value):
        return []
    return [value]


def _parse_jsonish_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _read_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
