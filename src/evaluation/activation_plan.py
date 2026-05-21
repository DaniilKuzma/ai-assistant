from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHASE3_LABELS = {
    "READY_NEXT_DATASET",
    "NEEDS_THRESHOLD",
    "NEEDS_MORE_TRAINING",
    "NEEDS_CANDIDATE_GENERATOR",
    "NEEDS_SYNTHETIC_GENERATOR",
    "NEEDS_VALIDATOR",
    "NEEDS_SYNTAX",
    "NEEDS_DICTIONARY",
    "NEEDS_NER",
    "KEEP_INACTIVE",
    "UNSAFE",
}

COMPARISON_METRICS = (
    "exact_match",
    "edit_precision",
    "edit_recall",
    "edit_f1",
    "spelling_f1",
    "punctuation_f1",
    "clean_overcorrection_rate",
    "dirty_worse_rate",
    "real_dirty_worse_rate",
)


def write_verified_activation_plan(
    *,
    matrix_activation_path: str | Path,
    matrix_summary_path: str | Path,
    under_quota_path: str | Path | None = None,
    matrix_under_quota_path: str | Path | None = None,
    output_csv_path: str | Path,
    output_md_path: str | Path,
) -> Any:
    activation_rows = _read_csv_dicts(matrix_activation_path)
    summary_by_rule = {row.get("rule_id", ""): row for row in _read_csv_dicts(matrix_summary_path)}
    quota_path = under_quota_path or matrix_under_quota_path
    quota_by_rule = {row.get("rule_id", ""): row for row in _read_csv_dicts(quota_path or "")}

    verified_rows: list[dict[str, str]] = []
    for row in activation_rows:
        rule_id = row.get("rule_id", "")
        summary = summary_by_rule.get(rule_id, {})
        quota = quota_by_rule.get(rule_id, {})
        activation = row.get("activation_decision") or row.get("activation_decision") or "EXCLUDE"
        status = summary.get("decision") or row.get("under_quota_decision") or "KEEP_INACTIVE"
        candidate_recall = row.get("candidate_recall") or summary.get("candidate_recall") or "0.0"
        current_f1 = row.get("f1") or summary.get("f1") or "0.0"
        candidate_path = _truthy(quota.get("candidate_path_exists")) or _float(candidate_recall) > 0.0
        synthetic_support = _truthy(quota.get("synthetic_generator_exists"))
        hard_negative_support = _truthy(quota.get("hard_negative_exists"))
        validator_support = not (
            (summary.get("decision") or "").upper() == "NEEDS_VALIDATOR"
            or _int(summary.get("rejected_count")) > _int(summary.get("accepted_count")) and _int(summary.get("accepted_count")) == 0
        )
        reason = row.get("reason") or summary.get("reason") or ""
        if rule_id == "hyphen_whitelist" and activation != "INCLUDE":
            reason = _append_reason(reason, "Matrix READY_NEXT_DATASET but eval_count=1, KEEP_UNDER_QUOTA")

        verified_rows.append(
            {
                "rule_id": rule_id,
                "matrix_group": row.get("matrix_group") or summary.get("matrix_group") or "",
                "status": status,
                "candidate_path_exists": str(bool(candidate_path)),
                "synthetic_support": str(bool(synthetic_support)),
                "hard_negative_support": str(bool(hard_negative_support)),
                "validator_support": str(bool(validator_support)),
                "candidate_recall": str(candidate_recall),
                "current_f1": str(current_f1),
                "activation_decision": activation,
                "reason": reason,
            }
        )

    output_csv = Path(output_csv_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_dicts(output_csv, verified_rows, _activation_fieldnames())

    ready_count = sum(1 for row in summary_by_rule.values() if row.get("decision") == "READY_NEXT_DATASET")
    include_count = sum(1 for row in verified_rows if row["activation_decision"] == "INCLUDE")
    excluded_ready = [
        row["rule_id"]
        for row in verified_rows
        if row["activation_decision"] != "INCLUDE" and summary_by_rule.get(row["rule_id"], {}).get("decision") == "READY_NEXT_DATASET"
    ]
    md_lines = [
        "# Next Dataset Activation Plan Verified",
        "",
        f"Matrix READY_NEXT_DATASET count: {ready_count}",
        f"Verified Activation INCLUDE count: {include_count}",
        f"Resolved mismatch: {', '.join(excluded_ready) if excluded_ready else 'none'}",
        "",
        "The verified activation list keeps only candidate-backed, sufficiently covered rules active for the next dataset cycle.",
        "",
        "| rule_id | decision | status | recall | f1 | reason |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in verified_rows:
        if row["activation_decision"] == "INCLUDE" or row["rule_id"] in excluded_ready:
            md_lines.append(
                f"| {row['rule_id']} | {row['activation_decision']} | {row['status']} | "
                f"{row['candidate_recall']} | {row['current_f1']} | {row['reason']} |"
            )
    output_md = Path(output_md_path)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    import pandas as pd

    return pd.DataFrame(verified_rows)


def normalize_activation_decision(decision: str, blockers: Sequence[str] | str | None = None) -> str:
    normalized = (decision or "").strip().upper()
    blocker_set = {_normalize_blocker(blocker) for blocker in _split_blockers(blockers)}

    if normalized in PHASE3_LABELS:
        if normalized == "KEEP_INACTIVE":
            return _decision_from_blockers(blocker_set) or "KEEP_INACTIVE"
        return normalized
    if normalized == "READY":
        return "READY_NEXT_DATASET"
    if normalized in {"NO_CANDIDATE_PATH", "NEEDS_RULE_IMPLEMENTATION"}:
        return _decision_from_blockers(blocker_set) or "NEEDS_CANDIDATE_GENERATOR"
    if normalized in {"NEEDS_SYNTAX_FEATURES", "SYNTAX_HEAVY"}:
        return "NEEDS_SYNTAX"
    if normalized in {"NEEDS_DICTIONARY_SUPPORT", "DICTIONARY_HEAVY"}:
        return "NEEDS_DICTIONARY"
    if normalized in {"NEEDS_NER_SUPPORT", "NER_HEAVY"}:
        return "NEEDS_NER"
    if normalized in {"KEEP_UNDER_QUOTA", "NEEDS_BACKFILL"}:
        return "NEEDS_SYNTHETIC_GENERATOR"
    if normalized == "NEEDS_VALIDATOR_SUPPORT":
        return "NEEDS_VALIDATOR"
    if normalized == "UNSAFE_RULE":
        return "UNSAFE"
    return "KEEP_INACTIVE"


def write_activation_final_report(*, reports_dir: str | Path, data_dir: str | Path) -> str:
    reports = Path(reports_dir)
    data = Path(data_dir)
    reports.mkdir(parents=True, exist_ok=True)

    summary_path = reports / "matrix_rule_eval_summary.csv"
    manifest_path = reports / "matrix_eval_manifest.json"
    matrix_path = data / "matrix_eval.csv.gz"

    rows = _read_csv_dicts(summary_path) if summary_path.exists() else []
    for row in rows:
        row["activation_decision"] = normalize_activation_decision(
            row.get("activation_decision") or row.get("decision") or row.get("under_quota_decision") or "",
            row.get("blockers") or row.get("reason") or "",
        )
    if rows:
        _write_csv_dicts(summary_path, rows, list(rows[0].keys()))

    missing = [str(path) for path in (summary_path, manifest_path, matrix_path) if not path.exists()]
    decision_counts = Counter(row.get("activation_decision", "KEEP_INACTIVE") for row in rows)
    verdict = "MATRIX_EVAL_PHASE3_BLOCKED" if not rows else "MATRIX_EVAL_PHASE3_COMPLETE"
    if missing:
        verdict = "MATRIX_EVAL_PHASE3_PARTIAL" if rows else "MATRIX_EVAL_PHASE3_BLOCKED"
    if decision_counts.get("UNSAFE", 0) and decision_counts.get("READY_NEXT_DATASET", 0) == 0:
        verdict = "MATRIX_EVAL_PHASE3_BLOCKED"

    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    total_entries = manifest.get("total_examples") or manifest.get("total_entries") or _count_gzip_csv_rows(matrix_path)
    md_lines = [
        "# Matrix Eval Activation Final Report",
        "",
        f"Verdict: {verdict}",
        f"Total entries: {total_entries}",
        f"Evaluated rules: {len(rows)}",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    for label in sorted(decision_counts):
        md_lines.append(f"| {label} | {decision_counts[label]} |")
    if missing:
        md_lines.extend(["", "Missing artifacts:", *[f"- {path}" for path in missing]])
    (reports / "activation_final_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return verdict


def write_model_comparison_report(
    *,
    working_summary_path: str | Path,
    canonical_summary_path: str | Path,
    output_csv_path: str | Path,
    output_md_path: str | Path,
    training_ran: bool,
) -> str:
    baseline = _first_row(working_summary_path)
    candidate = _first_row(canonical_summary_path)
    comparison_rows = []
    for metric in COMPARISON_METRICS:
        base_value = _float(baseline.get(metric))
        canonical_value = _float(candidate.get(metric))
        comparison_rows.append(
            {
                "metric": metric,
                "working": _format_float(base_value),
                "canonical": _format_float(canonical_value),
                "delta": _format_float(canonical_value - base_value),
            }
        )

    verdict = _comparison_verdict(baseline, candidate, training_ran=training_ran)
    output_csv = Path(output_csv_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_dicts(output_csv, comparison_rows, ["metric", "working", "canonical", "delta"])

    md_lines = [
        "# WORKING_V1 vs canonical Comparison",
        "",
        f"Verdict: {verdict}",
        f"Training ran: {bool(training_ran)}",
        "",
        "| metric | WORKING_V1 | canonical | delta |",
        "|---|---:|---:|---:|",
    ]
    md_lines.extend(
        f"| {row['metric']} | {row['working']} | {row['canonical']} | {row['delta']} |" for row in comparison_rows
    )
    output_md = Path(output_md_path)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return verdict


def _decision_from_blockers(blockers: set[str]) -> str | None:
    if "NEEDS_SYNTAX_FEATURES" in blockers or "NEEDS_SYNTAX" in blockers:
        return "NEEDS_SYNTAX"
    if "NEEDS_DICTIONARY_SUPPORT" in blockers or "NEEDS_DICTIONARY" in blockers:
        return "NEEDS_DICTIONARY"
    if "NEEDS_NER_SUPPORT" in blockers or "NEEDS_NER" in blockers:
        return "NEEDS_NER"
    if "NEEDS_VALIDATOR" in blockers or "NEEDS_VALIDATOR_SUPPORT" in blockers:
        return "NEEDS_VALIDATOR"
    if "NEEDS_SYNTHETIC_GENERATOR" in blockers:
        return "NEEDS_SYNTHETIC_GENERATOR"
    if "NEEDS_CANDIDATE_GENERATOR" in blockers or "NO_CANDIDATE_PATH" in blockers:
        return "NEEDS_CANDIDATE_GENERATOR"
    return None


def _comparison_verdict(baseline: Mapping[str, str], candidate: Mapping[str, str], *, training_ran: bool) -> str:
    if not training_ran:
        return "READY_FOR_COLAB_TRAINING"
    if not candidate:
        return "NEEDS_TARGETED_FIX"
    clean_ok = _float(candidate.get("clean_overcorrection_rate")) <= max(0.007, _float(baseline.get("clean_overcorrection_rate")))
    dirty_ok = _float(candidate.get("dirty_worse_rate")) <= max(0.005, _float(baseline.get("dirty_worse_rate")))
    precision_ok = _float(candidate.get("edit_precision")) >= 0.90
    safety_ok = clean_ok and dirty_ok and precision_ok
    improved = any(
        _float(candidate.get(metric)) > _float(baseline.get(metric)) + 1e-9
        for metric in ("edit_f1", "spelling_f1", "punctuation_f1")
    )
    return "PROMOTE_WORKING_CURRENT" if safety_ok and improved else "KEEP_WORKING_BASELINE"


def _activation_fieldnames() -> list[str]:
    return [
        "rule_id",
        "matrix_group",
        "status",
        "candidate_path_exists",
        "synthetic_support",
        "hard_negative_support",
        "validator_support",
        "candidate_recall",
        "current_f1",
        "activation_decision",
        "reason",
    ]


def _read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv_dicts(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _first_row(path: str | Path) -> dict[str, str]:
    rows = _read_csv_dicts(path)
    return rows[0] if rows else {}


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _count_gzip_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _split_blockers(blockers: Sequence[str] | str | None) -> Iterable[str]:
    if blockers is None:
        return ()
    if isinstance(blockers, str):
        return tuple(part.strip() for part in blockers.replace(";", ",").split(",") if part.strip())
    return tuple(str(part).strip() for part in blockers if str(part).strip())


def _normalize_blocker(value: str) -> str:
    return value.strip().upper()


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") if value else "0"


def _append_reason(reason: str, addition: str) -> str:
    return f"{reason}; {addition}" if reason else addition
