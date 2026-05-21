from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import json

import pandas as pd
import yaml

from src.rules.registry import rule_by_id
from src.rules.rule_ids import normalize_rule_id
from src.rules.syntax_synthetic import (
    RULE_TO_FAMILY,
    SUPPORTED_SYNTAX_RULE_IDS,
    build_syntax_candidate_recall,
    build_syntax_eval_examples,
    build_syntax_hard_negative_report,
    build_syntax_hard_negatives,
)


REPORT_COLUMNS = [
    "matrix_key",
    "title",
    "rule_id",
    "syntax_family",
    "candidate_path",
    "synthetic_support",
    "hard_negative_support",
    "candidate_recall",
    "training_eligible_now",
    "decision",
    "reason",
]
EXTRA_COLUMNS = [
    "source_section",
    "status",
    "entry_type",
    "coverage_bucket",
    "excluded_from_denominator",
]
INCLUDE_DECISIONS = {"INCLUDE_NOW", "INCLUDE_AFTER_THRESHOLD_CALIBRATION", "INCLUDE_AFTER_TRAINING"}
SUPPORTED_RULE_IDS = frozenset(SUPPORTED_SYNTAX_RULE_IDS)
BLOCK_REASON_BY_INVENTORY_ACTION = {
    "NEEDS_DICTIONARY": "needs_dictionary",
    "NEEDS_SEMANTIC_MODEL": "needs_semantic_model",
    "KEEP_METADATA_ONLY": "no_candidate_path",
    "BLOCK_FOR_NOW": "unsafe",
}


@dataclass(frozen=True)
class FinalSyntaxCapabilityResult:
    updated_config: dict[str, Any]
    rows: list[dict[str, Any]]
    eligible_rows: list[dict[str, Any]]
    blocked_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def build_final_syntax_capability(
    *,
    rules_config_path: str | Path = "configs/rules.yaml",
    syntax_inventory_path: str | Path = "reports/syntax_module/syntax_required_inventory.csv",
    clean_pool_path: str | Path = "data/processed/clean_sentence_pool.csv.gz",
    min_examples_per_rule: int = 50,
    min_hard_negatives_per_family: int = 50,
) -> FinalSyntaxCapabilityResult:
    rules_path = Path(rules_config_path)
    config = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    updated_config = deepcopy(config)

    examples = build_syntax_eval_examples(min_examples_per_rule=min_examples_per_rule, clean_pool_path=clean_pool_path)
    recall = build_syntax_candidate_recall(examples)
    hard_negatives = build_syntax_hard_negatives(
        min_examples_per_family=min_hard_negatives_per_family,
        clean_pool_path=clean_pool_path,
    )
    hard_report = build_syntax_hard_negative_report(hard_negatives)

    inventory = _inventory_by_key(syntax_inventory_path)
    support_by_rule = _support_by_rule(recall, hard_report, min_examples_per_rule)
    rows: list[dict[str, Any]] = []

    for section in ("orthography", "punctuation"):
        entries = updated_config.get(section, {})
        if not isinstance(entries, dict):
            continue
        for matrix_key, entry in entries.items():
            if not isinstance(entry, dict) or not _is_syntax_surface_entry(str(matrix_key), entry, inventory):
                continue
            entry_rows, update = _audit_entry(
                section=section,
                matrix_key=str(matrix_key),
                entry=entry,
                inventory=inventory.get(str(matrix_key), {}),
                support_by_rule=support_by_rule,
            )
            rows.extend(entry_rows)
            if update:
                entry.setdefault("dataset", {}).update(update)

    summary = _summary(updated_config, rows, hard_report)
    eligible_rows = [row for row in rows if bool(row["training_eligible_now"])]
    blocked_rows = [row for row in rows if not bool(row["training_eligible_now"])]
    return FinalSyntaxCapabilityResult(
        updated_config=updated_config,
        rows=rows,
        eligible_rows=eligible_rows,
        blocked_rows=blocked_rows,
        summary=summary,
    )


def write_final_syntax_capability_outputs(
    *,
    rules_config_path: str | Path = "configs/rules.yaml",
    reports_dir: str | Path = "reports/syntax_module",
    syntax_inventory_path: str | Path = "reports/syntax_module/syntax_required_inventory.csv",
    clean_pool_path: str | Path = "data/processed/clean_sentence_pool.csv.gz",
    min_examples_per_rule: int = 50,
    min_hard_negatives_per_family: int = 50,
    update_rules_yaml: bool = True,
) -> dict[str, str]:
    result = build_final_syntax_capability(
        rules_config_path=rules_config_path,
        syntax_inventory_path=syntax_inventory_path,
        clean_pool_path=clean_pool_path,
        min_examples_per_rule=min_examples_per_rule,
        min_hard_negatives_per_family=min_hard_negatives_per_family,
    )
    output = Path(reports_dir)
    output.mkdir(parents=True, exist_ok=True)

    capability_path = output / "final_syntax_capability.csv"
    eligible_path = output / "syntax_training_eligible_rules.csv"
    blocked_path = output / "syntax_blocked_rules.csv"
    summary_path = output / "final_syntax_capability_summary.md"

    _frame(result.rows).to_csv(capability_path, index=False)
    _frame(result.eligible_rows).to_csv(eligible_path, index=False)
    _frame(result.blocked_rows).to_csv(blocked_path, index=False)
    summary_path.write_text(_summary_markdown(result.summary, result.blocked_rows), encoding="utf-8")

    if update_rules_yaml:
        Path(rules_config_path).write_text(
            yaml.safe_dump(result.updated_config, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )

    return {
        "final_syntax_capability_summary": str(summary_path),
        "final_syntax_capability": str(capability_path),
        "syntax_training_eligible_rules": str(eligible_path),
        "syntax_blocked_rules": str(blocked_path),
    }


def _audit_entry(
    *,
    section: str,
    matrix_key: str,
    entry: dict[str, Any],
    inventory: dict[str, Any],
    support_by_rule: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    implementation = entry.get("implementation") or {}
    dataset = entry.get("dataset") or {}
    status = str(implementation.get("status") or "")
    requires = {str(item).lower() for item in implementation.get("requires", []) or []}
    rule_ids = [normalize_rule_id(item) for item in implementation.get("rule_ids", []) or []]
    supported_rule_ids = [rule_id for rule_id in rule_ids if rule_id in SUPPORTED_RULE_IDS]
    title = str(entry.get("title") or "")
    entry_type = str(entry.get("entry_type") or "")
    is_ner_blocked = status == "ner_required" or "ner" in requires
    is_metadata_blocked = _is_metadata_only_syntax_entry(entry, dataset, inventory)
    excluded = bool(is_ner_blocked or is_metadata_blocked or entry_type != "leaf_rule")
    rows: list[dict[str, Any]] = []

    if supported_rule_ids:
        eligible_rule_ids = [rule_id for rule_id in supported_rule_ids if _rule_is_supported(rule_id, support_by_rule)]
        for rule_id in supported_rule_ids:
            support = support_by_rule.get(rule_id, {})
            eligible = rule_id in eligible_rule_ids
            decision = _eligible_decision(dataset) if eligible else "BLOCK_NO_CANDIDATE"
            reason = "syntax eval support verified" if eligible else _blocked_reason(status, requires, inventory, dataset)
            rows.append(
                _row(
                    matrix_key=matrix_key,
                    title=title,
                    rule_id=rule_id,
                    syntax_family=str(support.get("syntax_family") or RULE_TO_FAMILY.get(rule_id, "")),
                    candidate_path=bool(support.get("candidate_path", False)),
                    synthetic_support=bool(support.get("synthetic_support", False)),
                    hard_negative_support=bool(support.get("hard_negative_support", False)),
                    candidate_recall=support.get("candidate_recall"),
                    training_eligible_now=eligible,
                    decision=decision,
                    reason=reason,
                    section=section,
                    status=status,
                    entry_type=entry_type,
                    coverage_bucket="syntax_plus_NER" if is_ner_blocked else "syntax_surface",
                    excluded_from_denominator=excluded,
                )
            )
        if eligible_rule_ids:
            min_recall = min(float(support_by_rule[rule_id]["candidate_recall"]) for rule_id in eligible_rule_ids)
            return rows, _eligible_dataset_update(dataset, min_recall)

    reason = _blocked_reason(status, requires, inventory, dataset)
    decision = _blocked_decision(reason, status, dataset)
    family = str(inventory.get("syntax_family") or _family_from_supported_rules(supported_rule_ids) or "")
    candidate_path = bool(supported_rule_ids and any(_candidate_path_exists(rule_id) for rule_id in supported_rule_ids))
    rows.append(
        _row(
            matrix_key=matrix_key,
            title=title,
            rule_id="",
            syntax_family=family,
            candidate_path=candidate_path,
            synthetic_support=False,
            hard_negative_support=False,
            candidate_recall=None,
            training_eligible_now=False,
            decision=decision,
            reason=reason,
            section=section,
            status=status,
            entry_type=entry_type,
            coverage_bucket="syntax_plus_NER" if is_ner_blocked else "syntax_surface",
            excluded_from_denominator=excluded,
        )
    )
    return rows, _blocked_dataset_update(reason, decision)


def _eligible_dataset_update(dataset: dict[str, Any], candidate_recall: float) -> dict[str, Any]:
    return {
        "reason": "syntax eval support verified",
        "training_eligible_now": True,
        "training_eligibility_decision": _eligible_decision(dataset),
        "training_eligibility_reason": "syntax eval support verified; candidate recall and hard-negative gates pass",
        "current_candidate_path": True,
        "current_synthetic_support": True,
        "current_hard_negative_support": True,
        "current_candidate_recall": round(candidate_recall, 4),
        "needs_before_training": [
            item
            for item in dataset.get("needs_before_training", []) or []
            if item not in {"synthetic_support", "hard_negatives", "candidate_recall"}
        ],
    }


def _blocked_dataset_update(reason: str, decision: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "training_eligible_now": False,
        "training_eligibility_decision": decision,
        "training_eligibility_reason": _blocked_reason_text(reason),
        "current_candidate_path": False,
        "current_synthetic_support": False,
        "current_hard_negative_support": False,
        "current_candidate_recall": None,
    }


def _eligible_decision(dataset: dict[str, Any]) -> str:
    current = str(dataset.get("training_eligibility_decision") or "")
    if current in {"INCLUDE_NOW", "INCLUDE_AFTER_TRAINING"}:
        return current
    return "INCLUDE_AFTER_THRESHOLD_CALIBRATION"


def _blocked_decision(reason: str, status: str, dataset: dict[str, Any]) -> str:
    if reason == "needs_dictionary":
        return "BLOCK_NEEDS_DICTIONARY"
    if reason == "needs_NER":
        return "BLOCK_NEEDS_NER"
    if reason == "no_candidate_path":
        current = str(dataset.get("training_eligibility_decision") or "")
        return "BLOCK_METADATA_ONLY" if current == "BLOCK_METADATA_ONLY" else "BLOCK_NO_CANDIDATE"
    if status == "planned":
        return "BLOCK_PLANNED"
    if status == "disabled":
        return "BLOCK_DISABLED"
    return "BLOCK_NEEDS_SYNTAX"


def _blocked_reason(status: str, requires: set[str], inventory: dict[str, Any], dataset: dict[str, Any]) -> str:
    if status == "ner_required" or "ner" in requires:
        return "needs_NER"
    action = str(inventory.get("recommended_action") or "")
    if action in BLOCK_REASON_BY_INVENTORY_ACTION:
        return BLOCK_REASON_BY_INVENTORY_ACTION[action]
    current_reason = str(dataset.get("reason") or "")
    current_decision = str(dataset.get("training_eligibility_decision") or "")
    if current_reason in {
        "no_candidate_path",
        "needs_semantic_model",
        "needs_dictionary",
        "needs_NER",
        "needs_decoder_constraint",
        "unsafe",
    }:
        return current_reason
    if "dictionary" in requires or current_decision == "BLOCK_NEEDS_DICTIONARY":
        return "needs_dictionary"
    if "decoder" in requires:
        return "needs_decoder_constraint"
    if status == "disabled":
        return "unsafe"
    return "no_candidate_path"


def _blocked_reason_text(reason: str) -> str:
    return {
        "no_candidate_path": "no bounded syntax candidate path is available",
        "needs_semantic_model": "semantic disambiguation is required before safe syntax candidate generation",
        "needs_dictionary": "dictionary-backed lexical or morphology support is required before training inclusion",
        "needs_NER": "NER/gazetteer support is required and excluded from syntax-module coverage",
        "needs_decoder_constraint": "decoder constraints are required before training inclusion",
        "unsafe": "hard-negative or policy risk blocks training inclusion",
    }.get(reason, reason)


def _support_by_rule(
    recall: pd.DataFrame,
    hard_report: pd.DataFrame,
    min_examples_per_rule: int,
) -> dict[str, dict[str, Any]]:
    hard_by_family = {str(row["syntax_family"]): row for row in hard_report.to_dict("records")}
    support: dict[str, dict[str, Any]] = {}
    for row in recall.to_dict("records"):
        rule_id = normalize_rule_id(row.get("rule_id", ""))
        family = str(row.get("syntax_family") or RULE_TO_FAMILY.get(rule_id, ""))
        hard = hard_by_family.get(family, {})
        recall_value = _float_or_none(row.get("candidate_recall"))
        gold_count = int(float(row.get("gold_count") or 0))
        hard_count = int(float(hard.get("hard_negative_count") or 0))
        unsafe_accepted = int(float(hard.get("unsafe_candidate_accepted_count") or 0))
        overcorrection = float(hard.get("hard_negative_overcorrection_rate") or 0.0)
        candidate_path = bool(rule_id in SUPPORTED_RULE_IDS and _candidate_path_exists(rule_id))
        synthetic_support = bool(gold_count >= min_examples_per_rule and recall_value is not None)
        hard_negative_support = bool(hard_count > 0 and unsafe_accepted == 0 and overcorrection == 0.0)
        support[rule_id] = {
            "syntax_family": family,
            "candidate_path": candidate_path,
            "synthetic_support": synthetic_support,
            "hard_negative_support": hard_negative_support,
            "candidate_recall": recall_value,
            "hard_negative_count": hard_count,
            "unsafe_candidate_accepted_count": unsafe_accepted,
            "hard_negative_overcorrection_rate": overcorrection,
        }
    return support


def _rule_is_supported(rule_id: str, support_by_rule: dict[str, dict[str, Any]]) -> bool:
    support = support_by_rule.get(rule_id, {})
    recall = support.get("candidate_recall")
    return bool(
        support.get("candidate_path")
        and support.get("synthetic_support")
        and support.get("hard_negative_support")
        and recall is not None
        and float(recall) >= 0.85
    )


def _summary(config: dict[str, Any], rows: list[dict[str, Any]], hard_report: pd.DataFrame) -> dict[str, Any]:
    entry_rows = _entry_level(rows)
    syntax_surface = [row for row in entry_rows if row["coverage_bucket"] == "syntax_surface" and not row["excluded_from_denominator"]]
    current_required = [
        row
        for row in entry_rows
        if row["status"] == "syntax_required" and row["entry_type"] == "leaf_rule" and not row["excluded_from_denominator"]
    ]
    total_required = [
        row
        for row in entry_rows
        if row["status"] == "syntax_required"
    ]
    ner_rows = [row for row in entry_rows if row["coverage_bucket"] == "syntax_plus_NER"]
    blockers = Counter(row["reason"] for row in syntax_surface if not row["training_eligible_now"])
    ner_blockers = Counter(row["reason"] for row in ner_rows if not row["training_eligible_now"])
    covered = sum(1 for row in syntax_surface if row["training_eligible_now"])
    current_covered = sum(1 for row in current_required if row["training_eligible_now"])
    unsafe_total = int(float(hard_report.get("unsafe_candidate_accepted_count", pd.Series(dtype=float)).sum())) if not hard_report.empty else 0
    overcorrection_max = (
        float(hard_report.get("hard_negative_overcorrection_rate", pd.Series(dtype=float)).max()) if not hard_report.empty else 0.0
    )
    coverage = _safe_percent(covered, len(syntax_surface))
    current_coverage = _safe_percent(current_covered, len(current_required))
    verdict = _verdict(coverage, unsafe_total, overcorrection_max, covered)
    return {
        "total_syntax_required_entries": len(total_required),
        "syntax_required_metadata_group_entries": sum(
            1
            for row in total_required
            if row["entry_type"] != "leaf_rule" or row["reason"] == "no_candidate_path" and row["decision"] == "BLOCK_METADATA_ONLY"
        ),
        "actionable_syntax_required_leaf_entries": len(current_required),
        "syntax_surface_total": len(syntax_surface),
        "syntax_surface_covered": covered,
        "syntax_surface_coverage_percent": coverage,
        "current_syntax_required_total": len(total_required),
        "current_syntax_required_covered": current_covered,
        "current_syntax_required_coverage_percent": current_coverage,
        "syntax_plus_NER_blocked_count": len([row for row in ner_rows if not row["training_eligible_now"]]),
        "syntax_plus_NER_blockers_by_reason": dict(sorted(ner_blockers.items())),
        "syntax_punctuation_covered_count": sum(
            1 for row in syntax_surface if row["source_section"] == "punctuation" and row["training_eligible_now"]
        ),
        "syntax_orthography_covered_count": sum(
            1 for row in syntax_surface if row["source_section"] == "orthography" and row["training_eligible_now"]
        ),
        "remaining_blockers_by_reason": dict(sorted(blockers.items())),
        "unsafe_hard_negative_accepted_count": unsafe_total,
        "hard_negative_overcorrection_rate_max": overcorrection_max,
        "verdict": verdict,
        "total_taxonomy_entries": sum(len(config.get(section, {}) or {}) for section in ("orthography", "punctuation")),
    }


def _entry_level(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["matrix_key"])].append(row)
    result: list[dict[str, Any]] = []
    for group in grouped.values():
        first = dict(group[0])
        eligible = [row for row in group if row["training_eligible_now"]]
        if eligible:
            first.update(eligible[0])
            first["training_eligible_now"] = True
        else:
            first["training_eligible_now"] = False
        result.append(first)
    return result


def _summary_markdown(summary: dict[str, Any], blocked_rows: list[dict[str, Any]]) -> str:
    ner_entries = sorted(
        {str(row["matrix_key"]) for row in blocked_rows if row["coverage_bucket"] == "syntax_plus_NER"}
    )
    blocker_lines = [
        f"- {reason}: {count}"
        for reason, count in sorted((summary.get("remaining_blockers_by_reason") or {}).items())
    ] or ["- none: 0"]
    ner_lines = [f"- {key}" for key in ner_entries] or ["- none"]
    lines = [
        "# Final Syntax Capability Summary",
        "",
        "This is an audit-only report. It does not build datasets, run training, change thresholds, or change checkpoints.",
        "",
        "## Summary",
        "",
        f"- total_syntax_required_entries: {summary['total_syntax_required_entries']}",
        f"- syntax_required metadata/group entries: {summary['syntax_required_metadata_group_entries']}",
        f"- actionable syntax_required leaf entries: {summary['actionable_syntax_required_leaf_entries']}",
        f"- syntax_surface_total: {summary['syntax_surface_total']}",
        f"- syntax_surface_covered: {summary['syntax_surface_covered']}",
        f"- syntax_surface_coverage_percent: {summary['syntax_surface_coverage_percent']:.2f}",
        f"- current_syntax_required_total: {summary['current_syntax_required_total']}",
        f"- current_syntax_required_covered: {summary['current_syntax_required_covered']}",
        f"- current_syntax_required_coverage_percent: {summary['current_syntax_required_coverage_percent']:.2f}",
        f"- syntax_plus_NER_blocked_count: {summary['syntax_plus_NER_blocked_count']}",
        f"- syntax punctuation covered count: {summary['syntax_punctuation_covered_count']}",
        f"- syntax orthography covered count: {summary['syntax_orthography_covered_count']}",
        f"- unsafe hard-negative accepted count: {summary['unsafe_hard_negative_accepted_count']}",
        f"- hard-negative overcorrection max: {summary['hard_negative_overcorrection_rate_max']:.4f}",
        "",
        "NER-required entries excluded from syntax coverage denominator; they are listed as separate blockers for a future NER/gazetteer module.",
        "",
        "## Remaining Blockers By Reason",
        "",
        *blocker_lines,
        "",
        "## Syntax Plus NER Blocked Entries",
        "",
        *ner_lines,
        "",
        f"Verdict: {summary['verdict']}",
        "",
    ]
    return "\n".join(lines)


def _verdict(coverage_percent: float, unsafe_total: int, overcorrection_max: float, covered: int) -> str:
    if covered == 0 or unsafe_total > 0 or overcorrection_max > 0.0:
        return "BLOCKED"
    if coverage_percent >= 50.0:
        return "SYNTAX_CAPABILITY_READY_FOR_DATASET"
    return "SYNTAX_CAPABILITY_PARTIAL"


def _is_syntax_surface_entry(matrix_key: str, entry: dict[str, Any], inventory: dict[str, dict[str, Any]]) -> bool:
    implementation = entry.get("implementation") or {}
    status = str(implementation.get("status") or "")
    requires = {str(item).lower() for item in implementation.get("requires", []) or []}
    rule_ids = {normalize_rule_id(item) for item in implementation.get("rule_ids", []) or []}
    return bool(status == "syntax_required" or "syntax" in requires or rule_ids & SUPPORTED_RULE_IDS or matrix_key in inventory)


def _is_metadata_only_syntax_entry(entry: dict[str, Any], dataset: dict[str, Any], inventory: dict[str, Any]) -> bool:
    return bool(
        str(entry.get("entry_type") or "") != "leaf_rule"
        or str(dataset.get("training_eligibility_decision") or "") == "BLOCK_METADATA_ONLY"
        or str(inventory.get("recommended_action") or "") == "KEEP_METADATA_ONLY"
    )


def _candidate_path_exists(rule_id: str) -> bool:
    rule = rule_by_id(rule_id)
    if rule is None:
        return False
    spec = getattr(rule, "spec", None)
    return bool(
        getattr(spec, "scope", "") == "punctuation_gap"
        or hasattr(rule, "generate_candidates")
        or hasattr(rule, "generate")
        or hasattr(rule, "generate_span")
    )


def _inventory_by_key(path: str | Path) -> dict[str, dict[str, Any]]:
    inventory_path = Path(path)
    if not inventory_path.exists():
        return {}
    frame = pd.read_csv(inventory_path)
    if "matrix_key" not in frame.columns:
        return {}
    return {str(row["matrix_key"]): row for row in frame.to_dict("records")}


def _row(
    *,
    matrix_key: str,
    title: str,
    rule_id: str,
    syntax_family: str,
    candidate_path: bool,
    synthetic_support: bool,
    hard_negative_support: bool,
    candidate_recall: float | None,
    training_eligible_now: bool,
    decision: str,
    reason: str,
    section: str,
    status: str,
    entry_type: str,
    coverage_bucket: str,
    excluded_from_denominator: bool,
) -> dict[str, Any]:
    row = {
        "matrix_key": matrix_key,
        "title": title,
        "rule_id": rule_id,
        "syntax_family": syntax_family,
        "candidate_path": bool(candidate_path),
        "synthetic_support": bool(synthetic_support),
        "hard_negative_support": bool(hard_negative_support),
        "candidate_recall": candidate_recall,
        "training_eligible_now": bool(training_eligible_now),
        "decision": decision,
        "reason": reason,
        "source_section": section,
        "status": status,
        "entry_type": entry_type,
        "coverage_bucket": coverage_bucket,
        "excluded_from_denominator": bool(excluded_from_denominator),
    }
    return {column: row.get(column) for column in [*REPORT_COLUMNS, *EXTRA_COLUMNS]}


def _family_from_supported_rules(rule_ids: Iterable[str]) -> str:
    families = sorted({RULE_TO_FAMILY.get(rule_id, "") for rule_id in rule_ids if RULE_TO_FAMILY.get(rule_id, "")})
    return "|".join(families)


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[*REPORT_COLUMNS, *EXTRA_COLUMNS])


def _safe_percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((float(numerator) / float(denominator)) * 100.0, 2)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final syntax capability audit without building data or training.")
    parser.add_argument("--rules", default="configs/rules.yaml")
    parser.add_argument("--reports-dir", default="reports/syntax_module")
    parser.add_argument("--syntax-inventory", default="reports/syntax_module/syntax_required_inventory.csv")
    parser.add_argument("--clean-pool", default="data/processed/clean_sentence_pool.csv.gz")
    parser.add_argument("--min-examples-per-rule", type=int, default=50)
    parser.add_argument("--min-hard-negatives-per-family", type=int, default=50)
    parser.add_argument("--no-update-rules-yaml", action="store_true")
    args = parser.parse_args()

    outputs = write_final_syntax_capability_outputs(
        rules_config_path=args.rules,
        reports_dir=args.reports_dir,
        syntax_inventory_path=args.syntax_inventory,
        clean_pool_path=args.clean_pool,
        min_examples_per_rule=args.min_examples_per_rule,
        min_hard_negatives_per_family=args.min_hard_negatives_per_family,
        update_rules_yaml=not args.no_update_rules_yaml,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
