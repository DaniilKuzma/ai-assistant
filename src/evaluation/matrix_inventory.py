from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.rules.registry import all_rules, rule_by_id
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id


INVENTORY_COLUMNS = [
    "section",
    "matrix_id",
    "orfogrammka_id",
    "parent_group",
    "group",
    "title",
    "status_from_yaml",
    "requires",
    "listed_rule_ids",
    "registry_rule_exists",
    "candidate_generator_support",
    "synthetic_corruption_support",
    "validator_guard_support",
    "current_dataset_count",
    "current_eval_count",
    "current_candidate_recall",
    "current_gap_coverage",
    "current_rule_precision",
    "current_rule_recall",
    "current_rule_f1",
    "current_predicted_count",
    "current_false_positive_count",
    "current_false_negative_count",
    "executable_status",
    "decision",
    "decision_reason",
]

ALIAS_AUDIT_COLUMNS = ["raw_rule_id", "canonical_rule_id", "source", "action", "notes"]

EXECUTABLE_YAML_STATUSES = {"implemented", "partial", "deterministic", "candidate_only"}
PLANNED_YAML_STATUSES = {"planned"}
METADATA_YAML_STATUSES = {"model_required", "syntax_required", "dictionary_model_required", "ner_required"}
REGISTRY_EXECUTABLE_MODES = {"deterministic", "candidate_only", "model_required"}
DICT_CANDIDATE_IDS = {
    "dictionary_fuzzy",
    "double_consonant_candidate",
    "keyboard_typo_candidate",
    "swapped_letters_candidate",
    "missing_letter_candidate",
    "extra_letter_candidate",
    "yo_e_candidate",
}
VALIDATOR_GUARD_RULE_IDS = {
    "capitalization_ner",
    "abbreviation_case_protection",
    "capitalization_sentence_start",
    "quote_open",
    "quote_close",
    "quote_pair_balance",
    "bracket_pair_balance",
    "punctuation_delete_replace",
    "dictionary_fuzzy",
    "double_consonant_candidate",
    "keyboard_typo_candidate",
    "swapped_letters_candidate",
    "missing_letter_candidate",
    "extra_letter_candidate",
}


def build_rule_matrix_inventory(
    *,
    rules_config_path: str | Path = "configs/rules.yaml",
    reports_dir: str | Path = "reports/eval_test_calibrated_v3_short_v2",
    dataset_manifest_path: str | Path = "reports/short_dataset_v2/dataset_manifest.json",
    current_dataset_path: str | Path = "data/processed/short_dataset_v2/correction_dataset.csv.gz",
) -> pd.DataFrame:
    """Build a non-production inventory for every coverage matrix group."""

    coverage = load_rules_coverage(rules_config_path)
    registry_rule_ids = {rule.spec.id for rule in all_rules()}
    synthetic_enabled = set(str(rule_id) for rule_id in coverage.get("synthetic_generation", {}).get("enabled", {}))
    synthetic_disabled = set(str(rule_id) for rule_id in coverage.get("synthetic_generation", {}).get("disabled", {}))
    dataset_counts, active_ids, excluded_ids = _current_dataset_state(dataset_manifest_path, current_dataset_path)
    report_state = _report_state(reports_dir)

    rows: list[dict[str, Any]] = []
    for section, group, entry in iter_coverage_entries(coverage):
        raw_rule_ids = [str(rule_id) for rule_id in entry.get("rules", [])]
        rule_ids = [normalize_rule_id(rule_id) for rule_id in raw_rule_ids]
        requires = [str(item) for item in entry.get("requires", [])]
        status = str(entry.get("status", ""))
        registry_exists = bool(rule_ids) and all(rule_id in registry_rule_ids for rule_id in rule_ids)
        candidate_support = _candidate_support(rule_ids)
        synthetic_support = _synthetic_support(rule_ids, synthetic_enabled, synthetic_disabled)
        validator_support = _validator_support(rule_ids, requires)
        current_dataset_count = sum(dataset_counts.get(rule_id, 0) for rule_id in rule_ids)
        metrics = _combined_metrics(rule_ids, report_state)
        executable_status, decision, reason = _classify_inventory_row(
            status=status,
            rule_ids=rule_ids,
            requires=requires,
            registry_exists=registry_exists,
            candidate_support=candidate_support,
            synthetic_support=synthetic_support,
            active_ids=active_ids,
            excluded_ids=excluded_ids,
            current_dataset_count=current_dataset_count,
            metrics=metrics,
        )
        rows.append(
            {
                "section": section,
                "matrix_id": group,
                "orfogrammka_id": str(entry.get("orfogrammka_id", "")),
                "parent_group": str(entry.get("parent_group", "")),
                "group": group,
                "title": str(entry.get("title", "")),
                "status_from_yaml": status,
                "requires": _json_list(requires),
                "listed_rule_ids": _json_list(rule_ids),
                "registry_rule_exists": registry_exists,
                "candidate_generator_support": candidate_support,
                "synthetic_corruption_support": synthetic_support,
                "validator_guard_support": validator_support,
                "current_dataset_count": current_dataset_count,
                "current_eval_count": metrics["current_eval_count"],
                "current_candidate_recall": metrics["current_candidate_recall"],
                "current_gap_coverage": metrics["current_gap_coverage"],
                "current_rule_precision": metrics["current_rule_precision"],
                "current_rule_recall": metrics["current_rule_recall"],
                "current_rule_f1": metrics["current_rule_f1"],
                "current_predicted_count": metrics["current_predicted_count"],
                "current_false_positive_count": metrics["current_false_positive_count"],
                "current_false_negative_count": metrics["current_false_negative_count"],
                "executable_status": executable_status,
                "decision": decision,
                "decision_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def write_rule_matrix_inventory_outputs(
    *,
    output_dir: str | Path = "reports/matrix_eval",
    rules_config_path: str | Path = "configs/rules.yaml",
    reports_dir: str | Path = "reports/eval_test_calibrated_v3_short_v2",
    dataset_manifest_path: str | Path = "reports/short_dataset_v2/dataset_manifest.json",
    current_dataset_path: str | Path = "data/processed/short_dataset_v2/correction_dataset.csv.gz",
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inventory = build_rule_matrix_inventory(
        rules_config_path=rules_config_path,
        reports_dir=reports_dir,
        dataset_manifest_path=dataset_manifest_path,
        current_dataset_path=current_dataset_path,
    )
    inventory_path = output / "rule_matrix_inventory.csv"
    summary_path = output / "rule_matrix_inventory_summary.md"
    alias_path = output / "rule_id_alias_audit.csv"
    inventory.to_csv(inventory_path, index=False)

    matrix_rule_ids = _matrix_rule_ids(rules_config_path)
    registry_rule_ids = sorted({rule.spec.id for rule in all_rules()})
    source_rule_ids = {
        "matrix": matrix_rule_ids,
        "registry": registry_rule_ids,
        "reports": _report_rule_ids(reports_dir),
        "dataset": sorted(_current_dataset_state(dataset_manifest_path, current_dataset_path)[0]),
    }
    alias_audit = build_rule_id_alias_audit(
        matrix_rule_ids=matrix_rule_ids,
        registry_rule_ids=registry_rule_ids,
        source_rule_ids=source_rule_ids,
    )
    alias_audit.to_csv(alias_path, index=False)
    _write_summary(summary_path, inventory, alias_audit)
    return {
        "inventory_path": str(inventory_path),
        "summary_path": str(summary_path),
        "alias_audit_path": str(alias_path),
    }


def build_rule_id_alias_audit(
    *,
    matrix_rule_ids: Iterable[str],
    registry_rule_ids: Iterable[str],
    source_rule_ids: dict[str, Iterable[str]],
) -> pd.DataFrame:
    matrix = {normalize_rule_id(rule_id) for rule_id in matrix_rule_ids if normalize_rule_id(rule_id) != UNKNOWN_RULE_ID}
    registry = {normalize_rule_id(rule_id) for rule_id in registry_rule_ids if normalize_rule_id(rule_id) != UNKNOWN_RULE_ID}
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, raw_values in source_rule_ids.items():
        for raw in raw_values:
            raw_rule_id = str(raw)
            key = (source, raw_rule_id)
            if key in seen:
                continue
            seen.add(key)
            canonical = normalize_rule_id(raw_rule_id)
            if canonical == UNKNOWN_RULE_ID:
                action = "unknown"
                notes = "legacy or external unknown row; not eligible as active target"
            elif canonical != raw_rule_id:
                action = "alias"
                notes = f"normalized alias {raw_rule_id} -> {canonical}"
            elif canonical in matrix or canonical in registry:
                action = "canonical"
                notes = ""
            else:
                action = "missing_in_matrix"
                notes = "source id has no matrix taxonomy row"
            rows.append(
                {
                    "raw_rule_id": raw_rule_id,
                    "canonical_rule_id": canonical,
                    "source": source,
                    "action": action,
                    "notes": notes,
                }
            )
    return pd.DataFrame(rows, columns=ALIAS_AUDIT_COLUMNS)


def _candidate_support(rule_ids: list[str]) -> str:
    if not rule_ids:
        return "false"
    statuses: list[str] = []
    for rule_id in rule_ids:
        rule = rule_by_id(rule_id)
        if rule is None:
            statuses.append("false")
            continue
        spec = getattr(rule, "spec", None)
        scope = getattr(spec, "scope", "")
        mode = getattr(spec, "mode", "")
        if rule_id in DICT_CANDIDATE_IDS:
            statuses.append("true")
        elif scope == "punctuation_gap":
            statuses.append("true")
        elif scope == "token" and (hasattr(rule, "generate_candidates") or hasattr(rule, "generate")):
            statuses.append("true")
        elif scope == "span" and hasattr(rule, "generate_span"):
            statuses.append("true")
        elif mode in REGISTRY_EXECUTABLE_MODES:
            statuses.append("unknown")
        else:
            statuses.append("false")
    if all(status == "true" for status in statuses):
        return "true"
    if any(status == "true" for status in statuses):
        return "unknown"
    if any(status == "unknown" for status in statuses):
        return "unknown"
    return "false"


def _synthetic_support(rule_ids: list[str], synthetic_enabled: set[str], synthetic_disabled: set[str]) -> str:
    if not rule_ids:
        return "false"
    statuses: list[str] = []
    for rule_id in rule_ids:
        rule = rule_by_id(rule_id)
        if rule_id in synthetic_enabled:
            statuses.append("true")
        elif rule_id in synthetic_disabled:
            statuses.append("false")
        elif rule is not None and (hasattr(rule, "generate_corruptions") or hasattr(rule, "transformations")):
            statuses.append("true")
        elif rule is not None:
            statuses.append("unknown")
        else:
            statuses.append("false")
    if all(status == "true" for status in statuses):
        return "true"
    if any(status == "true" for status in statuses):
        return "unknown"
    if any(status == "unknown" for status in statuses):
        return "unknown"
    return "false"


def _validator_support(rule_ids: list[str], requires: list[str]) -> str:
    if "validator" in requires or any(rule_id in VALIDATOR_GUARD_RULE_IDS for rule_id in rule_ids):
        return "true"
    if any(req in {"ner", "syntax", "dictionary"} for req in requires):
        return "unknown"
    return "unknown"


def _classify_inventory_row(
    *,
    status: str,
    rule_ids: list[str],
    requires: list[str],
    registry_exists: bool,
    candidate_support: str,
    synthetic_support: str,
    active_ids: set[str],
    excluded_ids: set[str],
    current_dataset_count: int,
    metrics: dict[str, Any],
) -> tuple[str, str, str]:
    if status in PLANNED_YAML_STATUSES:
        if "dictionary" in requires or "morphology" in requires:
            return "PLANNED_ONLY", "BACKLOG_IMPLEMENTATION", "planned taxonomy row needs implementation/data path"
        return "PLANNED_ONLY", "BACKLOG_DATA", "planned taxonomy row has no executable target"
    if status in METADATA_YAML_STATUSES and candidate_support != "true":
        if "ner" in requires:
            return "NEEDS_NER", "BACKLOG_IMPLEMENTATION", "metadata row requires NER-backed candidate path"
        if "syntax" in requires:
            return "NEEDS_SYNTAX", "BACKLOG_IMPLEMENTATION", "metadata row requires syntax-backed candidate path"
        if "dictionary" in requires:
            return "NEEDS_DICTIONARY", "BACKLOG_IMPLEMENTATION", "metadata row requires dictionary-backed candidate path"
        return "METADATA_ONLY", "BACKLOG_DATA", "metadata-only row has no executable candidate path"
    if rule_ids and not registry_exists:
        return "NEEDS_RULE_IMPLEMENTATION", "BACKLOG_IMPLEMENTATION", "matrix lists rule_id that is absent from registry"
    if candidate_support != "true":
        if "ner" in requires:
            return "NEEDS_NER", "BACKLOG_IMPLEMENTATION", "no verified candidate path and rule requires NER"
        if "syntax" in requires:
            return "NEEDS_SYNTAX", "BACKLOG_IMPLEMENTATION", "no verified candidate path and rule requires syntax"
        if "dictionary" in requires:
            return "NEEDS_DICTIONARY", "BACKLOG_IMPLEMENTATION", "no verified candidate path and rule requires dictionary"
        if rule_ids:
            return "NO_CANDIDATE_PATH", "BACKLOG_IMPLEMENTATION", "registry metadata exists but candidate generation is not executable"
        return "METADATA_ONLY", "BACKLOG_DATA", "no listed executable rule_id"
    if synthetic_support == "false" and current_dataset_count == 0:
        executable = "EXECUTABLE_INACTIVE" if set(rule_ids) & excluded_ids else "EXECUTABLE_ACTIVE"
        return executable, "EVALUATE_AFTER_BACKFILL", "candidate path exists but synthetic/eval coverage is missing"
    inactive = (set(rule_ids) & excluded_ids) or (active_ids and not (set(rule_ids) & active_ids))
    if inactive:
        return "EXECUTABLE_INACTIVE", "EVALUATE_AFTER_BACKFILL", "candidate path exists but current short-v2 target is inactive/excluded"
    if float(metrics.get("current_candidate_recall", 0.0)) < 0.85 and int(metrics.get("current_eval_count", 0)) > 0:
        return "EXECUTABLE_ACTIVE", "EVALUATE_NOW", "candidate path exists but current recall needs matrix audit"
    return "EXECUTABLE_ACTIVE", "EVALUATE_NOW", "candidate-backed executable rule group"


def _current_dataset_state(
    dataset_manifest_path: str | Path,
    current_dataset_path: str | Path,
) -> tuple[Counter[str], set[str], set[str]]:
    counts: Counter[str] = Counter()
    active_ids: set[str] = set()
    excluded_ids: set[str] = set()
    manifest_path = Path(dataset_manifest_path)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        for key in ("rule_id_counts", "rule_counts", "counts_by_rule_id"):
            raw_counts = manifest.get(key)
            if isinstance(raw_counts, dict):
                counts.update({normalize_rule_id(k): int(v) for k, v in raw_counts.items()})
        quota = manifest.get("active_rule_quota_summary") or manifest.get("quota_state") or {}
        if isinstance(quota, dict):
            active_ids.update(normalize_rule_id(value) for value in quota.get("active_rule_ids", []) or [])
            excluded_ids.update(normalize_rule_id(value) for value in quota.get("excluded_active_rule_ids", []) or [])
        active_ids.update(normalize_rule_id(value) for value in manifest.get("active_rule_ids", []) or [])
        excluded_ids.update(normalize_rule_id(value) for value in manifest.get("excluded_active_rule_ids", []) or [])
    dataset_path = Path(current_dataset_path)
    if dataset_path.exists():
        try:
            frame = pd.read_csv(dataset_path, usecols=lambda column: column in {"rule_id", "rule_ids"})
        except Exception:
            frame = pd.DataFrame()
        for row in frame.to_dict("records"):
            for rule_id in _row_rule_ids(row):
                counts[rule_id] += 1
    return counts, {item for item in active_ids if item != UNKNOWN_RULE_ID}, {item for item in excluded_ids if item != UNKNOWN_RULE_ID}


def _report_state(reports_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(reports_dir)
    files = {
        "rule_precision_recall": "rule_precision_recall.csv",
        "error_by_rule": "error_by_rule.csv",
        "candidate_recall": "candidate_recall_by_rule.csv",
        "gap_coverage": "gap_label_coverage_by_rule.csv",
    }
    result: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = base / filename
        if not path.exists():
            result[key] = pd.DataFrame()
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            frame = pd.DataFrame()
        if "rule_id" in frame.columns:
            frame["rule_id"] = frame["rule_id"].map(normalize_rule_id)
        result[key] = frame
    return result


def _combined_metrics(rule_ids: list[str], report_state: dict[str, pd.DataFrame]) -> dict[str, Any]:
    precision = report_state.get("rule_precision_recall", pd.DataFrame())
    recall = report_state.get("candidate_recall", pd.DataFrame())
    gap = report_state.get("gap_coverage", pd.DataFrame())
    selected_precision = precision[precision["rule_id"].isin(rule_ids)] if "rule_id" in precision.columns else pd.DataFrame()
    selected_recall = recall[recall["rule_id"].isin(rule_ids)] if "rule_id" in recall.columns else pd.DataFrame()
    selected_gap = gap[gap["rule_id"].isin(rule_ids)] if "rule_id" in gap.columns else pd.DataFrame()
    gold_count = _sum_column(selected_precision, "gold_count")
    predicted_count = _sum_column(selected_precision, "predicted_count")
    true_positive = _sum_column(selected_precision, "true_positive")
    false_positive = _sum_column(selected_precision, "false_positive")
    false_negative = _sum_column(selected_precision, "false_negative")
    return {
        "current_eval_count": gold_count,
        "current_candidate_recall": _weighted_rate(selected_recall, "candidate_present_count", "gold_count"),
        "current_gap_coverage": _weighted_rate(selected_gap, "candidate_gap_present_count", "gold_gap_count"),
        "current_rule_precision": _safe_rate(true_positive, predicted_count),
        "current_rule_recall": _safe_rate(true_positive, gold_count),
        "current_rule_f1": _f1(_safe_rate(true_positive, predicted_count), _safe_rate(true_positive, gold_count)),
        "current_predicted_count": predicted_count,
        "current_false_positive_count": false_positive,
        "current_false_negative_count": false_negative,
    }


def _write_summary(path: Path, inventory: pd.DataFrame, alias_audit: pd.DataFrame) -> None:
    counts = Counter(str(value) for value in inventory["executable_status"])
    decisions = Counter(str(value) for value in inventory["decision"])
    missing_registry = int((inventory["registry_rule_exists"] == False).sum())  # noqa: E712
    missing_candidate = int((inventory["candidate_generator_support"].astype(str) != "true").sum())
    missing_synthetic = int((inventory["synthetic_corruption_support"].astype(str) != "true").sum())
    low_recall = inventory[
        (inventory["current_eval_count"].astype(float) > 0)
        & (inventory["current_candidate_recall"].astype(float) < 0.85)
    ]
    high_fp = inventory[inventory["current_false_positive_count"].astype(float) > 0]
    eligible = inventory[inventory["decision"].isin(["EVALUATE_NOW", "EVALUATE_AFTER_BACKFILL"])]
    report_only = alias_audit[alias_audit["action"].isin(["missing_in_matrix", "unknown"])]
    lines = [
        "# Rule Matrix Inventory Summary",
        "",
        f"- total matrix groups: {len(inventory)}",
        f"- executable active: {counts.get('EXECUTABLE_ACTIVE', 0)}",
        f"- executable inactive: {counts.get('EXECUTABLE_INACTIVE', 0)}",
        f"- metadata-only: {counts.get('METADATA_ONLY', 0)}",
        f"- planned-only: {counts.get('PLANNED_ONLY', 0)}",
        f"- missing registry ids: {missing_registry}",
        f"- missing candidate path: {missing_candidate}",
        f"- missing synthetic support: {missing_synthetic}",
        f"- low-recall groups: {len(low_recall)}",
        f"- high-FP groups: {len(high_fp)}",
        f"- groups eligible for next dataset cycle: {len(eligible)}",
        "",
        "## Decisions",
        "",
        *[f"- {decision}: {count}" for decision, count in sorted(decisions.items())],
        "",
        "## Taxonomy Reference",
        "",
        "- Orfogrammka orthography: https://orfogrammka.ru/орфография/",
        "- Orfogrammka punctuation: https://orfogrammka.ru/пунктуация/",
        "",
    ]
    if not report_only.empty:
        lines.extend(
            [
                "## Report-Only Warnings",
                "",
                *[
                    f"- {row.source}: {row.raw_rule_id} -> {row.canonical_rule_id} ({row.action})"
                    for row in report_only.itertuples(index=False)
                ],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _matrix_rule_ids(rules_config_path: str | Path) -> list[str]:
    coverage = load_rules_coverage(rules_config_path)
    return sorted(
        {
            normalize_rule_id(rule_id)
            for _section, _group, entry in iter_coverage_entries(coverage)
            for rule_id in entry.get("rules", [])
            if normalize_rule_id(rule_id) != UNKNOWN_RULE_ID
        }
    )


def _report_rule_ids(reports_dir: str | Path) -> list[str]:
    base = Path(reports_dir)
    values: set[str] = set()
    for filename in (
        "rule_precision_recall.csv",
        "error_by_rule.csv",
        "candidate_recall_by_rule.csv",
        "gap_label_coverage_by_rule.csv",
    ):
        path = base / filename
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda column: column == "rule_id")
        except Exception:
            continue
        values.update(str(value) for value in frame.get("rule_id", []) if not pd.isna(value))
    return sorted(values)


def _row_rule_ids(row: dict[str, Any]) -> list[str]:
    if "rule_id" in row and not _is_missing(row.get("rule_id")):
        return [normalize_rule_id(row.get("rule_id"))]
    value = row.get("rule_ids")
    if _is_missing(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in stripped.split(",")]
    elif isinstance(value, list):
        parsed = value
    else:
        parsed = [value]
    return [normalize_rule_id(item) for item in parsed if normalize_rule_id(item) != UNKNOWN_RULE_ID]


def _json_list(values: Iterable[Any]) -> str:
    return json.dumps([str(value) for value in values], ensure_ascii=False)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _sum_column(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _weighted_rate(frame: pd.DataFrame, numerator_column: str, denominator_column: str) -> float:
    numerator = _sum_column(frame, numerator_column)
    denominator = _sum_column(frame, denominator_column)
    return _safe_rate(numerator, denominator)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False
