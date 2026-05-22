from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SYNTHETIC_OPEN_CLEAN = "synthetic_augmented_from_open_clean"
REAL_ERROR_PAIR = "real_error_pair"
CLEAN_IDENTITY_OPEN = "clean_identity_from_open_clean"
HARD_NEGATIVE_OPEN = "hard_negative_from_open_clean"
ARTIFICIAL_MARKER_KEYS = ("metka", "later_editor_checked_record", "random_filler_tokens")
METKA_SUBSTRING = "\u043c\u0435\u0442\u043a\u0430"
LATER_EDITOR_PATTERNS = (
    "\u043f\u043e\u0437\u0436\u0435 \u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440 "
    "\u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b \u0437\u0430\u043f\u0438\u0441\u044c",
    "\u043f\u043e\u0437\u0436\u0435 \u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440 "
    "\u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b \u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
)
RANDOM_FILLER_RE = re.compile(
    r"(?:\b(?:\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0435|\u0444\u0430\u0439\u043b\u0435)\s+[\u0430-\u044f\u0451]{2}\.)|"
    r"(?:\b\u043e\u0442\u0447[\u0435\u0451]\u0442\u0435\s+[\u0430-\u044f\u0451]{2}\s+\u0432\u0441\u0442\u0440\u0435\u0442\u0438\u043b\u043e\u0441\u044c)|"
    r"(?:\b\u0437\u0430\u043f\u0438\u0441\u0438\s+[\u0430-\u044f\u0451]{2}\.)|"
    r"(?:\b\u043f\u0438\u0441\u044c\u043c\u0435\s+[\u0430-\u044f\u0451]{2}\s+\u0431\u044b\u043b\u0430)|"
    r"(?:\b\u0437\u0430\u044f\u0432\u043b\u0435\u043d\u0438\u0438\s+[\u0430-\u044f\u0451]{2}\s+\u0443\u043a\u0430\u0437\u0430\u043b\u0438)|"
    r"(?:\u043f\u043e\u0437\u0436\u0435\s+\u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440\s+"
    r"\u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b\s+"
    r"(?:\u0437\u0430\u043f\u0438\u0441\u044c|\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b)\s+"
    r"[\u0430-\u044f\u0451]{2}\b)",
    re.IGNORECASE,
)

DASH_RULE_IDS = {"asyndetic_dash", "consequence_dash", "enumeration_dash"}
PUNCTUATION_RULE_MARKERS = (
    "comma",
    "dash",
    "colon",
    "semicolon",
    "speech",
    "quote",
    "bracket",
    "punctuation",
    "final_punctuation",
)
KNOWN_BAD_PHRASES = (
    "несогласен с выводом",
    "ненужно комиссии",
    "по-старому плану",
    "по-новому вариант",
    "по-новому договору",
    "по-старому адресу",
    "сохранил территория",
    "записал житель",
    "читал свежая сводка",
    "в закрытая заявка",
)
BAD_DASH_SPACING_RE = re.compile(r"\S—\s|\s—\S")


def audit_training_dataset(frame: pd.DataFrame, active_rule_ids: Iterable[str]) -> dict[str, Any]:
    active = sorted({str(rule_id) for rule_id in active_rule_ids})
    generation = generation_strategy_frame(frame, active)
    diversity = rule_diversity_frame(frame, active)
    extended = extended_quality_audit_frame(frame, active, diversity)
    known = known_quality_bug_counts(frame)
    artificial = artificial_marker_counts(frame)
    bearing_counts = error_bearing_sentence_source_counts(frame)

    synthetic = frame[frame["source_type"] == SYNTHETIC_OPEN_CLEAN] if "source_type" in frame else pd.DataFrame()
    bearing_total = max(1, int(bearing_counts.get("corpus", 0)) + int(bearing_counts.get("fallback_template", 0)))
    corpus_share = int(bearing_counts.get("corpus", 0)) / bearing_total
    fallback_share = int(bearing_counts.get("fallback_template", 0)) / bearing_total
    failed_diversity = diversity[~diversity["passes_required_gates"].astype(bool)] if not diversity.empty else diversity
    blocking = extended[extended["severity"].eq("blocking")] if not extended.empty else extended

    return {
        "known_quality_bugs": known,
        "artificial_marker_counts": artificial,
        "exact_clean_hard_duplicate_count": exact_clean_hard_duplicate_count(frame),
        "error_bearing_sentence_source_counts": bearing_counts,
        "generation_strategy_report": generation,
        "rule_diversity_report": diversity,
        "extended_quality_audit": extended,
        "corpus_opportunity_share": float(corpus_share),
        "fallback_template_share": float(fallback_share),
        "rule_diversity_summary": {
            "active_rule_count": int(len(active)),
            "failed_rule_count": int(len(failed_diversity)),
            "failed_rule_ids": sorted(failed_diversity["rule_id"].astype(str).tolist()) if not failed_diversity.empty else [],
        },
        "extended_quality_audit_summary": {
            "issue_count": int(len(extended)),
            "blocking_issue_count": int(len(blocking)),
        },
    }


def write_generation_strategy_report(audit: dict[str, Any], path: Path) -> None:
    frame = audit.get("generation_strategy_report")
    _frame_or_empty(frame).to_csv(path, index=False)


def write_rule_diversity_report(audit: dict[str, Any], path: Path) -> None:
    frame = audit.get("rule_diversity_report")
    _frame_or_empty(frame).to_csv(path, index=False)


def write_extended_quality_reports(audit: dict[str, Any], csv_path: Path, md_path: Path) -> None:
    frame = _frame_or_empty(audit.get("extended_quality_audit"))
    frame.to_csv(csv_path, index=False)
    lines = [
        "# Extended Quality Audit",
        "",
        f"- issue_count: {audit.get('extended_quality_audit_summary', {}).get('issue_count', 0)}",
        f"- blocking_issue_count: {audit.get('extended_quality_audit_summary', {}).get('blocking_issue_count', 0)}",
        "",
    ]
    if not frame.empty:
        lines.extend(["## Issues", ""])
        for row in frame.head(100).to_dict("records"):
            lines.append(
                f"- {row.get('severity')}: {row.get('check_name')} "
                f"rule={row.get('rule_id')} count={row.get('count')} example={row.get('example')}"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artificial_marker_reports(audit: dict[str, Any], csv_path: Path, md_path: Path) -> None:
    counts = dict(audit.get("artificial_marker_counts", {}) or {})
    rows = [{"marker_kind": key, "count": int(counts.get(key, 0) or 0)} for key in ARTIFICIAL_MARKER_KEYS]
    pd.DataFrame(rows, columns=["marker_kind", "count"]).to_csv(csv_path, index=False)
    lines = [
        "# Artificial Marker Audit",
        "",
        *[f"- {row['marker_kind']}: {row['count']}" for row in rows],
        f"- exact_clean_hard_duplicate_count: {int(audit.get('exact_clean_hard_duplicate_count', 0) or 0)}",
        (
            "- error_bearing_sentence_source_counts: "
            + json.dumps(audit.get("error_bearing_sentence_source_counts", {}), ensure_ascii=False, sort_keys=True)
        ),
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_known_quality_bugs_report(audit: dict[str, Any], path: Path) -> None:
    known = dict(audit.get("known_quality_bugs", {}) or {})
    lines = ["# Known Quality Bugs Report", ""]
    for name, count in sorted(known.items()):
        lines.append(f"- {name}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def known_quality_bug_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {
            "synthetic_positive_identity": 0,
            "final_punctuation_positive_identity": 0,
            "bad_dash_spacing": 0,
            "unsafe_known_phrases": 0,
            "n_nn_short_form_noun_rewrite": 0,
            "missing_generation_strategy": 0,
            "missing_error_bearing_sentence_source": 0,
            "synthetic_candidate_present_missing": 0,
            "artificial_marker_metka": 0,
            "artificial_marker_later_editor_checked_record": 0,
            "artificial_marker_random_filler_tokens": 0,
        }
    source = frame["source"].astype(str)
    target = frame["target"].astype(str)
    source_type = frame["source_type"].astype(str)
    metadata = frame["metadata"].map(_json_dict) if "metadata" in frame else pd.Series([{}] * len(frame))
    synthetic = source_type.eq(SYNTHETIC_OPEN_CLEAN)
    identity = source.eq(target)
    combined_lower = (source + "\n" + target).str.lower()

    final_positive = synthetic & _has_rule(frame, "final_punctuation_default")
    dash_rows = pd.Series(False, index=frame.index)
    for rule_id in DASH_RULE_IDS:
        dash_rows = dash_rows | _has_rule(frame, rule_id)
    unsafe_phrases = pd.Series(False, index=frame.index)
    for phrase in KNOWN_BAD_PHRASES:
        unsafe_phrases = unsafe_phrases | combined_lower.str.contains(phrase, regex=False, na=False)
    n_nn_short = _has_rule(frame, "n_nn_short_form")
    n_nn_bad = (
        n_nn_short
        & source.str.contains(r"\b(?:цены|страны)\b", regex=True, case=False, na=False)
        & target.str.contains(r"\b(?:ценны|странны)\b", regex=True, case=False, na=False)
    )
    missing_strategy = synthetic & metadata.map(lambda item: not item.get("generation_strategy"))
    missing_bearing_source = synthetic & metadata.map(lambda item: not item.get("error_bearing_sentence_source"))
    missing_candidate = synthetic & metadata.map(lambda item: item.get("candidate_present") is not True)
    artificial = artificial_marker_counts(frame)
    return {
        "synthetic_positive_identity": int((synthetic & identity).sum()),
        "final_punctuation_positive_identity": int((final_positive & identity).sum()),
        "bad_dash_spacing": int((dash_rows & target.str.contains(BAD_DASH_SPACING_RE, na=False)).sum()),
        "unsafe_known_phrases": int(unsafe_phrases.sum()),
        "n_nn_short_form_noun_rewrite": int(n_nn_bad.sum()),
        "missing_generation_strategy": int(missing_strategy.sum()),
        "missing_error_bearing_sentence_source": int(missing_bearing_source.sum()),
        "synthetic_candidate_present_missing": int(missing_candidate.sum()),
        "artificial_marker_metka": int(artificial.get("metka", 0)),
        "artificial_marker_later_editor_checked_record": int(artificial.get("later_editor_checked_record", 0)),
        "artificial_marker_random_filler_tokens": int(artificial.get("random_filler_tokens", 0)),
    }


def artificial_marker_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {key: 0 for key in ARTIFICIAL_MARKER_KEYS}
    source = frame["source"].astype(str) if "source" in frame else pd.Series([""] * len(frame), index=frame.index)
    target = frame["target"].astype(str) if "target" in frame else pd.Series([""] * len(frame), index=frame.index)
    combined = (source + "\n" + target).str.lower()
    source_type = frame["source_type"].astype(str) if "source_type" in frame else pd.Series([""] * len(frame), index=frame.index)
    synthetic = source_type.eq(SYNTHETIC_OPEN_CLEAN)
    later = pd.Series(False, index=frame.index)
    for phrase in LATER_EDITOR_PATTERNS:
        later = later | combined.str.contains(phrase, regex=False, na=False)
    return {
        "metka": int(combined.str.contains(METKA_SUBSTRING, regex=False, na=False).sum()),
        "later_editor_checked_record": int(later.sum()),
        "random_filler_tokens": int((synthetic & combined.str.contains(RANDOM_FILLER_RE, na=False)).sum()),
    }


def error_bearing_sentence_source_counts(frame: pd.DataFrame) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if frame.empty or "metadata" not in frame:
        return {}
    for value in frame["metadata"].tolist():
        item = _json_dict(value)
        source = str(item.get("error_bearing_sentence_source") or "")
        if source:
            counter[source] += 1
    return dict(sorted(counter.items()))


def exact_clean_hard_duplicate_count(frame: pd.DataFrame) -> int:
    if frame.empty or "source_type" not in frame:
        return 0
    clean = frame[frame["source_type"].astype(str).eq(CLEAN_IDENTITY_OPEN)]
    hard = frame[frame["source_type"].astype(str).eq(HARD_NEGATIVE_OPEN)]
    clean_pairs = set(zip(clean["source"].astype(str), clean["target"].astype(str)))
    hard_pairs = set(zip(hard["source"].astype(str), hard["target"].astype(str)))
    return int(len(clean_pairs & hard_pairs))


def generation_strategy_frame(frame: pd.DataFrame, active_rule_ids: Iterable[str]) -> pd.DataFrame:
    active = set(str(rule_id) for rule_id in active_rule_ids)
    rows: list[dict[str, Any]] = []
    synthetic = frame[frame["source_type"].eq(SYNTHETIC_OPEN_CLEAN)] if not frame.empty else frame
    exploded = _explode_rule_rows(synthetic)
    for rule_id in sorted(active):
        rule_rows = exploded[exploded["__rule_id"].eq(rule_id)]
        strategy_counts = _metadata_value_counts(rule_rows, "generation_strategy")
        bearing_counts = _metadata_value_counts(rule_rows, "error_bearing_sentence_source")
        total = int(len(rule_rows))
        rows.append(
            {
                "rule_id": rule_id,
                "is_active_rule": True,
                "synthetic_count": total,
                "corpus_opportunity_count": int(strategy_counts.get("corpus_opportunity", 0)),
                "fallback_template_count": int(strategy_counts.get("fallback_natural_template", 0)),
                "multi_error_stress_count": int(strategy_counts.get("multi_error_stress", 0)),
                "corpus_error_bearing_count": int(bearing_counts.get("corpus", 0)),
                "fallback_error_bearing_count": int(bearing_counts.get("fallback_template", 0)),
                "corpus_opportunity_share": _rate(bearing_counts.get("corpus", 0), max(1, bearing_counts.get("corpus", 0) + bearing_counts.get("fallback_template", 0))),
                "fallback_template_share": _rate(bearing_counts.get("fallback_template", 0), max(1, bearing_counts.get("corpus", 0) + bearing_counts.get("fallback_template", 0))),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "rule_id",
            "is_active_rule",
            "synthetic_count",
            "corpus_opportunity_count",
            "fallback_template_count",
            "multi_error_stress_count",
            "corpus_error_bearing_count",
            "fallback_error_bearing_count",
            "corpus_opportunity_share",
            "fallback_template_share",
        ],
    )


def rule_diversity_frame(frame: pd.DataFrame, active_rule_ids: Iterable[str]) -> pd.DataFrame:
    active = set(str(rule_id) for rule_id in active_rule_ids)
    rows: list[dict[str, Any]] = []
    exploded = _explode_rule_rows(frame)
    for rule_id in sorted(active):
        rule_rows = exploded[exploded["__rule_id"].eq(rule_id)]
        count = int(len(rule_rows))
        metadata = rule_rows["metadata"].map(_json_dict) if not rule_rows.empty else pd.Series(dtype=object)
        carrier_values = [
            str(item.get("carrier_sentence_hash") or item.get("original_clean_sentence") or "")
            for item in metadata.tolist()
        ]
        carrier_values = [value for value in carrier_values if value]
        template_counts = Counter(rule_rows["template_id"].astype(str).tolist()) if "template_id" in rule_rows else Counter()
        norm_counts = Counter(rule_rows["normalized_pair_hash"].astype(str).tolist()) if "normalized_pair_hash" in rule_rows else Counter()
        error_forms = [str(item.get("error_form") or "") for item in metadata.tolist()]
        target_forms = [str(item.get("target_form") or "") for item in metadata.tolist()]
        error_forms = [value for value in error_forms if value]
        target_forms = [value for value in target_forms if value]
        forms_applicable = _forms_applicable(rule_id, error_forms, target_forms)
        unique_carriers = len(set(carrier_values)) if carrier_values else int(rule_rows["target"].nunique()) if not rule_rows.empty else 0
        top_template_share = _top_share(template_counts, count)
        top_error_form_share = _top_share(Counter(error_forms), len(error_forms))
        top_target_form_share = _top_share(Counter(target_forms), len(target_forms))
        duplicate_rate = _duplicate_rate(norm_counts, count)
        passes = count >= 1000 and unique_carriers >= min(500, int(count * 0.5)) and top_template_share <= 0.10 and duplicate_rate <= 0.15
        if forms_applicable:
            passes = (
                passes
                and len(set(error_forms)) >= 30
                and len(set(target_forms)) >= 30
                and top_error_form_share <= 0.15
                and top_target_form_share <= 0.15
            )
        rows.append(
            {
                "rule_id": rule_id,
                "is_active_rule": True,
                "count": count,
                "unique_error_forms": len(set(error_forms)),
                "unique_target_forms": len(set(target_forms)),
                "unique_carrier_sentences": unique_carriers,
                "top_template_share": float(top_template_share),
                "top_error_form_share": float(top_error_form_share),
                "top_target_form_share": float(top_target_form_share),
                "fallback_template_share": _rule_fallback_share(rule_rows),
                "normalized_pair_duplicate_rate": float(duplicate_rate),
                "forms_applicable": bool(forms_applicable),
                "passes_required_gates": bool(passes),
            }
        )
    return pd.DataFrame(rows)


def extended_quality_audit_frame(frame: pd.DataFrame, active_rule_ids: Iterable[str], diversity: pd.DataFrame) -> pd.DataFrame:
    issues: list[dict[str, Any]] = []
    known = known_quality_bug_counts(frame)
    for name, count in known.items():
        if int(count):
            issues.append({"check_name": name, "rule_id": "", "severity": "blocking", "count": int(count), "example": ""})
    artificial = artificial_marker_counts(frame)
    for name, count in artificial.items():
        if int(count):
            issues.append({"check_name": f"artificial_marker_{name}", "rule_id": "", "severity": "blocking", "count": int(count), "example": ""})
    if not diversity.empty:
        failed = diversity[~diversity["passes_required_gates"].astype(bool)]
        for row in failed.to_dict("records"):
            issues.append(
                {
                    "check_name": "rule_diversity_gate_failed",
                    "rule_id": row.get("rule_id", ""),
                    "severity": "blocking",
                    "count": int(row.get("count", 0) or 0),
                    "example": "",
                }
            )
    return pd.DataFrame(issues, columns=["check_name", "rule_id", "severity", "count", "example"])


def _metadata_value_counts(frame: pd.DataFrame, key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if frame.empty or "metadata" not in frame:
        return {}
    for value in frame["metadata"].tolist():
        item = _json_dict(value)
        strategy = str(item.get(key) or "")
        if strategy:
            counter[strategy] += 1
    return dict(counter)


def _explode_rule_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "rule_ids" not in frame:
        return frame.assign(__rule_id=[])
    working = frame.copy()
    working["__rule_id"] = working["rule_ids"].map(lambda value: [str(item) for item in _json_list(value)])
    working = working.explode("__rule_id")
    return working[working["__rule_id"].notna()]


def _frame_or_empty(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _has_rule(frame: pd.DataFrame, rule_id: str) -> pd.Series:
    if "rule_ids" not in frame:
        return pd.Series(False, index=frame.index)
    return frame["rule_ids"].astype(str).str.contains(f'"{rule_id}"', regex=False, na=False)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def _top_share(counts: Counter[str], total: int) -> float:
    if total <= 0 or not counts:
        return 0.0
    return max(counts.values()) / total


def _duplicate_rate(counts: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(max(0, count - 1) for count in counts.values()) / total


def _forms_applicable(rule_id: str, error_forms: list[str], target_forms: list[str]) -> bool:
    if any(marker in rule_id for marker in PUNCTUATION_RULE_MARKERS):
        return False
    if not error_forms and not target_forms:
        return False
    if len(set(error_forms)) < 30 or len(set(target_forms)) < 30:
        return False
    return True


def _rule_fallback_share(rule_rows: pd.DataFrame) -> float:
    if rule_rows.empty:
        return 0.0
    counts = _metadata_value_counts(rule_rows[rule_rows["source_type"].eq(SYNTHETIC_OPEN_CLEAN)], "error_bearing_sentence_source")
    total = sum(counts.values())
    return _rate(counts.get("fallback_template", 0), total)
