from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.dataset_verifiers import is_detached_participial_comma_pair


SYNTHETIC_OPEN_CLEAN = "synthetic_augmented_from_open_clean"
REAL_ERROR_PAIR = "real_error_pair"
CLEAN_IDENTITY_OPEN = "clean_identity_from_open_clean"
HARD_NEGATIVE_OPEN = "hard_negative_from_open_clean"
ARTIFICIAL_MARKER_KEYS = ("metka", "later_editor_checked_record", "random_filler_tokens")
QUOTE_BRACKET_BALANCE_KEYS = (
    "unbalanced_target_guillemets",
    "unbalanced_target_ascii_quotes",
    "unbalanced_target_parentheses",
    "unbalanced_target_square_brackets",
    "unbalanced_target_curly_brackets",
)
QUOTE_BRACKET_BALANCE_SOURCE_TYPES = {SYNTHETIC_OPEN_CLEAN, REAL_ERROR_PAIR}
CLEAN_HARD_BALANCE_KEYS = (
    "unbalanced_guillemets",
    "unbalanced_ascii_quotes",
    "unbalanced_parentheses",
    "unbalanced_square_brackets",
    "unbalanced_curly_brackets",
)
CLEAN_HARD_BALANCE_SOURCE_TYPES = {CLEAN_IDENTITY_OPEN, HARD_NEGATIVE_OPEN}
RULE_SEMANTIC_ALIGNMENT_SOURCE_TYPES = {SYNTHETIC_OPEN_CLEAN}
SYNTAX_PUNCTUATION_ALIGNMENT_RULE_IDS = {
    "apposition_comma",
    "clarification_comma",
    "detached_participial_comma",
    "detached_adverbial_comma",
    "homogeneous_comma",
    "comparative_turnover_comma",
    "comma_subordinate",
    "comma_conjunction",
    "direct_speech_quotes",
    "subject_predicate_dash",
    "asyndetic_dash",
}
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
PUNCTUATION_CHARS = set(".,;:!?—-«»\"()[]{}")
PUNCTUATION_RULE_IDS = {
    "comma_subordinate",
    "comma_conjunction",
    "introductory_comma",
    "address_comma",
    "homogeneous_comma",
    "detached_adverbial_comma",
    "detached_participial_comma",
    "apposition_comma",
    "clarification_comma",
    "comparative_turnover_comma",
    "subject_predicate_dash",
    "asyndetic_dash",
    "consequence_dash",
    "explanation_colon",
    "enumeration_colon",
    "enumeration_dash",
    "semicolon",
    "direct_speech_colon",
    "direct_speech_dash",
    "direct_speech_quotes",
    "quote_pair_balance",
    "bracket_pair_balance",
    "punctuation_delete_replace",
    "final_punctuation_default",
}
CONTEXT_RULE_PAIRS = {
    "context_tak_zhe": (("так же", "также"), ("также", "так же")),
    "context_to_zhe": (("то же", "тоже"), ("тоже", "то же")),
    "context_chto_by": (("что бы", "чтобы"), ("чтобы", "что бы")),
    "context_za_to": (("за то", "зато"), ("зато", "за то")),
    "context_vsledstvie": (("в следствие", "вследствие"), ("вследствие", "в следствие")),
    "context_nesmotrya": (("не смотря", "несмотря"), ("несмотря", "не смотря")),
}
PATTERN_RULE_PAIRS = {
    "pattern_жы_жи": ("жы", "жи"),
    "pattern_шы_ши": ("шы", "ши"),
    "pattern_чя_ча": ("чя", "ча"),
    "pattern_щя_ща": ("щя", "ща"),
    "pattern_чю_чу": ("чю", "чу"),
    "pattern_щю_щу": ("щю", "щу"),
    "pattern_цы_ци": ("цы", "ци"),
    "pattern_жо_же": ("жо", "же"),
    "pattern_шо_ше": ("шо", "ше"),
    "pattern_чо_че": ("чо", "че"),
    "pattern_що_ще": ("що", "ще"),
}
SAFE_POL_POLU_TARGET_RE = re.compile(
    r"\bпол-[А-Яа-яЁё]+|"
    r"\bпол(?:часа|час|дня|дома|мира|литра|кило|килограмма|метра|яблока|лимона|арбуза|апельсина|"
    r"года|году|годом|годе|миллиона|миллиону|миллионом|миллиарда|века|веку|веком|день|дня|"
    r"ночи|ночь|минуты|минуту|пути|сотни|сотней|тонны|тонну)\b|"
    r"\bполу(?:финал|финала|финале|финалу|финалом|финальный|финальном|финальных|финалы|финалов|"
    r"защитник|защитника|защитнику|защитником|защитники|защитников|годие|годия|годовой|годового|"
    r"остров|острова|острове|круг|круга|месяц|месяца|сотни|сотней|века|миллион|миллиона|"
    r"миллионный|пустыня|пустыням|фабрикаты)\b",
    re.IGNORECASE,
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
    quote_bracket = quote_bracket_balance_counts(frame)
    quote_bracket_audit = quote_bracket_balance_audit_frame(frame)
    clean_hard_balance = clean_hard_balance_counts(frame)
    clean_hard_balance_audit = clean_hard_balance_audit_frame(frame)
    semantic_alignment_audit = rule_semantic_alignment_audit_frame(frame, active_rule_ids=active)
    semantic_alignment_summary = rule_semantic_alignment_summary(semantic_alignment_audit)
    bearing_counts = error_bearing_sentence_source_counts(frame)

    synthetic = frame[frame["source_type"] == SYNTHETIC_OPEN_CLEAN] if "source_type" in frame else pd.DataFrame()
    bearing_total = max(1, int(bearing_counts.get("corpus", 0)) + int(bearing_counts.get("fallback_template", 0)))
    corpus_share = int(bearing_counts.get("corpus", 0)) / bearing_total
    fallback_share = int(bearing_counts.get("fallback_template", 0)) / bearing_total
    failed_diversity = diversity[~diversity["passes_required_gates"].astype(bool)] if not diversity.empty else diversity
    blocking = extended[extended["severity"].eq("blocking")] if not extended.empty else extended

    return {
        "total_rows": int(len(frame)),
        "known_quality_bugs": known,
        "artificial_marker_counts": artificial,
        "quote_bracket_balance_bugs": quote_bracket,
        "quote_bracket_balance_audit": quote_bracket_audit,
        "clean_hard_balance_bugs": clean_hard_balance,
        "clean_hard_balance_audit": clean_hard_balance_audit,
        "rule_semantic_alignment_audit": semantic_alignment_audit,
        "rule_semantic_alignment": semantic_alignment_summary,
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


def write_quote_bracket_balance_reports(audit: dict[str, Any], csv_path: Path, md_path: Path) -> None:
    frame = _frame_or_empty(audit.get("quote_bracket_balance_audit"))
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "row_index",
                "split",
                "source_type",
                "rule_id",
                "source",
                "target",
                "unbalanced_guillemet",
                "unbalanced_ascii_quote",
                "unbalanced_parentheses",
                "unbalanced_square_bracket",
                "unbalanced_curly_bracket",
                "action",
                "reason",
            ]
        )
    frame.to_csv(csv_path, index=False)
    counts = dict(audit.get("quote_bracket_balance_bugs", {}) or {})
    lines = [
        "# Quote/Bracket Balance Audit",
        "",
        *[f"- {key}: {int(counts.get(key, 0) or 0)}" for key in QUOTE_BRACKET_BALANCE_KEYS],
        f"- issue_rows: {len(frame)}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_clean_hard_balance_report(audit: dict[str, Any], csv_path: Path) -> None:
    frame = _frame_or_empty(audit.get("clean_hard_balance_audit"))
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "row_index",
                "split",
                "source_type",
                "rule_id",
                "source",
                "target",
                "unbalanced_guillemet",
                "unbalanced_ascii_quote",
                "unbalanced_parentheses",
                "unbalanced_square_bracket",
                "unbalanced_curly_bracket",
                "action",
                "reason",
            ]
        )
    frame.to_csv(csv_path, index=False)


def write_rule_semantic_alignment_report(audit: dict[str, Any], csv_path: Path) -> None:
    frame = _frame_or_empty(audit.get("rule_semantic_alignment_audit"))
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "row_index",
                "split",
                "source_type",
                "rule_id",
                "source",
                "target",
                "detected_family",
                "expected_family",
                "alignment_pass",
                "reason",
            ]
        )
    frame.to_csv(csv_path, index=False)


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
            "quote_bracket_unbalanced_target_guillemets": 0,
            "quote_bracket_unbalanced_target_ascii_quotes": 0,
            "quote_bracket_unbalanced_target_parentheses": 0,
            "quote_bracket_unbalanced_target_square_brackets": 0,
            "quote_bracket_unbalanced_target_curly_brackets": 0,
        }
    source = frame["source"].astype(str)
    target = frame["target"].astype(str)
    source_type = frame["source_type"].astype(str)
    metadata = frame["metadata"].map(_json_dict) if "metadata" in frame else pd.Series([{}] * len(frame))
    synthetic = source_type.eq(SYNTHETIC_OPEN_CLEAN)
    identity = source.eq(target)
    combined_lower = (source + "\n" + target).str.lower()

    final_positive = synthetic & _has_rule(frame, "final_punctuation_default")
    dash_spacing_rows = source.str.contains(BAD_DASH_SPACING_RE, na=False) | target.str.contains(BAD_DASH_SPACING_RE, na=False)
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
    quote_bracket = quote_bracket_balance_counts(frame)
    return {
        "synthetic_positive_identity": int((synthetic & identity).sum()),
        "final_punctuation_positive_identity": int((final_positive & identity).sum()),
        "bad_dash_spacing": int(dash_spacing_rows.sum()),
        "unsafe_known_phrases": int(unsafe_phrases.sum()),
        "n_nn_short_form_noun_rewrite": int(n_nn_bad.sum()),
        "missing_generation_strategy": int(missing_strategy.sum()),
        "missing_error_bearing_sentence_source": int(missing_bearing_source.sum()),
        "synthetic_candidate_present_missing": int(missing_candidate.sum()),
        "artificial_marker_metka": int(artificial.get("metka", 0)),
        "artificial_marker_later_editor_checked_record": int(artificial.get("later_editor_checked_record", 0)),
        "artificial_marker_random_filler_tokens": int(artificial.get("random_filler_tokens", 0)),
        "quote_bracket_unbalanced_target_guillemets": int(quote_bracket.get("unbalanced_target_guillemets", 0)),
        "quote_bracket_unbalanced_target_ascii_quotes": int(quote_bracket.get("unbalanced_target_ascii_quotes", 0)),
        "quote_bracket_unbalanced_target_parentheses": int(quote_bracket.get("unbalanced_target_parentheses", 0)),
        "quote_bracket_unbalanced_target_square_brackets": int(quote_bracket.get("unbalanced_target_square_brackets", 0)),
        "quote_bracket_unbalanced_target_curly_brackets": int(quote_bracket.get("unbalanced_target_curly_brackets", 0)),
    }


def target_quote_bracket_balance_flags(text: str) -> dict[str, bool]:
    value = str(text)
    return {
        "unbalanced_target_guillemets": _ordered_pair_imbalance(value, "«", "»") > 0,
        "unbalanced_target_ascii_quotes": value.count('"') % 2 != 0,
        "unbalanced_target_parentheses": _ordered_pair_imbalance(value, "(", ")") > 0,
        "unbalanced_target_square_brackets": _ordered_pair_imbalance(value, "[", "]") > 0,
        "unbalanced_target_curly_brackets": _ordered_pair_imbalance(value, "{", "}") > 0,
    }


def target_has_quote_bracket_balance_bug(text: str) -> bool:
    return any(target_quote_bracket_balance_flags(text).values())


def text_has_quote_bracket_balance_bug(text: str) -> bool:
    return any(_plain_quote_bracket_balance_flags(text).values())


def quote_bracket_balance_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {key: 0 for key in QUOTE_BRACKET_BALANCE_KEYS}
    if frame.empty or "target" not in frame:
        return counts
    source_type = (
        frame["source_type"].astype(str)
        if "source_type" in frame
        else pd.Series([""] * len(frame), index=frame.index)
    )
    positive = frame[source_type.isin(QUOTE_BRACKET_BALANCE_SOURCE_TYPES)]
    for target in positive["target"].astype(str).tolist():
        flags = target_quote_bracket_balance_flags(target)
        for key, value in flags.items():
            if value:
                counts[key] += 1
    return counts


def quote_bracket_balance_audit_frame(frame: pd.DataFrame, *, action: str = "kept") -> pd.DataFrame:
    columns = [
        "row_index",
        "split",
        "source_type",
        "rule_id",
        "source",
        "target",
        "unbalanced_guillemet",
        "unbalanced_ascii_quote",
        "unbalanced_parentheses",
        "unbalanced_square_bracket",
        "unbalanced_curly_bracket",
        "action",
        "reason",
    ]
    if frame.empty or "target" not in frame:
        return pd.DataFrame(columns=columns)
    source_type = (
        frame["source_type"].astype(str)
        if "source_type" in frame
        else pd.Series([""] * len(frame), index=frame.index)
    )
    positive = frame[source_type.isin(QUOTE_BRACKET_BALANCE_SOURCE_TYPES)]
    rows: list[dict[str, Any]] = []
    for index, row in positive.iterrows():
        flags = target_quote_bracket_balance_flags(str(row.get("target", "")))
        if not any(flags.values()):
            continue
        reason = next((key for key in QUOTE_BRACKET_BALANCE_KEYS if flags.get(key)), "")
        rows.append(
            {
                "row_index": int(index) if isinstance(index, int) else str(index),
                "split": str(row.get("split", "")),
                "source_type": str(row.get("source_type", "")),
                "rule_id": str(row.get("rule_id", "")),
                "source": str(row.get("source", "")),
                "target": str(row.get("target", "")),
                "unbalanced_guillemet": bool(flags["unbalanced_target_guillemets"]),
                "unbalanced_ascii_quote": bool(flags["unbalanced_target_ascii_quotes"]),
                "unbalanced_parentheses": bool(flags["unbalanced_target_parentheses"]),
                "unbalanced_square_bracket": bool(flags["unbalanced_target_square_brackets"]),
                "unbalanced_curly_bracket": bool(flags["unbalanced_target_curly_brackets"]),
                "action": action,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def clean_hard_balance_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {key: 0 for key in CLEAN_HARD_BALANCE_KEYS}
    if frame.empty or "source" not in frame:
        return counts
    source_type = (
        frame["source_type"].astype(str)
        if "source_type" in frame
        else pd.Series([""] * len(frame), index=frame.index)
    )
    clean_hard = frame[source_type.isin(CLEAN_HARD_BALANCE_SOURCE_TYPES)]
    for source in clean_hard["source"].astype(str).tolist():
        flags = _plain_quote_bracket_balance_flags(source)
        for key, value in flags.items():
            if value:
                counts[key] += 1
    return counts


def clean_hard_balance_audit_frame(frame: pd.DataFrame, *, action: str = "kept") -> pd.DataFrame:
    columns = [
        "row_index",
        "split",
        "source_type",
        "rule_id",
        "source",
        "target",
        "unbalanced_guillemet",
        "unbalanced_ascii_quote",
        "unbalanced_parentheses",
        "unbalanced_square_bracket",
        "unbalanced_curly_bracket",
        "action",
        "reason",
    ]
    if frame.empty or "source" not in frame:
        return pd.DataFrame(columns=columns)
    source_type = (
        frame["source_type"].astype(str)
        if "source_type" in frame
        else pd.Series([""] * len(frame), index=frame.index)
    )
    clean_hard = frame[source_type.isin(CLEAN_HARD_BALANCE_SOURCE_TYPES)]
    rows: list[dict[str, Any]] = []
    for index, row in clean_hard.iterrows():
        flags = _plain_quote_bracket_balance_flags(str(row.get("source", "")))
        if not any(flags.values()):
            continue
        reason = next((key for key in CLEAN_HARD_BALANCE_KEYS if flags.get(key)), "")
        rows.append(
            {
                "row_index": int(index) if isinstance(index, int) else str(index),
                "split": str(row.get("split", "")),
                "source_type": str(row.get("source_type", "")),
                "rule_id": str(row.get("rule_id", "")),
                "source": str(row.get("source", "")),
                "target": str(row.get("target", "")),
                "unbalanced_guillemet": bool(flags["unbalanced_guillemets"]),
                "unbalanced_ascii_quote": bool(flags["unbalanced_ascii_quotes"]),
                "unbalanced_parentheses": bool(flags["unbalanced_parentheses"]),
                "unbalanced_square_bracket": bool(flags["unbalanced_square_brackets"]),
                "unbalanced_curly_bracket": bool(flags["unbalanced_curly_brackets"]),
                "action": action,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _plain_quote_bracket_balance_flags(text: str) -> dict[str, bool]:
    value = str(text)
    return {
        "unbalanced_guillemets": _ordered_pair_imbalance(value, "«", "»") > 0,
        "unbalanced_ascii_quotes": value.count('"') % 2 != 0,
        "unbalanced_parentheses": _ordered_pair_imbalance(value, "(", ")") > 0,
        "unbalanced_square_brackets": _ordered_pair_imbalance(value, "[", "]") > 0,
        "unbalanced_curly_brackets": _ordered_pair_imbalance(value, "{", "}") > 0,
    }


def rule_semantic_alignment_audit_frame(
    frame: pd.DataFrame,
    active_rule_ids: Iterable[str],
) -> pd.DataFrame:
    columns = [
        "row_index",
        "split",
        "source_type",
        "rule_id",
        "source",
        "target",
        "detected_family",
        "expected_family",
        "alignment_pass",
        "reason",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    active = {str(rule_id) for rule_id in active_rule_ids} & SYNTAX_PUNCTUATION_ALIGNMENT_RULE_IDS
    if not active:
        return pd.DataFrame(columns=columns)
    source_type = (
        frame["source_type"].astype(str)
        if "source_type" in frame
        else pd.Series([""] * len(frame), index=frame.index)
    )
    rows: list[dict[str, Any]] = []
    for index, row in frame[source_type.isin(RULE_SEMANTIC_ALIGNMENT_SOURCE_TYPES)].iterrows():
        row_rules = [rule_id for rule_id in _row_rule_ids(row) if rule_id in active]
        if not row_rules:
            continue
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        for rule_id in row_rules:
            passed, detected, reason = syntax_punctuation_rule_alignment(rule_id, source, target)
            rows.append(
                {
                    "row_index": int(index) if isinstance(index, int) else str(index),
                    "split": str(row.get("split", "")),
                    "source_type": str(row.get("source_type", "")),
                    "rule_id": rule_id,
                    "source": source,
                    "target": target,
                    "detected_family": detected,
                    "expected_family": _expected_syntax_family(rule_id),
                    "alignment_pass": bool(passed),
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def rule_semantic_alignment_summary(audit_frame: pd.DataFrame) -> dict[str, Any]:
    if audit_frame.empty:
        return {"failed_rows": 0, "failed_by_rule": {}, "rejected_rows": 0, "excluded_rules": []}
    failed = audit_frame[~audit_frame["alignment_pass"].astype(bool)]
    return {
        "failed_rows": int(len(failed)),
        "failed_by_rule": {
            str(rule_id): int(count)
            for rule_id, count in failed["rule_id"].astype(str).value_counts().sort_index().items()
        },
        "rejected_rows": 0,
        "excluded_rules": [],
    }


def syntax_punctuation_rule_alignment(rule_id: str, source: str, target: str) -> tuple[bool, str, str]:
    if rule_id not in SYNTAX_PUNCTUATION_ALIGNMENT_RULE_IDS:
        return True, "not_audited", "not_audited"
    if source == target:
        return False, "identity", "synthetic_positive_identity"
    detected = _detect_syntax_punctuation_family(source, target)
    if rule_id == "apposition_comma":
        passed = _is_apposition_comma(source, target)
    elif rule_id == "clarification_comma":
        passed = _is_clarification_comma(source, target)
    elif rule_id == "detached_participial_comma":
        passed = _is_detached_participial_comma(source, target)
    elif rule_id == "detached_adverbial_comma":
        passed = _is_detached_adverbial_comma(source, target)
    elif rule_id == "homogeneous_comma":
        passed = _is_homogeneous_comma(source, target)
    elif rule_id == "comparative_turnover_comma":
        passed = _is_comparative_turnover_comma(source, target)
    elif rule_id == "comma_subordinate":
        passed = _is_subordinate_comma(source, target)
    elif rule_id == "comma_conjunction":
        passed = _is_conjunction_comma(source, target)
    elif rule_id == "direct_speech_quotes":
        passed = _is_direct_speech_quotes(source, target)
    elif rule_id == "subject_predicate_dash":
        passed = _is_subject_predicate_dash(source, target)
    elif rule_id == "asyndetic_dash":
        passed = _is_asyndetic_dash(source, target)
    else:
        passed = True
    return bool(passed), detected, "ok" if passed else "expected_rule_specific_family"


def _expected_syntax_family(rule_id: str) -> str:
    return {
        "apposition_comma": "apposition",
        "clarification_comma": "clarification",
        "detached_participial_comma": "participial_turnover",
        "detached_adverbial_comma": "adverbial_participle_turnover",
        "homogeneous_comma": "homogeneous_members",
        "comparative_turnover_comma": "comparative_turnover",
        "comma_subordinate": "subordinate_clause",
        "comma_conjunction": "coordinating_conjunction",
        "direct_speech_quotes": "direct_speech_quotes",
        "subject_predicate_dash": "subject_predicate_dash",
        "asyndetic_dash": "asyndetic_dash",
    }.get(rule_id, rule_id)


def _detect_syntax_punctuation_family(source: str, target: str) -> str:
    if _is_direct_speech_quotes(source, target):
        return "direct_speech_quotes"
    if _is_subject_predicate_dash(source, target):
        return "subject_predicate_dash"
    if _is_asyndetic_dash(source, target):
        return "asyndetic_dash"
    if _is_detached_adverbial_comma(source, target):
        return "adverbial_participle_turnover"
    if _is_detached_participial_comma(source, target):
        return "participial_turnover"
    if _is_apposition_comma(source, target):
        return "apposition"
    if _is_clarification_comma(source, target):
        return "clarification"
    if _is_comparative_turnover_comma(source, target):
        return "comparative_turnover"
    if _is_homogeneous_comma(source, target):
        return "homogeneous_members"
    if _is_subordinate_comma(source, target):
        return "subordinate_clause"
    if _is_conjunction_comma(source, target):
        return "coordinating_conjunction"
    if _comma_delta(source, target):
        return "generic_comma"
    return "unknown"


def _comma_delta(source: str, target: str) -> bool:
    return target.count(",") > source.count(",")


def _is_subordinate_comma(source: str, target: str) -> bool:
    lower = target.lower()
    if not _comma_delta(source, target):
        return False
    return bool(
        re.search(
            r",\s*(?:что|чтобы|когда|если|потому\s+что|так\s+как|котор(?:ый|ая|ое|ые|ого|ому|ым|ой|ую|ых|ыми)|где|куда|откуда|пока|хотя)\b",
            lower,
        )
    )


def _is_conjunction_comma(source: str, target: str) -> bool:
    lower = target.lower()
    if not _comma_delta(source, target):
        return False
    if _is_subordinate_comma(source, target):
        return False
    return bool(re.search(r",\s*(?:но|а|однако|зато)\b", lower))


def _is_apposition_comma(source: str, target: str) -> bool:
    lower = target.lower()
    if target.count(",") - source.count(",") < 2:
        return False
    if _has_generic_comma_family_marker(lower):
        return False
    apposition_heads = (
        "директор",
        "руководитель",
        "глава",
        "редактор",
        "эксперт",
        "представитель",
        "основатель",
        "автор",
        "инженер",
        "аналитик",
        "город",
        "столица",
        "компания",
        "организация",
    )
    return bool(re.search(r",\s*(?:" + "|".join(apposition_heads) + r")\b[^,]{2,80},", lower))


def _is_clarification_comma(source: str, target: str) -> bool:
    lower = target.lower()
    if not _comma_delta(source, target):
        return False
    if re.search(r",\s*(?:то\s+есть|а\s+именно|в\s+частности|именно|например)\b", lower):
        return True
    if re.search(r"\b(?:в\s+понедельник|во\s+вторник|в\s+среду|в\s+четверг|в\s+пятницу|в\s+субботу|в\s+воскресенье|утром|вечером|днем|ночью),\s*\d{1,2}\s+[а-яё]+", lower):
        return True
    if re.search(r"\b(?:там|здесь|тут|туда|оттуда),\s*(?:на|в|у|около)\b", lower):
        return True
    return False


def _is_detached_participial_comma(source: str, target: str) -> bool:
    return is_detached_participial_comma_pair(source, target)


def _is_detached_adverbial_comma(source: str, target: str) -> bool:
    lower = target.lower()
    if not _comma_delta(source, target):
        return False
    gerunds = (
        "проверив",
        "получив",
        "закончив",
        "обсудив",
        "изучив",
        "сравнив",
        "подготовив",
        "согласовав",
        "прочитав",
        "вернувшись",
        "учитывая",
        "используя",
        "рассмотрев",
        "оценив",
    )
    return bool(re.search(r"^\s*(?:" + "|".join(gerunds) + r")\b[^,]{1,120},", lower))


def _is_homogeneous_comma(source: str, target: str) -> bool:
    lower = target.lower()
    if not _comma_delta(source, target):
        return False
    if _has_generic_comma_family_marker(lower):
        return False
    if re.search(r"\b(?:и|ни|либо)\s+[^,]{1,40},\s*(?:и|ни|либо)\s+", lower):
        return True
    words = re.findall(r"[а-яё]+", lower)
    if len(words) < 5:
        return False
    return bool(
        re.search(
            r"\b[а-яё]{4,}(?:ы|а|я|и|ов|ев)?\s*,\s*[а-яё]{4,}(?:ы|а|я|и|ов|ев)?\s+(?:и|или)\s+[а-яё]{4,}",
            lower,
        )
    )


def _is_comparative_turnover_comma(source: str, target: str) -> bool:
    lower = target.lower()
    if not _comma_delta(source, target):
        return False
    if re.search(r"\b(?:работает|служит|выступает|назначен|стал|был|была|были)\s+как\s+(?:инженер|врач|директор|эксперт|редактор)\b", lower):
        return False
    if re.search(r"\b(?:посмотрите|проверьте|узнали|спросили|рассказали)\s+как\b", lower):
        return False
    return bool(re.search(r",\s*(?:словно|будто|как\s+будто|будто\s+бы|как)\b", lower))


def _is_direct_speech_quotes(source: str, target: str) -> bool:
    if target_has_quote_bracket_balance_bug(target):
        return False
    quote_count = target.count("«") + target.count("»") + target.count('"')
    return quote_count >= 2 and quote_count > source.count("«") + source.count("»") + source.count('"')


def _is_subject_predicate_dash(source: str, target: str) -> bool:
    if " — " not in target or " — " in source:
        return False
    lower = target.lower()
    if re.search(r"\b[а-яё]{3,}(?:\s+[а-яё]{3,}){0,3}\s+—\s+это\s+[а-яё]{3,}", lower):
        return True
    predicate_heads = (
        "столица",
        "основа",
        "цель",
        "задача",
        "пример",
        "результат",
        "причина",
        "решение",
        "документ",
        "проект",
        "система",
        "команда",
    )
    return bool(re.search(r"\b[а-яё]{3,}(?:\s+[а-яё]{3,}){0,3}\s+—\s+(?:" + "|".join(predicate_heads) + r")\b", lower))


def _is_asyndetic_dash(source: str, target: str) -> bool:
    if " — " not in target or " — " in source:
        return False
    if _is_subject_predicate_dash(source, target):
        return False
    left, right = target.split(" — ", 1)
    return len(left.split()) >= 2 and len(right.split()) >= 2


def _has_generic_comma_family_marker(lower: str) -> bool:
    return bool(
        re.search(
            r",\s*(?:что|чтобы|когда|если|потому\s+что|котор(?:ый|ая|ое|ые|ого|ому|ым|ой|ую|ых|ыми)|но|однако|зато|как\s+и\s+ожидалось)\b",
            lower,
        )
    )


def _ordered_pair_imbalance(text: str, open_char: str, close_char: str) -> int:
    balance = 0
    unmatched_close = 0
    for char in text:
        if char == open_char:
            balance += 1
        elif char == close_char:
            if balance:
                balance -= 1
            else:
                unmatched_close += 1
    return balance + unmatched_close


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
        fallback_template_share = _rule_fallback_share(rule_rows)
        passes = (
            count > 0
            and unique_carriers >= min(500, int(count * 0.5))
            and top_template_share <= 0.10
            and fallback_template_share <= 0.25
            and duplicate_rate <= 0.15
        )
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
                "fallback_template_share": float(fallback_template_share),
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
    quote_bracket = quote_bracket_balance_counts(frame)
    quote_audit = quote_bracket_balance_audit_frame(frame)
    for name, count in quote_bracket.items():
        if int(count):
            example = ""
            if not quote_audit.empty:
                examples = quote_audit[quote_audit["reason"].eq(name)]
                if not examples.empty:
                    example = str(examples.iloc[0].get("target", ""))
            issues.append(
                {
                    "check_name": f"quote_bracket_balance_{name}",
                    "rule_id": "",
                    "severity": "blocking",
                    "count": int(count),
                    "example": example,
                }
            )
    clean_hard = clean_hard_balance_counts(frame)
    clean_hard_audit = clean_hard_balance_audit_frame(frame)
    for name, count in clean_hard.items():
        if int(count):
            example = ""
            if not clean_hard_audit.empty:
                examples = clean_hard_audit[clean_hard_audit["reason"].eq(name)]
                if not examples.empty:
                    example = str(examples.iloc[0].get("source", ""))
            issues.append(
                {
                    "check_name": f"clean_hard_balance_{name}",
                    "rule_id": "",
                    "severity": "blocking",
                    "count": int(count),
                    "example": example,
                }
            )
    semantic_audit = rule_semantic_alignment_audit_frame(frame, active_rule_ids)
    semantic = rule_semantic_alignment_summary(semantic_audit)
    for rule_id, count in dict(semantic.get("failed_by_rule", {}) or {}).items():
        example = ""
        if not semantic_audit.empty:
            examples = semantic_audit[
                semantic_audit["rule_id"].astype(str).eq(str(rule_id))
                & ~semantic_audit["alignment_pass"].astype(bool)
            ]
            if not examples.empty:
                example = str(examples.iloc[0].get("target", ""))
        issues.append(
            {
                "check_name": "rule_semantic_alignment_failed",
                "rule_id": rule_id,
                "severity": "blocking",
                "count": int(count),
                "example": example,
            }
        )
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


def _row_rule_semantic_issue(row: dict[str, Any] | pd.Series) -> tuple[str, str, list[dict[str, Any]]] | None:
    source = str(row.get("source", ""))
    target = str(row.get("target", ""))
    if source == target:
        return None
    rule_ids = _row_rule_ids(row)
    if not rule_ids:
        return None
    edits = _row_edits(row)
    for rule_id in rule_ids:
        if rule_id in {"unknown", "unknown_real_validated", "clean_identity", "clean_identity_hard_negative"}:
            continue
        matching = _matching_edits(rule_id, edits)
        if not matching:
            return rule_id, "missing_matching_edit_for_rule_id", edits
        if not _rule_semantic_matches(rule_id, matching, source, target):
            return rule_id, "edit_does_not_match_rule_family", matching
    return None


def _row_rule_ids(row: dict[str, Any] | pd.Series) -> list[str]:
    raw = _json_list(row.get("rule_ids", ""))
    if not raw:
        raw = _json_list(_json_dict(row.get("metadata", "")).get("rule_ids", []))
    primary = str(row.get("rule_id") or "")
    values = [str(item) for item in raw if str(item)]
    if primary and primary not in values:
        values.insert(0, primary)
    result: list[str] = []
    for rule_id in values:
        if rule_id and rule_id not in result:
            result.append(rule_id)
    return result


def _row_edits(row: dict[str, Any] | pd.Series) -> list[dict[str, Any]]:
    for key in ("edits", "edit_operations"):
        parsed = _json_list(row.get(key, ""))
        edits = [item for item in parsed if isinstance(item, dict)]
        if edits:
            return edits
    return []


def _matching_edits(rule_id: str, edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matching: list[dict[str, Any]] = []
    for edit in edits:
        edit_rule = str(edit.get("rule_id") or "")
        edit_rules = [str(item) for item in _json_list(edit.get("rule_ids", []))]
        if edit_rule == rule_id or rule_id in edit_rules:
            matching.append(edit)
    return matching


def _rule_semantic_matches(rule_id: str, edits: list[dict[str, Any]], source: str, target: str) -> bool:
    if rule_id in PUNCTUATION_RULE_IDS or any(marker in rule_id for marker in PUNCTUATION_RULE_MARKERS):
        return _punctuation_rule_matches(rule_id, edits, source, target)
    if rule_id.startswith("hyphen_"):
        return _hyphen_rule_matches(edits, source, target)
    if rule_id == "pol_polu_compounds":
        return _pol_polu_rule_matches(edits, source, target)
    if rule_id.startswith("ne_") or rule_id.startswith("ni_"):
        return _ne_ni_rule_matches(rule_id, edits, source, target)
    if rule_id.startswith("n_nn_"):
        return _n_nn_rule_matches(edits, source, target)
    if rule_id.startswith("tsya_"):
        return _tsya_rule_matches(edits, source, target)
    if rule_id in CONTEXT_RULE_PAIRS:
        return _context_rule_matches(rule_id, edits, source, target)
    if rule_id in PATTERN_RULE_PAIRS:
        return _pattern_rule_matches(rule_id, edits, source, target)
    if rule_id.startswith("prefix_"):
        return _prefix_rule_matches(rule_id, edits, source, target)
    if rule_id in {"missing_hard_sign", "soft_to_hard_sign"}:
        return _hard_soft_sign_rule_matches(edits)
    if rule_id == "capitalization_sentence_start":
        return _case_rule_matches(edits) and bool(target[:1].isupper())
    if rule_id == "abbreviation_case_protection":
        return _case_rule_matches(edits) and bool(re.search(r"\b[А-ЯЁ]{2,}\b", target))
    if rule_id in {
        "dictionary_fuzzy",
        "frequent_error_exact",
        "missing_letter_candidate",
        "extra_letter_candidate",
        "swapped_letters_candidate",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
    }:
        return _lexical_candidate_rule_matches(rule_id, edits)
    return _has_non_punctuation_edit(edits) or _punctuation_rule_matches(rule_id, edits, source, target)


def _punctuation_rule_matches(rule_id: str, edits: list[dict[str, Any]], source: str, target: str) -> bool:
    edit_text = _combined_edit_text(edits)
    edit_types = _edit_types(edits)
    if rule_id == "final_punctuation_default":
        return any(item == "final_punctuation" for item in edit_types) or source.rstrip()[-1:] != target.rstrip()[-1:]
    if rule_id in DASH_RULE_IDS:
        return " — " in target and "—" in edit_text + source + target
    if rule_id == "semicolon":
        return ";" in edit_text
    if "comma" in rule_id:
        return "," in edit_text or any("punctuation" in item for item in edit_types)
    if "colon" in rule_id:
        return ":" in edit_text
    if "quote" in rule_id or "speech" in rule_id:
        return any(char in edit_text + target for char in ('"', "«", "»", "—", ":"))
    if "bracket" in rule_id:
        return any(char in edit_text + target for char in "()[]{}")
    return any("punctuation" in item or any(char in _edit_pair_text(edit) for char in PUNCTUATION_CHARS) for item in edit_types for edit in edits)


def _hyphen_rule_matches(edits: list[dict[str, Any]], source: str, target: str) -> bool:
    if not any("-" in _edit_pair_text(edit) or "hyphen" in str(edit.get("edit_type") or "") for edit in edits):
        return False
    return source.replace("-", "") != target.replace("-", "") or source != target


def _pol_polu_rule_matches(edits: list[dict[str, Any]], source: str, target: str) -> bool:
    if not SAFE_POL_POLU_TARGET_RE.search(target):
        return False
    edit_text = _combined_edit_text(edits).lower()
    combined = f"{source}\n{target}".lower()
    return bool(re.search(r"\bпол(?:у)?\s+[а-яё]+|\bпол-[а-яё]+|\bполу[а-яё]+", edit_text + "\n" + combined))


def _ne_ni_rule_matches(rule_id: str, edits: list[dict[str, Any]], source: str, target: str) -> bool:
    text = _combined_edit_text(edits).lower()
    combined = f"{source}\n{target}".lower()
    particle = "ни" if rule_id.startswith("ni_") else "не"
    if particle not in text and particle not in combined:
        return False
    if rule_id == "ne_verb":
        return bool(re.search(r"\bне\s+[а-яё]+", target.lower()) or re.search(r"\bне[а-яё]+", source.lower()))
    return True


def _n_nn_rule_matches(edits: list[dict[str, Any]], source: str, target: str) -> bool:
    for edit in edits:
        left = str(edit.get("source") or "").lower()
        right = str(edit.get("replacement") or "").lower()
        if "н" in left + right and abs(left.count("н") - right.count("н")) in {1, 2}:
            return True
    return _contains_n_nn_delta(source, target)


def _tsya_rule_matches(edits: list[dict[str, Any]], source: str, target: str) -> bool:
    text = f"{_combined_edit_text(edits)}\n{source}\n{target}".lower()
    return "тся" in text and "ться" in text


def _context_rule_matches(rule_id: str, edits: list[dict[str, Any]], source: str, target: str) -> bool:
    haystack = f"{_combined_edit_text(edits)}\n{source}\n{target}".lower()
    return any(left in haystack and right in haystack for left, right in CONTEXT_RULE_PAIRS[rule_id])


def _pattern_rule_matches(rule_id: str, edits: list[dict[str, Any]], source: str, target: str) -> bool:
    bad, good = PATTERN_RULE_PAIRS[rule_id]
    text = f"{_combined_edit_text(edits)}\n{source}\n{target}".lower()
    if bad in text and good in text:
        return True
    if rule_id.endswith(("_же", "_ше", "_че", "_ще")) and bad in text and bad[:-1] + "ё" in text:
        return True
    return False


def _prefix_rule_matches(rule_id: str, edits: list[dict[str, Any]], source: str, target: str) -> bool:
    text = f"{_combined_edit_text(edits)}\n{source}\n{target}".lower()
    if rule_id == "prefix_pre_pri":
        return bool(re.search(r"\bпр[еи]", text))
    if rule_id in {"prefix_s_to_z", "prefix_z_to_s"}:
        return "с" in text and "з" in text and _has_non_punctuation_edit(edits)
    return _has_non_punctuation_edit(edits)


def _hard_soft_sign_rule_matches(edits: list[dict[str, Any]]) -> bool:
    text = _combined_edit_text(edits).lower()
    return "ъ" in text or "ь" in text


def _case_rule_matches(edits: list[dict[str, Any]]) -> bool:
    for edit in edits:
        left = str(edit.get("source") or "")
        right = str(edit.get("replacement") or "")
        if str(edit.get("edit_type") or "") == "case_change" or (left.lower() == right.lower() and left != right):
            return True
    return False


def _lexical_candidate_rule_matches(rule_id: str, edits: list[dict[str, Any]]) -> bool:
    if not _has_non_punctuation_edit(edits):
        return False
    if rule_id in {"dictionary_fuzzy", "frequent_error_exact", "keyboard_typo_candidate"}:
        return True
    for edit in edits:
        raw_left = str(edit.get("source") or "").lower()
        raw_right = str(edit.get("replacement") or "").lower()
        left = re.sub(r"\s+", "", str(edit.get("source") or "").lower())
        right = re.sub(r"\s+", "", str(edit.get("replacement") or "").lower())
        if not left or not right:
            continue
        if left == right:
            continue
        if rule_id == "missing_letter_candidate" and len(right) > len(left) and _is_subsequence(left, right):
            return True
        if rule_id == "extra_letter_candidate" and len(left) > len(right) and _is_subsequence(right, left):
            return True
        if rule_id == "swapped_letters_candidate" and len(left) == len(right) and _is_adjacent_swap(left, right):
            return True
        if rule_id == "double_consonant_candidate" and _double_consonant_delta(left, right):
            return True
        if rule_id in {"dictionary_fuzzy", "frequent_error_exact", "keyboard_typo_candidate"}:
            return True
    return False


def _has_non_punctuation_edit(edits: list[dict[str, Any]]) -> bool:
    return any("punctuation" not in str(edit.get("edit_type") or "") for edit in edits)


def _combined_edit_text(edits: list[dict[str, Any]]) -> str:
    return "\n".join(_edit_pair_text(edit) for edit in edits)


def _edit_pair_text(edit: dict[str, Any]) -> str:
    return f"{edit.get('source', '')}\n{edit.get('replacement', '')}"


def _edit_types(edits: list[dict[str, Any]]) -> list[str]:
    return [str(edit.get("edit_type") or "") for edit in edits]


def _is_subsequence(needle: str, haystack: str) -> bool:
    iterator = iter(haystack)
    return all(char in iterator for char in needle)


def _is_adjacent_swap(left: str, right: str) -> bool:
    diffs = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
    return len(diffs) == 2 and diffs[1] == diffs[0] + 1 and left[diffs[0]] == right[diffs[1]] and left[diffs[1]] == right[diffs[0]]


def _double_consonant_delta(left: str, right: str) -> bool:
    consonants = "бвгджзйклмнпрстфхцчшщ"
    for char in consonants:
        if abs(left.count(char) - right.count(char)) == 1 and (char * 2 in left or char * 2 in right):
            return True
    return False


def _contains_n_nn_delta(source: str, target: str) -> bool:
    source_words = re.findall(r"[А-Яа-яЁё]+", source.lower())
    target_words = re.findall(r"[А-Яа-яЁё]+", target.lower())
    for left, right in zip(source_words, target_words):
        if left == right:
            continue
        if "н" in left + right and abs(left.count("н") - right.count("н")) in {1, 2}:
            return True
    return False


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
