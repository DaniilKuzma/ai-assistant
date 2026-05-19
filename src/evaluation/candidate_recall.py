from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from src.alignment.punctuation_label_builder import PUNCT_LABELS
from src.candidates.candidate_generator import CandidateGenerator
from src.candidates.candidate_ranking import rank_candidates_for_budget
from src.candidates.matching import candidate_matches_edit
from src.evaluation.rule_metrics import _gold_edits, _load_rule_groups
from src.preprocessing.tokenizer import Token, tokenize_words
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import PUNCTUATION_TYPES


CANDIDATE_RECALL_COLUMNS = [
    "rule_id",
    "group",
    "gold_count",
    "candidate_present_count",
    "candidate_recall",
    "missing_count",
    "missing_examples",
]
GAP_LABEL_COVERAGE_COLUMNS = [
    "rule_id",
    "group",
    "gold_gap_count",
    "candidate_gap_present_count",
    "gap_candidate_recall",
    "missing_examples",
]


def build_candidate_recall_reports(
    rows: Iterable[dict[str, Any]],
    candidate_generator: CandidateGenerator | None = None,
    max_candidates: int | None = None,
    rules_config_path: str | Path = "configs/rules.yaml",
    max_missing_examples: int = 20,
) -> dict[str, pd.DataFrame]:
    generator = candidate_generator or CandidateGenerator()
    rule_groups = _load_rule_groups(rules_config_path)
    analyzer = DiffAnalyzer()
    gold_counter: Counter[str] = Counter()
    present_counter: Counter[str] = Counter()
    missing_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gold_gap_counter: Counter[str] = Counter()
    present_gap_counter: Counter[str] = Counter()
    missing_gap_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_match_cache: dict[tuple[str, str, str, str, str], bool] = {}
    gap_match_cache: dict[tuple[str, str, str, str, str, str], bool] = {}

    for row in rows:
        source = str(row.get("source", ""))
        gold_edits = _gold_edits(row, analyzer)
        if not gold_edits:
            continue

        candidates: list[Any] | None = None
        candidate_gap_keys: set[tuple[int | None, str, str]] | None = None

        def get_candidates() -> list[Any]:
            nonlocal candidates
            if candidates is None:
                generated = generator.generate(source)
                candidates = rank_candidates_for_budget(generated, int(max_candidates)) if max_candidates is not None else generated
            return candidates

        def get_candidate_gap_keys() -> set[tuple[int | None, str, str]]:
            nonlocal candidate_gap_keys
            if candidate_gap_keys is None:
                candidate_gap_keys = {
                    (candidate.gap_index, candidate.action, candidate.label)
                    for candidate in get_candidates()
                    if candidate.gap_index is not None and candidate.action
                }
            return candidate_gap_keys

        for edit in gold_edits:
            rule_id = normalize_rule_id(edit.rule_id)
            gold_counter[rule_id] += 1
            cache_key = _candidate_match_cache_key(source, rule_id, edit)
            candidate_present = candidate_match_cache.get(cache_key)
            if candidate_present is None:
                candidate_present = any(candidate_matches_edit(candidate, edit) for candidate in get_candidates())
                candidate_match_cache[cache_key] = candidate_present
            if candidate_present:
                present_counter[rule_id] += 1
            else:
                _append_missing_example(missing_examples[rule_id], row, edit, max_missing_examples)

        words = tokenize_words(source)
        for edit in gold_edits:
            if edit.edit_type not in PUNCTUATION_TYPES:
                continue
            gap_index = _punctuation_edit_gap_index(words, edit)
            action, label = _punctuation_action_label(edit)
            if gap_index < 0 or not action:
                continue
            rule_id = normalize_rule_id(edit.rule_id)
            gold_gap_counter[rule_id] += 1
            gap_cache_key = _gap_match_cache_key(source, rule_id, edit, action, label)
            gap_present = gap_match_cache.get(gap_cache_key)
            if gap_present is None:
                gap_present = (gap_index, action, label) in get_candidate_gap_keys()
                gap_match_cache[gap_cache_key] = gap_present
            if gap_present:
                present_gap_counter[rule_id] += 1
            else:
                _append_missing_gap_example(
                    missing_gap_examples[rule_id],
                    row,
                    edit,
                    gap_index,
                    action,
                    label,
                    max_missing_examples,
                )

    return {
        "candidate_recall_by_rule": _candidate_recall_frame(
            gold_counter,
            present_counter,
            missing_examples,
            rule_groups,
        ),
        "gap_label_coverage_by_rule": _gap_label_coverage_frame(
            gold_gap_counter,
            present_gap_counter,
            missing_gap_examples,
            rule_groups,
        ),
    }


def _candidate_match_cache_key(source: str, rule_id: str, edit: Edit) -> tuple[str, str, str, str, str]:
    return (
        _normalize_recall_source(source),
        rule_id,
        edit.edit_type,
        edit.source,
        edit.replacement,
    )


def _gap_match_cache_key(source: str, rule_id: str, edit: Edit, action: str, label: str) -> tuple[str, str, str, str, str, str]:
    return (
        _normalize_recall_source(source),
        rule_id,
        edit.edit_type,
        edit.replacement,
        action,
        label,
    )


def _normalize_recall_source(source: str) -> str:
    text = source.lower().strip()
    text = re.sub(r"\d+", "<NUM>", text)
    return re.sub(r"\s+", " ", text)


def write_candidate_recall_reports(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    candidate_generator: CandidateGenerator | None = None,
    max_candidates: int | None = None,
    rules_config_path: str | Path = "configs/rules.yaml",
    max_missing_examples: int = 20,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = build_candidate_recall_reports(
        rows,
        candidate_generator=candidate_generator,
        max_candidates=max_candidates,
        rules_config_path=rules_config_path,
        max_missing_examples=max_missing_examples,
    )
    reports["candidate_recall_by_rule"].to_csv(output / "candidate_recall_by_rule.csv", index=False)
    reports["gap_label_coverage_by_rule"].to_csv(output / "gap_label_coverage_by_rule.csv", index=False)


def _candidate_recall_frame(
    gold_counter: Counter[str],
    present_counter: Counter[str],
    missing_examples: dict[str, list[dict[str, Any]]],
    rule_groups: dict[str, str],
) -> pd.DataFrame:
    rows = [
        {
            "rule_id": rule_id,
            "group": rule_groups.get(rule_id, UNKNOWN_RULE_ID),
            "gold_count": gold_counter[rule_id],
            "candidate_present_count": present_counter[rule_id],
            "candidate_recall": _safe_rate(present_counter[rule_id], gold_counter[rule_id]),
            "missing_count": max(0, gold_counter[rule_id] - present_counter[rule_id]),
            "missing_examples": _json_examples(missing_examples.get(rule_id, [])),
        }
        for rule_id in sorted(gold_counter)
    ]
    return pd.DataFrame(rows, columns=CANDIDATE_RECALL_COLUMNS)


def _gap_label_coverage_frame(
    gold_gap_counter: Counter[str],
    present_gap_counter: Counter[str],
    missing_gap_examples: dict[str, list[dict[str, Any]]],
    rule_groups: dict[str, str],
) -> pd.DataFrame:
    rows = [
        {
            "rule_id": rule_id,
            "group": rule_groups.get(rule_id, UNKNOWN_RULE_ID),
            "gold_gap_count": gold_gap_counter[rule_id],
            "candidate_gap_present_count": present_gap_counter[rule_id],
            "gap_candidate_recall": _safe_rate(present_gap_counter[rule_id], gold_gap_counter[rule_id]),
            "missing_examples": _json_examples(missing_gap_examples.get(rule_id, [])),
        }
        for rule_id in sorted(gold_gap_counter)
    ]
    return pd.DataFrame(rows, columns=GAP_LABEL_COVERAGE_COLUMNS)


def _append_missing_example(
    examples: list[dict[str, Any]],
    row: dict[str, Any],
    edit: Edit,
    limit: int,
) -> None:
    if len(examples) >= limit:
        return
    examples.append(_missing_example(row, edit))


def _append_missing_gap_example(
    examples: list[dict[str, Any]],
    row: dict[str, Any],
    edit: Edit,
    gap_index: int,
    action: str,
    label: str,
    limit: int,
) -> None:
    if len(examples) >= limit:
        return
    example = _missing_example(row, edit)
    example.update({"gap_index": gap_index, "action": action, "label": label})
    examples.append(example)


def _missing_example(row: dict[str, Any], edit: Edit) -> dict[str, Any]:
    source = str(row.get("source", ""))
    return {
        "source": source,
        "target": str(row.get("target", "")),
        "edit_type": edit.edit_type,
        "start": edit.start,
        "end": edit.end,
        "source_fragment": _source_fragment(source, edit),
        "replacement": edit.replacement,
    }


def _source_fragment(source: str, edit: Edit) -> str:
    if edit.start >= 0 and edit.end >= edit.start:
        return source[edit.start : edit.end]
    return edit.source


def _punctuation_action_label(edit: Edit) -> tuple[str, str]:
    if edit.edit_type == "punctuation_delete":
        return "DELETE", "NONE"
    if edit.edit_type == "punctuation_replace":
        return "REPLACE", PUNCT_LABELS.get(edit.replacement, "NONE")
    if edit.edit_type == "final_punctuation":
        return ("REPLACE" if edit.source else "INSERT"), PUNCT_LABELS.get(edit.replacement, "NONE")
    if edit.edit_type == "punctuation_insert":
        return "INSERT", PUNCT_LABELS.get(edit.replacement, "NONE")
    return "", "NONE"


def _punctuation_edit_gap_index(words: list[Token], edit: Edit) -> int:
    if not words:
        return -1
    if edit.edit_type == "final_punctuation":
        return len(words) - 1
    return _gap_index_for_position(words, edit.start)


def _gap_index_for_position(words: list[Token], position: int) -> int:
    if position < 0:
        return 0
    gap = 0
    for index, word in enumerate(words):
        if word.end <= position:
            gap = index
        elif word.start > position:
            break
    return max(0, min(gap, len(words) - 1))


def _json_examples(examples: list[dict[str, Any]]) -> str:
    return json.dumps(examples, ensure_ascii=False)


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
