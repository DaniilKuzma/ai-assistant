from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any
import unicodedata


def normalize_text_for_eval(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).replace("\r\n", "\n").replace("\r", "\n").strip()


def char_accuracy(source: str, target: str, prediction: str) -> float:
    del source
    normalized_target = normalize_text_for_eval(target)
    normalized_prediction = normalize_text_for_eval(prediction)
    if not normalized_target and not normalized_prediction:
        return 1.0
    if not normalized_target or not normalized_prediction:
        return 0.0
    return SequenceMatcher(None, normalized_target, normalized_prediction).ratio()


def exact_match(target: str, prediction: str) -> bool:
    return normalize_text_for_eval(target) == normalize_text_for_eval(prediction)


def edit_precision_recall_f1(
    gold_edits: Iterable[Any],
    predicted_edits: Iterable[Any],
) -> dict[str, float]:
    counts = edit_counts(gold_edits, predicted_edits)
    precision = _safe_rate(counts["true_positive"], counts["predicted_count"])
    recall = _safe_rate(counts["true_positive"], counts["gold_count"])
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(counts["true_positive"]),
        "false_positive": float(counts["false_positive"]),
        "false_negative": float(counts["false_negative"]),
        "gold_count": float(counts["gold_count"]),
        "predicted_count": float(counts["predicted_count"]),
    }


def token_label_accuracy(gold_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    return _label_accuracy(gold_labels, predicted_labels)


def gap_label_accuracy(gold_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    return _label_accuracy(gold_labels, predicted_labels)


def rule_breakdown(examples: Sequence[Any], predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[tuple[Any, Mapping[str, Any]]]] = defaultdict(list)
    for example, prediction in zip(examples, predictions, strict=False):
        buckets[_example_attr(example, "primary_rule_id", "unknown")].append((example, prediction))

    rows: list[dict[str, Any]] = []
    for rule_id in sorted(buckets):
        pairs = buckets[rule_id]
        gold_edits: list[Any] = []
        predicted_edits: list[Any] = []
        changed_examples = 0
        changed_when_needed = 0
        clean_examples = 0
        unchanged_when_clean = 0
        guarded_examples = 0
        overcorrected = 0
        exact = 0
        char_total = 0.0
        rejected_by_threshold = 0
        rejected_by_scope_guard = 0

        for row_index, (example, prediction) in enumerate(pairs):
            source_text = _example_attr(example, "source_text", "")
            target_text = _example_attr(example, "target_text", "")
            predicted_text = str(prediction.get("predicted_text", ""))
            normalized_source = normalize_text_for_eval(source_text)
            normalized_target = normalize_text_for_eval(target_text)
            normalized_prediction = normalize_text_for_eval(predicted_text)
            is_exact = normalized_prediction == normalized_target
            exact += int(is_exact)
            char_total += char_accuracy(source_text, target_text, predicted_text)

            if normalized_source != normalized_target:
                changed_examples += 1
                changed_when_needed += int(normalized_prediction != normalized_source)
            else:
                clean_examples += 1
                unchanged_when_clean += int(normalized_prediction == normalized_source)

            if _example_attr(example, "mode", "") in {"clean_identity", "hard_negative"}:
                guarded_examples += 1
                overcorrected += int(not is_exact)

            gold_edits.extend(_row_tagged_edits(prediction.get("gold_edits", []), row_index))
            predicted_edits.extend(_row_tagged_edits(prediction.get("predicted_edits", []), row_index))
            rejected_by_threshold += int(prediction.get("rejected_by_threshold", 0) or 0)
            rejected_by_scope_guard += int(prediction.get("rejected_by_scope_guard", 0) or 0)

        edit_metrics = edit_precision_recall_f1(gold_edits, predicted_edits)
        rows.append(
            {
                "rule_id": rule_id,
                "example_count": len(pairs),
                "exact_match": _safe_rate(exact, len(pairs)),
                "char_accuracy": _safe_rate_float(char_total, len(pairs)),
                "changed_when_needed": _safe_rate(changed_when_needed, changed_examples),
                "unchanged_when_clean": _safe_rate(unchanged_when_clean, clean_examples),
                "overcorrection_rate": _safe_rate(overcorrected, guarded_examples),
                "edit_precision": edit_metrics["precision"],
                "edit_recall": edit_metrics["recall"],
                "edit_f1": edit_metrics["f1"],
                "rejected_by_threshold": rejected_by_threshold,
                "rejected_by_scope_guard": rejected_by_scope_guard,
            }
        )
    return rows


def edit_counts(gold_edits: Iterable[Any], predicted_edits: Iterable[Any]) -> dict[str, int]:
    gold = Counter(_edit_key(edit) for edit in gold_edits)
    predicted = Counter(_edit_key(edit) for edit in predicted_edits)
    true_positive = sum((gold & predicted).values())
    gold_count = sum(gold.values())
    predicted_count = sum(predicted.values())
    return {
        "true_positive": true_positive,
        "false_positive": max(0, predicted_count - true_positive),
        "false_negative": max(0, gold_count - true_positive),
        "gold_count": gold_count,
        "predicted_count": predicted_count,
    }


def _label_accuracy(gold_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    if not gold_labels:
        return 0.0
    matches = sum(
        1
        for index, gold_label in enumerate(gold_labels)
        if index < len(predicted_labels) and str(predicted_labels[index]) == str(gold_label)
    )
    return matches / len(gold_labels)


def _edit_key(edit: Any) -> tuple[Any, ...]:
    row_id = _edit_value(edit, "row_id", None)
    key = (
        int(_edit_value(edit, "start", 0)),
        int(_edit_value(edit, "end", 0)),
        str(_edit_value(edit, "source", "")),
        str(_edit_value(edit, "replacement", "")),
        str(_edit_value(edit, "edit_type", "")),
        str(_edit_value(edit, "rule_id", "")),
    )
    if row_id is None:
        return key
    return (row_id, *key)


def _edit_value(edit: Any, name: str, default: Any) -> Any:
    if isinstance(edit, Mapping):
        return edit.get(name, default)
    return getattr(edit, name, default)


def _example_attr(example: Any, name: str, default: Any) -> Any:
    if isinstance(example, Mapping):
        return example.get(name, default)
    return getattr(example, name, default)


def _row_tagged_edits(edits: Iterable[Any], row_index: int) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for edit in edits:
        if isinstance(edit, Mapping):
            row = dict(edit)
        else:
            row = {
                "start": getattr(edit, "start", 0),
                "end": getattr(edit, "end", 0),
                "source": getattr(edit, "source", ""),
                "replacement": getattr(edit, "replacement", ""),
                "edit_type": getattr(edit, "edit_type", ""),
                "rule_id": getattr(edit, "rule_id", ""),
            }
        row["row_id"] = row_index
        tagged.append(row)
    return tagged


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _safe_rate_float(numerator: float, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


__all__ = [
    "char_accuracy",
    "edit_counts",
    "edit_precision_recall_f1",
    "exact_match",
    "gap_label_accuracy",
    "normalize_text_for_eval",
    "rule_breakdown",
    "token_label_accuracy",
]
