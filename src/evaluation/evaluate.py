from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.config.load_config import load_config
from src.evaluation.direct_metrics import (
    char_accuracy,
    edit_precision_recall_f1,
    exact_match,
    normalize_text_for_eval,
    rule_breakdown,
)
from src.runtime.corrector import Corrector
from src.runtime.edit_realizer import apply_gap_labels, apply_token_edit_labels
from src.schema import GeneratedExample, RuntimeEdit
from src.schema.serialization import read_jsonl_examples


@dataclass
class _RuntimeDiagnostics:
    rejected_by_threshold: int = 0
    rejected_by_scope_guard: int = 0


def evaluate_corrector(
    config_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    examples = read_jsonl_examples(dataset_path)
    corrector = Corrector.from_config(config)
    diagnostics = _instrument_corrector(corrector)

    rows: list[dict[str, Any]] = []
    for row_id, example in enumerate(examples):
        threshold_before = diagnostics.rejected_by_threshold
        scope_before = diagnostics.rejected_by_scope_guard
        result = corrector.correct(example.source_text)
        threshold_count = diagnostics.rejected_by_threshold - threshold_before
        scope_count = diagnostics.rejected_by_scope_guard - scope_before
        if threshold_count == 0:
            threshold_count = _metadata_count(result.metadata, "rejected_by_threshold", "rejected_by_threshold_count")
        if scope_count == 0:
            scope_count = _metadata_count(
                result.metadata,
                "rejected_by_scope_guard",
                "rejected_by_scope_guard_count",
                "scope_guard_rejections",
            )

        predicted_text = result.corrected_text
        gold_edits = _gold_runtime_edits(example)
        predicted_edits = list(result.edits)
        rows.append(
            {
                "row_id": row_id,
                "example": example,
                "source_text": example.source_text,
                "target_text": example.target_text,
                "predicted_text": predicted_text,
                "primary_rule_id": example.primary_rule_id,
                "mode": example.mode,
                "exact_match": exact_match(example.target_text, predicted_text),
                "char_accuracy": char_accuracy(example.source_text, example.target_text, predicted_text),
                "gold_edits": [_serialize_edit(edit) for edit in gold_edits],
                "predicted_edits": [_serialize_edit(edit) for edit in predicted_edits],
                "rejected_by_threshold": threshold_count,
                "rejected_by_scope_guard": scope_count,
                "metadata": dict(example.metadata),
                "runtime_metadata": dict(result.metadata),
            }
        )

    summary = _summary_metrics(examples, rows)
    per_rule = rule_breakdown(examples, rows)
    _write_outputs(Path(output_dir), summary, per_rule, rows)
    return summary


def _summary_metrics(examples: Sequence[GeneratedExample], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        edit_metrics = edit_precision_recall_f1([], [])
        return {
            "example_count": 0,
            "exact_match": 0.0,
            "char_accuracy": 0.0,
            "changed_when_needed": 0.0,
            "unchanged_when_clean": 0.0,
            "overcorrection_rate": 0.0,
            "edit_precision": edit_metrics["precision"],
            "edit_recall": edit_metrics["recall"],
            "edit_f1": edit_metrics["f1"],
            "rejected_by_threshold": 0,
            "rejected_by_scope_guard": 0,
        }

    exact_total = 0
    char_total = 0.0
    changed_examples = 0
    changed_when_needed = 0
    clean_examples = 0
    unchanged_when_clean = 0
    guarded_examples = 0
    overcorrected = 0
    all_gold_edits: list[dict[str, Any]] = []
    all_predicted_edits: list[dict[str, Any]] = []

    for row in rows:
        row_id = int(row["row_id"])
        source_text = str(row["source_text"])
        target_text = str(row["target_text"])
        predicted_text = str(row["predicted_text"])
        normalized_source = normalize_text_for_eval(source_text)
        normalized_target = normalize_text_for_eval(target_text)
        normalized_prediction = normalize_text_for_eval(predicted_text)

        exact_total += int(row["exact_match"])
        char_total += float(row["char_accuracy"])
        if normalized_source != normalized_target:
            changed_examples += 1
            changed_when_needed += int(normalized_prediction != normalized_source)
        else:
            clean_examples += 1
            unchanged_when_clean += int(normalized_prediction == normalized_source)

        if str(row["mode"]) in {"clean_identity", "hard_negative"}:
            guarded_examples += 1
            overcorrected += int(not bool(row["exact_match"]))

        all_gold_edits.extend(_row_tagged_edits(row.get("gold_edits", []), row_id))
        all_predicted_edits.extend(_row_tagged_edits(row.get("predicted_edits", []), row_id))

    edit_metrics = edit_precision_recall_f1(all_gold_edits, all_predicted_edits)
    return {
        "example_count": len(rows),
        "exact_match": exact_total / len(rows),
        "char_accuracy": char_total / len(rows),
        "changed_when_needed": _safe_rate(changed_when_needed, changed_examples),
        "unchanged_when_clean": _safe_rate(unchanged_when_clean, clean_examples),
        "overcorrection_rate": _safe_rate(overcorrected, guarded_examples),
        "edit_precision": edit_metrics["precision"],
        "edit_recall": edit_metrics["recall"],
        "edit_f1": edit_metrics["f1"],
        "rejected_by_threshold": sum(int(row.get("rejected_by_threshold", 0) or 0) for row in rows),
        "rejected_by_scope_guard": sum(int(row.get("rejected_by_scope_guard", 0) or 0) for row in rows),
    }


def _gold_runtime_edits(example: GeneratedExample) -> list[RuntimeEdit]:
    confidences = [1.0] * len(example.source_tokens)
    _token_text, token_edits = apply_token_edit_labels(
        example.source_text,
        example.source_tokens,
        example.token_edit_labels,
        confidences,
        threshold=0.0,
        rule_ids=example.rule_ids,
    )
    _gap_text, gap_edits = apply_gap_labels(
        example.source_text,
        example.source_tokens,
        example.gap_labels,
        confidences,
        threshold=0.0,
        rule_ids=example.rule_ids,
    )
    return [*token_edits, *gap_edits]


def _write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    per_rule: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_per_rule_metrics(output_dir / "per_rule_metrics.csv", per_rule)
    _write_worst_examples(output_dir / "worst_examples.jsonl", rows)


def _write_per_rule_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "rule_id",
        "example_count",
        "exact_match",
        "char_accuracy",
        "changed_when_needed",
        "unchanged_when_clean",
        "overcorrection_rate",
        "edit_precision",
        "edit_recall",
        "edit_f1",
        "rejected_by_threshold",
        "rejected_by_scope_guard",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_worst_examples(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    worst_rows = sorted(
        (row for row in rows if not bool(row.get("exact_match", False))),
        key=lambda row: (float(row.get("char_accuracy", 0.0)), int(row.get("row_id", 0))),
    )
    with path.open("w", encoding="utf-8") as handle:
        for row in worst_rows:
            handle.write(
                json.dumps(
                    {
                        "source_text": row["source_text"],
                        "target_text": row["target_text"],
                        "predicted_text": row["predicted_text"],
                        "primary_rule_id": row["primary_rule_id"],
                        "mode": row["mode"],
                        "edits": row["predicted_edits"],
                        "metadata": row["metadata"],
                        "runtime_metadata": row["runtime_metadata"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")


def _instrument_corrector(corrector: Any) -> _RuntimeDiagnostics:
    diagnostics = _RuntimeDiagnostics()
    thresholds = getattr(corrector, "thresholds", None)
    if thresholds is not None and hasattr(thresholds, "should_apply"):
        corrector.thresholds = _CountingThresholds(thresholds, diagnostics)
    scope_guard = getattr(corrector, "scope_guard", None)
    if scope_guard is not None:
        corrector.scope_guard = _CountingScopeGuard(scope_guard, diagnostics)
    return diagnostics


class _CountingThresholds:
    def __init__(self, wrapped: Any, diagnostics: _RuntimeDiagnostics) -> None:
        self._wrapped = wrapped
        self._diagnostics = diagnostics

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def should_apply(self, confidence: float, margin: float, rule_id: str, edit_type: str) -> bool:
        accepted = bool(self._wrapped.should_apply(confidence, margin, rule_id, edit_type))
        if not accepted:
            self._diagnostics.rejected_by_threshold += 1
        return accepted


class _CountingScopeGuard:
    def __init__(self, wrapped: Any, diagnostics: _RuntimeDiagnostics) -> None:
        self._wrapped = wrapped
        self._diagnostics = diagnostics

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def validate_edit(self, source_text: str, edit: RuntimeEdit) -> bool:
        accepted = bool(self._wrapped.validate_edit(source_text, edit))
        if not accepted:
            self._diagnostics.rejected_by_scope_guard += 1
        return accepted

    def validate_result(
        self,
        source_text: str,
        corrected_text: str,
        edits: Sequence[RuntimeEdit],
    ) -> tuple[bool, list[str]]:
        accepted, reasons = self._wrapped.validate_result(source_text, corrected_text, edits)
        if not accepted:
            self._diagnostics.rejected_by_scope_guard += max(1, len(reasons))
        return accepted, reasons


def _serialize_edit(edit: Any) -> dict[str, Any]:
    return {
        "start": int(getattr(edit, "start", 0)),
        "end": int(getattr(edit, "end", 0)),
        "source": str(getattr(edit, "source", "")),
        "replacement": str(getattr(edit, "replacement", "")),
        "edit_type": str(getattr(edit, "edit_type", "")),
        "rule_id": str(getattr(edit, "rule_id", "")),
        "confidence": float(getattr(edit, "confidence", 0.0)),
        "explanation": str(getattr(edit, "explanation", "")),
    }


def _row_tagged_edits(edits: Sequence[Mapping[str, Any]], row_id: int) -> list[dict[str, Any]]:
    tagged = []
    for edit in edits:
        row = dict(edit)
        row["row_id"] = row_id
        tagged.append(row)
    return tagged


def _metadata_count(metadata: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key in metadata:
            return _count_value(metadata[key])
    return 0


def _count_value(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return 1 if value else 0
    if isinstance(value, Mapping):
        return sum(_count_value(item) for item in value.values())
    if isinstance(value, Sequence):
        return len(value)
    return 0


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate runtime Corrector on frozen GeneratedExample JSONL.")
    parser.add_argument("config")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary = evaluate_corrector(args.config, args.dataset, args.output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_corrector"]
