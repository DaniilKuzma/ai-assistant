from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.evaluation.direct_metrics import exact_match, normalize_text_for_eval
from src.runtime.corrector import Corrector
from src.schema import GeneratedExample
from src.schema.serialization import read_jsonl_examples


DEFAULT_THRESHOLD_GRID = tuple(round(index / 20, 2) for index in range(0, 21))
OBJECTIVE_PENALTY = 2.0


def tune_thresholds(
    config_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path = "reports/threshold_tuning.json",
    threshold_grid: list[float] | tuple[float, ...] | None = None,
    sweep_per_rule: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    dataset = read_jsonl_examples(dataset_path)
    grid = _normalized_grid(threshold_grid or DEFAULT_THRESHOLD_GRID)

    selected = _current_thresholds(config)
    token_trials = _sweep_global(config, dataset, grid, selected, "token_edit")
    selected["token_edit"] = _best_trial(token_trials)["threshold"]

    punctuation_trials = _sweep_global(config, dataset, grid, selected, "punctuation")
    selected["punctuation"] = _best_trial(punctuation_trials)["threshold"]

    per_rule_trials: dict[str, list[dict[str, Any]]] = {}
    accepted_rule_thresholds: dict[str, float] = {}
    if sweep_per_rule:
        selected["rule_thresholds"] = {}
        baseline_summary = _evaluate_config(_config_with_thresholds(config, selected), dataset)
        for rule_id in _enabled_rule_ids(dataset):
            trials = _sweep_rule(config, dataset, grid, selected, rule_id)
            per_rule_trials[rule_id] = trials
            best_trial = _best_trial(trials)
            if _is_meaningful_rule_improvement(best_trial, baseline_summary):
                threshold = float(best_trial["threshold"])
                selected.setdefault("rule_thresholds", {})[rule_id] = threshold
                accepted_rule_thresholds[rule_id] = threshold
                baseline_summary = _evaluate_config(_config_with_thresholds(config, selected), dataset)

    final_config = _config_with_thresholds(config, selected)
    final_summary = _evaluate_config(final_config, dataset)
    selected["combined_score"] = final_summary["combined_score"]

    output_obj = {
        "runtime": {
            "confidence_thresholds": _threshold_yaml_payload(selected),
        }
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(output_obj, allow_unicode=True, sort_keys=True), encoding="utf-8")

    report = {
        "dataset_path": str(dataset_path),
        "example_count": len(dataset),
        "objective": "exact_match - 2.0 * overcorrection_rate",
        "selected_thresholds": selected,
        "final_metrics": final_summary,
        "threshold_grid": grid,
        "token_edit_trials": token_trials,
        "punctuation_trials": punctuation_trials,
        "per_rule_trials": per_rule_trials,
        "accepted_rule_thresholds": accepted_rule_thresholds,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _sweep_global(
    config: dict[str, Any],
    dataset: list[GeneratedExample],
    grid: list[float],
    selected: dict[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for threshold in grid:
        candidate = copy.deepcopy(selected)
        candidate[name] = threshold
        metrics = _evaluate_config(_config_with_thresholds(config, candidate), dataset)
        trials.append({"threshold": threshold, **metrics})
    return trials


def _sweep_rule(
    config: dict[str, Any],
    dataset: list[GeneratedExample],
    grid: list[float],
    selected: dict[str, Any],
    rule_id: str,
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for threshold in grid:
        candidate = copy.deepcopy(selected)
        candidate.setdefault("rule_thresholds", {})[rule_id] = threshold
        metrics = _evaluate_config(_config_with_thresholds(config, candidate), dataset)
        trials.append({"threshold": threshold, **metrics})
    return trials


def _evaluate_config(config: dict[str, Any], dataset: list[GeneratedExample]) -> dict[str, Any]:
    corrector = Corrector.from_config(config)
    exact_count = 0
    guarded_count = 0
    overcorrected_count = 0

    for example in dataset:
        result = corrector.correct(example.source_text)
        is_exact = exact_match(example.target_text, result.corrected_text)
        exact_count += int(is_exact)
        if example.mode in {"clean_identity", "hard_negative"}:
            guarded_count += 1
            source = normalize_text_for_eval(example.source_text)
            prediction = normalize_text_for_eval(result.corrected_text)
            overcorrected_count += int(prediction != source)

    exact_rate = _safe_rate(exact_count, len(dataset))
    overcorrection_rate = _safe_rate(overcorrected_count, guarded_count)
    return {
        "exact_match": exact_rate,
        "overcorrection_rate": overcorrection_rate,
        "combined_score": exact_rate - OBJECTIVE_PENALTY * overcorrection_rate,
    }


def _best_trial(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("threshold grid must not be empty")
    return max(trials, key=lambda trial: (float(trial["combined_score"]), float(trial["exact_match"]), -float(trial["threshold"])))


def _is_meaningful_rule_improvement(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    epsilon = 1e-12
    return (
        float(candidate["combined_score"]) > float(baseline["combined_score"]) + epsilon
        and float(candidate["exact_match"]) >= float(baseline["exact_match"]) - epsilon
        and float(candidate["overcorrection_rate"]) <= float(baseline["overcorrection_rate"]) + epsilon
    )


def _current_thresholds(config: dict[str, Any]) -> dict[str, Any]:
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    raw = runtime.get("confidence_thresholds", {}) if isinstance(runtime, dict) else {}
    thresholds = raw if isinstance(raw, dict) else {}
    rule_thresholds = thresholds.get("rule_thresholds", {})
    if not isinstance(rule_thresholds, dict):
        rule_thresholds = {}
    return {
        "token_edit": float(thresholds.get("token_edit", 0.7)),
        "punctuation": float(thresholds.get("punctuation", 0.7)),
        "min_margin": float(thresholds.get("min_margin", 0.0)),
        "rule_thresholds": {str(rule_id): float(value) for rule_id, value in rule_thresholds.items()},
    }


def _config_with_thresholds(config: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    runtime = cloned.setdefault("runtime", {})
    confidence_thresholds = runtime.setdefault("confidence_thresholds", {})
    confidence_thresholds["token_edit"] = float(thresholds["token_edit"])
    confidence_thresholds["punctuation"] = float(thresholds["punctuation"])
    confidence_thresholds["min_margin"] = float(thresholds.get("min_margin", 0.0))
    rule_thresholds = thresholds.get("rule_thresholds", {})
    if rule_thresholds:
        confidence_thresholds["rule_thresholds"] = {
            str(rule_id): float(value)
            for rule_id, value in sorted(rule_thresholds.items())
        }
    else:
        confidence_thresholds.pop("rule_thresholds", None)
    return cloned


def _threshold_yaml_payload(thresholds: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "min_margin": float(thresholds.get("min_margin", 0.0)),
        "punctuation": float(thresholds["punctuation"]),
        "token_edit": float(thresholds["token_edit"]),
    }
    rule_thresholds = thresholds.get("rule_thresholds", {})
    if rule_thresholds:
        payload["rule_thresholds"] = {
            str(rule_id): float(value)
            for rule_id, value in sorted(rule_thresholds.items())
        }
    return payload


def _enabled_rule_ids(dataset: list[GeneratedExample]) -> list[str]:
    rule_ids: set[str] = set()
    for example in dataset:
        rule_ids.add(example.primary_rule_id)
        rule_ids.update(example.rule_ids)
    return sorted(rule_id for rule_id in rule_ids if rule_id not in {"", "none", "clean_identity"})


def _normalized_grid(values: list[float] | tuple[float, ...]) -> list[float]:
    grid = sorted({round(float(value), 4) for value in values})
    if not grid:
        raise ValueError("threshold grid must not be empty")
    for value in grid:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"threshold values must be between 0 and 1: {value}")
    return grid


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune runtime confidence thresholds on frozen validation JSONL.")
    parser.add_argument("config", help="Path to YAML config.")
    parser.add_argument("--dataset", required=True, help="Frozen validation GeneratedExample JSONL.")
    parser.add_argument("--output", required=True, help="YAML path for selected runtime thresholds.")
    parser.add_argument("--per-rule", action="store_true", help="Also tune per-rule thresholds for rules present in eval JSONL.")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Frozen eval dataset not found: {dataset_path}", file=sys.stderr)
        print(
            "Run scripts/build_frozen_eval.py first, for example: "
            "python scripts/build_frozen_eval.py configs/config.yaml --split val --count 5000 "
            "--output data/generated_eval/val.jsonl",
            file=sys.stderr,
        )
        return 2

    report = tune_thresholds(
        config_path=args.config,
        dataset_path=dataset_path,
        output_path=args.output,
        sweep_per_rule=args.per_rule,
    )
    print(json.dumps(report["selected_thresholds"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["tune_thresholds"]
