from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import time
from typing import Any, Iterable

import pandas as pd
from rapidfuzz.distance import Levenshtein
import yaml

from src.app.streamlit_app import build_streamlit_corrector
from src.candidates.candidate_generator import CandidateGenerator
from src.config.load_config import load_config
from src.evaluation.fast_eval import (
    _candidate_predictions_from_feature,
    _punctuation_candidates_from_feature,
    _punctuation_predictions_from_feature,
)
from src.inference.model_corrector import (
    ModelPunctuationPrediction,
    TrainedModelCorrector,
    _apply_candidates,
    _apply_punctuation_predictions,
    _select_candidates_with_trace,
)
from src.inference.postprocess import normalize_spacing
from src.model.encoder import load_tokenizer, EncoderLoadConfig
from src.training.tensorization import build_features_from_rows
from src.validation.diff_analyzer import DiffAnalyzer


ROOT = Path(".")
BASE_CONFIG_PATH = Path("configs/config.yaml")
SEED_CONFIG_PATH = Path("configs/config_threshold_calibrated_gui_spelling_v2.yaml")
OUTPUT_CONFIG_PATH = Path("configs/config_threshold_calibrated_gui_spelling_v3.yaml")
DATASET_PATH = Path("data/processed/short_dataset_v2/correction_dataset.csv.gz")
OUT_DIR = Path("reports/threshold_calibration_large_v3")
MODE_NAME = "calibrated_gui_spelling_large_v3"
RANDOM_SEED = 20260521


PUNCTUATION_EDIT_TYPES = {
    "punctuation_insert",
    "punctuation_delete",
    "punctuation_replace",
    "final_punctuation",
}
WORD_EDIT_TYPES = {"spelling", "split_join", "hyphen", "case"}
HIGH_RISK_THRESHOLD_KEYS = {
    "context_pair_threshold",
    "tsya_threshold",
    "ne_adjective_threshold",
    "ne_participle_threshold",
    "ne_adverb_threshold",
    "quote_threshold",
    "quote_open_threshold",
    "quote_close_threshold",
    "quote_pair_balance_threshold",
    "bracket_threshold",
    "bracket_pair_balance_threshold",
    "punctuation_delete_threshold",
    "punctuation_delete_replace_threshold",
    "capitalization_ner_threshold",
}
RULE_FAMILY_ALIASES = {
    "dictionary_fuzzy": "dictionary_fuzzy_threshold",
    "double_consonant_candidate": "double_consonant_candidate_threshold",
    "keyboard_typo_candidate": "keyboard_typo_candidate_threshold",
    "swapped_letters_candidate": "swapped_letters_candidate_threshold",
    "missing_letter_candidate": "missing_letter_candidate_threshold",
    "extra_letter_candidate": "extra_letter_candidate_threshold",
    "yo_e_candidate": "yo_e_candidate_threshold",
    "frequent_error_exact": "frequent_error_exact_threshold",
    "hyphen_koe_koy": "hyphen_koe_koy_threshold",
    "hyphen_particles": "hyphen_particles_threshold",
    "hyphen_po_adverbs": "hyphen_po_adverbs_threshold",
    "introductory_comma": "introductory_word_threshold",
    "subject_predicate_dash": "subject_predicate_dash_threshold",
}
PUNCTUATION_LABEL_KEYS = {
    "COMMA": "comma_threshold",
    "DOT": "final_punctuation_threshold",
    "QUESTION": "final_punctuation_threshold",
    "EXCLAMATION": "final_punctuation_threshold",
    "ELLIPSIS": "final_punctuation_threshold",
    "COLON": "colon_threshold",
    "DASH": "dash_threshold",
    "SEMICOLON": "semicolon_threshold",
    "QUOTE_OPEN": "quote_threshold",
    "QUOTE_CLOSE": "quote_threshold",
    "BRACKET_OPEN": "bracket_threshold",
    "BRACKET_CLOSE": "bracket_threshold",
}
EXPLICIT_RULE_KEYS = {
    "comma_subordinate",
    "comma_conjunction",
    "direct_speech_colon",
    "direct_speech_quotes",
    "subject_predicate_dash",
    "dash_subject_predicate",
    "introductory_comma",
    "capitalization_sentence_start",
    "n_nn_adjective",
    "n_nn_deverbal_adjective",
    "n_nn_participle",
    "n_nn_short_form",
    "cy_exception",
}
THRESHOLD_FLOORS = {
    "default_threshold": 0.40,
    "spelling_threshold": 0.35,
    "dictionary_threshold": 0.35,
    "dictionary_fuzzy_threshold": 0.35,
    "frequent_error_exact_threshold": 0.05,
    "double_consonant_candidate_threshold": 0.25,
    "keyboard_typo_candidate_threshold": 0.25,
    "swapped_letters_candidate_threshold": 0.25,
    "missing_letter_candidate_threshold": 0.25,
    "extra_letter_candidate_threshold": 0.25,
    "split_join_threshold": 0.35,
    "hyphen_threshold": 0.35,
    "hyphen_koe_koy_threshold": 0.20,
    "hyphen_particles_threshold": 0.20,
    "hyphen_po_adverbs_threshold": 0.20,
    "case_threshold": 0.85,
    "capitalization_sentence_start_threshold": 0.85,
    "comma_threshold": 0.65,
    "punctuation_threshold": 0.65,
    "final_punctuation_threshold": 0.65,
    "colon_threshold": 0.75,
    "dash_threshold": 0.75,
    "semicolon_threshold": 0.85,
    "introductory_word_threshold": 0.70,
    "comma_subordinate_threshold": 0.70,
    "comma_conjunction_threshold": 0.85,
    "direct_speech_colon_threshold": 0.95,
    "direct_speech_quotes_threshold": 0.95,
    "subject_predicate_dash_threshold": 0.85,
    "dash_subject_predicate_threshold": 0.85,
    "context_pair_threshold": 0.95,
    "tsya_threshold": 0.95,
    "ne_adjective_threshold": 0.98,
    "ne_participle_threshold": 0.98,
    "ne_adverb_threshold": 0.98,
    "quote_threshold": 0.95,
    "quote_open_threshold": 0.99,
    "quote_close_threshold": 0.99,
    "bracket_threshold": 0.95,
    "punctuation_delete_threshold": 0.98,
    "punctuation_delete_replace_threshold": 0.99,
    "capitalization_ner_threshold": 0.999,
}


@dataclass(frozen=True)
class CandidateDecision:
    split: str
    row_id: int
    threshold_key: str
    score: float
    is_positive: bool
    rule_id: str
    edit_type: str
    source: str
    replacement: str


def main() -> None:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_config(BASE_CONFIG_PATH)
    seed_config_path = SEED_CONFIG_PATH if SEED_CONFIG_PATH.exists() else BASE_CONFIG_PATH
    seed_config = load_config(seed_config_path)
    base_mode = str(base_config["thresholds"]["mode"])
    seed_mode = str(seed_config["thresholds"]["mode"])
    base_profile = deepcopy(base_config["thresholds"][base_mode])
    seed_profile = deepcopy(seed_config["thresholds"][seed_mode])

    rows = _load_rows()
    _write_csv(OUT_DIR / "large_calibration_rows.csv", rows)
    print(
        f"Loaded {len(rows)} rows: {Counter(row['split'] for row in rows)} "
        f"from {DATASET_PATH}",
        flush=True,
    )

    corrector = build_streamlit_corrector(seed_config_path).corrector
    if not isinstance(corrector, TrainedModelCorrector):
        raise RuntimeError(f"Expected trained corrector, got {type(corrector).__name__}")

    features = _build_features(seed_config, corrector, rows)
    scores, model_forward_time = corrector.backend.score_features_batched(
        features,
        batch_size=max(1, int(seed_config.get("evaluation", {}).get("batch_size", 128))),
        mixed_precision=bool(seed_config.get("training", {}).get("mixed_precision", False)),
    )
    print(f"Model scoring finished in {model_forward_time:.2f}s", flush=True)

    decisions = _collect_decisions(corrector, rows, features, scores, seed_profile)
    _write_csv(OUT_DIR / "decision_summary_by_threshold.csv", _decision_summary(decisions))
    print(f"Collected {len(decisions)} threshold decisions", flush=True)

    tuned_profile, threshold_rows = _tune_profile(seed_profile, decisions, calibration_split="val")
    _write_csv(OUT_DIR / "threshold_search_large_v3.csv", threshold_rows)

    profiles = {
        "baseline": base_profile,
        "seed_v2": seed_profile,
        "large_v3": tuned_profile,
    }
    eval_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for profile_name, profile in profiles.items():
        print(f"Evaluating profile {profile_name}", flush=True)
        evaluated = _evaluate_profile(corrector, rows, features, scores, profile, profile_name)
        eval_rows.extend(evaluated)
        category_rows.extend(_category_summary(evaluated, profile_name))
        sample_rows.extend(_interesting_samples(evaluated, profile_name))
    _write_csv(OUT_DIR / "profile_eval_rows.csv", eval_rows)
    _write_csv(OUT_DIR / "profile_comparison_by_category.csv", category_rows)
    _write_csv(OUT_DIR / "interesting_samples.csv", sample_rows)

    summaries = _profile_summaries(eval_rows)
    _write_json(
        OUT_DIR / "comparison_summary_large_v3.json",
        {
            "dataset_path": str(DATASET_PATH),
            "rows": len(rows),
            "seed_config": str(seed_config_path),
            "output_config": str(OUTPUT_CONFIG_PATH),
            "base_mode": base_mode,
            "seed_mode": seed_mode,
            "output_mode": MODE_NAME,
            "model_forward_time_sec": model_forward_time,
            "total_time_sec": time.perf_counter() - started,
            "summaries": summaries,
            "changed_thresholds": _changed_thresholds(seed_profile, tuned_profile),
        },
    )
    _write_report(
        OUT_DIR / "comparison_report_large_v3.md",
        summaries,
        threshold_rows,
        _changed_thresholds(seed_profile, tuned_profile),
    )
    _write_output_config(seed_config, tuned_profile)
    print(f"Wrote {OUTPUT_CONFIG_PATH}", flush=True)
    print(f"Done in {time.perf_counter() - started:.2f}s", flush=True)


def _load_rows() -> list[dict[str, Any]]:
    frame = pd.read_csv(DATASET_PATH)
    frame = frame[frame["split"].isin(["val", "test"])].copy()
    frame["source"] = frame["source"].map(_clean_text)
    frame["target"] = frame["target"].map(_clean_text)
    frame = frame[(frame["source"].str.len() > 0) & (frame["target"].str.len() > 0)]
    frame = frame.sort_values(["split", "rule_id", "source"]).reset_index(drop=True)
    rows = []
    for row_id, row in enumerate(frame.to_dict("records")):
        rows.append(
            {
                "row_id": row_id,
                "source": row["source"],
                "target": row["target"],
                "split": row["split"],
                "rule_id": str(row.get("rule_id", "")),
                "error_type": str(row.get("error_type", "")),
                "source_dataset": str(row.get("source_dataset", "")),
                "is_clean": bool(row.get("is_clean", False)),
                "is_hard_negative": bool(row.get("is_hard_negative", False)),
            }
        )
    return rows


def _build_features(config: dict[str, Any], corrector: TrainedModelCorrector, rows: list[dict[str, Any]]) -> list[Any]:
    model_config = config.get("model", {}) or {}
    tokenizer = load_tokenizer(
        EncoderLoadConfig(
            model_name=model_config.get("primary_encoder", "ai-forever/ruRoberta-large"),
            fallback_model_name=model_config.get("fallback_encoder", "ai-forever/ruRoberta-large"),
            local_files_only=bool(model_config.get("local_files_only", False)),
        )
    )
    return build_features_from_rows(
        rows,
        tokenizer=tokenizer,
        punctuation_label_map=config.get("labels", {}).get("punctuation", {}),
        punctuation_action_label_map=config.get("labels", {}).get("punctuation_actions", {}),
        error_type_label_map=config.get("labels", {}).get("error_types", {}),
        max_length=int(model_config.get("max_sequence_length", 128)),
        max_candidates=int(model_config.get("max_candidates", 16)),
        candidate_generator=CandidateGenerator.from_config(config),
        show_progress=True,
        split="eval",
    )


def _collect_decisions(
    corrector: TrainedModelCorrector,
    rows: list[dict[str, Any]],
    features: list[Any],
    scores: list[Any],
    profile: dict[str, Any],
) -> list[CandidateDecision]:
    decisions: list[CandidateDecision] = []
    action_by_id = {value: key for key, value in (corrector.backend.punctuation_action_labels or {}).items()}
    label_by_id = corrector.backend.punctuation_by_id
    for row, feature, feature_scores in zip(rows, features, scores, strict=False):
        for index, prediction in enumerate(_candidate_predictions_from_feature(feature, feature_scores)):
            candidate = prediction.candidate
            if candidate.edit_type == "keep":
                continue
            key = _threshold_key_for_word(candidate.rule_id, candidate.edit_type, profile)
            if not key:
                continue
            label = bool(index < len(feature.candidate_labels) and float(feature.candidate_labels[index]) >= 0.5)
            decisions.append(
                CandidateDecision(
                    split=str(row["split"]),
                    row_id=int(row["row_id"]),
                    threshold_key=key,
                    score=float(prediction.score),
                    is_positive=label,
                    rule_id=candidate.rule_id,
                    edit_type=candidate.edit_type,
                    source=candidate.source,
                    replacement=candidate.replacement,
                )
            )
        punct_index = _punctuation_candidate_index(_punctuation_candidates_from_feature(feature))
        for prediction in _punctuation_predictions_from_feature(corrector, feature, feature_scores):
            if prediction.action in {"KEEP_NONE", "KEEP_EXISTING"} or prediction.label == "NONE":
                continue
            matched = punct_index.get((prediction.gap_index, prediction.action, prediction.label))
            if matched is None:
                continue
            rule_id = matched.rule_id
            key = _threshold_key_for_punctuation(rule_id, prediction.action, prediction.label, profile)
            if not key:
                continue
            gap_index = int(prediction.gap_index)
            true_label = label_by_id.get(int(feature.punctuation_labels[gap_index]), "NONE")
            true_action = action_by_id.get(int(feature.punctuation_action_labels[gap_index]), "KEEP_NONE")
            label = true_label == prediction.label and true_action == prediction.action
            decisions.append(
                CandidateDecision(
                    split=str(row["split"]),
                    row_id=int(row["row_id"]),
                    threshold_key=key,
                    score=float(prediction.confidence),
                    is_positive=label,
                    rule_id=rule_id,
                    edit_type=matched.edit_type,
                    source=matched.source,
                    replacement=matched.replacement,
                )
            )
    return decisions


def _tune_profile(
    seed_profile: dict[str, Any],
    decisions: list[CandidateDecision],
    *,
    calibration_split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[CandidateDecision]] = defaultdict(list)
    for decision in decisions:
        if decision.split == calibration_split:
            grouped[decision.threshold_key].append(decision)

    tuned = deepcopy(seed_profile)
    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        positives = sum(1 for item in items if item.is_positive)
        negatives = len(items) - positives
        if positives == 0:
            continue
        current = float(seed_profile.get(key, _fallback_threshold(key, seed_profile)))
        floor = _threshold_floor(key, current)
        grid = _threshold_grid(floor)
        best: tuple[float, float, float, int, int, int, float] | None = None
        for threshold in grid:
            tp = sum(1 for item in items if item.score >= threshold and item.is_positive)
            fp = sum(1 for item in items if item.score >= threshold and not item.is_positive)
            fn = positives - tp
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / positives if positives else 0.0
            objective = _threshold_objective(key, tp, fp, fn, precision, recall)
            candidate = (objective, precision, recall, -fp, tp, -fn, threshold)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            continue
        objective, precision, recall, neg_fp, tp, neg_fn, threshold = best
        fp = -neg_fp
        fn = -neg_fn
        if tp == 0:
            threshold = current
        tuned[key] = float(threshold)
        rows.append(
            {
                "threshold_key": key,
                "current": current,
                "tuned": float(threshold),
                "floor": floor,
                "decision_count": len(items),
                "positive_count": positives,
                "negative_count": negatives,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "objective": objective,
            }
        )
    return tuned, rows


def _evaluate_profile(
    corrector: TrainedModelCorrector,
    rows: list[dict[str, Any]],
    features: list[Any],
    scores: list[Any],
    profile: dict[str, Any],
    profile_name: str,
) -> list[dict[str, Any]]:
    profile_corrector = TrainedModelCorrector(
        corrector.backend,
        thresholds=profile,
        max_passes=1,
        candidate_generator=corrector.candidates,
    )
    analyzer = DiffAnalyzer()
    evaluated: list[dict[str, Any]] = []
    for row, feature, feature_scores in zip(rows, features, scores, strict=False):
        predictions = _candidate_predictions_from_feature(feature, feature_scores)
        selected, _decisions = _select_candidates_with_trace(predictions, profile)
        proposed = _apply_candidates(str(row["source"]), selected)
        proposed, punctuation_edits = _apply_punctuation_predictions(
            proposed,
            _punctuation_predictions_from_feature(profile_corrector, feature, feature_scores),
            profile,
            punctuation_candidates=_punctuation_candidates_from_feature(feature),
        )
        proposed = normalize_spacing(proposed)
        validation = profile_corrector.validator.validate(
            str(row["source"]),
            proposed,
            trusted_edits=[*selected, *punctuation_edits],
        )
        corrected = validation.apply_accepted()
        gold_edits = analyzer.analyze(row["source"], row["target"])
        predicted_edits = analyzer.analyze(row["source"], corrected)
        accepted = [edit for edit in validation.edits if edit.status == "accepted"]
        metrics = _row_metrics(str(row["source"]), str(row["target"]), corrected)
        evaluated.append(
            {
                "profile": profile_name,
                **row,
                "prediction": corrected,
                **metrics,
                "gold_edit_count": len(gold_edits),
                "predicted_edit_count": len(predicted_edits),
                "accepted_edit_count": len(accepted),
                "accepted_rules": "|".join(sorted({edit.rule_id for edit in accepted if edit.rule_id})),
            }
        )
    return evaluated


def _profile_summaries(eval_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for profile in sorted({row["profile"] for row in eval_rows}):
        for split in ["val", "test", "all"]:
            rows = [row for row in eval_rows if row["profile"] == profile and (split == "all" or row["split"] == split)]
            result[f"{profile}:{split}"] = _summary(rows)
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dirty = [row for row in rows if row["dirty"]]
    clean = [row for row in rows if row["clean"]]
    return {
        "rows": len(rows),
        "dirty_rows": len(dirty),
        "clean_rows": len(clean),
        "exact_match_rate": _rate(rows, "exact"),
        "dirty_exact_rate": _rate(dirty, "exact"),
        "dirty_improved_rate": _rate(dirty, "improved"),
        "dirty_worse_rate": _rate(dirty, "worse"),
        "clean_overcorrection_rate": _rate(clean, "clean_overcorrected"),
        "mean_distance_delta": sum(float(row["base_distance"]) - float(row["corrected_distance"]) for row in rows)
        / len(rows)
        if rows
        else 0.0,
        "accepted_edits_per_row": sum(int(row["accepted_edit_count"]) for row in rows) / len(rows) if rows else 0.0,
    }


def _category_summary(rows: list[dict[str, Any]], profile_name: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        for rule_id in sorted({str(row["rule_id"]) for row in split_rows}):
            group = [row for row in split_rows if str(row["rule_id"]) == rule_id]
            if len(group) < 3:
                continue
            summary = _summary(group)
            result.append({"profile": profile_name, "split": split, "rule_id": rule_id, **summary})
    return result


def _interesting_samples(rows: list[dict[str, Any]], profile_name: str) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        if row["split"] != "test":
            continue
        if row["clean_overcorrected"] or row["worse"] or (row["dirty"] and row["improved"]):
            samples.append(
                {
                    "profile": profile_name,
                    "row_id": row["row_id"],
                    "rule_id": row["rule_id"],
                    "error_type": row["error_type"],
                    "source": row["source"],
                    "target": row["target"],
                    "prediction": row["prediction"],
                    "exact": row["exact"],
                    "improved": row["improved"],
                    "worse": row["worse"],
                    "clean_overcorrected": row["clean_overcorrected"],
                    "accepted_rules": row["accepted_rules"],
                }
            )
        if len(samples) >= 300:
            break
    return samples


def _decision_summary(decisions: list[CandidateDecision]) -> list[dict[str, Any]]:
    result = []
    for (split, key), items in sorted(_group_by(decisions, lambda item: (item.split, item.threshold_key)).items()):
        positives = sum(1 for item in items if item.is_positive)
        result.append(
            {
                "split": split,
                "threshold_key": key,
                "decision_count": len(items),
                "positive_count": positives,
                "negative_count": len(items) - positives,
                "min_score": min(item.score for item in items),
                "max_score": max(item.score for item in items),
                "mean_score": sum(item.score for item in items) / len(items),
            }
        )
    return result


def _changed_thresholds(seed_profile: dict[str, Any], tuned_profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(seed_profile) | set(tuned_profile)):
        old = seed_profile.get(key)
        new = tuned_profile.get(key)
        if old is None or new is None:
            continue
        if abs(float(old) - float(new)) <= 1e-12:
            continue
        rows.append({"threshold_key": key, "seed_v2": float(old), "large_v3": float(new)})
    return rows


def _write_output_config(seed_config: dict[str, Any], tuned_profile: dict[str, Any]) -> None:
    output = deepcopy(seed_config)
    output.setdefault("thresholds", {})["mode"] = MODE_NAME
    output["thresholds"][MODE_NAME] = tuned_profile
    output.setdefault("paths", {})["reports_dir"] = str(OUT_DIR)
    with OUTPUT_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, allow_unicode=True, sort_keys=False)


def _threshold_key_for_word(rule_id: str, edit_type: str, profile: dict[str, Any]) -> str:
    exact = f"{rule_id}_threshold" if rule_id else ""
    if exact and (exact in profile or rule_id in EXPLICIT_RULE_KEYS):
        return exact
    if rule_id in RULE_FAMILY_ALIASES:
        return RULE_FAMILY_ALIASES[rule_id]
    if edit_type == "spelling":
        return "spelling_threshold"
    if edit_type == "split_join":
        return "split_join_threshold"
    if edit_type == "hyphen":
        return "hyphen_threshold"
    if edit_type == "case":
        return "case_threshold"
    return "default_threshold"


def _threshold_key_for_punctuation(rule_id: str, action: str, label: str, profile: dict[str, Any]) -> str:
    exact = f"{rule_id}_threshold" if rule_id else ""
    if exact and (exact in profile or rule_id in EXPLICIT_RULE_KEYS):
        return exact
    if rule_id in RULE_FAMILY_ALIASES:
        return RULE_FAMILY_ALIASES[rule_id]
    if action == "DELETE":
        return "punctuation_delete_threshold"
    return PUNCTUATION_LABEL_KEYS.get(label, "punctuation_threshold")


def _threshold_objective(
    key: str,
    tp: int,
    fp: int,
    fn: int,
    precision: float,
    recall: float,
) -> float:
    false_positive_penalty = 16.0 if key in HIGH_RISK_THRESHOLD_KEYS else 9.0
    false_negative_penalty = 1.0
    true_positive_reward = 3.0
    if key.startswith("frequent_error_exact"):
        false_positive_penalty = 12.0
        true_positive_reward = 4.0
    if key.startswith("hyphen"):
        false_positive_penalty = 10.0
    if "punctuation" in key or "comma" in key or key in {"colon_threshold", "dash_threshold"}:
        false_positive_penalty = max(false_positive_penalty, 12.0)
    objective = true_positive_reward * tp - false_positive_penalty * fp - false_negative_penalty * fn
    min_precision = 0.92 if key in HIGH_RISK_THRESHOLD_KEYS else 0.86
    if tp > 0 and precision < min_precision:
        objective -= 10_000.0
    if fp > 0 and key in HIGH_RISK_THRESHOLD_KEYS:
        objective -= 10_000.0
    return objective


def _threshold_grid(floor: float) -> list[float]:
    values = {round(x / 100, 2) for x in range(5, 100)}
    values.update({0.995, 0.999, 0.9999, 0.99995, 0.99999, 0.999999})
    return sorted(value for value in values if value >= floor)


def _threshold_floor(key: str, current: float) -> float:
    if key in THRESHOLD_FLOORS:
        return THRESHOLD_FLOORS[key]
    if key in HIGH_RISK_THRESHOLD_KEYS:
        return max(0.95, min(current, 0.999999))
    if key.endswith("_threshold") and ("quote" in key or "bracket" in key):
        return 0.95
    return 0.20


def _fallback_threshold(key: str, profile: dict[str, Any]) -> float:
    if key.startswith("hyphen"):
        return float(profile.get("hyphen_threshold", 0.90))
    if "dictionary" in key or key in {
        "frequent_error_exact_threshold",
        "double_consonant_candidate_threshold",
        "keyboard_typo_candidate_threshold",
        "swapped_letters_candidate_threshold",
        "missing_letter_candidate_threshold",
        "extra_letter_candidate_threshold",
    }:
        return float(profile.get("dictionary_threshold", profile.get("spelling_threshold", 0.90)))
    if "comma" in key:
        return float(profile.get("comma_threshold", profile.get("punctuation_threshold", 0.90)))
    if key in {"colon_threshold", "dash_threshold", "semicolon_threshold"}:
        return float(profile.get(key, profile.get("punctuation_threshold", 0.90)))
    return float(profile.get("default_threshold", 0.90))


def _punctuation_candidate_index(candidates: Iterable[Any]) -> dict[tuple[int | None, str, str], Any]:
    result = {}
    for candidate in candidates:
        result[(candidate.gap_index, candidate.action, candidate.label)] = candidate
    return result


def _row_metrics(source: str, target: str, prediction: str) -> dict[str, Any]:
    base_distance = Levenshtein.distance(source, target)
    corrected_distance = Levenshtein.distance(prediction, target)
    dirty = source != target
    clean = not dirty
    return {
        "exact": prediction == target,
        "changed": prediction != source,
        "dirty": dirty,
        "clean": clean,
        "improved": dirty and corrected_distance < base_distance,
        "worse": corrected_distance > base_distance,
        "clean_overcorrected": clean and prediction != source,
        "base_distance": base_distance,
        "corrected_distance": corrected_distance,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if bool(row[key])) / len(rows) if rows else 0.0


def _clean_text(value: Any) -> str:
    return str(value).replace("\ufeff", "").strip()


def _group_by(values: Iterable[Any], key_fn: Any) -> dict[Any, list[Any]]:
    result: dict[Any, list[Any]] = defaultdict(list)
    for value in values:
        result[key_fn(value)].append(value)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_report(
    path: Path,
    summaries: dict[str, dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    changed_thresholds: list[dict[str, Any]],
) -> None:
    lines = [
        "# Large Threshold Calibration v3",
        "",
        f"- Dataset: `{DATASET_PATH}`",
        "- Calibration split: `val`",
        "- Holdout split: `test`",
        f"- Output config: `{OUTPUT_CONFIG_PATH}`",
        "",
        "## Profile Summary",
        "",
        "| profile:split | rows | dirty exact | dirty improved | dirty worse | clean overcorr | mean distance delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['rows']} | {summary['dirty_exact_rate']:.4f} | "
            f"{summary['dirty_improved_rate']:.4f} | {summary['dirty_worse_rate']:.4f} | "
            f"{summary['clean_overcorrection_rate']:.4f} | {summary['mean_distance_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Changed Thresholds",
            "",
            "| threshold | seed_v2 | large_v3 |",
            "|---|---:|---:|",
        ]
    )
    for row in changed_thresholds:
        lines.append(f"| {row['threshold_key']} | {row['seed_v2']:.6f} | {row['large_v3']:.6f} |")
    lines.extend(
        [
            "",
            "## Search Evidence",
            "",
            "| threshold | current | tuned | val decisions | positives | fp | precision | recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_rows:
        if abs(float(row["current"]) - float(row["tuned"])) <= 1e-12:
            continue
        lines.append(
            f"| {row['threshold_key']} | {row['current']:.6f} | {row['tuned']:.6f} | "
            f"{row['decision_count']} | {row['positive_count']} | {row['fp']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
