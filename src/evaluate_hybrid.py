"""Evaluation for the final hybrid corrector."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from edit_labels import action_replace_rank, build_training_example
from hybrid_corrector import HybridCorrector
from quality_guard import levenshtein_distance
from text_utils import normalize_word


PUNCT_CHARS = set('.,!?;:-"()[]{}—–«»')


def char_error_rate(predicted: str, target: str) -> float:
    target = str(target)
    if not target:
        return 0.0 if not predicted else 1.0
    return levenshtein_distance(str(predicted), target) / len(target)


def word_error_rate(predicted: str, target: str) -> float:
    pred_words = str(predicted).split()
    target_words = str(target).split()
    if not target_words:
        return 0.0 if not pred_words else 1.0
    return levenshtein_distance(" ".join(pred_words), " ".join(target_words)) / max(len(" ".join(target_words)), 1)


def punctuation_only(text: str) -> str:
    return "".join(ch for ch in str(text) if ch in PUNCT_CHARS)


def punctuation_similarity(predicted: str, target: str) -> float:
    return SequenceMatcher(None, punctuation_only(predicted), punctuation_only(target)).ratio()


def punctuation_count_error(predicted: str, target: str) -> int:
    pred_counts = Counter(punctuation_only(predicted))
    target_counts = Counter(punctuation_only(target))
    return sum(abs(pred_counts[ch] - target_counts[ch]) for ch in set(pred_counts) | set(target_counts))


def normalize_without_spacing(text: str) -> str:
    return "".join(str(text).split())


def is_word_edit_action(action: str) -> bool:
    action = str(action)
    return action == "DELETE" or action == "REPLACE" or action.startswith("REPLACE_")


def explode_error_types(value: str) -> list[str]:
    value = str(value or "unknown")
    labels: list[str] = []
    for chunk in value.replace("|", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            labels.append(chunk)
    return labels or ["unknown"]


def summarize(df: pd.DataFrame, include_slices: bool = False) -> dict:
    clean_mask = df["input_cer"] == 0
    wrong_mask = df["input_cer"] > 0
    result = {
        "examples": int(len(df)),
        "exact_match": float(df["exact_match"].mean()),
        "mean_input_cer": float(df["input_cer"].mean()),
        "mean_pred_cer": float(df["pred_cer"].mean()),
        "mean_cer_delta": float(df["cer_delta"].mean()),
        "improved_rate": float(df["improved"].mean()),
        "worse_rate": float(df["worse"].mean()),
        "unchanged_rate": float(df["unchanged"].mean()),
        "unchanged_wrong_rate": float(df.loc[wrong_mask, "unchanged"].mean()) if wrong_mask.any() else None,
        "clean_overcorrection_rate": float(df.loc[clean_mask, "clean_overcorrection"].mean()) if clean_mask.any() else None,
        "word_overcorrection_rate": float(df.loc[clean_mask, "word_overcorrection"].mean()) if clean_mask.any() and "word_overcorrection" in df else None,
        "punct_overcorrection_rate": float(df.loc[clean_mask, "punct_overcorrection"].mean()) if clean_mask.any() and "punct_overcorrection" in df else None,
        "punct_input_similarity": float(df["punct_input_similarity"].mean()),
        "punct_pred_similarity": float(df["punct_pred_similarity"].mean()),
        "punct_similarity_delta": float(df["punct_pred_similarity"].mean() - df["punct_input_similarity"].mean()),
        "mean_punct_count_error": float(df["punct_count_error"].mean()),
        "mean_confidence": float(df["confidence"].mean()),
        "accepted_rate": float(df["accepted"].mean()),
        "spacing_only_overcorrection_rate": (
            float(df.loc[clean_mask, "spacing_only_overcorrection"].mean())
            if clean_mask.any() and "spacing_only_overcorrection" in df
            else None
        ),
        "layout_changed_without_edits_count": int(df.get("layout_changed_without_edits", pd.Series(dtype=bool)).sum()),
    }
    if include_slices:
        result["clean_slice"] = summarize(df.loc[clean_mask].copy(), include_slices=False) if clean_mask.any() else None
        result["dirty_slice"] = summarize(df.loc[wrong_mask].copy(), include_slices=False) if wrong_mask.any() else None
    return result


def summarize_by_error_type(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, value in enumerate(df["error_types"].fillna("unknown")):
        for label in explode_error_types(value):
            groups[label].append(idx)

    rows = []
    for label, indices in groups.items():
        if len(indices) < min_count:
            continue
        part = df.iloc[indices]
        item = summarize(part)
        item["error_type"] = label
        rows.append(item)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(["mean_cer_delta", "worse_rate"], ascending=[True, False])


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def candidate_coverage_for_pair(
    error_text: str,
    correct_text: str,
    generator,
    candidate_top_k: int,
) -> dict[str, object]:
    example = build_training_example(error_text, correct_text)
    stats = {
        "replace_token_count": 0,
        "target_candidate_hit_count": 0,
        "target_rank_distribution": Counter(),
    }
    if example is None:
        return stats

    for idx, (source, action) in enumerate(zip(example.source_words, example.action_labels)):
        if action_replace_rank(action) is None:
            continue
        target = example.target_words[idx] if idx < len(example.target_words) else ""
        if not target or normalize_word(source) == normalize_word(target):
            continue
        stats["replace_token_count"] += 1
        candidates = generator.get_candidates(source, max_candidates=candidate_top_k)
        target_norm = normalize_word(target)
        rank = next(
            (i for i, candidate in enumerate(candidates) if normalize_word(candidate.text) == target_norm),
            -1,
        )
        stats["target_rank_distribution"][str(rank)] += 1
        if rank >= 0:
            stats["target_candidate_hit_count"] += 1
    return stats


def evaluate_hybrid(
    dataset_path: str = "data/processed/test.csv",
    sample_size: int | None = 500,
    output_dir: str = "report",
    strictness: str = "normal",
    punctuation_mode: str = "conservative",
    min_dictionary_score: float = 1.0,
    use_morphology_guard: bool = True,
    candidate_top_k: int = 5,
    context_reranker_enabled: bool = True,
    context_model_name: str = "DeepPavlov/rubert-base-cased",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    corrector = HybridCorrector(
        strictness=strictness,
        punctuation_mode=punctuation_mode,
        min_dictionary_score=min_dictionary_score,
        use_morphology_guard=use_morphology_guard,
        candidate_top_k=candidate_top_k,
        context_reranker_enabled=context_reranker_enabled,
        context_model_name=context_model_name,
    )
    if not corrector.is_trained:
        raise RuntimeError("Hybrid model is missing or obsolete. Retrain from the notebook before evaluation.")

    rows = []
    word_edit_rows = []
    punct_decision_rows = []
    coverage_totals = Counter()
    target_rank_distribution: Counter[str] = Counter()
    start = time.time()
    for i, row in df.iterrows():
        error_text = str(row["error_text"])
        correct_text = str(row["correct_text"])
        result = corrector.correct(error_text, return_details=True)
        predicted_text = result.corrected
        coverage = candidate_coverage_for_pair(
            error_text,
            correct_text,
            corrector.candidate_generator,
            candidate_top_k,
        )
        coverage_totals["replace_token_count"] += int(coverage["replace_token_count"])
        coverage_totals["target_candidate_hit_count"] += int(coverage["target_candidate_hit_count"])
        target_rank_distribution.update(coverage["target_rank_distribution"])

        input_cer = char_error_rate(error_text, correct_text)
        pred_cer = char_error_rate(predicted_text, correct_text)
        exact = predicted_text.strip() == correct_text.strip()
        unchanged = predicted_text.strip() == error_text.strip()
        is_clean = input_cer == 0

        word_edit_count = sum(1 for edit in result.edits if is_word_edit_action(edit.action))
        punct_edit_count = sum(1 for edit in result.edits if edit.action == "PUNCT")

        row_data = {
            "error_text": error_text,
            "correct_text": correct_text,
            "predicted_text": predicted_text,
            "difficulty": row.get("difficulty", "unknown"),
            "error_types": row.get("error_types", "unknown"),
            "input_cer": input_cer,
            "pred_cer": pred_cer,
            "cer_delta": input_cer - pred_cer,
            "exact_match": exact,
            "improved": pred_cer < input_cer,
            "unchanged": unchanged,
            "worse": pred_cer > input_cer,
            "clean_overcorrection": bool(is_clean and not unchanged),
            "spacing_only_overcorrection": bool(
                is_clean
                and not unchanged
                and normalize_without_spacing(predicted_text) == normalize_without_spacing(correct_text)
            ),
            "layout_changed_without_edits": bool(len(result.edits) == 0 and predicted_text != error_text),
            "word_overcorrection": bool(is_clean and word_edit_count > 0),
            "punct_overcorrection": bool(is_clean and punct_edit_count > 0),
            "punct_input_similarity": punctuation_similarity(error_text, correct_text),
            "punct_pred_similarity": punctuation_similarity(predicted_text, correct_text),
            "punct_count_error": punctuation_count_error(predicted_text, correct_text),
            "confidence": result.confidence,
            "accepted": result.accepted,
            "guard_reason": result.guard_reason,
            "edit_count": len(result.edits),
            "word_edit_count": word_edit_count,
            "punct_edit_count": punct_edit_count,
            "replace_token_count": int(coverage["replace_token_count"]),
            "target_candidate_hit_count": int(coverage["target_candidate_hit_count"]),
        }
        rows.append(row_data)
        for edit in result.edits:
            if not is_word_edit_action(edit.action):
                continue
            word_edit_rows.append(
                {
                    "row_index": i,
                    "difficulty": row_data["difficulty"],
                    "error_types": row_data["error_types"],
                    "is_clean": is_clean,
                    "row_improved": row_data["improved"],
                    "row_worse": row_data["worse"],
                    "row_unchanged": row_data["unchanged"],
                    "input_cer": input_cer,
                    "pred_cer": pred_cer,
                    "cer_delta": row_data["cer_delta"],
                    "source": edit.source,
                    "target": edit.target,
                    "action": edit.action,
                    "action_conf": edit.confidence,
                    "source_len": len(str(edit.source)),
                    "source_norm": str(edit.source).lower().replace("ё", "е"),
                    "source_is_lower": str(edit.source) == str(edit.source).lower(),
                    "source_is_title": str(edit.source).istitle(),
                    "source_is_upper": str(edit.source).isupper(),
                    "candidate_source": edit.candidate_source,
                    "candidate_distance": edit.candidate_distance,
                    "candidate_score": edit.candidate_score,
                    "candidate_rank": edit.candidate_rank,
                    "context_score": edit.context_score,
                    "context_margin": edit.context_margin,
                    "reranked": edit.reranked,
                    "reranker_reason": edit.reranker_reason,
                    "text": error_text,
                    "predicted": predicted_text,
                    "target_text": correct_text,
                }
            )
        for decision in result.punctuation_diagnostics:
            punct_decision_rows.append(
                {
                    "row_index": i,
                    "difficulty": row_data["difficulty"],
                    "error_types": row_data["error_types"],
                    "is_clean": is_clean,
                    "row_improved": row_data["improved"],
                    "row_worse": row_data["worse"],
                    "row_unchanged": row_data["unchanged"],
                    "index": decision.index,
                    "word": decision.word,
                    "source_punct": decision.source_punct,
                    "predicted_punct": decision.predicted_punct,
                    "confidence": decision.confidence,
                    "margin": decision.margin,
                    "applied": decision.applied,
                    "blocked_reason": decision.blocked_reason,
                    "is_final": decision.is_final,
                    "protected_gap": decision.protected_gap,
                    "punct_position_type": decision.punct_position_type,
                    "window_id": decision.window_id,
                    "window_conflict_resolved": decision.window_conflict_resolved,
                    "text": error_text,
                    "predicted": predicted_text,
                    "target_text": correct_text,
                }
            )
        if (i + 1) % 50 == 0:
            print(f"Evaluated {i + 1}/{len(df)} examples")

    analysis = pd.DataFrame(rows)
    analysis_path = Path(output_dir) / "error_analysis.csv"
    analysis.to_csv(analysis_path, index=False, encoding="utf-8")
    word_edit_columns = [
        "row_index", "difficulty", "error_types", "is_clean", "row_improved",
        "row_worse", "row_unchanged", "input_cer", "pred_cer", "cer_delta",
        "source", "target", "action", "action_conf", "source_len",
        "source_norm", "source_is_lower", "source_is_title", "source_is_upper",
        "candidate_source", "candidate_distance", "candidate_score", "candidate_rank",
        "context_score", "context_margin", "reranked", "reranker_reason", "text",
        "predicted", "target_text",
    ]
    word_edit_df = pd.DataFrame(word_edit_rows, columns=word_edit_columns)
    word_edit_df.to_csv(Path(output_dir) / "word_edit_diagnostics.csv", index=False, encoding="utf-8")
    punct_decision_columns = [
        "row_index", "difficulty", "error_types", "is_clean", "row_improved",
        "row_worse", "row_unchanged", "index", "word", "source_punct",
        "predicted_punct", "confidence", "margin", "applied",
        "blocked_reason", "is_final", "protected_gap", "punct_position_type", "window_id",
        "window_conflict_resolved", "text", "predicted", "target_text",
    ]
    punct_decision_df = pd.DataFrame(punct_decision_rows, columns=punct_decision_columns)
    punct_decision_df.to_csv(Path(output_dir) / "punct_edit_diagnostics.csv", index=False, encoding="utf-8")

    summary = summarize(analysis, include_slices=True)
    summary["elapsed_minutes"] = (time.time() - start) / 60
    summary["guard_reason_counts"] = value_counts_dict(analysis["guard_reason"])
    summary["accepted_guard_reason_counts"] = value_counts_dict(
        analysis.loc[analysis["accepted"], "guard_reason"]
    )
    summary["rejected_guard_reason_counts"] = value_counts_dict(
        analysis.loc[~analysis["accepted"], "guard_reason"]
    )
    summary["word_edit_candidate_source_counts"] = (
        value_counts_dict(word_edit_df["candidate_source"])
        if not word_edit_df.empty
        else {}
    )
    punct_change_mask = (
        punct_decision_df["predicted_punct"] != punct_decision_df["source_punct"]
        if not punct_decision_df.empty
        else pd.Series(dtype=bool)
    )
    summary["punct_applied_count"] = int(punct_decision_df["applied"].sum()) if not punct_decision_df.empty else 0
    summary["punct_change_candidate_count"] = int(punct_change_mask.sum()) if not punct_decision_df.empty else 0
    summary["punct_blocked_reason_counts"] = (
        value_counts_dict(punct_decision_df.loc[~punct_decision_df["applied"], "blocked_reason"])
        if not punct_decision_df.empty
        else {}
    )
    summary["protected_punct_gap_count"] = (
        int(punct_decision_df["protected_gap"].sum()) if not punct_decision_df.empty and "protected_gap" in punct_decision_df else 0
    )
    replace_tokens = int(coverage_totals["replace_token_count"])
    summary["candidate_coverage_rate"] = (
        float(coverage_totals["target_candidate_hit_count"] / replace_tokens) if replace_tokens else None
    )
    summary["target_rank_distribution"] = {str(key): int(value) for key, value in sorted(target_rank_distribution.items())}
    runtime_stats = getattr(corrector, "runtime_stats", Counter())
    summary["reranker_called_count"] = int(runtime_stats.get("reranker_called_count", 0))
    summary["reranker_changed_top1_count"] = int(runtime_stats.get("reranker_changed_top1_count", 0))
    summary["reranker_source_preferred_count"] = int(runtime_stats.get("reranker_source_preferred_count", 0))
    summary["reranker_fail_open_count"] = int(runtime_stats.get("reranker_fail_open_count", 0))
    split_seen = int(runtime_stats.get("split_candidates_seen", 0))
    split_applied = int((word_edit_df["candidate_source"] == "split").sum()) if not word_edit_df.empty else 0
    summary["split_candidate_accept_rate"] = float(split_applied / split_seen) if split_seen else None
    clean_rows = max(1, int((analysis["input_cer"] == 0).sum()))
    summary["internal_period_overcorrection_rate"] = (
        float(
            len(
                punct_decision_df[
                    (punct_decision_df["is_clean"] == True)
                    & (punct_decision_df["applied"] == True)
                    & (punct_decision_df["punct_position_type"] == "internal")
                    & (punct_decision_df["predicted_punct"] == ".")
                ]
            )
            / clean_rows
        )
        if not punct_decision_df.empty
        else 0.0
    )
    period_to_comma_mask = analysis["error_types"].astype(str).str.contains("punct_wrong_period_to_comma", na=False)
    summary["punct_wrong_period_to_comma_improved_rate"] = (
        float(analysis.loc[period_to_comma_mask, "improved"].mean()) if period_to_comma_mask.any() else None
    )
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    by_type = summarize_by_error_type(analysis)
    by_type.to_csv(Path(output_dir) / "error_metrics_by_type.csv", index=False, encoding="utf-8")
    analysis.sort_values("cer_delta").head(50).to_csv(
        Path(output_dir) / "worst_cases.csv",
        index=False,
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the hybrid corrector.")
    parser.add_argument("--dataset", default="data/processed/test.csv")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--all", action="store_true", help="Evaluate the full dataset.")
    parser.add_argument("--output-dir", default="report")
    parser.add_argument("--strictness", default="strict", choices=["strict", "normal", "aggressive"])
    parser.add_argument("--punctuation-mode", default="conservative", choices=["conservative", "none", "legacy"])
    parser.add_argument("--min-dictionary-score", type=float, default=1.0)
    parser.add_argument("--candidate-top-k", type=int, default=5)
    parser.add_argument("--disable-morphology-guard", action="store_true")
    parser.add_argument("--disable-context-reranker", action="store_true")
    parser.add_argument("--context-model-name", default="DeepPavlov/rubert-base-cased")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_hybrid(
        dataset_path=args.dataset,
        sample_size=None if args.all else args.sample_size,
        output_dir=args.output_dir,
        strictness=args.strictness,
        punctuation_mode=args.punctuation_mode,
        min_dictionary_score=args.min_dictionary_score,
        use_morphology_guard=not args.disable_morphology_guard,
        candidate_top_k=args.candidate_top_k,
        context_reranker_enabled=not args.disable_context_reranker,
        context_model_name=args.context_model_name,
    )


if __name__ == "__main__":
    main()
