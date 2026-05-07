"""HF-style benchmark metrics for the hybrid corrector."""

from __future__ import annotations

import json
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from external_datasets import AI_FOREVER_REPO, AI_FOREVER_TEST_FILES, load_external_pairs
from hybrid_corrector import HybridCorrector
from text_utils import extract_word_slots, normalize_word, punctuation_labels_for_slots


METRIC_TYPES = ("spell", "punct", "case")


def _case_signature(word: str) -> tuple[str, ...]:
    return tuple("U" if ch.isupper() else "L" if ch.islower() else "O" for ch in str(word))


def _is_case_change(source_word: str, target_word: str) -> bool:
    if normalize_word(source_word) != normalize_word(target_word):
        return False
    return _case_signature(source_word) != _case_signature(target_word)


def _add_punct_edit(
    counters: dict[str, Counter[tuple[Any, ...]]],
    source_index: int,
    source_punct: str,
    target_punct: str,
) -> None:
    if str(source_punct or "") != str(target_punct or ""):
        counters["punct"][(source_index, str(target_punct or ""))] += 1


def extract_edit_counters(source_text: str, output_text: str) -> dict[str, Counter[tuple[Any, ...]]]:
    """Extract approximate spell/punctuation/case edits from source to output.

    The official SAGE evaluator is not shipped with the model card. This function
    keeps every example in the benchmark and uses token-level SequenceMatcher
    fallbacks when a pair is not compatible with the local edit architecture.
    """

    counters: dict[str, Counter[tuple[Any, ...]]] = {name: Counter() for name in METRIC_TYPES}
    source_slots = extract_word_slots(source_text)
    output_slots = extract_word_slots(output_text)
    source_words = [slot.word for slot in source_slots]
    output_words = [slot.word for slot in output_slots]
    source_norm = [normalize_word(word) for word in source_words]
    output_norm = [normalize_word(word) for word in output_words]
    source_punct = punctuation_labels_for_slots(source_text, source_slots) if source_slots else []
    output_punct = punctuation_labels_for_slots(output_text, output_slots) if output_slots else []

    matcher = SequenceMatcher(None, source_norm, output_norm)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for source_index, output_index in zip(range(i1, i2), range(j1, j2)):
                if _is_case_change(source_words[source_index], output_words[output_index]):
                    counters["case"][(source_index, output_words[output_index])] += 1
                _add_punct_edit(
                    counters,
                    source_index,
                    source_punct[source_index] if source_index < len(source_punct) else "",
                    output_punct[output_index] if output_index < len(output_punct) else "",
                )
            continue

        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for source_index, output_index in zip(range(i1, i2), range(j1, j2)):
                if source_norm[source_index] != output_norm[output_index]:
                    counters["spell"][(source_index, output_norm[output_index])] += 1
                elif _is_case_change(source_words[source_index], output_words[output_index]):
                    counters["case"][(source_index, output_words[output_index])] += 1
                _add_punct_edit(
                    counters,
                    source_index,
                    source_punct[source_index] if source_index < len(source_punct) else "",
                    output_punct[output_index] if output_index < len(output_punct) else "",
                )
            continue

        if tag == "replace":
            counters["spell"][("replace_span", i1, i2, tuple(output_norm[j1:j2]))] += 1
            source_punct_span = tuple(source_punct[i1:i2])
            output_punct_span = tuple(output_punct[j1:j2])
            if source_punct_span != output_punct_span:
                counters["punct"][("replace_span", i1, i2, output_punct_span)] += 1
            continue

        if tag == "delete":
            counters["spell"][("delete_span", i1, i2)] += 1
            source_punct_span = tuple(source_punct[i1:i2])
            if any(source_punct_span):
                counters["punct"][("delete_span", i1, i2)] += 1
            continue

        if tag == "insert":
            counters["spell"][("insert_at", i1, tuple(output_norm[j1:j2]))] += 1
            output_punct_span = tuple(output_punct[j1:j2])
            if any(output_punct_span):
                counters["punct"][("insert_at", i1, output_punct_span)] += 1

    return counters


def _counter_overlap(gold: Counter[tuple[Any, ...]], predicted: Counter[tuple[Any, ...]]) -> int:
    return int(sum((gold & predicted).values()))


def score_prediction(source_text: str, correct_text: str, predicted_text: str) -> dict[str, dict[str, int]]:
    gold = extract_edit_counters(source_text, correct_text)
    predicted = extract_edit_counters(source_text, predicted_text)
    scores: dict[str, dict[str, int]] = {}
    for metric_type in METRIC_TYPES:
        tp = _counter_overlap(gold[metric_type], predicted[metric_type])
        predicted_total = int(sum(predicted[metric_type].values()))
        gold_total = int(sum(gold[metric_type].values()))
        scores[metric_type] = {
            "tp": tp,
            "fp": predicted_total - tp,
            "fn": gold_total - tp,
            "support": gold_total,
        }
    return scores


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _wide_metric_row(dataset_name: str, examples: int, totals: dict[str, Counter[str]]) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset": dataset_name, "examples": int(examples)}
    labels = {"spell": "spell", "punct": "punc", "case": "case"}
    for metric_type in METRIC_TYPES:
        counts = totals[metric_type]
        precision, recall, f1 = _precision_recall_f1(
            int(counts["tp"]),
            int(counts["fp"]),
            int(counts["fn"]),
        )
        label = labels[metric_type]
        row[f"Pr. ({label})"] = precision
        row[f"Rec. ({label})"] = recall
        row[f"F1 ({label})"] = f1
        row[f"tp ({label})"] = int(counts["tp"])
        row[f"fp ({label})"] = int(counts["fp"])
        row[f"fn ({label})"] = int(counts["fn"])
        row[f"support ({label})"] = int(counts["support"])
    return row


def evaluate_hybrid_ai_forever_benchmark_hf_style(
    *,
    output_dir: str | Path = "report/hybrid_ai_forever_benchmark_hf_style",
    cache_dir: str | Path = "data/external",
    strictness: str = "normal",
    punctuation_mode: str = "conservative",
    min_dictionary_score: float = 0.25,
    use_morphology_guard: bool = True,
    candidate_top_k: int = 16,
    context_reranker_enabled: bool = True,
    context_model_name: str = "DeepPavlov/rubert-base-cased",
    context_device: str = "auto",
    context_margin: float = 0.25,
    use_entity_guard: bool = True,
    inference_batch_size: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    corrector = HybridCorrector(
        strictness=strictness,
        punctuation_mode=punctuation_mode,
        min_dictionary_score=min_dictionary_score,
        use_morphology_guard=use_morphology_guard,
        candidate_top_k=candidate_top_k,
        context_reranker_enabled=context_reranker_enabled,
        context_model_name=context_model_name,
        context_device=context_device,
        context_margin=context_margin,
        use_entity_guard=use_entity_guard,
    )
    if not corrector.is_trained:
        raise RuntimeError("Hybrid model is missing or obsolete. Retrain from the notebook before evaluation.")

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    batch_size = max(1, int(inference_batch_size))
    start = time.time()

    for dataset_name, split_name, remote_path in AI_FOREVER_TEST_FILES:
        pairs = load_external_pairs(
            AI_FOREVER_REPO,
            [(dataset_name, split_name, remote_path)],
            cache_dir=cache_dir,
        )
        print(f"{dataset_name}: loaded {len(pairs)} examples from {AI_FOREVER_REPO}/{remote_path}")
        texts = [pair.error_text for pair in pairs]
        results = []
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            results.extend(corrector.correct_many(texts[batch_start:batch_end], batch_size=batch_size))
            print(f"{dataset_name}: corrected {batch_end}/{len(texts)} examples")
        if len(results) != len(pairs):
            raise RuntimeError(f"{dataset_name}: corrector returned {len(results)} results for {len(pairs)} inputs.")

        totals: dict[str, Counter[str]] = {name: Counter() for name in METRIC_TYPES}
        for row_index, (pair, result) in enumerate(zip(pairs, results)):
            predicted_text = result.corrected
            item_scores = score_prediction(pair.error_text, pair.correct_text, predicted_text)
            for metric_type, values in item_scores.items():
                totals[metric_type].update(values)
            prediction_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "row_index": row_index,
                    "source": pair.error_text,
                    "correction": pair.correct_text,
                    "predicted_text": predicted_text,
                    "domain": pair.source_domain,
                    "confidence": result.confidence,
                    "accepted": result.accepted,
                    "guard_reason": result.guard_reason,
                    "edit_count": len(result.edits),
                }
            )
        metric_rows.append(_wide_metric_row(dataset_name, len(pairs), totals))

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df.to_csv(output_path / "metrics.csv", index=False, encoding="utf-8")
    predictions_df.to_csv(output_path / "predictions.csv", index=False, encoding="utf-8")
    metrics_payload = {
        "repo": AI_FOREVER_REPO,
        "runtime_version": "hybrid_hf_style_metrics_v1",
        "elapsed_minutes": (time.time() - start) / 60,
        "metrics": metric_rows,
    }
    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    return metrics_df, predictions_df
