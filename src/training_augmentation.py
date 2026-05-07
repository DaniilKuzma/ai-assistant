"""Candidate-aware training helpers and clean corpus diagnostics."""

from __future__ import annotations

from dataclasses import replace
import math
import random
from typing import Iterable, Sequence

import pandas as pd

from candidate_generator import Candidate, CandidateGenerator, RUSSIAN_ALPHABET, bounded_damerau_levenshtein
from edit_labels import (
    ACTION_KEEP,
    ACTION_DELETE,
    HybridTrainingExample,
    action_replace_rank,
    candidate_list,
    replace_action,
)
from morphology_guard import is_morphological_dictionary_word, is_same_lemma_inflection
from text_utils import apply_case_like, is_word, normalize_word


FOREIGN_NAME_PARTS = {"делла", "де", "ди", "фон", "ван", "дер", "ле", "ла", "аль", "ибн"}


def is_candidate_eligible_word(word: str) -> bool:
    word = str(word)
    norm = normalize_word(word)
    if len(norm) < 5 or not is_word(word):
        return False
    if word != word.lower() or word.isupper():
        return False
    if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in word):
        return False
    return not any(ch.isdigit() for ch in word)


def best_hard_negative_candidate(
    generator: CandidateGenerator,
    word: str,
    *,
    min_dictionary_score: float = 0.25,
    max_candidates: int = 8,
) -> Candidate | None:
    if not is_candidate_eligible_word(word):
        return None

    candidates = fast_distance_one_dictionary_candidates(generator, word)
    if not candidates:
        return None

    def priority(candidate: Candidate) -> tuple[int, int, float]:
        same_lemma = is_same_lemma_inflection(word, candidate.text)
        low_score = candidate.score < min_dictionary_score
        return (0 if same_lemma else 1, 0 if low_score else 1, -candidate.score)

    candidates.sort(key=priority)
    return candidates[0]


def populate_top_k_candidates(
    examples: Sequence[HybridTrainingExample],
    generator: CandidateGenerator,
    *,
    candidate_top_k: int = 16,
) -> tuple[list[HybridTrainingExample], dict[str, int]]:
    """Populate candidate lists and align replacement labels with candidate ranks.

    Earlier versions always injected the oracle target at rank 0, so the model
    rarely learned to use higher replacement ranks. V7 keeps the generated order when
    the target is already present and only injects the oracle into a free/last
    slot when generation missed it.
    """
    top_k = max(1, int(candidate_top_k))
    stats = {
        "examples": len(examples),
        "tokens": 0,
        "oracle_replace_tokens": 0,
        "target_candidate_hits": 0,
        "oracle_injected_tokens": 0,
        "candidate_miss_tokens": 0,
        "generated_candidate_tokens": 0,
        "cache_hits": 0,
        "candidate_searches": 0,
        "target_rank_distribution": {},
    }
    cache: dict[str, list[Candidate]] = {}
    populated: list[HybridTrainingExample] = []

    for example in examples:
        next_candidates: list[list[str]] = []
        next_actions = list(example.action_labels)
        target_ranks: list[int] = []
        oracle_injected_flags: list[bool] = []
        for idx, source in enumerate(example.source_words):
            stats["tokens"] += 1
            action = example.action_labels[idx]
            target_word = example.target_words[idx] if idx < len(example.target_words) else ""
            norm = normalize_word(source)
            if norm in cache:
                generated = cache[norm]
                stats["cache_hits"] += 1
            else:
                generated = generator.get_candidates(
                    source,
                    max_candidates=top_k,
                    include_known_dictionary=False,
                    allow_long_oov=False,
                )
                cache[norm] = generated
                stats["candidate_searches"] += 1

            items: list[str] = []
            replace_rank = action_replace_rank(action)

            for candidate in generated:
                if normalize_word(candidate.text) == normalize_word(source):
                    continue
                if candidate.text and candidate.text not in items:
                    items.append(candidate.text)

            target_rank = -1
            oracle_injected = False
            if replace_rank is not None and action != ACTION_DELETE and target_word:
                stats["oracle_replace_tokens"] += 1
                target_norm = normalize_word(target_word)
                target_rank = next(
                    (
                        rank
                        for rank, item in enumerate(items[:top_k])
                        if normalize_word(item) == target_norm
                    ),
                    -1,
                )
                if target_rank < 0 and _should_try_long_oov_target(generator, norm, target_norm):
                    stats["targeted_long_oov_hits"] = stats.get("targeted_long_oov_hits", 0) + 1
                    if target_word not in items:
                        if len(items) < top_k:
                            items.append(target_word)
                        else:
                            items[-1] = target_word
                    target_rank = next(
                        (
                            rank
                            for rank, item in enumerate(items[:top_k])
                            if normalize_word(item) == target_norm
                        ),
                        -1,
                    )
                if target_rank >= 0:
                    stats["target_candidate_hits"] += 1
                else:
                    oracle_injected = True
                    stats["oracle_injected_tokens"] += 1
                    if len(items) < top_k:
                        items.append(target_word)
                        target_rank = len(items) - 1
                    else:
                        items = items[:top_k]
                        items[-1] = target_word
                        target_rank = top_k - 1
                    if target_rank < 0:
                        stats["candidate_miss_tokens"] += 1
                next_actions[idx] = replace_action(target_rank)

            if not items:
                items = candidate_list(example.candidate_words[idx])
            if not items:
                items = [source]

            if len(items) > top_k:
                items = items[:top_k]
            next_candidates.append(items)
            target_ranks.append(target_rank)
            oracle_injected_flags.append(oracle_injected)
            if target_rank >= 0:
                key = str(target_rank)
                stats["target_rank_distribution"][key] = stats["target_rank_distribution"].get(key, 0) + 1
            if any(normalize_word(item) != normalize_word(source) for item in items):
                stats["generated_candidate_tokens"] += 1

        if stats["oracle_replace_tokens"]:
            stats["candidate_coverage_rate"] = stats["target_candidate_hits"] / stats["oracle_replace_tokens"]
        else:
            stats["candidate_coverage_rate"] = 1.0

        populated.append(
            replace(
                example,
                candidate_words=next_candidates,
                action_labels=next_actions,
                target_candidate_ranks=target_ranks,
                oracle_injected_flags=oracle_injected_flags,
            )
        )

    return populated, stats


def _should_try_long_oov_target(generator: CandidateGenerator, source_norm: str, target_norm: str) -> bool:
    if not source_norm or not target_norm:
        return False
    if generator.is_known(source_norm):
        return False
    min_length = int(getattr(generator, "long_oov_min_length", 8))
    max_distance = int(getattr(generator, "long_oov_max_distance", 1))
    if max_distance <= 1 or len(source_norm) < min_length:
        return False
    if abs(len(source_norm) - len(target_norm)) > max_distance:
        return False
    return bounded_damerau_levenshtein(source_norm, target_norm, max_distance) <= max_distance


def fast_distance_one_dictionary_candidates(
    generator: CandidateGenerator,
    word: str,
) -> list[Candidate]:
    """Find dictionary neighbors at edit distance 1 without scanning buckets."""
    norm = normalize_word(word)
    if len(norm) < getattr(generator, "min_dictionary_word_length", 4):
        return []

    seen: set[str] = set()
    candidates: list[Candidate] = []

    def add(candidate_norm: str) -> None:
        if candidate_norm == norm or candidate_norm in seen:
            return
        freq = generator.frequencies.get(candidate_norm, 0)
        if freq < generator.min_freq:
            return
        seen.add(candidate_norm)
        score = math.log1p(freq) - 2.0
        candidates.append(
            Candidate(
                text=apply_case_like(candidate_norm, word),
                score=score,
                source="dictionary",
                distance=1,
            )
        )

    for i in range(len(norm)):
        add(norm[:i] + norm[i + 1 :])

    for i in range(len(norm) + 1):
        prefix = norm[:i]
        suffix = norm[i:]
        for ch in RUSSIAN_ALPHABET:
            add(prefix + ch + suffix)

    for i, original in enumerate(norm):
        prefix = norm[:i]
        suffix = norm[i + 1 :]
        for ch in RUSSIAN_ALPHABET:
            if ch != original:
                add(prefix + ch + suffix)

    for i in range(len(norm) - 1):
        if norm[i] != norm[i + 1]:
            add(norm[:i] + norm[i + 1] + norm[i] + norm[i + 2 :])

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def augment_keep_candidates(
    examples: Sequence[HybridTrainingExample],
    generator: CandidateGenerator,
    *,
    enabled: bool = True,
    probability: float = 0.35,
    dirty_probability: float | None = None,
    max_per_example: int = 3,
    max_candidate_searches: int = 8_000,
    clean_only: bool = True,
    seed: int = 42,
    min_dictionary_score: float = 0.25,
) -> tuple[list[HybridTrainingExample], dict[str, int]]:
    """Inject hard dictionary candidates into KEEP labels.

    This keeps the target/action unchanged, but teaches the model that a
    plausible edit-distance neighbor can still be a word that must be kept.
    """
    stats = {
        "examples": len(examples),
        "changed_examples": 0,
        "eligible_tokens": 0,
        "augmented_tokens": 0,
        "clean_augmented_tokens": 0,
        "dirty_augmented_tokens": 0,
        "candidate_searches": 0,
        "cache_hits": 0,
        "budget_skips": 0,
    }
    if not enabled:
        return list(examples), stats

    rng = random.Random(seed)
    dirty_prob = probability / 3.0 if dirty_probability is None else dirty_probability
    augmented: list[HybridTrainingExample] = []
    candidate_cache: dict[str, Candidate | None] = {}

    for example in examples:
        if clean_only and not example.is_clean:
            augmented.append(example)
            continue

        candidate_words = [candidate_list(value) for value in example.candidate_words]
        eligible_indices = [
            i
            for i, (word, candidate, action) in enumerate(
                zip(example.source_words, example.candidate_words, example.action_labels)
            )
            if action == ACTION_KEEP and is_candidate_eligible_word(word)
        ]
        stats["eligible_tokens"] += len(eligible_indices)
        rng.shuffle(eligible_indices)

        changed = 0
        per_token_probability = probability if example.is_clean else dirty_prob
        for idx in eligible_indices:
            if changed >= max_per_example:
                break
            if rng.random() > per_token_probability:
                continue

            norm = normalize_word(example.source_words[idx])
            if norm in candidate_cache:
                hard_candidate = candidate_cache[norm]
                stats["cache_hits"] += 1
            elif stats["candidate_searches"] >= max_candidate_searches:
                stats["budget_skips"] += 1
                continue
            else:
                hard_candidate = best_hard_negative_candidate(
                    generator,
                    example.source_words[idx],
                    min_dictionary_score=min_dictionary_score,
                )
                candidate_cache[norm] = hard_candidate
                stats["candidate_searches"] += 1
            if hard_candidate is None:
                continue

            current = candidate_words[idx]
            candidate_words[idx] = [hard_candidate.text] + [
                item for item in current if normalize_word(item) != normalize_word(hard_candidate.text)
            ]
            changed += 1

        if changed:
            stats["changed_examples"] += 1
            stats["augmented_tokens"] += changed
            if example.is_clean:
                stats["clean_augmented_tokens"] += changed
            else:
                stats["dirty_augmented_tokens"] += changed
            augmented.append(replace(example, candidate_words=candidate_words))
        else:
            augmented.append(example)

    return augmented, stats


def build_clean_candidate_audit(
    rows: pd.DataFrame,
    generator: CandidateGenerator,
    *,
    min_dictionary_score: float = 0.25,
    max_rows: int = 5000,
    max_word_searches: int = 5000,
) -> pd.DataFrame:
    """Find clean corpus words that have suspicious dictionary neighbors."""
    records: list[dict[str, object]] = []
    seen_texts: set[str] = set()
    candidate_cache: dict[str, Candidate | None] = {}
    searches = 0

    for _, row in rows.iterrows():
        text = str(row.get("correct_text", "")).strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)

        for word in _iter_words(text):
            if not is_candidate_eligible_word(word):
                continue
            norm = normalize_word(word)
            if norm in candidate_cache:
                candidate = candidate_cache[norm]
            else:
                if searches >= max_word_searches:
                    return pd.DataFrame(records)
                candidate = best_hard_negative_candidate(
                    generator,
                    word,
                    min_dictionary_score=min_dictionary_score,
                )
                candidate_cache[norm] = candidate
                searches += 1
            if candidate is None or candidate.score < min_dictionary_score:
                continue

            same_lemma = is_same_lemma_inflection(word, candidate.text)
            if same_lemma:
                continue

            records.append(
                {
                    "split": row.get("split", ""),
                    "word": word,
                    "candidate": candidate.text,
                    "candidate_score": candidate.score,
                    "candidate_distance": candidate.distance,
                    "source_known": generator.is_known(word),
                    "text": text,
                }
            )
            if len(records) >= max_rows:
                return pd.DataFrame(records)

    return pd.DataFrame(records)


def filter_noisy_clean_rows(
    rows: pd.DataFrame,
    generator: CandidateGenerator,
    *,
    min_dictionary_score: float = 0.25,
    max_word_searches: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Remove likely typo-contaminated clean source rows from training data."""
    records: list[dict[str, object]] = []
    suspicious_texts: set[str] = set()
    candidate_cache: dict[str, Candidate | None] = {}
    searches = 0

    unique_texts = rows["correct_text"].dropna().astype(str).drop_duplicates().tolist()
    for text in unique_texts:
        for word in _iter_words(text):
            if not is_clean_noise_candidate_word(word):
                continue
            norm = normalize_word(word)
            if norm in candidate_cache:
                candidate = candidate_cache[norm]
            else:
                if searches >= max_word_searches:
                    candidate = None
                else:
                    candidate = best_hard_negative_candidate(
                        generator,
                        word,
                        min_dictionary_score=min_dictionary_score,
                    )
                    candidate_cache[norm] = candidate
                    searches += 1
            if candidate is None or candidate.score < min_dictionary_score:
                continue
            if is_same_lemma_inflection(word, candidate.text):
                continue

            suspicious_texts.add(text)
            records.append(
                {
                    "word": word,
                    "candidate": candidate.text,
                    "candidate_score": candidate.score,
                    "candidate_distance": candidate.distance,
                    "text": text,
                }
            )
            break

    if suspicious_texts:
        filtered = rows.loc[~rows["correct_text"].astype(str).isin(suspicious_texts)].copy()
    else:
        filtered = rows.copy()
    return filtered.reset_index(drop=True), pd.DataFrame(records), suspicious_texts


def is_clean_noise_candidate_word(word: str) -> bool:
    if not is_candidate_eligible_word(word):
        return False
    norm = normalize_word(word)
    if norm in FOREIGN_NAME_PARTS:
        return False
    if is_morphological_dictionary_word(word):
        return False
    return True


def _iter_words(text: str) -> Iterable[str]:
    from text_utils import word_tokens

    yield from word_tokens(text)
