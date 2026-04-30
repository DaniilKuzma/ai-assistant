"""Main runtime corrector for the hybrid edit-based assistant."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from candidate_generator import CandidateGenerator
from context_reranker import ContextReranker
from edit_labels import ACTION_DELETE, ACTION_KEEP, action_replace_rank
from hybrid_preprocessor import HybridPreprocessor
from morphology_guard import is_morphological_dictionary_word, is_same_lemma_inflection
from quality_guard import QualityGuard
from text_utils import (
    apply_case_like,
    extract_word_slots,
    is_protected_punctuation_gap,
    is_trainable_punctuation_gap,
    normalize_word,
    punctuation_labels_for_slots,
    rebuild_preserving_layout,
)


@dataclass
class CorrectionEdit:
    index: int
    source: str
    target: str
    action: str
    confidence: float
    candidate_source: str = ""
    candidate_distance: int = 0
    candidate_score: float = 0.0
    candidate_rank: int = -1
    context_score: float = 0.0
    context_margin: float = 0.0
    reranked: bool = False
    reranker_reason: str = ""


@dataclass(frozen=True)
class RuntimeCandidate:
    text: str
    source: str = "original"
    distance: int = 0
    score: float = 0.0
    rank: int = -1


@dataclass(frozen=True)
class PunctuationDecision:
    index: int
    word: str
    source_punct: str
    predicted_punct: str
    confidence: float
    margin: float
    applied: bool
    blocked_reason: str
    is_final: bool
    protected_gap: bool = False
    window_id: int = 0
    window_conflict_resolved: bool = False
    punct_position_type: str = "internal"


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    confidence: float
    accepted: bool
    guard_reason: str
    edits: List[CorrectionEdit]
    punctuation_diagnostics: List[PunctuationDecision] = field(default_factory=list)


class HybridCorrector:
    """One public correction interface for the project."""

    def __init__(
        self,
        model_path: str = "models/hybrid_corrector.keras",
        preprocessor_path: str = "models/hybrid_preprocessor.pkl",
        candidate_generator_path: str = "models/candidate_generator.pkl",
        strictness: str = "normal",
        punctuation_mode: str = "conservative",
        candidate_top_k: int = 5,
        min_dictionary_score: float = 1.0,
        use_morphology_guard: bool = True,
        context_reranker_enabled: bool = True,
        context_model_name: str = "DeepPavlov/rubert-base-cased",
        context_margin: float = 0.25,
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.candidate_generator_path = candidate_generator_path
        self.strictness = strictness
        self.punctuation_mode = punctuation_mode
        self.candidate_top_k = max(1, int(candidate_top_k))
        self.min_dictionary_score = min_dictionary_score
        self.use_morphology_guard = use_morphology_guard
        self.context_reranker_enabled = context_reranker_enabled
        self.context_model_name = context_model_name
        self.context_margin = float(context_margin)
        self.thresholds = self._thresholds(strictness)
        self.runtime_stats: Counter[str] = Counter()
        self.context_reranker = ContextReranker(
            model_name=context_model_name,
            enabled=context_reranker_enabled,
        )

        self.candidate_generator = self._load_candidate_generator(candidate_generator_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)
        if self.preprocessor is not None:
            self.candidate_top_k = getattr(self.preprocessor, "candidate_top_k", self.candidate_top_k)
        self.model = self._load_model(model_path)
        self.guard = QualityGuard(
            min_confidence=self.thresholds["guard_confidence"],
            max_change_ratio=self.thresholds["max_change_ratio"],
            max_length_delta_ratio=self.thresholds["max_length_delta_ratio"],
            max_simple_punct_drop=int(self.thresholds["max_simple_punct_drop"]),
            min_punct_similarity=self.thresholds["min_punct_similarity"],
        )

    @staticmethod
    def _thresholds(strictness: str) -> Dict[str, float]:
        table = {
            "strict": {
                "action": 0.85,
                "dictionary_action": 0.93,
                "rule_action": 0.88,
                "action_margin": 0.18,
                "delete": 0.995,
                "punct": 0.92,
                "punct_delete": 0.995,
                "punct_margin": 0.12,
                "punct_delete_margin": 0.45,
                "guard_confidence": 0.70,
                "max_change_ratio": 0.12,
                "max_length_delta_ratio": 0.10,
                "max_simple_punct_drop": 1,
                "min_punct_similarity": 0.70,
            },
            "normal": {
                "action": 0.80,
                "dictionary_action": 0.90,
                "rule_action": 0.84,
                "action_margin": 0.14,
                "delete": 0.995,
                "punct": 0.94,
                "punct_delete": 0.99,
                "punct_margin": 0.20,
                "punct_delete_margin": 0.40,
                "guard_confidence": 0.65,
                "max_change_ratio": 0.18,
                "max_length_delta_ratio": 0.12,
                "max_simple_punct_drop": 2,
                "min_punct_similarity": 0.55,
            },
            "aggressive": {
                "action": 0.70,
                "dictionary_action": 0.84,
                "rule_action": 0.78,
                "action_margin": 0.10,
                "delete": 0.98,
                "punct": 0.88,
                "punct_delete": 0.97,
                "punct_margin": 0.14,
                "punct_delete_margin": 0.30,
                "guard_confidence": 0.55,
                "max_change_ratio": 0.28,
                "max_length_delta_ratio": 0.20,
                "max_simple_punct_drop": 3,
                "min_punct_similarity": 0.45,
            },
        }
        return table.get(strictness, table["normal"])

    def _load_candidate_generator(self, path: str) -> CandidateGenerator:
        if os.path.exists(path):
            return CandidateGenerator.load(path)

        texts = self._load_clean_texts_for_runtime_dictionary()
        generator = CandidateGenerator.from_texts(
            texts,
            min_freq=1,
            max_distance=1,
            allow_split_candidates=True,
            safe_split_only=True,
        )
        return generator

    def _load_preprocessor(self, path: str) -> HybridPreprocessor | None:
        if not os.path.exists(path):
            return None
        return HybridPreprocessor.load(path)

    def _load_model(self, path: str):
        if not os.path.exists(path) or self.preprocessor is None:
            return None
        config_path = Path(path).with_name("hybrid_config.json")
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if config.get("model_version", 0) < 6:
                    print("Hybrid model is obsolete: missing v6 top-k/reranker changes. Retrain the notebook.")
                    return None
                if not config.get("requires_source_punct_ids", False):
                    print("Hybrid model is obsolete: missing source_punct_ids input. Retrain the notebook.")
                    return None
                if not config.get("requires_top_k_candidates", False):
                    print("Hybrid model is obsolete: missing top-k candidate input. Retrain the notebook.")
                    return None
            except Exception:
                pass
        try:
            import hybrid_model  # noqa: F401  # registers custom Keras layers
            from tensorflow.keras.models import load_model

            model = load_model(path)
            input_names = {inp.name.split(":")[0].split("/")[-1] for inp in model.inputs}
            if "source_punct_ids" not in input_names:
                print("Hybrid model is obsolete: missing source_punct_ids input. Retrain the notebook.")
                return None
            candidate_inputs = [inp for inp in model.inputs if inp.name.split(":")[0].split("/")[-1] == "candidate_ids"]
            if candidate_inputs and len(candidate_inputs[0].shape) < 3:
                print("Hybrid model is obsolete: missing top-k candidate input. Retrain the notebook.")
                return None
            return model
        except Exception as exc:
            print(f"Hybrid model could not be loaded: {exc}")
            return None

    @staticmethod
    def _load_clean_texts_for_runtime_dictionary() -> List[str]:
        candidates = [
            Path("data/processed/train.csv"),
            Path("data/raw/texts.txt"),
            Path("models/texts.txt"),
        ]
        for path in candidates:
            if not path.exists():
                continue
            if path.suffix == ".csv":
                try:
                    import pandas as pd

                    df = pd.read_csv(path)
                    if "correct_text" in df.columns:
                        return df["correct_text"].dropna().astype(str).tolist()
                except Exception:
                    continue
            else:
                return path.read_text(encoding="utf-8").splitlines()
        return []

    @property
    def is_trained(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def correct(self, text: str, return_details: bool = False):
        result = self.correct_with_details(text)
        return result if return_details else result.corrected

    def correct_with_details(self, text: str) -> CorrectionResult:
        original = "" if text is None else str(text)
        if not original.strip():
            return CorrectionResult(original, original, 1.0, True, "empty", [])

        corrected_lines: List[str] = []
        all_edits: List[CorrectionEdit] = []
        all_punctuation_diagnostics: List[PunctuationDecision] = []
        confidences: List[float] = []
        accepted = True
        reasons: List[str] = []

        for line in original.splitlines() or [original]:
            if not line.strip():
                corrected_lines.append(line)
                continue

            line_result = self._correct_segment(line)
            corrected_lines.append(line_result.corrected)
            all_edits.extend(line_result.edits)
            all_punctuation_diagnostics.extend(line_result.punctuation_diagnostics)
            confidences.append(line_result.confidence)
            accepted = accepted and line_result.accepted
            reasons.append(line_result.guard_reason)

        corrected = "\n".join(corrected_lines)
        confidence = float(np.mean(confidences)) if confidences else 1.0
        decision = self.guard.check(original, corrected, confidence, len(all_edits))
        if not decision.accepted:
            return CorrectionResult(
                original,
                original,
                confidence,
                False,
                decision.reason,
                [],
                all_punctuation_diagnostics,
            )

        reason = "accepted" if accepted else ",".join(sorted(set(reasons)))
        return CorrectionResult(
            original,
            corrected,
            confidence,
            accepted,
            reason,
            all_edits,
            all_punctuation_diagnostics,
        )

    def _correct_segment(self, text: str) -> CorrectionResult:
        slots = extract_word_slots(text)
        if not slots:
            return CorrectionResult(text, text, 1.0, True, "no_words", [])

        source_words = [slot.word for slot in slots]
        source_puncts = punctuation_labels_for_slots(text, slots)
        source_gaps = [
            text[slot.end : (slots[i + 1].start if i + 1 < len(slots) else len(text))]
            for i, slot in enumerate(slots)
        ]
        candidate_options = self._candidate_options(source_words)

        if self.is_trained:
            corrected_words, corrected_puncts, edits, punctuation_diagnostics, confidence = self._model_correct(
                source_words,
                candidate_options,
                source_puncts,
                source_gaps,
            )
        else:
            corrected_words, corrected_puncts, edits, confidence = self._dictionary_correct(
                source_words,
                candidate_options,
                source_puncts,
                source_gaps,
            )
            punctuation_diagnostics = []

        if not edits:
            return CorrectionResult(text, text, confidence, True, "unchanged", [], punctuation_diagnostics)

        corrected = rebuild_preserving_layout(
            text,
            slots,
            corrected_words,
            corrected_puncts,
            punctuation_mode=self.punctuation_mode,
            source_puncts=source_puncts,
            rewrite_punct_indices=[edit.index for edit in edits if edit.action == "PUNCT"],
        )
        decision = self.guard.check(text, corrected, confidence, len(edits))
        if not decision.accepted:
            return CorrectionResult(text, text, confidence, False, decision.reason, [], punctuation_diagnostics)

        return CorrectionResult(text, corrected, confidence, True, decision.reason, edits, punctuation_diagnostics)

    def _candidate_options_for_word(self, word: str) -> List[RuntimeCandidate]:
        raw_candidates = self.candidate_generator.get_candidates(word, max_candidates=self.candidate_top_k)
        if any(candidate.source == "split" for candidate in raw_candidates):
            self._increment_stat("split_candidates_seen")
        candidates = [
            RuntimeCandidate(
                text=apply_case_like(candidate.text, word),
                source=candidate.source,
                distance=candidate.distance,
                score=candidate.score,
                rank=rank,
            )
            for rank, candidate in enumerate(raw_candidates[: self.candidate_top_k])
        ]
        if not candidates:
            candidates.append(RuntimeCandidate(text=word, rank=0))
        return self._renumber_candidates(candidates)

    def _candidate_options(self, words: Sequence[str]) -> List[List[RuntimeCandidate]]:
        candidates = [self._candidate_options_for_word(word) for word in words]
        normalized = [word.lower().replace("ё", "е") for word in words]

        phrase_map = getattr(self.candidate_generator, "phrase_confusions", {})
        for source_phrase, target_phrases in phrase_map.items():
            source_parts = source_phrase.split()
            if len(source_parts) <= 1:
                continue
            for i in range(0, len(words) - len(source_parts) + 1):
                if normalized[i : i + len(source_parts)] != source_parts:
                    continue
                target_parts = str(target_phrases[0]).split()
                if len(target_parts) == len(source_parts):
                    for offset, target_word in enumerate(target_parts):
                        phrase_candidate = RuntimeCandidate(
                            text=apply_case_like(target_word, words[i + offset]),
                            source="phrase",
                            distance=0,
                            score=100.0,
                            rank=0,
                        )
                        candidates[i + offset] = self._renumber_candidates([phrase_candidate] + [
                            candidate
                            for candidate in candidates[i + offset]
                            if normalize_word(candidate.text) != normalize_word(phrase_candidate.text)
                        ][: self.candidate_top_k - 1])
                elif len(target_parts) == 1:
                    phrase_candidate = RuntimeCandidate(
                        text=apply_case_like(target_parts[0], words[i]),
                        source="phrase",
                        distance=0,
                        score=100.0,
                        rank=0,
                    )
                    candidates[i] = self._renumber_candidates([phrase_candidate] + [
                        candidate
                        for candidate in candidates[i]
                        if normalize_word(candidate.text) != normalize_word(phrase_candidate.text)
                    ][: self.candidate_top_k - 1])
        return candidates

    @staticmethod
    def _renumber_candidates(candidates: Sequence[RuntimeCandidate]) -> List[RuntimeCandidate]:
        return [
            RuntimeCandidate(
                text=candidate.text,
                source=candidate.source,
                distance=candidate.distance,
                score=candidate.score,
                rank=rank,
            )
            for rank, candidate in enumerate(candidates)
        ]

    def _increment_stat(self, key: str, value: int = 1) -> None:
        if not hasattr(self, "runtime_stats"):
            self.runtime_stats = Counter()
        self.runtime_stats[key] += value

    def _dictionary_correct(
        self,
        source_words: Sequence[str],
        candidate_options: Sequence[Sequence[RuntimeCandidate]],
        source_puncts: Sequence[str],
        source_gaps: Sequence[str],
    ) -> tuple[List[str], List[str], List[CorrectionEdit], float]:
        corrected_words: List[str] = []
        corrected_puncts: List[str] = []
        edits: List[CorrectionEdit] = []
        confidences: List[float] = []

        for i, (source, candidates, punct) in enumerate(zip(source_words, candidate_options, source_puncts)):
            selected = None
            context_score = 0.0
            context_margin = 0.0
            reranked = False
            reranker_reason = ""
            prev_word = source_words[i - 1] if i > 0 else ""
            next_word = source_words[i + 1] if i + 1 < len(source_words) else ""
            for candidate in candidates:
                if candidate.text != source and self._can_apply_candidate(
                    source,
                    candidate,
                    0.90,
                    0.25,
                    prev_word=prev_word,
                    next_word=next_word,
                ):
                    selected, context_score, context_margin, reranked, reranker_reason = self._maybe_rerank_candidate(
                        source_words,
                        i,
                        source,
                        candidate,
                        candidates,
                        confidence=0.90,
                        margin=0.25,
                        prev_word=prev_word,
                        next_word=next_word,
                    )
                    break
            if selected is not None:
                corrected_words.append(selected.text)
                edits.append(
                    CorrectionEdit(
                        i,
                        source,
                        selected.text,
                        f"REPLACE_{max(0, selected.rank)}",
                        0.55,
                        selected.source,
                        selected.distance,
                        selected.score,
                        selected.rank,
                        context_score,
                        context_margin,
                        reranked,
                        reranker_reason,
                    )
                )
                confidences.append(0.55)
            else:
                corrected_words.append(source)
                confidences.append(0.90)
            corrected_puncts.append(punct)

        return corrected_words, corrected_puncts, edits, float(np.mean(confidences) if confidences else 1.0)

    def _model_correct(
        self,
        source_words: Sequence[str],
        candidate_options: Sequence[Sequence[RuntimeCandidate]],
        source_puncts: Sequence[str],
        source_gaps: Sequence[str],
    ) -> tuple[List[str], List[str], List[CorrectionEdit], List[PunctuationDecision], float]:
        assert self.preprocessor is not None
        length = len(source_words)
        corrected_words = list(source_words)
        corrected_puncts = list(source_puncts)
        selected_edits: List[List[CorrectionEdit]] = [[] for _ in range(length)]
        selected_punct_diagnostics: List[PunctuationDecision | None] = [None] * length
        selected_scores: List[tuple[int, float]] = [(-1, -1.0)] * length
        selected_confidences: List[float] = [0.0] * length
        conflict_flags = [False] * length

        for window_id, (start, end) in enumerate(self._window_ranges(length, self.preprocessor.max_length)):
            local_words = list(source_words[start:end])
            local_candidates = [candidate_options[i] for i in range(start, end)]
            local_candidate_words = [[candidate.text for candidate in candidates] for candidates in local_candidates]
            local_puncts = list(source_puncts[start:end])
            inputs = self.preprocessor.vectorize_inference(local_words, local_candidate_words, local_puncts)
            prediction = self.model.predict(inputs, verbose=0)
            if isinstance(prediction, dict):
                action_probs = prediction["action"][0]
                punct_probs = prediction["punct"][0]
            else:
                action_probs, punct_probs = prediction
                action_probs = action_probs[0]
                punct_probs = punct_probs[0]

            window_len = end - start
            for local_i in range(window_len):
                i = start + local_i
                source = source_words[i]
                candidates = candidate_options[i]
                prev_word = source_words[i - 1] if i > 0 else ""
                next_word = source_words[i + 1] if i + 1 < length else ""

                action_id = int(np.argmax(action_probs[local_i]))
                action_conf = float(np.max(action_probs[local_i]))
                action_margin = probability_margin(action_probs[local_i])
                action = self.preprocessor.id_to_action.get(action_id, ACTION_KEEP)

                output_word = source
                token_edits: List[CorrectionEdit] = []
                replace_rank = action_replace_rank(action)
                if replace_rank is not None and replace_rank < len(candidates):
                    candidate = candidates[replace_rank]
                    context_score = 0.0
                    context_margin = 0.0
                    reranked = False
                    reranker_reason = ""
                    if (
                        candidate.text != source
                        and self._can_apply_candidate(
                            source,
                            candidate,
                            action_conf,
                            action_margin,
                            prev_word=prev_word,
                            next_word=next_word,
                        )
                    ):
                        candidate, context_score, context_margin, reranked, reranker_reason = self._maybe_rerank_candidate(
                            source_words,
                            i,
                            source,
                            candidate,
                            candidates,
                            confidence=action_conf,
                            margin=action_margin,
                            prev_word=prev_word,
                            next_word=next_word,
                        )
                        if candidate is not None and self._can_apply_candidate(
                            source,
                            candidate,
                            action_conf,
                            action_margin,
                            prev_word=prev_word,
                            next_word=next_word,
                        ):
                            output_word = candidate.text
                            token_edits.append(
                                CorrectionEdit(
                                    i,
                                    source,
                                    candidate.text,
                                    f"REPLACE_{max(0, candidate.rank)}",
                                    action_conf,
                                    candidate.source,
                                    candidate.distance,
                                    candidate.score,
                                    candidate.rank,
                                    context_score,
                                    context_margin,
                                    reranked,
                                    reranker_reason,
                                )
                            )
                elif action == ACTION_DELETE and action_conf >= self.thresholds["delete"]:
                    output_word = ""
                    token_edits.append(CorrectionEdit(i, source, "", action, action_conf))

                punct_id = int(np.argmax(punct_probs[local_i]))
                punct_conf = float(np.max(punct_probs[local_i]))
                punct_margin = probability_margin(punct_probs[local_i])
                predicted_punct = self.preprocessor.id_to_punct.get(punct_id, "")
                output_punct = source_puncts[i]
                gap = source_gaps[i] if i < len(source_gaps) else ""
                is_final = i + 1 >= length
                protected_gap = is_protected_punctuation_gap(
                    gap,
                    word=source,
                    next_word=next_word,
                    is_final=is_final,
                )
                can_apply_punct, punct_reason = self._punctuation_decision(
                    source_puncts[i],
                    predicted_punct,
                    gap,
                    source,
                    next_word,
                    is_final=is_final,
                    confidence=punct_conf,
                    margin=punct_margin,
                    protected_gap=protected_gap,
                )
                if can_apply_punct and predicted_punct != source_puncts[i]:
                    output_punct = predicted_punct
                    token_edits.append(CorrectionEdit(i, source_puncts[i], output_punct, "PUNCT", punct_conf))

                punct_diag = PunctuationDecision(
                    index=i,
                    word=source,
                    source_punct=source_puncts[i],
                    predicted_punct=predicted_punct,
                    confidence=punct_conf,
                    margin=punct_margin,
                    applied=can_apply_punct and output_punct != source_puncts[i],
                    blocked_reason=punct_reason,
                    is_final=is_final,
                    protected_gap=protected_gap,
                    window_id=window_id,
                    window_conflict_resolved=False,
                    punct_position_type=self._punct_position_type(protected_gap, is_final),
                )

                token_conf = max(action_conf, punct_conf)
                edge_distance = min(local_i, window_len - 1 - local_i)
                score = (edge_distance, token_conf)
                current_score = selected_scores[i]
                should_replace = score[0] > current_score[0] or (
                    score[0] == current_score[0] and score[1] > current_score[1]
                )
                if should_replace:
                    if selected_punct_diagnostics[i] is not None and (
                        corrected_words[i] != output_word or corrected_puncts[i] != output_punct
                    ):
                        conflict_flags[i] = True
                    corrected_words[i] = output_word
                    corrected_puncts[i] = output_punct
                    selected_edits[i] = token_edits
                    selected_punct_diagnostics[i] = punct_diag
                    selected_scores[i] = score
                    selected_confidences[i] = token_conf

        edits: List[CorrectionEdit] = []
        punctuation_diagnostics: List[PunctuationDecision] = []
        for i, diag in enumerate(selected_punct_diagnostics):
            edits.extend(selected_edits[i])
            if diag is not None:
                punctuation_diagnostics.append(
                    PunctuationDecision(
                        index=diag.index,
                        word=diag.word,
                        source_punct=diag.source_punct,
                        predicted_punct=diag.predicted_punct,
                        confidence=diag.confidence,
                        margin=diag.margin,
                        applied=diag.applied,
                        blocked_reason=diag.blocked_reason,
                        is_final=diag.is_final,
                        protected_gap=diag.protected_gap,
                        window_id=diag.window_id,
                        window_conflict_resolved=conflict_flags[i],
                        punct_position_type=diag.punct_position_type,
                    )
                )

        return (
            corrected_words,
            corrected_puncts,
            edits,
            punctuation_diagnostics,
            float(np.mean([c for c in selected_confidences if c > 0]) if selected_confidences else 1.0),
        )

    @staticmethod
    def _window_ranges(length: int, max_length: int, overlap: int = 24) -> List[tuple[int, int]]:
        if length <= 0:
            return []
        max_length = max(1, int(max_length))
        if length <= max_length:
            return [(0, length)]
        stride = max(1, max_length - max(0, int(overlap)))
        ranges: List[tuple[int, int]] = []
        start = 0
        while start < length:
            end = min(start + max_length, length)
            ranges.append((start, end))
            if end >= length:
                break
            start += stride
        return ranges

    def _maybe_rerank_candidate(
        self,
        source_words: Sequence[str],
        index: int,
        source: str,
        selected: RuntimeCandidate,
        candidates: Sequence[RuntimeCandidate],
        *,
        confidence: float,
        margin: float,
        prev_word: str = "",
        next_word: str = "",
    ) -> tuple[RuntimeCandidate | None, float, float, bool, str]:
        if selected.source not in {"dictionary", "split"} or not getattr(self, "context_reranker_enabled", False):
            return selected, 0.0, 0.0, False, "not_called"

        viable = [
            candidate
            for candidate in candidates
            if candidate.source in {"dictionary", "split"} and normalize_word(candidate.text) != normalize_word(source)
        ]
        if not viable:
            return selected, 0.0, 0.0, False, "no_viable_candidates"

        self._increment_stat("reranker_called_count")
        texts = [source] + [candidate.text for candidate in viable]
        scores = self.context_reranker.score_candidates(source_words, index, source, texts)
        if not scores or not all(score.available for score in scores):
            self._increment_stat("reranker_fail_open_count")
            reason = scores[0].reason if scores else "no_scores"
            return selected, 0.0, 0.0, False, reason

        score_by_norm = {normalize_word(score.text): score for score in scores}
        ranked = sorted(scores, key=lambda item: item.score, reverse=True)
        best = ranked[0]
        second_score = ranked[1].score if len(ranked) > 1 else best.score
        context_margin = float(best.score - second_score)
        source_score = score_by_norm.get(normalize_word(source))
        selected_score = score_by_norm.get(normalize_word(selected.text))
        context_score = float(selected_score.score) if selected_score is not None else 0.0

        if (
            source_score is not None
            and normalize_word(best.text) == normalize_word(source)
            and context_margin >= self.context_margin
        ):
            self._increment_stat("reranker_source_preferred_count")
            return None, float(source_score.score), context_margin, True, "context_prefers_source"

        if normalize_word(best.text) != normalize_word(source) and context_margin >= self.context_margin:
            best_candidate = next(
                (candidate for candidate in viable if normalize_word(candidate.text) == normalize_word(best.text)),
                None,
            )
            if best_candidate is not None and self._can_apply_candidate(
                source,
                best_candidate,
                confidence,
                margin,
                prev_word=prev_word,
                next_word=next_word,
                skip_context=True,
            ):
                changed = normalize_word(best_candidate.text) != normalize_word(selected.text)
                if changed:
                    self._increment_stat("reranker_changed_top1_count")
                return best_candidate, float(best.score), context_margin, changed, (
                    "context_changed_top1" if changed else "context_confirmed"
                )

        return selected, context_score, context_margin, False, "context_low_margin"

    def _can_apply_candidate(
        self,
        source: str,
        candidate: RuntimeCandidate,
        confidence: float,
        margin: float,
        *,
        prev_word: str = "",
        next_word: str = "",
        skip_context: bool = False,
    ) -> bool:
        if candidate.source == "original" or candidate.text == source:
            return False
        if confidence < self.thresholds["action"] or margin < self.thresholds["action_margin"]:
            return False

        norm = normalize_word(source)
        known = self.candidate_generator.is_known(source)
        is_lower_word = source == source.lower()
        has_latin = any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in source)
        is_upper = source.isupper()

        if candidate.source in {"dictionary", "split"}:
            return (
                confidence >= self.thresholds["dictionary_action"]
                and (candidate.source == "split" or candidate.score >= self.min_dictionary_score)
                and not known
                and not is_morphological_dictionary_word(source)
                and is_lower_word
                and not has_latin
                and not is_upper
                and len(norm) >= 5
                and (candidate.source == "split" or candidate.distance == 1)
                and not (
                    self.use_morphology_guard
                    and is_same_lemma_inflection(source, candidate.text)
                )
            )

        if candidate.source in {"rule", "phrase"}:
            if confidence < self.thresholds["rule_action"] or has_latin or is_upper:
                return False
            if normalize_word(source) == "так" and normalize_word(candidate.text) == "также" and normalize_word(next_word) == "же":
                return False
            if self._looks_like_name_context(source, prev_word, next_word):
                return False
            if self._blocked_tsya_context(source, candidate.text, prev_word):
                return False
            return True

        return False

    @staticmethod
    def _looks_like_name_context(source: str, prev_word: str = "", next_word: str = "") -> bool:
        source_s = str(source or "")
        if not source_s[:1].isupper():
            return False
        return str(prev_word or "")[:1].isupper() or str(next_word or "")[:1].isupper()

    @staticmethod
    def _blocked_tsya_context(source: str, target: str, prev_word: str = "") -> bool:
        src = normalize_word(source)
        tgt = normalize_word(target)
        prev = normalize_word(prev_word)
        infinitive_cues = {"может", "могут", "будет", "будут", "должен", "должна", "должно", "должны", "нужно", "надо", "риск", "риске", "чтобы"}
        return src.endswith("ться") and tgt.endswith("тся") and prev in infinitive_cues

    @staticmethod
    def _punct_position_type(protected_gap: bool, is_final: bool) -> str:
        if protected_gap:
            return "protected"
        if is_final:
            return "final"
        return "internal"

    def _can_apply_punctuation(
        self,
        source_punct: str,
        predicted_punct: str,
        gap: str,
        next_word: str,
        *,
        is_final: bool,
        confidence: float,
        margin: float,
    ) -> bool:
        return self._punctuation_decision(
            source_punct,
            predicted_punct,
            gap,
            "",
            next_word,
            is_final=is_final,
            confidence=confidence,
            margin=margin,
            protected_gap=is_protected_punctuation_gap(gap, next_word=next_word, is_final=is_final),
        )[0]

    def _punctuation_decision(
        self,
        source_punct: str,
        predicted_punct: str,
        gap: str,
        word: str,
        next_word: str,
        *,
        is_final: bool,
        confidence: float,
        margin: float,
        protected_gap: bool = False,
    ) -> tuple[bool, str]:
        if self.punctuation_mode != "conservative":
            return False, "mode_disabled"
        if predicted_punct == source_punct:
            return False, "same_as_source"
        if predicted_punct not in {"", ",", ".", "?", "!", ":", ";"}:
            return False, "unsupported_punctuation"
        if protected_gap:
            return False, "protected_gap"
        if not is_trainable_punctuation_gap(gap, word=word, next_word=next_word, is_final=is_final):
            return False, "structural_gap"
        if is_final and predicted_punct in {",", ":", ";"}:
            return False, "final_non_terminal_punctuation"
        if not is_final:
            if predicted_punct in {".", "?", "!"} and not (source_punct == "." and predicted_punct == ","):
                return False, "internal_sentence_end"

        threshold = self.thresholds["punct_delete"] if predicted_punct == "" else self.thresholds["punct"]
        margin_threshold = self.thresholds["punct_delete_margin"] if predicted_punct == "" else self.thresholds["punct_margin"]
        if confidence < threshold or margin < margin_threshold:
            return False, "low_confidence_or_margin"

        if is_final and source_punct in {".", "!", "?"} and predicted_punct == "":
            return False, "final_punctuation_delete"
        if is_final and predicted_punct and predicted_punct not in {".", "?", "!"}:
            return False, "final_non_terminal_punctuation"
        if source_punct and predicted_punct == "" and self._is_protected_next_token(next_word):
            return False, "delete_before_protected"
        return True, "applied"

    @staticmethod
    def _is_protected_next_token(word: str) -> bool:
        word = str(word)
        if not word:
            return False
        if any(ch.isdigit() for ch in word):
            return True
        if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in word):
            return True
        return word.isupper()

    def get_confidence(self, text: str) -> float:
        return self.correct_with_details(text).confidence


def probability_margin(probs: np.ndarray) -> float:
    values = np.asarray(probs, dtype=np.float32)
    if values.size < 2:
        return 1.0
    top_two = np.sort(np.partition(values, -2)[-2:])
    return float(top_two[-1] - top_two[-2])
