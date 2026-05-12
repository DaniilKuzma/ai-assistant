"""Main runtime corrector for the hybrid edit-based assistant."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from typing import Dict, List, Sequence

import numpy as np

from candidate_generator import CandidateGenerator
from context_reranker import ContextReranker
from edit_labels import ACTION_DELETE, ACTION_KEEP, action_replace_rank, iter_span_phrase_rewrites
from entity_guard import ProtectedLexicon
from hybrid_preprocessor import HybridPreprocessor
from morphology_guard import is_morphological_dictionary_word, is_same_lemma_inflection
from quality_guard import QualityGuard
from text_utils import (
    TRAINABLE_PUNCT_LABELS,
    apply_case_like,
    extract_word_slots,
    is_protected_punctuation_gap,
    is_trainable_punctuation_gap,
    normalize_word,
    punctuation_labels_for_slots,
    rebuild_preserving_layout,
)


GLUED_SERVICE_TOKENS = {
    "в",
    "во",
    "на",
    "не",
    "ни",
    "о",
    "об",
    "обо",
    "по",
    "за",
    "до",
    "к",
    "ко",
    "с",
    "со",
    "из",
    "от",
    "их",
}
ALLOWED_PHRASE_JOININGS = {
    ("так", "же", "также"),
    ("что", "бы", "чтобы"),
    ("по", "новому", "по-новому"),
    ("не", "смотря", "несмотря"),
    ("в", "виду", "ввиду"),
    ("мини", "футбол", "мини-футбол"),
}
OFFICIAL_PRESIDENT_NEXT = {"рф", "российской"}
SAFE_COMMA_DELETE_SERVICE_WORDS = {
    "в",
    "во",
    "на",
    "к",
    "ко",
    "с",
    "со",
    "от",
    "до",
    "из",
    "за",
    "по",
    "при",
    "для",
    "без",
    "у",
    "о",
    "об",
    "обо",
}
SAFE_COMMA_DELETE_MONTHS = {
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
}
DICTIONARY_LIKE_CANDIDATE_SOURCES = {"dictionary", "orthographic", "mined"}
CONTEXT_RERANK_CANDIDATE_SOURCES = DICTIONARY_LIKE_CANDIDATE_SOURCES | {"split", "phrase_safe", "phrase", "rule"}
CONTEXT_REQUIRED_CANDIDATE_SOURCES = {"split", "phrase_safe", "phrase"}


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
    span_len: int = 1


@dataclass(frozen=True)
class RuntimeCandidate:
    text: str
    source: str = "original"
    distance: int = 0
    score: float = 0.0
    rank: int = -1
    span_len: int = 1


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


@dataclass(frozen=True)
class WordDecision:
    index: int
    word: str
    predicted_action: str
    confidence: float
    margin: float
    selected_rank: int = -1
    selected_text: str = ""
    selected_source: str = ""
    selected_score: float = 0.0
    applied: bool = False
    blocked_reason: str = ""
    candidate_texts: List[str] = field(default_factory=list)
    candidate_sources: List[str] = field(default_factory=list)
    candidate_scores: List[float] = field(default_factory=list)
    context_best: str = ""
    context_scores: List[str] = field(default_factory=list)
    context_margin: float = 0.0
    reranker_reason: str = ""
    context_needed: bool = False
    window_id: int = 0
    window_conflict_resolved: bool = False
    protected_source: bool = False
    entity_context: bool = False
    clean_lexicon_frequency: int = 0


@dataclass(frozen=True)
class RerankOutcome:
    selected: RuntimeCandidate | None
    context_score: float = 0.0
    context_margin: float = 0.0
    reranked: bool = False
    reason: str = ""
    context_best: str = ""
    context_scores: List[str] = field(default_factory=list)


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    confidence: float
    accepted: bool
    guard_reason: str
    edits: List[CorrectionEdit]
    punctuation_diagnostics: List[PunctuationDecision] = field(default_factory=list)
    word_diagnostics: List[WordDecision] = field(default_factory=list)


class HybridCorrector:
    """One public correction interface for the project."""

    def __init__(
        self,
        model_path: str = "models/hybrid_corrector.keras",
        preprocessor_path: str = "models/hybrid_preprocessor.pkl",
        candidate_generator_path: str = "models/candidate_generator.pkl",
        strictness: str = "normal",
        punctuation_mode: str = "conservative",
        candidate_top_k: int = 16,
        min_dictionary_score: float = 0.25,
        use_morphology_guard: bool = True,
        context_reranker_enabled: bool = True,
        context_model_name: str = "DeepPavlov/rubert-base-cased",
        context_device: str = "cpu",
        context_margin: float = 0.25,
        use_entity_guard: bool = True,
        protected_lexicon_paths: Sequence[str] | None = None,
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
        self.context_device = context_device
        self.context_margin = float(context_margin)
        self.use_entity_guard = use_entity_guard
        self.thresholds = self._thresholds(strictness)
        self.runtime_stats: Counter[str] = Counter()
        self.entity_guard = self._load_entity_guard(protected_lexicon_paths)
        self.context_reranker = ContextReranker(
            model_name=context_model_name,
            enabled=context_reranker_enabled,
            device=context_device,
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
                "action": 0.82,
                "dictionary_action": 0.88,
                "rule_action": 0.88,
                "action_margin": 0.18,
                "delete": 0.995,
                "punct": 0.92,
                "punct_delete": 0.995,
                "comma_delete": 0.9995,
                "punct_margin": 0.12,
                "punct_delete_margin": 0.45,
                "comma_delete_margin": 0.70,
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
                "comma_delete": 0.999,
                "punct_margin": 0.20,
                "punct_delete_margin": 0.40,
                "comma_delete_margin": 0.60,
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
                "comma_delete": 0.995,
                "punct_margin": 0.14,
                "punct_delete_margin": 0.30,
                "comma_delete_margin": 0.45,
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
            long_oov_max_distance=2,
            long_oov_min_length=8,
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
                if config.get("model_version", 0) < 10:
                    print("Hybrid model is obsolete: missing V10 punctuation/orthography labels. Retrain the notebook.")
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

                    df = pd.read_csv(path, low_memory=False)
                    if "correct_text" in df.columns:
                        return df["correct_text"].dropna().astype(str).tolist()
                except Exception:
                    continue
            else:
                return path.read_text(encoding="utf-8").splitlines()
        return []

    def _load_entity_guard(self, paths: Sequence[str] | None = None) -> ProtectedLexicon:
        if not getattr(self, "use_entity_guard", True):
            return ProtectedLexicon.empty()
        try:
            if paths is None:
                return ProtectedLexicon.from_default_paths()
            return ProtectedLexicon.from_paths(paths)
        except Exception:
            return ProtectedLexicon.empty()

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

        line_results: List[CorrectionResult] = []
        for line in original.splitlines() or [original]:
            if not line.strip():
                line_results.append(CorrectionResult(line, line, 1.0, True, "empty", []))
                continue
            line_results.append(self._correct_segment(line))

        return self._combine_line_results(original, line_results)

    def correct_many(self, texts: Sequence[str], batch_size: int = 256) -> List[CorrectionResult]:
        originals = ["" if text is None else str(text) for text in texts]
        results: List[CorrectionResult | None] = [None] * len(originals)
        line_results_by_text: List[List[CorrectionResult]] = [[] for _ in originals]
        prepared_segments: List[dict[str, object]] = []

        if not self.is_trained:
            return [self.correct_with_details(original) for original in originals]

        for text_index, original in enumerate(originals):
            if not original.strip():
                results[text_index] = CorrectionResult(original, original, 1.0, True, "empty", [])
                continue

            for line in original.splitlines() or [original]:
                if not line.strip():
                    line_results_by_text[text_index].append(CorrectionResult(line, line, 1.0, True, "empty", []))
                    continue

                slots = extract_word_slots(line)
                if not slots:
                    line_results_by_text[text_index].append(CorrectionResult(line, line, 1.0, True, "no_words", []))
                    continue

                source_words = [slot.word for slot in slots]
                source_puncts = punctuation_labels_for_slots(line, slots)
                source_gaps = [
                    line[slot.end : (slots[i + 1].start if i + 1 < len(slots) else len(line))]
                    for i, slot in enumerate(slots)
                ]
                prepared_segments.append(
                    {
                        "text_index": text_index,
                        "line": line,
                        "slots": slots,
                        "source_words": source_words,
                        "source_puncts": source_puncts,
                        "source_gaps": source_gaps,
                        "candidate_options": self._candidate_options(source_words),
                        "window_predictions": [],
                    }
                )

        self._predict_prepared_segments(prepared_segments, batch_size=batch_size)

        for segment in prepared_segments:
            text_index = int(segment["text_index"])
            line = str(segment["line"])
            source_words = segment["source_words"]
            source_puncts = segment["source_puncts"]
            source_gaps = segment["source_gaps"]
            candidate_options = segment["candidate_options"]
            corrected_words, corrected_puncts, edits, punctuation_diagnostics, word_diagnostics, confidence = (
                self._decode_model_predictions(
                    source_words,
                    candidate_options,
                    source_puncts,
                    source_gaps,
                    segment["window_predictions"],
                )
            )
            line_results_by_text[text_index].append(
                self._finish_segment_correction(
                    line,
                    segment["slots"],
                    source_puncts,
                    corrected_words,
                    corrected_puncts,
                    edits,
                    punctuation_diagnostics,
                    word_diagnostics,
                    confidence,
                )
            )

        for text_index, original in enumerate(originals):
            if results[text_index] is None:
                results[text_index] = self._combine_line_results(original, line_results_by_text[text_index])

        return [result for result in results if result is not None]

    def _combine_line_results(self, original: str, line_results: Sequence[CorrectionResult]) -> CorrectionResult:
        corrected_lines: List[str] = []
        all_edits: List[CorrectionEdit] = []
        all_punctuation_diagnostics: List[PunctuationDecision] = []
        all_word_diagnostics: List[WordDecision] = []
        confidences: List[float] = []
        accepted = True
        reasons: List[str] = []

        for line_result in line_results:
            corrected_lines.append(line_result.corrected)
            all_edits.extend(line_result.edits)
            all_punctuation_diagnostics.extend(line_result.punctuation_diagnostics)
            all_word_diagnostics.extend(line_result.word_diagnostics)
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
                all_word_diagnostics,
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
            all_word_diagnostics,
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
            corrected_words, corrected_puncts, edits, punctuation_diagnostics, word_diagnostics, confidence = self._model_correct(
                source_words,
                candidate_options,
                source_puncts,
                source_gaps,
            )
        else:
            corrected_words, corrected_puncts, edits, word_diagnostics, confidence = self._dictionary_correct(
                source_words,
                candidate_options,
                source_puncts,
                source_gaps,
            )
            punctuation_diagnostics = []

        return self._finish_segment_correction(
            text,
            slots,
            source_puncts,
            corrected_words,
            corrected_puncts,
            edits,
            punctuation_diagnostics,
            word_diagnostics,
            confidence,
        )

    def _finish_segment_correction(
        self,
        text: str,
        slots,
        source_puncts: Sequence[str],
        corrected_words: Sequence[str],
        corrected_puncts: Sequence[str],
        edits: List[CorrectionEdit],
        punctuation_diagnostics: List[PunctuationDecision],
        word_diagnostics: List[WordDecision],
        confidence: float,
    ) -> CorrectionResult:
        if edits:
            corrected = rebuild_preserving_layout(
                text,
                slots,
                corrected_words,
                corrected_puncts,
                punctuation_mode=self.punctuation_mode,
                source_puncts=source_puncts,
                rewrite_punct_indices=[edit.index for edit in edits if edit.action == "PUNCT"],
            )
            if any(edit.action == ACTION_DELETE or int(getattr(edit, "span_len", 1)) > 1 for edit in edits):
                corrected = self._normalize_span_deleted_spacing(corrected)
        else:
            corrected = text

        if not edits:
            return CorrectionResult(text, text, confidence, True, "unchanged", [], punctuation_diagnostics, word_diagnostics)

        decision = self.guard.check(text, corrected, confidence, len(edits))
        if not decision.accepted:
            return CorrectionResult(text, text, confidence, False, decision.reason, [], punctuation_diagnostics, word_diagnostics)

        return CorrectionResult(text, corrected, confidence, True, decision.reason, edits, punctuation_diagnostics, word_diagnostics)

    @staticmethod
    def _normalize_span_deleted_spacing(text: str) -> str:
        text = re.sub(r"[ \t]{2,}", " ", str(text))
        text = re.sub(r"\s+([,.;:!?»)\]])", r"\1", text)
        text = re.sub(r"([«(\[])\s+", r"\1", text)
        return text.strip()

    def _candidate_options_for_word(self, word: str, prev_word: str = "", next_word: str = "") -> List[RuntimeCandidate]:
        norm = normalize_word(word)
        if norm and not self.candidate_generator.is_known(norm) and not self.candidate_generator._has_rule_candidate(norm):
            entity_meta = self._entity_metadata(word, prev_word, next_word)
            if (
                bool(entity_meta["protected_source"])
                or bool(entity_meta["entity_context"])
                or is_morphological_dictionary_word(norm)
            ):
                return [RuntimeCandidate(text=word, rank=0)]
        raw_candidates = self.candidate_generator.get_candidates(word, max_candidates=self.candidate_top_k)
        if any(candidate.source == "split" for candidate in raw_candidates):
            self._increment_stat("split_candidates_seen")
        if any(candidate.source == "phrase_safe" for candidate in raw_candidates):
            self._increment_stat("phrase_safe_candidates_seen")
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
        candidates = [
            self._candidate_options_for_word(
                word,
                prev_word=words[i - 1] if i > 0 else "",
                next_word=words[i + 1] if i + 1 < len(words) else "",
            )
            for i, word in enumerate(words)
        ]
        normalized = [normalize_word(word) for word in words]

        phrase_map = getattr(self.candidate_generator, "phrase_confusions", {})
        for source_phrase, target_phrases in phrase_map.items():
            source_parts = source_phrase.split()
            if len(source_parts) <= 1:
                continue
            for i in range(0, len(words) - len(source_parts) + 1):
                if normalized[i : i + len(source_parts)] != source_parts:
                    continue
                candidates = self._inject_span_phrase_candidates(
                    candidates,
                    words,
                    i,
                    len(source_parts),
                    str(target_phrases[0]).split(),
                    score=100.0,
                )

        for start, source_len, target_parts in iter_span_phrase_rewrites(words):
            candidates = self._inject_span_phrase_candidates(
                candidates,
                words,
                start,
                source_len,
                list(target_parts),
                score=120.0,
            )
        return candidates

    def _inject_span_phrase_candidates(
        self,
        candidates: List[List[RuntimeCandidate]],
        words: Sequence[str],
        start: int,
        source_len: int,
        target_parts: Sequence[str],
        *,
        score: float,
    ) -> List[List[RuntimeCandidate]]:
        source_parts = [normalize_word(word) for word in words[start : start + source_len]]
        for offset, span_len, candidate_text in self._span_phrase_candidate_specs(source_parts, target_parts):
            index = start + offset
            if index >= len(candidates):
                continue
            phrase_candidate = RuntimeCandidate(
                text=apply_case_like(candidate_text, words[index]),
                source="phrase",
                distance=0,
                score=score,
                rank=0,
                span_len=span_len,
            )
            if normalize_word(phrase_candidate.text) == normalize_word(words[index]):
                continue
            candidates[index] = self._prepend_candidate(candidates[index], phrase_candidate)
        return candidates

    @staticmethod
    def _span_phrase_candidate_specs(
        source_parts: Sequence[str],
        target_parts: Sequence[str],
    ) -> List[tuple[int, int, str]]:
        source_parts = [normalize_word(part) for part in source_parts]
        target_parts = [normalize_word(part) for part in target_parts]
        if not source_parts or not target_parts or source_parts == target_parts:
            return []

        prefix_len = 0
        while (
            prefix_len < len(source_parts)
            and prefix_len < len(target_parts)
            and source_parts[prefix_len] == target_parts[prefix_len]
        ):
            prefix_len += 1

        suffix_len = 0
        while (
            suffix_len < len(source_parts) - prefix_len
            and suffix_len < len(target_parts) - prefix_len
            and source_parts[len(source_parts) - 1 - suffix_len] == target_parts[len(target_parts) - 1 - suffix_len]
        ):
            suffix_len += 1

        source_mid_len = len(source_parts) - prefix_len - suffix_len
        target_mid_end = len(target_parts) - suffix_len if suffix_len else len(target_parts)
        target_mid_parts = target_parts[prefix_len:target_mid_end]
        if source_mid_len <= 0 or not target_mid_parts:
            return []

        if source_mid_len == len(target_mid_parts):
            return [
                (prefix_len + offset, 1, target_word)
                for offset, target_word in enumerate(target_mid_parts)
                if source_parts[prefix_len + offset] != target_word
            ]

        return [(prefix_len, source_mid_len, " ".join(target_mid_parts))]

    def _prepend_candidate(
        self,
        candidates: Sequence[RuntimeCandidate],
        candidate: RuntimeCandidate,
    ) -> List[RuntimeCandidate]:
        normalized_candidate = normalize_word(candidate.text)
        return self._renumber_candidates(
            [candidate]
            + [
                existing
                for existing in candidates
                if normalize_word(existing.text) != normalized_candidate
            ][: self.candidate_top_k - 1]
        )

    @staticmethod
    def _renumber_candidates(candidates: Sequence[RuntimeCandidate]) -> List[RuntimeCandidate]:
        return [
            RuntimeCandidate(
                text=candidate.text,
                source=candidate.source,
                distance=candidate.distance,
                score=candidate.score,
                rank=rank,
                span_len=max(1, int(getattr(candidate, "span_len", 1))),
            )
            for rank, candidate in enumerate(candidates)
        ]

    def _increment_stat(self, key: str, value: int = 1) -> None:
        if not hasattr(self, "runtime_stats"):
            self.runtime_stats = Counter()
        self.runtime_stats[key] += value

    def _entity_metadata(self, word: str, prev_word: str = "", next_word: str = "") -> dict[str, object]:
        guard = getattr(self, "entity_guard", None)
        if not getattr(self, "use_entity_guard", True) or guard is None:
            return {"protected_source": False, "entity_context": False, "clean_lexicon_frequency": 0}
        return guard.metadata(word, prev_word=prev_word, next_word=next_word)

    def _dictionary_correct(
        self,
        source_words: Sequence[str],
        candidate_options: Sequence[Sequence[RuntimeCandidate]],
        source_puncts: Sequence[str],
        source_gaps: Sequence[str],
    ) -> tuple[List[str], List[str], List[CorrectionEdit], List[WordDecision], float]:
        corrected_words: List[str] = []
        corrected_puncts: List[str] = []
        edits: List[CorrectionEdit] = []
        word_diagnostics: List[WordDecision] = []
        confidences: List[float] = []

        for i, (source, candidates, punct) in enumerate(zip(source_words, candidate_options, source_puncts)):
            selected = None
            context_score = 0.0
            context_margin = 0.0
            reranked = False
            reranker_reason = ""
            context_best = ""
            context_scores: List[str] = []
            blocked_reason = "no_candidate"
            prev_word = source_words[i - 1] if i > 0 else ""
            next_word = source_words[i + 1] if i + 1 < len(source_words) else ""
            for candidate in candidates:
                blocked_reason = self._candidate_block_reason(
                    source,
                    candidate,
                    0.90,
                    0.25,
                    prev_word=prev_word,
                    next_word=next_word,
                    skip_context=True,
                )
                if candidate.text != source and not blocked_reason:
                    outcome = self._maybe_rerank_candidate(
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
                    selected = outcome.selected
                    context_score = outcome.context_score
                    context_margin = outcome.context_margin
                    reranked = outcome.reranked
                    reranker_reason = outcome.reason
                    context_best = outcome.context_best
                    context_scores = outcome.context_scores
                    if selected is None:
                        blocked_reason = outcome.reason
                    else:
                        blocked_reason = self._candidate_block_reason(
                            source,
                            selected,
                            0.90,
                            0.25,
                            prev_word=prev_word,
                            next_word=next_word,
                            context_margin=outcome.context_margin,
                        )
                    break
            if selected is not None and not blocked_reason:
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
                        span_len=selected.span_len,
                    )
                )
                confidences.append(0.55)
                blocked_reason = "applied"
            else:
                corrected_words.append(source)
                confidences.append(0.90)
            if self._should_log_word_decision(candidates, "REPLACE_0"):
                entity_meta = self._entity_metadata(source, prev_word, next_word)
                word_diagnostics.append(
                    WordDecision(
                        index=i,
                        word=source,
                        predicted_action="REPLACE_0",
                        confidence=0.90,
                        margin=0.25,
                        selected_rank=selected.rank if selected is not None else -1,
                        selected_text=selected.text if selected is not None else "",
                        selected_source=selected.source if selected is not None else "",
                        selected_score=selected.score if selected is not None else 0.0,
                        applied=selected is not None and blocked_reason == "applied",
                        blocked_reason=blocked_reason,
                        candidate_texts=[candidate.text for candidate in candidates],
                        candidate_sources=[candidate.source for candidate in candidates],
                        candidate_scores=[candidate.score for candidate in candidates],
                        context_best=context_best,
                        context_scores=context_scores,
                        context_margin=context_margin,
                        reranker_reason=reranker_reason,
                        context_needed=self._decision_context_needed(candidates, selected),
                        protected_source=bool(entity_meta["protected_source"]),
                        entity_context=bool(entity_meta["entity_context"]),
                        clean_lexicon_frequency=int(entity_meta["clean_lexicon_frequency"]),
                    )
                )
            corrected_puncts.append(punct)

        if edits:
            edits_by_index: List[List[CorrectionEdit]] = [[] for _ in source_words]
            for edit in edits:
                if 0 <= edit.index < len(edits_by_index):
                    edits_by_index[edit.index].append(edit)
            self._apply_span_deletions(
                source_words,
                source_puncts,
                corrected_words,
                corrected_puncts,
                edits_by_index,
            )
            edits = [edit for group in edits_by_index for edit in group]

        return corrected_words, corrected_puncts, edits, word_diagnostics, float(np.mean(confidences) if confidences else 1.0)

    @staticmethod
    def _apply_span_deletions(
        source_words: Sequence[str],
        source_puncts: Sequence[str],
        corrected_words: List[str],
        corrected_puncts: List[str],
        edits_by_index: List[List[CorrectionEdit]],
    ) -> None:
        length = len(source_words)
        for index, token_edits in enumerate(list(edits_by_index)):
            span_edits = [
                edit
                for edit in token_edits
                if edit.action.startswith("REPLACE_") and int(getattr(edit, "span_len", 1)) > 1
            ]
            if not span_edits:
                continue
            edit = span_edits[0]
            span_len = min(int(getattr(edit, "span_len", 1)), length - index)
            if span_len <= 1:
                continue

            last_index = index + span_len - 1
            if last_index < len(corrected_puncts) and corrected_puncts[last_index]:
                corrected_puncts[index] = corrected_puncts[last_index]

            for delete_index in range(index + 1, index + span_len):
                corrected_words[delete_index] = ""
                corrected_puncts[delete_index] = ""
                edits_by_index[delete_index] = [
                    CorrectionEdit(
                        delete_index,
                        source_words[delete_index],
                        "",
                        ACTION_DELETE,
                        edit.confidence,
                        edit.candidate_source,
                        edit.candidate_distance,
                        edit.candidate_score,
                        edit.candidate_rank,
                        edit.context_score,
                        edit.context_margin,
                        edit.reranked,
                        edit.reranker_reason,
                    )
                ]

    def _model_correct(
        self,
        source_words: Sequence[str],
        candidate_options: Sequence[Sequence[RuntimeCandidate]],
        source_puncts: Sequence[str],
        source_gaps: Sequence[str],
    ) -> tuple[List[str], List[str], List[CorrectionEdit], List[PunctuationDecision], List[WordDecision], float]:
        assert self.preprocessor is not None
        window_predictions = []
        for window_id, (start, end) in enumerate(self._window_ranges(len(source_words), self.preprocessor.max_length)):
            local_words = list(source_words[start:end])
            local_candidates = [candidate_options[i] for i in range(start, end)]
            local_candidate_words = [[candidate.text for candidate in candidates] for candidates in local_candidates]
            local_puncts = list(source_puncts[start:end])
            inputs = self.preprocessor.vectorize_inference(local_words, local_candidate_words, local_puncts)
            action_probs, punct_probs = self._predict_window(inputs)
            window_predictions.append((window_id, start, end, action_probs, punct_probs))
        return self._decode_model_predictions(
            source_words,
            candidate_options,
            source_puncts,
            source_gaps,
            window_predictions,
        )

    def _decode_model_predictions(
        self,
        source_words: Sequence[str],
        candidate_options: Sequence[Sequence[RuntimeCandidate]],
        source_puncts: Sequence[str],
        source_gaps: Sequence[str],
        window_predictions,
    ) -> tuple[List[str], List[str], List[CorrectionEdit], List[PunctuationDecision], List[WordDecision], float]:
        assert self.preprocessor is not None
        length = len(source_words)
        corrected_words = list(source_words)
        corrected_puncts = list(source_puncts)
        selected_edits: List[List[CorrectionEdit]] = [[] for _ in range(length)]
        selected_punct_diagnostics: List[PunctuationDecision | None] = [None] * length
        selected_word_diagnostics: List[WordDecision | None] = [None] * length
        selected_scores: List[tuple[int, float]] = [(-1, -1.0)] * length
        selected_confidences: List[float] = [0.0] * length
        conflict_flags = [False] * length

        for window_id, start, end, action_probs, punct_probs in window_predictions:
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
                word_applied = False
                word_blocked_reason = "keep"
                word_selected: RuntimeCandidate | None = None
                context_score = 0.0
                context_margin = 0.0
                reranked = False
                reranker_reason = ""
                context_best = ""
                context_scores: List[str] = []
                replace_rank = action_replace_rank(action)
                if replace_rank is not None and replace_rank < len(candidates):
                    candidate = candidates[replace_rank]
                    word_selected = candidate
                    word_blocked_reason = self._candidate_block_reason(
                        source,
                        candidate,
                        action_conf,
                        action_margin,
                        prev_word=prev_word,
                        next_word=next_word,
                        skip_context=True,
                    )
                    if candidate.text != source and not word_blocked_reason:
                        outcome = self._maybe_rerank_candidate(
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
                        candidate = outcome.selected
                        context_score = outcome.context_score
                        context_margin = outcome.context_margin
                        reranked = outcome.reranked
                        reranker_reason = outcome.reason
                        context_best = outcome.context_best
                        context_scores = outcome.context_scores
                        word_selected = candidate
                        if candidate is None:
                            word_blocked_reason = outcome.reason
                        else:
                            word_blocked_reason = self._candidate_block_reason(
                                source,
                                candidate,
                                action_conf,
                                action_margin,
                                prev_word=prev_word,
                                next_word=next_word,
                                context_margin=outcome.context_margin,
                            )
                        if candidate is not None and not word_blocked_reason:
                            output_word = candidate.text
                            word_applied = True
                            word_blocked_reason = "applied"
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
                                    span_len=candidate.span_len,
                                )
                            )
                elif replace_rank is not None:
                    word_blocked_reason = "replace_rank_out_of_range"
                elif action == ACTION_DELETE and action_conf >= self.thresholds["delete"]:
                    output_word = ""
                    word_applied = True
                    word_blocked_reason = "applied"
                    token_edits.append(CorrectionEdit(i, source, "", action, action_conf))
                elif action == ACTION_DELETE:
                    word_blocked_reason = "low_delete_confidence"

                word_diag = None
                if self._should_log_word_decision(candidates, action):
                    entity_meta = self._entity_metadata(source, prev_word, next_word)
                    word_diag = WordDecision(
                        index=i,
                        word=source,
                        predicted_action=action,
                        confidence=action_conf,
                        margin=action_margin,
                        selected_rank=word_selected.rank if word_selected is not None else -1,
                        selected_text=word_selected.text if word_selected is not None else "",
                        selected_source=word_selected.source if word_selected is not None else "",
                        selected_score=word_selected.score if word_selected is not None else 0.0,
                        applied=word_applied,
                        blocked_reason=word_blocked_reason,
                        candidate_texts=[candidate.text for candidate in candidates],
                        candidate_sources=[candidate.source for candidate in candidates],
                        candidate_scores=[candidate.score for candidate in candidates],
                        context_best=context_best,
                        context_scores=context_scores,
                        context_margin=context_margin,
                        reranker_reason=reranker_reason,
                        context_needed=self._decision_context_needed(candidates, word_selected),
                        window_id=window_id,
                        window_conflict_resolved=False,
                        protected_source=bool(entity_meta["protected_source"]),
                        entity_context=bool(entity_meta["entity_context"]),
                        clean_lexicon_frequency=int(entity_meta["clean_lexicon_frequency"]),
                    )

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
                    selected_word_diagnostics[i] = word_diag
                    selected_scores[i] = score
                    selected_confidences[i] = token_conf

        self._apply_span_deletions(
            source_words,
            source_puncts,
            corrected_words,
            corrected_puncts,
            selected_edits,
        )

        edits: List[CorrectionEdit] = []
        punctuation_diagnostics: List[PunctuationDecision] = []
        word_diagnostics: List[WordDecision] = []
        for i, diag in enumerate(selected_punct_diagnostics):
            edits.extend(selected_edits[i])
            word_diag = selected_word_diagnostics[i]
            if word_diag is not None:
                word_diagnostics.append(
                    WordDecision(
                        index=word_diag.index,
                        word=word_diag.word,
                        predicted_action=word_diag.predicted_action,
                        confidence=word_diag.confidence,
                        margin=word_diag.margin,
                        selected_rank=word_diag.selected_rank,
                        selected_text=word_diag.selected_text,
                        selected_source=word_diag.selected_source,
                        selected_score=word_diag.selected_score,
                        applied=word_diag.applied,
                        blocked_reason=word_diag.blocked_reason,
                        candidate_texts=word_diag.candidate_texts,
                        candidate_sources=word_diag.candidate_sources,
                        candidate_scores=word_diag.candidate_scores,
                        context_best=word_diag.context_best,
                        context_scores=word_diag.context_scores,
                        context_margin=word_diag.context_margin,
                        reranker_reason=word_diag.reranker_reason,
                        context_needed=word_diag.context_needed,
                        window_id=word_diag.window_id,
                        window_conflict_resolved=conflict_flags[i],
                        protected_source=word_diag.protected_source,
                        entity_context=word_diag.entity_context,
                        clean_lexicon_frequency=word_diag.clean_lexicon_frequency,
                    )
                )
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
            word_diagnostics,
            float(np.mean([c for c in selected_confidences if c > 0]) if selected_confidences else 1.0),
        )

    def _predict_window(self, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if callable(self.model):
            prediction = self.model(inputs, training=False)
        else:
            prediction = self.model.predict(inputs, verbose=0)
        if isinstance(prediction, dict):
            action_probs = prediction["action"]
            punct_probs = prediction["punct"]
        else:
            action_probs, punct_probs = prediction
        return np.asarray(action_probs)[0], np.asarray(punct_probs)[0]

    def _predict_prepared_segments(self, prepared_segments: Sequence[dict[str, object]], batch_size: int = 256) -> None:
        if not prepared_segments:
            return
        assert self.preprocessor is not None

        batch_words: List[List[str]] = []
        batch_candidate_words: List[List[List[str]]] = []
        batch_puncts: List[List[str]] = []
        window_refs: List[tuple[dict[str, object], int, int, int]] = []

        for segment in prepared_segments:
            source_words = segment["source_words"]
            source_puncts = segment["source_puncts"]
            candidate_options = segment["candidate_options"]
            for window_id, (start, end) in enumerate(self._window_ranges(len(source_words), self.preprocessor.max_length)):
                local_candidates = [candidate_options[i] for i in range(start, end)]
                batch_words.append(list(source_words[start:end]))
                batch_candidate_words.append([[candidate.text for candidate in candidates] for candidates in local_candidates])
                batch_puncts.append(list(source_puncts[start:end]))
                window_refs.append((segment, window_id, start, end))

        chunk_size = max(1, int(batch_size))
        for batch_start in range(0, len(window_refs), chunk_size):
            batch_end = min(batch_start + chunk_size, len(window_refs))
            inputs = self.preprocessor.vectorize_inference_batch(
                batch_words[batch_start:batch_end],
                batch_candidate_words[batch_start:batch_end],
                batch_puncts[batch_start:batch_end],
            )
            action_batch, punct_batch = self._predict_windows_batch(inputs, batch_size=chunk_size)
            for local_i, ref in enumerate(window_refs[batch_start:batch_end]):
                segment, window_id, start, end = ref
                segment["window_predictions"].append((window_id, start, end, action_batch[local_i], punct_batch[local_i]))

    def _predict_windows_batch(self, inputs: dict[str, np.ndarray], batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        if callable(self.model):
            prediction = self.model(inputs, training=False)
        else:
            prediction = self.model.predict(inputs, batch_size=max(1, int(batch_size)), verbose=0)
        if isinstance(prediction, dict):
            action_probs = prediction["action"]
            punct_probs = prediction["punct"]
        else:
            action_probs, punct_probs = prediction
        return np.asarray(action_probs), np.asarray(punct_probs)

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

    @staticmethod
    def _context_score_payload(scores) -> List[str]:
        payload = []
        for score in scores or []:
            value = score.score
            value_text = f"{float(value):.4f}" if isinstance(value, (int, float)) and math.isfinite(value) else str(value)
            payload.append(f"{score.text}:{value_text}:{score.reason}")
        return payload

    @staticmethod
    def _best_context_text(scores) -> str:
        finite = [score for score in scores or [] if math.isfinite(float(score.score))]
        if not finite:
            return ""
        return max(finite, key=lambda item: item.score).text

    def _should_relax_source_veto(
        self,
        source: str,
        selected: RuntimeCandidate,
        confidence: float,
        context_margin: float,
        *,
        prev_word: str = "",
        next_word: str = "",
    ) -> bool:
        if selected.source not in DICTIONARY_LIKE_CANDIDATE_SOURCES:
            return False
        if context_margin >= 0.75 or confidence < 0.95:
            return False
        if selected.distance != 1:
            return False
        if source != source.lower() or any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in source):
            return False
        if self.candidate_generator.is_known(source) or is_morphological_dictionary_word(source):
            return False
        entity_meta = self._entity_metadata(source, prev_word, next_word)
        return not bool(entity_meta["protected_source"] or entity_meta["entity_context"])

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
    ) -> RerankOutcome:
        context_required = self._requires_context_confirmation(selected)
        if context_required:
            self._increment_stat("context_needed_count")
        if selected.source not in CONTEXT_RERANK_CANDIDATE_SOURCES:
            return RerankOutcome(selected=selected, reason="not_called")
        if not getattr(self, "context_reranker_enabled", False):
            if context_required:
                self._increment_context_required_block(selected)
                return RerankOutcome(selected=None, reason=f"{selected.source}_context_disabled")
            return RerankOutcome(selected=selected, reason="not_called")

        viable = [
            candidate
            for candidate in candidates
            if candidate.source in CONTEXT_RERANK_CANDIDATE_SOURCES and normalize_word(candidate.text) != normalize_word(source)
        ]
        if not viable:
            return RerankOutcome(selected=selected, reason="no_viable_candidates")

        self._increment_stat("reranker_called_count")
        source_context_text = self._source_span_text(source_words, index, getattr(selected, "span_len", 1))
        texts = [source_context_text] + [candidate.text for candidate in viable]
        span_lengths = [max(1, int(getattr(selected, "span_len", 1)))] + [
            max(1, int(getattr(candidate, "span_len", 1)))
            for candidate in viable
        ]
        try:
            scores = self.context_reranker.score_candidates(
                source_words,
                index,
                source_context_text,
                texts,
                candidate_span_lengths=span_lengths,
            )
        except TypeError as exc:
            if "candidate_span_lengths" not in str(exc):
                raise
            scores = self.context_reranker.score_candidates(
                source_words,
                index,
                source_context_text,
                texts,
            )
        context_payload = self._context_score_payload(scores)
        context_best = self._best_context_text(scores)
        for score in scores or []:
            if not score.available or score.reason != "ok" or not math.isfinite(float(score.score)):
                self._increment_stat(f"reranker_error_reason:{score.reason}")
        if not scores or not all(score.available for score in scores):
            self._increment_stat("reranker_fail_open_count")
            reason = scores[0].reason if scores else "no_scores"
            if context_required:
                self._increment_context_required_block(selected)
                return RerankOutcome(
                    selected=None,
                    reason=f"{selected.source}_context_unavailable:{reason}",
                    context_best=context_best,
                    context_scores=context_payload,
                )
            return RerankOutcome(selected=selected, reason=reason, context_best=context_best, context_scores=context_payload)

        finite_scores = [score for score in scores if math.isfinite(float(score.score))]
        non_finite_count = len(scores) - len(finite_scores)
        if non_finite_count:
            self._increment_stat("reranker_non_finite_count", non_finite_count)
        if len(finite_scores) < 2:
            reason = "context_non_finite"
            if context_required:
                self._increment_context_required_block(selected)
                reason = f"{selected.source}_context_non_finite"
                return RerankOutcome(
                    selected=None,
                    reason=reason,
                    context_best=context_best,
                    context_scores=context_payload,
                )
            return RerankOutcome(
                selected=selected,
                reason=reason,
                context_best=context_best,
                context_scores=context_payload,
            )

        score_by_norm = {normalize_word(score.text): score for score in finite_scores}
        ranked = sorted(finite_scores, key=lambda item: item.score, reverse=True)
        best = ranked[0]
        second_score = ranked[1].score if len(ranked) > 1 else best.score
        context_margin = float(best.score - second_score)
        source_score = score_by_norm.get(normalize_word(source_context_text))
        selected_score = score_by_norm.get(normalize_word(selected.text))
        context_score = float(selected_score.score) if selected_score is not None else -math.inf

        if (
            source_score is not None
            and normalize_word(best.text) == normalize_word(source_context_text)
            and context_margin >= self.context_margin
        ):
            self._increment_stat("reranker_source_preferred_count")
            if self._should_relax_source_veto(
                source,
                selected,
                confidence,
                context_margin,
                prev_word=prev_word,
                next_word=next_word,
            ):
                self._increment_stat("context_source_veto_relaxed_count")
                return RerankOutcome(
                    selected=selected,
                    context_score=context_score,
                    context_margin=context_margin,
                    reranked=False,
                    reason="context_source_veto_relaxed",
                    context_best=best.text,
                    context_scores=context_payload,
                )
            if context_required:
                self._increment_context_required_block(selected)
            return RerankOutcome(
                selected=None,
                context_score=float(source_score.score),
                context_margin=context_margin,
                reranked=True,
                reason="context_prefers_source",
                context_best=best.text,
                context_scores=context_payload,
            )

        if normalize_word(best.text) != normalize_word(source_context_text) and context_margin >= self.context_margin:
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
                return RerankOutcome(
                    selected=best_candidate,
                    context_score=float(best.score),
                    context_margin=context_margin,
                    reranked=changed,
                    reason="context_changed_top1" if changed else "context_confirmed",
                    context_best=best.text,
                    context_scores=context_payload,
                )

        if context_required:
            self._increment_context_required_block(selected)
            return RerankOutcome(
                selected=None,
                context_score=context_score,
                context_margin=context_margin,
                reranked=False,
                reason=f"{selected.source}_context_low_margin",
                context_best=best.text,
                context_scores=context_payload,
            )

        return RerankOutcome(
            selected=selected,
            context_score=context_score,
            context_margin=context_margin,
            reranked=False,
            reason="context_low_margin",
            context_best=best.text,
            context_scores=context_payload,
        )

    @staticmethod
    def _requires_context_confirmation(candidate: RuntimeCandidate) -> bool:
        return HybridCorrector._candidate_needs_context(candidate)

    @staticmethod
    def _candidate_needs_context(candidate: RuntimeCandidate | None) -> bool:
        if candidate is None:
            return False
        return (
            candidate.source in CONTEXT_REQUIRED_CANDIDATE_SOURCES
            or " " in str(candidate.text or "")
            or int(getattr(candidate, "span_len", 1)) > 1
        )

    @staticmethod
    def _decision_context_needed(
        candidates: Sequence[RuntimeCandidate],
        selected: RuntimeCandidate | None = None,
    ) -> bool:
        if HybridCorrector._candidate_needs_context(selected):
            return True
        return any(HybridCorrector._candidate_needs_context(candidate) for candidate in candidates)

    @staticmethod
    def _source_span_text(words: Sequence[str], index: int, span_len: int = 1) -> str:
        span_len = max(1, int(span_len))
        return " ".join(str(word) for word in words[index : min(len(words), index + span_len)]) or (
            str(words[index]) if 0 <= index < len(words) else ""
        )

    def _increment_context_required_block(self, candidate: RuntimeCandidate) -> None:
        if candidate.source == "split":
            self._increment_stat("split_blocked_by_context_count")
        elif candidate.source in {"phrase", "phrase_safe"}:
            self._increment_stat("phrase_blocked_by_context_count")
        else:
            self._increment_stat("unsafe_split_blocked_count")

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
        context_margin: float | None = None,
    ) -> bool:
        return not self._candidate_block_reason(
            source,
            candidate,
            confidence,
            margin,
            prev_word=prev_word,
            next_word=next_word,
            skip_context=skip_context,
            context_margin=context_margin,
        )

    def _candidate_block_reason(
        self,
        source: str,
        candidate: RuntimeCandidate,
        confidence: float,
        margin: float,
        *,
        prev_word: str = "",
        next_word: str = "",
        skip_context: bool = False,
        context_margin: float | None = None,
    ) -> str:
        if candidate.source == "original" or candidate.text == source:
            return "same_as_source"

        norm = normalize_word(source)
        known = self.candidate_generator.is_known(source)
        morph_known = is_morphological_dictionary_word(source)
        is_lower_word = source == source.lower()
        has_latin = any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in source)
        is_upper = source.isupper()
        entity_meta = self._entity_metadata(source, prev_word, next_word)
        low_action_recovery = self._is_safe_low_action_dictionary_recovery(
            source,
            candidate,
            confidence,
            margin,
            known=known,
            morph_known=morph_known,
            has_latin=has_latin,
            is_upper=is_upper,
            entity_meta=entity_meta,
            prev_word=prev_word,
            next_word=next_word,
        )
        if confidence < self.thresholds["action"] or margin < self.thresholds["action_margin"]:
            if not low_action_recovery:
                return "low_action_confidence_or_margin"
            if not skip_context:
                self._increment_stat("low_action_dictionary_recovery_count")

        if candidate.source in DICTIONARY_LIKE_CANDIDATE_SOURCES or candidate.source == "split":
            if bool(entity_meta["entity_context"]):
                self._increment_stat("entity_guard_blocked_count")
                return "entity_context_guard"
            if bool(entity_meta["protected_source"]):
                self._increment_stat("entity_guard_blocked_count")
                return "protected_lexicon_word"
            if candidate.source in DICTIONARY_LIKE_CANDIDATE_SOURCES and self._dictionary_drops_glued_service_token(source, candidate.text):
                return "dictionary_drops_glued_service_token"
            dirty_recovery = self._is_dirty_dictionary_recovery(
                source,
                candidate,
                confidence,
                known=known,
                morph_known=morph_known,
                has_latin=has_latin,
                is_upper=is_upper,
                entity_meta=entity_meta,
                prev_word=prev_word,
                next_word=next_word,
            )
            short_recovery = self._is_safe_short_dictionary_recovery(
                source,
                candidate,
                confidence,
                margin,
                known=known,
                morph_known=morph_known,
                has_latin=has_latin,
                is_upper=is_upper,
                entity_meta=entity_meta,
                prev_word=prev_word,
                next_word=next_word,
                context_margin=context_margin,
            )
            low_score_recovery = self._is_low_score_dictionary_recovery(
                source,
                candidate,
                confidence,
                margin,
                known=known,
                morph_known=morph_known,
                has_latin=has_latin,
                is_upper=is_upper,
                entity_meta=entity_meta,
                prev_word=prev_word,
                next_word=next_word,
                skip_context=skip_context,
                context_margin=context_margin,
            )
            if confidence < self.thresholds["dictionary_action"] and not (
                dirty_recovery or short_recovery or low_action_recovery
            ):
                return "low_dictionary_confidence"
            if candidate.source in DICTIONARY_LIKE_CANDIDATE_SOURCES and candidate.score < self.min_dictionary_score and not low_score_recovery:
                return "low_dictionary_score"
            if (
                candidate.source in DICTIONARY_LIKE_CANDIDATE_SOURCES
                and candidate.score < 1.0
                and not skip_context
                and (context_margin is None or context_margin < 0.50)
                and not low_score_recovery
            ):
                return "low_score_needs_context_margin"
            if known or morph_known:
                return "known_or_dictionary_word"
            if candidate.source in DICTIONARY_LIKE_CANDIDATE_SOURCES and (not is_lower_word or has_latin or is_upper):
                if not (short_recovery or low_action_recovery or self._can_apply_titlecase_dictionary_candidate(
                    source,
                    candidate,
                    confidence,
                    prev_word=prev_word,
                    next_word=next_word,
                    skip_context=skip_context,
                    context_margin=context_margin,
                )):
                    return "protected_word_shape"
            if candidate.source == "split" and (
                has_latin
                or is_upper
                or (
                    not is_lower_word
                    and not self._can_apply_titlecase_split_candidate(
                        source,
                        candidate,
                        prev_word=prev_word,
                        next_word=next_word,
                    )
                )
            ):
                return "protected_word_shape"
            if len(norm) < 5 and not short_recovery:
                return "too_short"
            if candidate.source in DICTIONARY_LIKE_CANDIDATE_SOURCES and not (
                candidate.distance == 1 or (candidate.distance == 2 and len(norm) >= 8)
            ):
                return "bad_dictionary_distance"
            if self.use_morphology_guard and is_same_lemma_inflection(source, candidate.text):
                return "same_lemma_inflection"
            return ""

        if candidate.source in {"rule", "phrase", "phrase_safe"}:
            if confidence < self.thresholds["rule_action"] or has_latin or is_upper:
                return "low_rule_confidence_or_shape"
            if candidate.source == "rule" and self._is_case_rule_candidate(source, candidate.text):
                return self._case_rule_guard_reason(source, candidate.text, next_word)
            if bool(entity_meta["entity_context"]) or bool(entity_meta["protected_source"]):
                self._increment_stat("entity_guard_blocked_count")
                return "entity_context_guard" if bool(entity_meta["entity_context"]) else "protected_lexicon_word"
            if (
                candidate.source in {"phrase", "phrase_safe"}
                and self._collapses_next_word(source, candidate.text, next_word)
                and (
                    int(getattr(candidate, "span_len", 1)) <= 1
                    or not self._is_allowed_phrase_joining(source, candidate.text, next_word)
                )
            ):
                return "phrase_context_guard"
            if self._looks_like_name_context(source, prev_word, next_word):
                return "name_context_guard"
            if self._blocked_tsya_context(source, candidate.text, prev_word):
                return "tsya_context_guard"
            return ""

        return "unsupported_candidate_source"

    @staticmethod
    def _is_case_rule_candidate(source: str, candidate: str) -> bool:
        return str(source or "") != str(candidate or "") and normalize_word(source) == normalize_word(candidate)

    @staticmethod
    def _case_rule_guard_reason(source: str, candidate: str, next_word: str = "") -> str:
        src = normalize_word(source)
        tgt = str(candidate or "")
        next_norm = normalize_word(next_word)
        if src == "президент" and tgt == "Президент" and next_norm not in OFFICIAL_PRESIDENT_NEXT:
            return "president_case_context_guard"
        return ""

    @staticmethod
    def _is_allowed_phrase_joining(source: str, candidate: str, next_word: str = "") -> bool:
        key = (normalize_word(source), normalize_word(next_word), normalize_word(candidate))
        return key in ALLOWED_PHRASE_JOININGS

    def _is_dirty_dictionary_recovery(
        self,
        source: str,
        candidate: RuntimeCandidate,
        confidence: float,
        *,
        known: bool,
        morph_known: bool,
        has_latin: bool,
        is_upper: bool,
        entity_meta: dict[str, object],
        prev_word: str = "",
        next_word: str = "",
    ) -> bool:
        if candidate.source not in DICTIONARY_LIKE_CANDIDATE_SOURCES:
            return False
        if self._candidate_rank(candidate) > 1:
            return False
        if candidate.distance != 1 or candidate.score < 1.0:
            return False
        if confidence < self.thresholds["action"]:
            return False
        return self._is_dirty_oov_source_shape(
            source,
            candidate,
            known=known,
            morph_known=morph_known,
            has_latin=has_latin,
            is_upper=is_upper,
            entity_meta=entity_meta,
            prev_word=prev_word,
            next_word=next_word,
            allow_short=False,
            allow_titlecase=False,
        )

    def _is_low_score_dictionary_recovery(
        self,
        source: str,
        candidate: RuntimeCandidate,
        confidence: float,
        margin: float,
        *,
        known: bool,
        morph_known: bool,
        has_latin: bool,
        is_upper: bool,
        entity_meta: dict[str, object],
        prev_word: str = "",
        next_word: str = "",
        skip_context: bool = False,
        context_margin: float | None = None,
    ) -> bool:
        if candidate.source not in DICTIONARY_LIKE_CANDIDATE_SOURCES:
            return False
        if self._candidate_rank(candidate) > 1:
            return False
        if candidate.score < -1.5 or confidence < 0.95 or margin < self.thresholds["action_margin"]:
            return False
        if not (candidate.distance == 1 or (candidate.distance == 2 and len(normalize_word(source)) >= 8)):
            return False
        if not skip_context and (context_margin is None or context_margin < 0.75):
            return False
        return self._is_dirty_oov_source_shape(
            source,
            candidate,
            known=known,
            morph_known=morph_known,
            has_latin=has_latin,
            is_upper=is_upper,
            entity_meta=entity_meta,
            prev_word=prev_word,
            next_word=next_word,
            allow_short=False,
            allow_titlecase=False,
        )

    def _is_safe_low_action_dictionary_recovery(
        self,
        source: str,
        candidate: RuntimeCandidate,
        confidence: float,
        margin: float,
        *,
        known: bool,
        morph_known: bool,
        has_latin: bool,
        is_upper: bool,
        entity_meta: dict[str, object],
        prev_word: str = "",
        next_word: str = "",
    ) -> bool:
        if candidate.source not in DICTIONARY_LIKE_CANDIDATE_SOURCES:
            return False
        if self._candidate_rank(candidate) != 0:
            return False
        norm = normalize_word(source)
        candidate_norm = normalize_word(candidate.text)
        if not norm or not candidate_norm or norm == candidate_norm:
            return False
        if len(norm) < 5 or " " in str(candidate.text or ""):
            return False
        if candidate.score < 1.2:
            return False
        if not (candidate.distance == 1 or (candidate.distance == 2 and len(norm) >= 8)):
            return False
        if not self._is_safe_low_action_edit_shape(norm, candidate_norm):
            return False

        standard_signal = confidence >= 0.55 and margin >= 0.10
        extra_letter_signal = (
            self._has_single_extra_letter(norm, candidate_norm)
            and confidence >= 0.45
            and margin >= 0.05
        )
        if not (standard_signal or extra_letter_signal):
            return False

        if self.use_morphology_guard and is_same_lemma_inflection(source, candidate.text):
            return False

        return self._is_dirty_oov_source_shape(
            source,
            candidate,
            known=known,
            morph_known=morph_known,
            has_latin=has_latin,
            is_upper=is_upper,
            entity_meta=entity_meta,
            prev_word=prev_word,
            next_word=next_word,
            allow_short=False,
            allow_titlecase=True,
        )

    @staticmethod
    def _is_safe_low_action_edit_shape(source_norm: str, candidate_norm: str) -> bool:
        if not source_norm or not candidate_norm:
            return False
        if source_norm[0] != candidate_norm[0] or source_norm[-1] != candidate_norm[-1]:
            return False
        if HybridCorrector._has_single_extra_letter(source_norm, candidate_norm):
            return True
        if HybridCorrector._has_single_extra_letter(candidate_norm, source_norm):
            return True
        if len(source_norm) != len(candidate_norm):
            return False

        diff_indices = [
            index
            for index, (source_char, candidate_char) in enumerate(zip(source_norm, candidate_norm))
            if source_char != candidate_char
        ]
        if len(diff_indices) == 1:
            return True
        if len(diff_indices) == 2:
            first, second = diff_indices
            return (
                second == first + 1
                and source_norm[first] == candidate_norm[second]
                and source_norm[second] == candidate_norm[first]
            )
        return False

    @staticmethod
    def _has_single_extra_letter(longer: str, shorter: str) -> bool:
        if len(longer) != len(shorter) + 1:
            return False
        longer_index = 0
        shorter_index = 0
        skipped = False
        while longer_index < len(longer) and shorter_index < len(shorter):
            if longer[longer_index] == shorter[shorter_index]:
                longer_index += 1
                shorter_index += 1
                continue
            if skipped:
                return False
            skipped = True
            longer_index += 1
        return True

    def _is_safe_short_dictionary_recovery(
        self,
        source: str,
        candidate: RuntimeCandidate,
        confidence: float,
        margin: float,
        *,
        known: bool,
        morph_known: bool,
        has_latin: bool,
        is_upper: bool,
        entity_meta: dict[str, object],
        prev_word: str = "",
        next_word: str = "",
        context_margin: float | None = None,
    ) -> bool:
        norm = normalize_word(source)
        if candidate.source not in DICTIONARY_LIKE_CANDIDATE_SOURCES:
            return False
        if not (3 <= len(norm) <= 4):
            return False
        if self._candidate_rank(candidate) > 1:
            return False
        if candidate.distance != 1 or candidate.score < 1.0:
            return False
        if confidence < 0.95 or margin < 0.50:
            return False
        if context_margin is not None and context_margin < 0.50:
            return False
        return self._is_dirty_oov_source_shape(
            source,
            candidate,
            known=known,
            morph_known=morph_known,
            has_latin=has_latin,
            is_upper=is_upper,
            entity_meta=entity_meta,
            prev_word=prev_word,
            next_word=next_word,
            allow_short=True,
            allow_titlecase=True,
        )

    def _is_dirty_oov_source_shape(
        self,
        source: str,
        candidate: RuntimeCandidate,
        *,
        known: bool,
        morph_known: bool,
        has_latin: bool,
        is_upper: bool,
        entity_meta: dict[str, object],
        prev_word: str = "",
        next_word: str = "",
        allow_short: bool = False,
        allow_titlecase: bool = False,
    ) -> bool:
        norm = normalize_word(source)
        if known or morph_known:
            return False
        if bool(entity_meta["protected_source"]) or bool(entity_meta["entity_context"]):
            return False
        if " " in candidate.text or has_latin or is_upper:
            return False
        if len(norm) < 5 and not allow_short:
            return False
        if source == source.lower():
            return True
        if not allow_titlecase:
            return False
        if not source[:1].isupper() or source.isupper():
            return False
        if prev_word:
            return False
        if self._looks_like_name_context(source, prev_word, next_word):
            return False
        if not candidate.text[:1].isupper() or candidate.text.isupper():
            return False
        return True

    @staticmethod
    def _candidate_rank(candidate: RuntimeCandidate) -> int:
        return int(candidate.rank) if int(candidate.rank) >= 0 else 0

    def _dictionary_drops_glued_service_token(self, source: str, candidate: str) -> bool:
        source_norm = normalize_word(source)
        candidate_norm = normalize_word(candidate)
        if not source_norm or not candidate_norm or " " in str(candidate):
            return False
        if len(source_norm) >= 2 and source_norm[0] == source_norm[1] and candidate_norm == source_norm[1:]:
            return False
        if len(candidate_norm) < 5 or len(source_norm) <= len(candidate_norm):
            return False
        if source_norm.startswith(candidate_norm):
            dropped = source_norm[len(candidate_norm) :]
            return self._is_real_glued_service_pair(candidate_norm, dropped)
        elif source_norm.endswith(candidate_norm):
            dropped = source_norm[: -len(candidate_norm)]
            return self._is_real_glued_service_pair(dropped, candidate_norm)
        return False

    def _is_real_glued_service_pair(self, left: str, right: str) -> bool:
        left_norm = normalize_word(left)
        right_norm = normalize_word(right)
        if left_norm in GLUED_SERVICE_TOKENS and len(left_norm) >= 2 and len(right_norm) >= 3:
            return self.candidate_generator.is_known(right_norm)
        if right_norm in GLUED_SERVICE_TOKENS and len(right_norm) >= 2 and len(left_norm) >= 5:
            return self.candidate_generator.is_known(left_norm)
        return False

    @staticmethod
    def _can_apply_titlecase_split_candidate(
        source: str,
        candidate: RuntimeCandidate,
        *,
        prev_word: str = "",
        next_word: str = "",
    ) -> bool:
        if candidate.source != "split" or prev_word:
            return False
        source_s = str(source or "")
        candidate_s = str(candidate.text or "")
        if not source_s[:1].isupper() or source_s.isupper() or source_s[1:] != source_s[1:].lower():
            return False
        if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in source_s + candidate_s):
            return False
        parts = candidate_s.split()
        if len(parts) != 2:
            return False
        if not parts[0][:1].isupper() or parts[0].isupper():
            return False
        if parts[1] != parts[1].lower():
            return False
        return True

    def _can_apply_titlecase_dictionary_candidate(
        self,
        source: str,
        candidate: RuntimeCandidate,
        confidence: float,
        *,
        prev_word: str = "",
        next_word: str = "",
        skip_context: bool = False,
        context_margin: float | None = None,
    ) -> bool:
        if candidate.source not in DICTIONARY_LIKE_CANDIDATE_SOURCES:
            return False
        if confidence < 0.995:
            return False
        if not skip_context and (context_margin is None or context_margin < 0.75):
            return False
        if self._looks_like_name_context(source, prev_word, next_word):
            return False
        if self._entity_metadata(source, prev_word, next_word)["entity_context"]:
            return False
        if " " in candidate.text:
            return False
        if not source[:1].isupper() or source.isupper():
            return False
        if not candidate.text[:1].isupper() or candidate.text.isupper():
            return False
        return True

    @staticmethod
    def _collapses_next_word(source: str, candidate: str, next_word: str = "") -> bool:
        if not next_word:
            return False
        return normalize_word(candidate) == normalize_word(source + next_word)

    @staticmethod
    def _should_log_word_decision(candidates: Sequence[RuntimeCandidate], action: str) -> bool:
        if action != ACTION_KEEP:
            return True
        return any(
            candidate.source != "original" and normalize_word(candidate.text) != ""
            for candidate in candidates
        )

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
        if predicted_punct not in TRAINABLE_PUNCT_LABELS:
            return False, "unsupported_punctuation"
        if protected_gap:
            return False, "protected_gap"
        if not is_trainable_punctuation_gap(gap, word=word, next_word=next_word, is_final=is_final):
            return False, "structural_gap"
        if is_final and predicted_punct in {",", ":", ";", "—", ":—", ":«", ':"', "«", '"', "("}:
            return False, "final_non_terminal_punctuation"
        if not is_final:
            if predicted_punct in {".", "?", "!"} and not (source_punct == "." and predicted_punct == ","):
                return False, "internal_sentence_end"
        if {source_punct, predicted_punct} == {":", ";"}:
            return False, "structural_punctuation_swap_guard"

        comma_delete = source_punct == "," and predicted_punct == ""
        safe_comma_delete_recovery_reason = ""
        if comma_delete:
            comma_guard_reason = self._comma_delete_guard_reason(word, next_word)
            if comma_guard_reason:
                self._increment_stat("comma_delete_blocked_count")
                return False, comma_guard_reason
            safe_comma_delete_recovery_reason = self._safe_comma_delete_recovery_reason(
                word,
                next_word,
                is_final=is_final,
                confidence=confidence,
                margin=margin,
            )

        comma_insert = source_punct == "" and predicted_punct == ","
        if comma_insert:
            comma_insert_guard = self._comma_insert_guard_reason(word, next_word)
            if comma_insert_guard:
                self._increment_stat("comma_insert_blocked_count")
                return False, comma_insert_guard

        safe_final_period_recovery = self._is_safe_final_period_recovery(
            source_punct,
            predicted_punct,
            is_final=is_final,
            confidence=confidence,
            margin=margin,
        )

        if comma_delete:
            threshold = self.thresholds["comma_delete"]
            margin_threshold = self.thresholds["comma_delete_margin"]
        else:
            threshold = self.thresholds["punct_delete"] if predicted_punct == "" else self.thresholds["punct"]
            margin_threshold = self.thresholds["punct_delete_margin"] if predicted_punct == "" else self.thresholds["punct_margin"]
        if confidence < threshold or margin < margin_threshold:
            if comma_delete:
                if safe_comma_delete_recovery_reason:
                    self._increment_stat("safe_comma_delete_recovery_count")
                    if safe_comma_delete_recovery_reason != "safe_comma_delete_recovery":
                        self._increment_stat(f"{safe_comma_delete_recovery_reason}_count")
                    return True, safe_comma_delete_recovery_reason
                self._increment_stat("comma_delete_blocked_count")
                return False, "low_comma_delete_confidence_or_margin"
            if safe_final_period_recovery:
                self._increment_stat("safe_final_period_recovery_count")
                return True, "safe_final_period_recovery"
            return False, "low_confidence_or_margin"

        if is_final and source_punct in {".", "!", "?"} and predicted_punct == "":
            return False, "final_punctuation_delete"
        if is_final and predicted_punct and not self._is_final_punctuation_label(predicted_punct):
            return False, "final_non_terminal_punctuation"
        if source_punct and predicted_punct == "" and self._is_protected_next_token(next_word):
            return False, "delete_before_protected"
        return True, "applied"

    def _safe_comma_delete_recovery_reason(
        self,
        word: str,
        next_word: str,
        *,
        is_final: bool,
        confidence: float,
        margin: float,
    ) -> str:
        if is_final:
            return ""
        source = str(word or "").strip()
        if source == "О":
            return ""
        word_norm = normalize_word(word)
        next_norm = normalize_word(next_word)
        if self._is_protected_next_token(next_word):
            return ""
        if self._is_safe_date_comma_delete_recovery(word_norm, next_norm, confidence=confidence, margin=margin):
            return "safe_date_comma_delete_recovery"
        if word_norm not in SAFE_COMMA_DELETE_SERVICE_WORDS or not next_norm:
            return ""
        if confidence >= 0.985 and margin >= 0.95:
            return "safe_comma_delete_recovery"
        if source == source.lower() and confidence >= 0.95 and margin >= 0.90:
            return "safe_service_comma_delete_recovery"
        return ""

    @staticmethod
    def _is_safe_date_comma_delete_recovery(
        word_norm: str,
        next_norm: str,
        *,
        confidence: float,
        margin: float,
    ) -> bool:
        if not word_norm.isdigit() or next_norm not in SAFE_COMMA_DELETE_MONTHS:
            return False
        day = int(word_norm)
        return 1 <= day <= 31 and confidence >= 0.86 and margin >= 0.70

    @staticmethod
    def _is_safe_final_period_recovery(
        source_punct: str,
        predicted_punct: str,
        *,
        is_final: bool,
        confidence: float,
        margin: float,
    ) -> bool:
        if not is_final or predicted_punct != ".":
            return False
        if source_punct in {"", ","}:
            return confidence >= 0.82 and margin >= 0.70
        return False

    @staticmethod
    def _is_final_punctuation_label(label: str) -> bool:
        return any(ch in str(label or "") for ch in ".?!")

    @staticmethod
    def _comma_delete_guard_reason(word: str, next_word: str) -> str:
        next_norm = normalize_word(next_word)
        if not next_norm:
            return ""
        if next_norm.startswith("котор") or next_norm in {"что", "чтобы"}:
            return "comma_delete_clause_guard"
        if next_norm in {"пожалуй", "наверное", "конечно", "возможно", "вероятно", "однако", "например"}:
            return "comma_delete_introductory_guard"
        if HybridCorrector._looks_like_participle_or_descriptor(next_norm):
            return "comma_delete_participle_guard"
        if str(word or "")[:1].isupper() and str(next_word or "")[:1].isupper():
            return "comma_delete_titlecase_list_guard"
        return ""

    @staticmethod
    def _comma_insert_guard_reason(word: str, next_word: str) -> str:
        word_norm = normalize_word(word)
        next_norm = normalize_word(next_word)
        if not next_norm:
            return ""
        if word_norm == "того" and next_norm == "чтобы":
            return "comma_insert_fixed_phrase_guard"
        if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in str(word or "")):
            return "comma_insert_protected_token_guard"
        if any(ch.isdigit() for ch in str(next_word or "")):
            return "comma_insert_protected_token_guard"
        if next_norm.startswith("котор") or next_norm in {"что", "чтобы"}:
            return ""
        if str(word or "")[:1].isupper() and HybridCorrector._looks_like_finite_verb(next_norm):
            return "comma_insert_subject_predicate_guard"
        if word_norm in {
            "сообщалось",
            "сообщается",
            "сообщили",
            "сообщил",
            "сообщила",
            "сообщают",
            "сообщает",
        } and next_norm in {"о", "об", "обо", "ранее"}:
            return "comma_insert_after_reporting_verb_guard"
        if word_norm in {"словам", "данным", "информации", "источника"}:
            return "comma_insert_source_phrase_guard"
        if word_norm in {"того", "тому"} and next_norm in {"или", "чтобы", "самого", "самой", "самое", "самые", "стоит", "времени"}:
            return "comma_insert_fixed_phrase_guard"
        if next_norm == "же":
            return "comma_insert_fixed_phrase_guard"
        if (word_norm, next_norm) in {
            ("башни", "не"),
            ("подвал", "не"),
            ("словом", "не"),
            ("прессы", "не"),
            ("тем", "не"),
            ("античности", "геммы"),
            ("очередь", "министр"),
            ("образом", "биржа"),
            ("мотивация", "была"),
            ("почерковеды", "работали"),
            ("значит", "возраст"),
            ("работаешь", "сейчас"),
            ("ответьте", "мне"),
            ("выяснялись", "отношения"),
            ("контекст", "достаточно"),
            ("средство", "освоения"),
            ("руставели", "их"),
        }:
            return "comma_insert_fixed_phrase_guard"
        if word_norm in {
            "утверждают",
            "утверждалось",
            "заявил",
            "заявила",
            "заявили",
            "заявляет",
            "заявляют",
            "добавил",
            "добавила",
            "добавили",
            "подтвердил",
            "подтвердила",
            "подтвердили",
            "ответил",
            "ответила",
            "ответили",
            "выразил",
            "выразила",
            "выразили",
            "подчеркнул",
            "подчеркнула",
            "подчеркнули",
            "спрашивает",
            "сказал",
            "сказала",
            "сказали",
            "понимают",
            "слушают",
            "сожалею",
        }:
            return "comma_insert_verb_argument_guard"
        if word_norm in {"заявил", "заявила", "заявили", "заявляет", "заявляют"} and next_norm in {
            "сегодня",
            "вчера",
            "ранее",
            "тогда",
        }:
            return "comma_insert_reporting_tail_guard"
        if word_norm in {"добавил", "добавила", "добавили"} and next_norm in {"он", "она", "они"}:
            return "comma_insert_reporting_tail_guard"
        if word_norm in {"напомнил", "напомнила", "напомнили"} and next_norm in {"ей", "ему", "им", "о", "об", "обо"}:
            return "comma_insert_verb_argument_guard"
        if word_norm in {"рассказал", "рассказала", "рассказали"} and next_norm in {"женам", "жене", "мужу", "детям", "ему", "ей", "им"}:
            return "comma_insert_verb_argument_guard"
        if word_norm in {"спросил", "спросила", "спросили"} and next_norm in {"у"}:
            return "comma_insert_verb_argument_guard"
        if word_norm in {"договорился", "договорилась", "договорились"} and next_norm in {"о", "об", "обо"}:
            return "comma_insert_verb_argument_guard"
        if next_norm == "также":
            return "comma_insert_before_takzhe_guard"
        if next_norm in {"на", "с", "со", "в", "во", "по", "при", "для", "о", "об", "обо", "у", "к", "ко", "от", "до", "из", "за"}:
            return "comma_insert_preposition_phrase_guard"
        if HybridCorrector._looks_like_adjective_like(next_norm):
            return "comma_insert_adjective_phrase_guard"
        if len(next_norm) >= 5 and next_norm.endswith(("ть", "ти")):
            return "comma_insert_infinitive_argument_guard"
        if next_norm.startswith("друг"):
            return "comma_insert_descriptor_guard"
        if HybridCorrector._looks_like_adjective_like(word_norm) and len(next_norm) >= 4:
            return "comma_insert_adjective_phrase_guard"
        if word_norm.endswith(("ен", "на", "но", "ны", "ый", "ий", "ая", "ое", "ые")) and len(next_norm) >= 4:
            return "comma_insert_descriptor_guard"
        return ""

    @staticmethod
    def _looks_like_adjective_like(norm: str) -> bool:
        endings = (
            "ых",
            "их",
            "ого",
            "его",
            "ому",
            "ему",
            "ым",
            "им",
            "ыми",
            "ими",
            "ой",
            "ую",
            "юю",
        )
        return len(norm) >= 5 and norm.endswith(endings)

    @staticmethod
    def _looks_like_finite_verb(norm: str) -> bool:
        endings = (
            "л",
            "ла",
            "ло",
            "ли",
            "ет",
            "ют",
            "ит",
            "ят",
            "ется",
            "ются",
            "ем",
            "им",
            "ете",
            "ите",
        )
        return len(norm) >= 5 and norm.endswith(endings)

    @staticmethod
    def _looks_like_participle_or_descriptor(norm: str) -> bool:
        endings = (
            "вший",
            "вшая",
            "вшее",
            "вшие",
            "нный",
            "нная",
            "нное",
            "нные",
            "енный",
            "енная",
            "енное",
            "енные",
            "емый",
            "емая",
            "емое",
            "емые",
            "ющий",
            "ющая",
            "ющее",
            "ющие",
            "ший",
            "шая",
            "шее",
            "шие",
            "тый",
            "тая",
            "тое",
            "тые",
        )
        return norm.endswith(endings)

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
