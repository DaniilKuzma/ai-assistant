from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config.load_config import load_config
from src.runtime.deterministic_rules import DeterministicRuleEngine
from src.runtime.edit_realizer import (
    apply_gap_labels,
    apply_runtime_edits,
    apply_token_edit_labels,
    gap_edit_type_for_label,
    token_edit_type_for_label,
)
from src.runtime.explanations import attach_explanations
from src.runtime.neural_backend import DirectNeuralBackend
from src.runtime.scope_guard import ScopeGuard
from src.runtime.thresholds import RuntimeThresholds
from src.runtime.tokenization import tokenize_runtime_words
from src.schema.edits import CorrectionResult, RuntimeEdit


class Corrector:
    def __init__(
        self,
        deterministic_engine: DeterministicRuleEngine | None = None,
        neural_backend: Any | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = dict(config or {})
        self.runtime_config = self.config.get("runtime", {}) if isinstance(self.config.get("runtime", {}), Mapping) else {}
        self.max_passes = max(1, int(self.runtime_config.get("max_passes", 2)))
        self.deterministic_engine = deterministic_engine or DeterministicRuleEngine.from_config(self.config)
        self.neural_backend = neural_backend
        self.thresholds = RuntimeThresholds.from_config(self.config)
        self.scope_guard = ScopeGuard()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | str | Path) -> "Corrector":
        if isinstance(config, (str, Path)):
            config = load_config(config)
        runtime = config.get("runtime", {}) if isinstance(config, Mapping) else {}
        backend = None
        if bool(runtime.get("neural_token_edits", True) or runtime.get("neural_punctuation", True)):
            try:
                backend = DirectNeuralBackend.from_config(config)
            except Exception:
                backend = None
        return cls(
            deterministic_engine=DeterministicRuleEngine.from_config(config),
            neural_backend=backend,
            config=config,
        )

    def correct(self, text: str) -> CorrectionResult:
        if not text:
            return CorrectionResult(source_text=text, corrected_text=text, edits=[])

        source_text = text
        current = text
        edits: list[RuntimeEdit] = []

        current, deterministic_edits = self._apply_deterministic(current)
        edits.extend(deterministic_edits)

        if self.neural_backend is not None:
            current, token_edits = self._apply_neural_token_edits(current)
            edits.extend(token_edits)
            current, gap_edits = self._apply_neural_gap_edits(current)
            edits.extend(gap_edits)

        ok, reasons = self.scope_guard.validate_result(source_text, current, edits)
        if not ok:
            return CorrectionResult(
                source_text=source_text,
                corrected_text=source_text,
                edits=[],
                metadata={"scope_guard_rejections": reasons},
            )

        attach_explanations(edits)
        metadata = {"scope_guard_warnings": reasons} if reasons else {}
        return CorrectionResult(source_text=source_text, corrected_text=current, edits=edits, metadata=metadata)

    def _apply_deterministic(self, text: str) -> tuple[str, list[RuntimeEdit]]:
        current = text
        accepted: list[RuntimeEdit] = []
        for _ in range(self.max_passes):
            proposed = [
                edit
                for edit in self.deterministic_engine.propose_edits(current)
                if self.scope_guard.validate_edit(current, edit)
            ]
            if not proposed:
                break
            updated = apply_runtime_edits(current, proposed)
            accepted.extend(proposed)
            if updated == current:
                break
            current = updated
        return current, accepted

    def _apply_neural_token_edits(self, text: str) -> tuple[str, list[RuntimeEdit]]:
        if not bool(self.runtime_config.get("neural_token_edits", True)):
            return text, []
        prediction = self.neural_backend.predict(text)
        tokens = tokenize_runtime_words(text)
        labels = _list_attr(prediction, "token_labels", len(tokens), "KEEP")
        confidences = _float_list_attr(prediction, "token_confidences", len(tokens), 0.0)
        margins = _float_list_attr(prediction, "token_margins", len(tokens), 1.0)
        rule_ids = _list_attr(prediction, "rule_ids", len(tokens), "none")
        accepted_labels = self._gate_token_labels(text, tokens, labels, confidences, margins, rule_ids)
        return apply_token_edit_labels(text, tokens, accepted_labels, confidences, threshold=0.0, rule_ids=rule_ids)

    def _apply_neural_gap_edits(self, text: str) -> tuple[str, list[RuntimeEdit]]:
        if not bool(self.runtime_config.get("neural_punctuation", True)):
            return text, []
        prediction = self.neural_backend.predict(text)
        tokens = tokenize_runtime_words(text)
        labels = _list_attr(prediction, "gap_labels", len(tokens), "NONE")
        confidences = _float_list_attr(prediction, "gap_confidences", len(tokens), 0.0)
        margins = _float_list_attr(prediction, "gap_margins", len(tokens), 1.0)
        rule_ids = _list_attr(prediction, "rule_ids", len(tokens), "none")
        accepted_labels = self._gate_gap_labels(text, tokens, labels, confidences, margins, rule_ids)
        return apply_gap_labels(text, tokens, accepted_labels, confidences, threshold=0.0, rule_ids=rule_ids)

    def _gate_token_labels(
        self,
        text: str,
        tokens: Sequence[Any],
        labels: Sequence[str],
        confidences: Sequence[float],
        margins: Sequence[float],
        rule_ids: Sequence[str],
    ) -> list[str]:
        accepted = ["KEEP"] * len(tokens)
        consumed: set[int] = set()
        for index, label in enumerate(labels[: len(tokens)]):
            if index in consumed or label in {"KEEP", "SKIP_MERGED", "DICT_REPLACE"}:
                continue
            rule_id = _rule_id(rule_ids, index)
            edit_type = token_edit_type_for_label(label)
            if not self.thresholds.should_apply(float(confidences[index]), float(margins[index]), rule_id, edit_type):
                continue
            candidate_labels = ["KEEP"] * len(tokens)
            candidate_labels[index] = label
            if _label_consumes_next(label) and index + 1 < len(tokens):
                candidate_labels[index + 1] = "SKIP_MERGED"
            _candidate_text, candidate_edits = apply_token_edit_labels(
                text,
                tokens,
                candidate_labels,
                [1.0] * len(tokens),
                threshold=0.0,
                rule_ids=rule_ids,
            )
            if candidate_edits and all(self.scope_guard.validate_edit(text, edit) for edit in candidate_edits):
                accepted[index] = label
                consumed.add(index)
                if _label_consumes_next(label) and index + 1 < len(tokens):
                    accepted[index + 1] = "SKIP_MERGED"
                    consumed.add(index + 1)
        return accepted

    def _gate_gap_labels(
        self,
        text: str,
        tokens: Sequence[Any],
        labels: Sequence[str],
        confidences: Sequence[float],
        margins: Sequence[float],
        rule_ids: Sequence[str],
    ) -> list[str]:
        accepted = ["NONE"] * len(tokens)
        for index, label in enumerate(labels[: len(tokens)]):
            if label == "NONE":
                continue
            rule_id = _rule_id(rule_ids, index)
            edit_type = gap_edit_type_for_label(label)
            if not self.thresholds.should_apply(float(confidences[index]), float(margins[index]), rule_id, edit_type):
                continue
            candidate_labels = ["NONE"] * len(tokens)
            candidate_labels[index] = label
            _candidate_text, candidate_edits = apply_gap_labels(
                text,
                tokens,
                candidate_labels,
                [1.0] * len(tokens),
                threshold=0.0,
                rule_ids=rule_ids,
            )
            if candidate_edits and all(self.scope_guard.validate_edit(text, edit) for edit in candidate_edits):
                accepted[index] = label
        return accepted


def _list_attr(prediction: Any, name: str, length: int, default: str) -> list[str]:
    values = list(getattr(prediction, name, []) or [])
    if len(values) < length:
        values.extend([default] * (length - len(values)))
    return [str(value) for value in values[:length]]


def _float_list_attr(prediction: Any, name: str, length: int, default: float) -> list[float]:
    values = list(getattr(prediction, name, []) or [])
    if len(values) < length:
        values.extend([default] * (length - len(values)))
    return [float(value) for value in values[:length]]


def _rule_id(rule_ids: Sequence[str], index: int) -> str:
    if index < len(rule_ids) and rule_ids[index] not in {"", "none"}:
        return str(rule_ids[index])
    return "none"


def _label_consumes_next(label: str) -> bool:
    return label in {
        "MERGE_TAK_ZHE_TO_TAKZHE",
        "MERGE_TO_ZHE_TO_TOZHE",
        "MERGE_ZA_TO_TO_ZATO",
        "HYPHENATE_PARTICLE_TO",
        "HYPHENATE_PARTICLE_LIBO",
        "HYPHENATE_PARTICLE_NIBUD",
        "HYPHENATE_KOE",
        "HYPHENATE_PO_ADVERB",
    }


__all__ = ["CorrectionResult", "Corrector", "RuntimeEdit"]
