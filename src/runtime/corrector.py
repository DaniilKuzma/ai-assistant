from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config.load_config import load_config
from src.runtime.deterministic_rules import DeterministicRuleEngine
from src.runtime.edit_realizer import (
    MAX_LEXICON_SPAN_TOKENS,
    apply_boundary_and_token_edit_labels,
    apply_boundary_labels,
    apply_gap_labels,
    apply_runtime_edits,
    apply_token_edit_labels,
    gap_edit_type_for_label,
    boundary_edit_type_for_label,
    token_edit_type_for_label,
)
from src.runtime.explanations import attach_explanations
from src.runtime.neural_backend import DirectNeuralBackend
from src.runtime.orthographic_lexicon import CorrectionEntry, OrthographicCorrectionLexicon
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
        self.orthographic_lexicon = OrthographicCorrectionLexicon.from_config(self.config)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str | Path,
        *,
        strict_neural: bool = False,
        allow_fallback: bool = False,
    ) -> "Corrector":
        if isinstance(config, (str, Path)):
            config = load_config(config)
        runtime = config.get("runtime", {}) if isinstance(config, Mapping) else {}
        backend = None
        if bool(runtime.get("neural_token_edits", True) or runtime.get("neural_punctuation", True)):
            try:
                backend = DirectNeuralBackend.from_config(config)
            except Exception as exc:
                if strict_neural and not allow_fallback:
                    raise RuntimeError(
                        "Neural runtime is enabled, but DirectNeuralBackend could not be loaded. "
                        "Evaluation refuses to use deterministic fallback unless --allow-fallback is set."
                    ) from exc
                backend = None
        return cls(
            deterministic_engine=DeterministicRuleEngine.from_config(config),
            neural_backend=backend,
            config=config,
        )

    def correct(self, text: str) -> CorrectionResult:
        runtime_metadata = self._runtime_metadata()
        if not text:
            return CorrectionResult(source_text=text, corrected_text=text, edits=[], metadata=runtime_metadata)

        source_text = text
        current = text
        edits: list[RuntimeEdit] = []

        current, deterministic_edits = self._apply_deterministic(current)
        edits.extend(deterministic_edits)

        current, lexicon_edits = self._apply_deterministic_lexicon(current)
        edits.extend(lexicon_edits)

        if self.neural_backend is not None:
            current, token_edits = self._apply_neural_token_edits(current)
            edits.extend(token_edits)
            current, gap_edits = self._apply_neural_gap_edits(current)
            edits.extend(gap_edits)

        ok, reasons = self.scope_guard.validate_result(source_text, current, edits)
        if not ok:
            metadata = dict(runtime_metadata)
            metadata["scope_guard_rejections"] = reasons
            return CorrectionResult(
                source_text=source_text,
                corrected_text=source_text,
                edits=[],
                metadata=metadata,
            )

        attach_explanations(edits)
        metadata = dict(runtime_metadata)
        if reasons:
            metadata["scope_guard_warnings"] = reasons
        return CorrectionResult(source_text=source_text, corrected_text=current, edits=edits, metadata=metadata)

    def _runtime_metadata(self) -> dict[str, Any]:
        backend = self.neural_backend
        return {
            "backend_kind": "direct_neural" if backend is not None else "deterministic_fallback",
            "model_loaded": backend is not None,
            "adapter_path": str(getattr(backend, "adapter_path", "") or ""),
            "heads_path": str(getattr(backend, "heads_path", "") or ""),
            "selected_epoch": getattr(backend, "selected_epoch", None),
        }

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

    def _apply_deterministic_lexicon(self, text: str) -> tuple[str, list[RuntimeEdit]]:
        if not bool(self.runtime_config.get("deterministic_lexicon", False)):
            return text, []

        current = text
        accepted: list[RuntimeEdit] = []
        for _ in range(self.max_passes):
            proposed = [
                edit
                for edit in self._trusted_lexicon_edits(current)
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

    def _trusted_lexicon_edits(self, text: str) -> list[RuntimeEdit]:
        tokens = tokenize_runtime_words(text)
        edits: list[RuntimeEdit] = []
        consumed: set[int] = set()
        for index, token in enumerate(tokens):
            if index in consumed:
                continue
            edit = self._trusted_lexicon_edit_at(text, tokens, index)
            if edit is None:
                continue
            edits.append(edit)
            for position in range(index, len(tokens)):
                if tokens[position].start >= edit.start and tokens[position].end <= edit.end:
                    consumed.add(position)
        return edits

    def _trusted_lexicon_edit_at(self, text: str, tokens: Sequence[Any], index: int) -> RuntimeEdit | None:
        max_stop = min(len(tokens), index + MAX_LEXICON_SPAN_TOKENS)
        for stop in range(max_stop, index, -1):
            start = int(tokens[index].start)
            end = int(tokens[stop - 1].end)
            source = text[start:end]
            entries = [
                entry
                for entry in self.orthographic_lexicon.lookup(source)
                if _trusted_deterministic_entry(entry, text)
            ]
            if not entries:
                continue
            targets = {entry.target for entry in entries}
            if len(targets) != 1:
                return None
            entry = entries[0]
            replacement = _match_case(source, next(iter(targets)))
            if not replacement or replacement == source:
                return None
            return RuntimeEdit(
                start=start,
                end=end,
                source=source,
                replacement=replacement,
                edit_type=_edit_type_for_lexicon_operation(entry.operation, source=source, replacement=replacement),
                rule_id=entry.rule_id,
                confidence=entry.confidence,
            )
        return None

    def _apply_neural_token_edits(self, text: str) -> tuple[str, list[RuntimeEdit]]:
        if not bool(self.runtime_config.get("neural_token_edits", True)):
            return text, []
        prediction = self.neural_backend.predict(text)
        tokens = tokenize_runtime_words(text)
        labels = _list_attr(prediction, "token_labels", len(tokens), "KEEP")
        confidences = _float_list_attr(prediction, "token_confidences", len(tokens), 0.0)
        margins = _float_list_attr(prediction, "token_margins", len(tokens), 1.0)
        boundary_before_labels = _list_attr(prediction, "boundary_before_labels", len(tokens), "NONE")
        boundary_before_confidences = _float_list_attr(prediction, "boundary_before_confidences", len(tokens), 0.0)
        boundary_before_margins = _float_list_attr(prediction, "boundary_before_margins", len(tokens), 1.0)
        boundary_after_labels = _list_attr(prediction, "boundary_after_labels", len(tokens), "NONE")
        boundary_after_confidences = _float_list_attr(prediction, "boundary_after_confidences", len(tokens), 0.0)
        boundary_after_margins = _float_list_attr(prediction, "boundary_after_margins", len(tokens), 1.0)
        rule_ids = _list_attr(prediction, "rule_ids", len(tokens), "none")
        accepted_labels = self._gate_token_labels(text, tokens, labels, confidences, margins, rule_ids)
        accepted_boundary_before_labels = self._gate_boundary_labels(
            text,
            tokens,
            boundary_before_labels,
            boundary_before_confidences,
            boundary_before_margins,
            rule_ids,
            side="before",
        )
        accepted_boundary_after_labels = self._gate_boundary_labels(
            text,
            tokens,
            boundary_after_labels,
            boundary_after_confidences,
            boundary_after_margins,
            rule_ids,
            side="after",
        )
        return apply_boundary_and_token_edit_labels(
            text,
            tokens,
            accepted_labels,
            confidences,
            threshold=0.0,
            boundary_before_labels=accepted_boundary_before_labels,
            boundary_after_labels=accepted_boundary_after_labels,
            boundary_before_confidences=boundary_before_confidences,
            boundary_after_confidences=boundary_after_confidences,
            rule_ids=rule_ids,
            orthographic_lexicon=self.orthographic_lexicon,
        )

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
            if index in consumed or label in {"KEEP", "SKIP_MERGED"}:
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
                orthographic_lexicon=self.orthographic_lexicon,
            )
            if candidate_edits and all(self.scope_guard.validate_edit(text, edit) for edit in candidate_edits):
                accepted[index] = label
                edit_consumed = _accepted_consumed_indexes(tokens, index, label, candidate_edits)
                consumed.update(edit_consumed)
                for consumed_index in edit_consumed:
                    if consumed_index != index and consumed_index < len(accepted):
                        accepted[consumed_index] = "SKIP_MERGED"
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

    def _gate_boundary_labels(
        self,
        text: str,
        tokens: Sequence[Any],
        labels: Sequence[str],
        confidences: Sequence[float],
        margins: Sequence[float],
        rule_ids: Sequence[str],
        *,
        side: str,
    ) -> list[str]:
        accepted = ["NONE"] * len(tokens)
        for index, label in enumerate(labels[: len(tokens)]):
            if label == "NONE":
                continue
            rule_id = _rule_id(rule_ids, index)
            edit_type = boundary_edit_type_for_label(label)
            if not self.thresholds.should_apply(float(confidences[index]), float(margins[index]), rule_id, edit_type):
                continue
            before = ["NONE"] * len(tokens)
            after = ["NONE"] * len(tokens)
            if side == "before":
                before[index] = label
            else:
                after[index] = label
            _candidate_text, candidate_edits = apply_boundary_labels(
                text,
                tokens,
                before,
                after,
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


def _accepted_consumed_indexes(
    tokens: Sequence[Any],
    index: int,
    label: str,
    edits: Sequence[RuntimeEdit],
) -> set[int]:
    if label == "SPAN_REPLACE_BY_LEXICON" and edits:
        edit = edits[0]
        return {
            position
            for position in range(index, len(tokens))
            if getattr(tokens[position], "start", -1) >= getattr(tokens[index], "start", -1)
            and getattr(tokens[position], "end", -1) <= edit.end
        }
    if _label_consumes_next(label) and index + 1 < len(tokens):
        return {index, index + 1}
    return {index}


DETERMINISTIC_LEXICON_RULE_PREFIXES = ("dictionary_", "typo_", "compound_")
DETERMINISTIC_MORPHEME_RULE_IDS = frozenset(
    {
        "morpheme_hissing_vowels",
        "morpheme_soft_hard_signs",
        "morpheme_root_vowels",
        "morpheme_prefixes",
        "morpheme_suffixes",
        "morpheme_consonants",
        "morpheme_endings",
        "morpheme_n_nn",
    }
)
CONTEXT_DEPENDENT_N_NN_SUB_RULE_IDS = frozenset({"dependent_word"})


def _trusted_deterministic_entry(entry: CorrectionEntry, text: str) -> bool:
    rule_id = str(entry.rule_id)
    if not (
        rule_id.startswith(DETERMINISTIC_LEXICON_RULE_PREFIXES)
        or rule_id in DETERMINISTIC_MORPHEME_RULE_IDS
    ):
        return False
    if rule_id == "morpheme_endings":
        safe_patterns = entry.safe_context_patterns or ()
        lowered = text.casefold()
        if safe_patterns and any(str(item).casefold() in lowered for item in safe_patterns):
            pass
        elif not _is_unknown_single_russian_word(entry.source):
            return False
    if rule_id == "morpheme_n_nn" and entry.sub_rule_id in CONTEXT_DEPENDENT_N_NN_SUB_RULE_IDS:
        return False
    forbidden = entry.forbidden_contexts or ()
    if forbidden:
        lowered = text.casefold()
        if any(str(item).casefold() in lowered for item in forbidden if str(item).strip()):
            return False
    return True


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _edit_type_for_lexicon_operation(operation: str, *, source: str = "", replacement: str = "") -> str:
    if source and replacement:
        source_letters = _letters_only(source)
        replacement_letters = _letters_only(replacement)
        if source_letters and replacement_letters and source_letters != replacement_letters:
            return "spelling"
    if operation in {"split_join", "split", "merge", "join"}:
        return "split_join"
    if operation in {"hyphen", "hyphenate", "unhyphen", "unhyphenate", "dehyphen"}:
        return "hyphen"
    return "spelling"


def _word_count(value: str) -> int:
    import re

    return len(re.findall(r"[А-Яа-яЁё]+", value))


def _letters_only(value: str) -> str:
    import re

    return "".join(re.findall(r"[А-Яа-яЁё]+", value.casefold())).replace("ё", "е")


@lru_cache(maxsize=4096)
def _is_unknown_single_russian_word(value: str) -> bool:
    import re

    if not re.fullmatch(r"[А-Яа-яЁё]+", value):
        return False
    analyzer = _morph_analyzer()
    if analyzer is None:
        return False
    parses = analyzer.parse(value)
    return bool(parses) and not any(bool(getattr(parse, "is_known", False)) for parse in parses)


@lru_cache(maxsize=1)
def _morph_analyzer() -> Any | None:
    try:
        from pymorphy3 import MorphAnalyzer

        return MorphAnalyzer()
    except Exception:
        return None


__all__ = ["CorrectionResult", "Corrector", "RuntimeEdit"]
