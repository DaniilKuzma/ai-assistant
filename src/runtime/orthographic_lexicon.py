from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from src.orthography_gen.lexeme_cards import LexemeCard, load_lexeme_cards
from src.orthography_gen.rule_specs import DEFAULT_ORTHOGRAPHY_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CorrectionEntry:
    source: str
    target: str
    rule_id: str
    explanation_id: str = ""
    context_class: str = ""
    ambiguity_level: str = "unambiguous"
    operation: str = "dict_replace"
    sub_rule_id: str = ""
    confidence: float = 1.0
    forbidden_contexts: tuple[str, ...] = ()
    safe_context_patterns: tuple[str, ...] = ()


class OrthographicCorrectionLexicon:
    def __init__(self, entries: Iterable[CorrectionEntry]) -> None:
        entries = _merge_entry_context_guards(tuple(entries))
        mapping: dict[str, list[CorrectionEntry]] = defaultdict(list)
        for entry in entries:
            if not entry.source or not entry.target or entry.source == entry.target:
                continue
            mapping[entry.source.lower()].append(entry)
        self._mapping = {key: tuple(value) for key, value in mapping.items()}

    @classmethod
    def default(cls) -> "OrthographicCorrectionLexicon":
        return cls.from_dir(DEFAULT_ORTHOGRAPHY_DIR)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "OrthographicCorrectionLexicon":
        paths = config.get("paths", {}) if isinstance(config, Mapping) else {}
        lexicon_dir = paths.get("lexicon_dir", "lexicon") if isinstance(paths, Mapping) else "lexicon"
        base = Path(str(lexicon_dir))
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return cls.from_root(base, seed=_generation_seed(config))

    @classmethod
    def from_root(cls, path: str | Path, *, seed: int | None = None) -> "OrthographicCorrectionLexicon":
        base = Path(path)
        entries: list[CorrectionEntry] = []
        orthography_dir = base / "orthography"
        if orthography_dir.exists():
            entries.extend(_entries_from_cards(load_lexeme_cards(orthography_dir)))
        entries.extend(_entries_from_layer_corrections(base / "layers"))
        entries.extend(_entries_from_compound_spelling_specs(base / "layers"))
        entries.extend(_entries_from_dictionary_typo(base / "layers", seed=seed))
        return cls(entries)

    @classmethod
    def from_dir(cls, path: str | Path) -> "OrthographicCorrectionLexicon":
        base = Path(path)
        if not base.exists():
            return cls(())
        return cls(_entries_from_cards(load_lexeme_cards(base)))

    def lookup(
        self,
        source: str,
        *,
        rule_id: str | None = None,
        context_class: str | None = None,
        operation: str | None = None,
        sub_rule_id: str | None = None,
    ) -> list[CorrectionEntry]:
        entries = list(self._mapping.get(source.lower(), ()))
        if rule_id and rule_id != "none":
            entries = [entry for entry in entries if entry.rule_id == rule_id]
        if context_class:
            entries = [entry for entry in entries if entry.context_class == context_class]
        if operation:
            entries = [entry for entry in entries if entry.operation == operation]
        if sub_rule_id:
            entries = [entry for entry in entries if entry.sub_rule_id == sub_rule_id]
        ambiguity = "ambiguous" if len({entry.target for entry in entries}) > 1 else "unambiguous"
        return [replace(entry, ambiguity_level=ambiguity) for entry in entries]

    def lookup_any(self, sources: Iterable[str]) -> list[CorrectionEntry]:
        result: list[CorrectionEntry] = []
        for source in sources:
            result.extend(self.lookup(source))
        return result


def _entries_from_cards(cards: Iterable[LexemeCard]) -> list[CorrectionEntry]:
    entries: list[CorrectionEntry] = []
    for card in cards:
        contexts = card.safe_contexts or [{}]
        for forms in card.forms.values():
            source = str(forms.get("wrong") or "")
            target = str(forms.get("correct") or "")
            if not source or not target or source == target:
                continue
            for context in contexts:
                entries.append(
                    CorrectionEntry(
                        source=source,
                        target=target,
                        rule_id=card.rule_id,
                        explanation_id=card.explanation_id,
                        context_class=str(context.get("context_class") or "") if isinstance(context, Mapping) else "",
                        ambiguity_level="unambiguous",
                        operation="dict_replace",
                        sub_rule_id=card.sub_rule_id,
                        confidence=1.0,
                        safe_context_patterns=_safe_context_patterns(context, source),
                    )
                )
    return entries


def _merge_entry_context_guards(entries: tuple[CorrectionEntry, ...]) -> tuple[CorrectionEntry, ...]:
    forbidden_by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    safe_by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for entry in entries:
        key = _entry_context_key(entry)
        forbidden_by_key[key].update(entry.forbidden_contexts)
        safe_by_key[key].update(entry.safe_context_patterns)
    return tuple(
        replace(
            entry,
            forbidden_contexts=tuple(sorted(forbidden_by_key[_entry_context_key(entry)])),
            safe_context_patterns=tuple(sorted(safe_by_key[_entry_context_key(entry)])),
        )
        for entry in entries
    )


def _entry_context_key(entry: CorrectionEntry) -> tuple[str, str, str, str]:
    return (
        entry.source.casefold(),
        entry.target.casefold(),
        entry.rule_id,
        entry.sub_rule_id,
    )


def _entries_from_layer_corrections(path: str | Path) -> list[CorrectionEntry]:
    base = Path(path)
    if not base.exists():
        return []

    entries: list[CorrectionEntry] = []
    for yaml_path in sorted(base.rglob("corrections.yaml")):
        if yaml_path.parent.name == "dictionary_typo":
            continue
        with yaml_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        raw_entries = raw.get("corrections", []) if isinstance(raw, Mapping) else []
        if raw_entries is None:
            continue
        if not isinstance(raw_entries, list):
            raise ValueError(f"corrections must be a list: {yaml_path}")
        entries.extend(_entry_from_layer_mapping(item, yaml_path) for item in raw_entries)
    return entries


def _entries_from_dictionary_typo(path: str | Path, *, seed: int | None = None) -> list[CorrectionEntry]:
    try:
        from src.rule_layers.dictionary_typo import load_dictionary_typo_corrections
    except Exception:
        return []

    base = Path(path)
    if not (base / "dictionary_typo").exists():
        return []

    entries: list[CorrectionEntry] = []
    for item in load_dictionary_typo_corrections(base, seed=seed):
        entries.append(
            CorrectionEntry(
                source=item.source,
                target=item.target,
                rule_id=item.rule_id,
                explanation_id=item.explanation_id,
                operation=item.operation,
                sub_rule_id=item.sub_rule_id,
                confidence=1.0,
            )
        )
    return entries


def _entries_from_compound_spelling_specs(path: str | Path) -> list[CorrectionEntry]:
    try:
        from src.rule_layers.compound_spelling import load_compound_spelling_specs
    except Exception:
        return []

    entries: list[CorrectionEntry] = []
    try:
        specs = load_compound_spelling_specs(path)
    except Exception:
        return []
    for spec in specs:
        for case in spec.cases:
            if case.mode != "positive":
                continue
            metadata = dict(case.metadata)
            source = str(metadata.get("source") or "")
            target = str(metadata.get("target") or "")
            if not source or not target:
                for operation in case.token_operations:
                    source = source or operation.source_pattern
                    target = target or operation.target_pattern
                    if source and target:
                        break
            if not source or not target or source == target:
                continue
            entries.append(
                CorrectionEntry(
                    source=source,
                    target=target,
                    rule_id=case.rule_id,
                    operation=str(metadata.get("operation") or "replace"),
                    explanation_id=case.rule_id,
                    sub_rule_id=case.sub_rule_id,
                    confidence=1.0,
                    forbidden_contexts=_string_tuple(metadata.get("forbidden_contexts")),
                )
            )
    return entries


def _entry_from_layer_mapping(data: Any, path: Path) -> CorrectionEntry:
    if not isinstance(data, Mapping):
        raise ValueError(f"Layer correction entry must be a mapping: {path}")
    source = _required_layer_str(data, "source", path)
    target = _required_layer_str(data, "target", path)
    rule_id = _required_layer_str(data, "rule_id", path)
    operation = _required_layer_str(data, "operation", path)
    return CorrectionEntry(
        source=source,
        target=target,
        rule_id=rule_id,
        operation=operation,
        explanation_id=str(data.get("explanation_id") or rule_id),
        sub_rule_id=str(data.get("sub_rule_id") or ""),
        context_class=str(data.get("context_class") or ""),
        confidence=_confidence(data.get("confidence")),
        forbidden_contexts=_string_tuple(data.get("forbidden_contexts")),
        safe_context_patterns=_string_tuple(data.get("safe_context_patterns")),
    )


def _safe_context_patterns(context: Mapping[str, Any], source: str) -> tuple[str, ...]:
    template = str(context.get("template") or "").strip() if isinstance(context, Mapping) else ""
    if not template:
        return ()
    try:
        rendered = template.format(word=source, variant="")
    except (KeyError, IndexError, ValueError):
        return ()
    return (rendered.strip(),) if rendered.strip() else ()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _required_layer_str(data: Mapping[str, Any], key: str, path: Path) -> str:
    value = str(data.get(key) or "")
    if not value.strip():
        raise ValueError(f"Layer correction entry is missing {key!r}: {path}")
    return value


def _confidence(value: Any) -> float:
    if value is None:
        return 1.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 1.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _generation_seed(config: Mapping[str, Any] | None) -> int | None:
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    if not isinstance(generation, Mapping) or "seed" not in generation:
        return None
    try:
        return int(generation.get("seed"))
    except (TypeError, ValueError):
        return None


__all__ = ["CorrectionEntry", "OrthographicCorrectionLexicon"]
