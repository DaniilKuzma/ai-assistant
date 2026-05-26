from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.orthography_gen.lexeme_cards import LexemeCard, load_lexeme_cards
from src.orthography_gen.rule_specs import DEFAULT_ORTHOGRAPHY_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CorrectionEntry:
    source: str
    target: str
    rule_id: str
    explanation_id: str
    context_class: str
    ambiguity_level: str


class OrthographicCorrectionLexicon:
    def __init__(self, entries: Iterable[CorrectionEntry]) -> None:
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
        return cls.from_dir(base / "orthography")

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
    ) -> list[CorrectionEntry]:
        entries = list(self._mapping.get(source.lower(), ()))
        if rule_id and rule_id != "none":
            entries = [entry for entry in entries if entry.rule_id == rule_id]
        if context_class:
            entries = [entry for entry in entries if entry.context_class == context_class]
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
                    )
                )
    return entries


__all__ = ["CorrectionEntry", "OrthographicCorrectionLexicon"]
