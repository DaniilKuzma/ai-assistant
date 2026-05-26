from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.orthography_gen.lexeme_cards import LexemeCard


class WordFormGenerator:
    def __init__(self, *, use_pymorphy: bool = True) -> None:
        self._morph = None
        if use_pymorphy:
            try:
                from pymorphy3 import MorphAnalyzer

                self._morph = MorphAnalyzer()
            except Exception:
                self._morph = None

    def form(self, card: LexemeCard, grammemes: str | Iterable[str]) -> str:
        key = _form_key(grammemes)
        curated = card.forms.get(key, {}).get("correct")
        if curated:
            return curated
        inflected = self._inflect(card.correct_lemma, card.pos, key)
        if inflected:
            return inflected
        raise ValueError(f"Missing critical correct form {key!r} for {card.lexeme_card_id}.")

    def wrong_form(self, card: LexemeCard, grammemes: str | Iterable[str]) -> str:
        key = _form_key(grammemes)
        curated = card.forms.get(key, {}).get("wrong")
        if curated:
            return curated
        raise ValueError(f"Missing critical wrong form {key!r} for {card.lexeme_card_id}.")

    def available_forms(self, card: LexemeCard) -> tuple[str, ...]:
        return tuple(sorted(card.forms))

    def _inflect(self, lemma: str, pos: str, form_key: str) -> str | None:
        if self._morph is None:
            return None
        tags = _pymorphy_tags(form_key)
        if not tags:
            return None
        parses = self._morph.parse(lemma)
        preferred = _preferred_pos(pos)
        parse = next((item for item in parses if getattr(item.tag, "POS", None) == preferred), None)
        if parse is None:
            return None
        inflected = parse.inflect(tags)
        word = str(getattr(inflected, "word", "")) if inflected is not None else ""
        return word or None


def _form_key(grammemes: str | Iterable[str]) -> str:
    if isinstance(grammemes, str):
        return grammemes
    return "_".join(str(item) for item in grammemes)


def _preferred_pos(pos: str) -> str:
    return {
        "ADJ": "ADJF",
        "ADJF": "ADJF",
        "PRTF": "PRTF",
        "NOUN": "NOUN",
    }.get(pos, pos)


def _pymorphy_tags(form_key: str) -> set[str]:
    tags: set[str] = set()
    for part in form_key.split("_"):
        mapped = {
            "nom": "nomn",
            "nomn": "nomn",
            "gent": "gent",
            "datv": "datv",
            "accs": "accs",
            "ablt": "ablt",
            "loct": "loct",
            "masc": "masc",
            "femn": "femn",
            "fem": "femn",
            "neut": "neut",
            "sing": "sing",
            "plur": "plur",
        }.get(part)
        if mapped:
            tags.add(mapped)
    return tags


__all__ = ["WordFormGenerator"]
