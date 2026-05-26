from __future__ import annotations

from dataclasses import dataclass

from src.orthography_gen.lexeme_cards import LexemeCard
from src.orthography_gen.wordform_generator import WordFormGenerator


@dataclass(frozen=True)
class OrthographicReplacement:
    source: str
    target: str
    form_key: str


class OrthographicErrorInjector:
    def __init__(self, wordforms: WordFormGenerator | None = None) -> None:
        self.wordforms = wordforms or WordFormGenerator()

    def make_positive(self, card: LexemeCard, form_key: str) -> OrthographicReplacement:
        return OrthographicReplacement(
            source=self.wordforms.wrong_form(card, form_key),
            target=self.wordforms.form(card, form_key),
            form_key=form_key,
        )

    def make_hard_negative(self, card: LexemeCard, form_key: str) -> OrthographicReplacement:
        correct = self.wordforms.form(card, form_key)
        return OrthographicReplacement(source=correct, target=correct, form_key=form_key)

    def make_minimal_pair(self, card: LexemeCard, context_class: str) -> OrthographicReplacement:
        del context_class
        form_key = self.wordforms.available_forms(card)[0]
        return self.make_positive(card, form_key)


__all__ = ["OrthographicErrorInjector", "OrthographicReplacement"]
