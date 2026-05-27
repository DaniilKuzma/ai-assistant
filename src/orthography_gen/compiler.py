from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.rules.common import gap_labels_from_text
from src.orthography_gen.context_wrapper import ContextWrapper
from src.orthography_gen.error_injector import OrthographicErrorInjector
from src.orthography_gen.lexeme_cards import LexemeCard, load_lexeme_cards
from src.orthography_gen.rule_specs import DEFAULT_ORTHOGRAPHY_DIR, OrthographyRuleSpec, load_rule_specs
from src.rule_layers.context_variation import contextualize_example
from src.schema import GeneratedExample


class OrthographicScenarioCompiler:
    def __init__(
        self,
        specs: dict[str, OrthographyRuleSpec],
        cards: Iterable[LexemeCard],
        *,
        injector: OrthographicErrorInjector | None = None,
        wrapper: ContextWrapper | None = None,
    ) -> None:
        self.specs = dict(specs)
        self.cards = list(cards)
        self.cards_by_rule: dict[str, list[LexemeCard]] = defaultdict(list)
        for card in self.cards:
            if card.rule_id not in self.specs:
                raise ValueError(f"LexemeCard references unknown rule spec: {card.rule_id}")
            self.cards_by_rule[card.rule_id].append(card)
        self.injector = injector or OrthographicErrorInjector()
        self.wrapper = wrapper or ContextWrapper()

    @classmethod
    def default(cls) -> "OrthographicScenarioCompiler":
        return cls.from_dir(DEFAULT_ORTHOGRAPHY_DIR)

    @classmethod
    def from_dir(cls, path: str | Path) -> "OrthographicScenarioCompiler":
        return cls(load_rule_specs(path), load_lexeme_cards(path))

    def compile_example(
        self,
        rule_id: str,
        mode: GenerationMode | str,
        rng: RandomSource,
    ) -> GeneratedExample:
        generation_mode = _coerce_mode(mode)
        cards = self._cards_for(rule_id, generation_mode)
        card = rng.choice(tuple(cards))
        form_key = rng.choice(self.injector.wordforms.available_forms(card))
        if generation_mode is GenerationMode.HARD_NEGATIVE:
            replacement = self.injector.make_hard_negative(card, form_key)
            context = _choose_context(card.hard_negative_contexts, rng)
        else:
            replacement = self.injector.make_positive(card, form_key)
            context = _choose_context(card.safe_contexts, rng)
        context.setdefault("variant", rng.randint(1, 100_000))
        context.setdefault("rule_id", card.rule_id)
        context.setdefault("semantic_class", card.semantic_class or "")
        context.setdefault("gender", card.gender or "")
        context.setdefault("derivational_base", card.derivational_base or "")
        context.setdefault("correct_lemma", card.correct_lemma)
        context.setdefault("wrong_lemma", card.wrong_lemma)
        context.setdefault("sub_rule_id", card.sub_rule_id)

        wrapped = self.wrapper.wrap(
            source_word=replacement.source,
            target_word=replacement.target,
            pos=card.pos,
            context=context,
        )
        labels = ["KEEP"] * len(wrapped.source_tokens)
        if generation_mode is GenerationMode.POSITIVE:
            labels[wrapped.edited_token_index] = "DICT_REPLACE"

        metadata = self._metadata(
            card,
            context,
            wrapped.construction_id,
            replacement.source,
            replacement.target,
            form_key,
            expected_edits=1 if generation_mode is GenerationMode.POSITIVE else 0,
        )
        example = GeneratedExample(
            source_text=wrapped.source_text,
            target_text=wrapped.target_text,
            source_tokens=wrapped.source_tokens,
            token_edit_labels=labels,
            gap_labels=gap_labels_from_text(wrapped.source_text, wrapped.source_tokens),
            rule_ids=[rule_id] * len(wrapped.source_tokens),
            primary_rule_id=rule_id,
            mode=generation_mode.value,
            explanation_ids=[card.explanation_id or self.specs[rule_id].explanation_id],
            metadata=metadata,
        )
        return contextualize_example(example, rng)

    def compile_batch(
        self,
        count: int,
        rules: list[str] | tuple[str, ...] | None = None,
        *,
        rng: RandomSource | None = None,
        mode: GenerationMode | str = GenerationMode.POSITIVE,
    ) -> list[GeneratedExample]:
        if count < 0:
            raise ValueError("count must be non-negative.")
        random_source = rng or RandomSource()
        rule_ids = tuple(rules or self._rules_with_cards(_coerce_mode(mode)))
        if not rule_ids:
            raise ValueError("No orthography rules are available for batch compilation.")
        return [
            self.compile_example(random_source.choice(rule_ids), mode, random_source)
            for _ in range(count)
        ]

    def has_mode(self, rule_id: str, mode: GenerationMode | str) -> bool:
        return bool(self._cards_for(rule_id, _coerce_mode(mode), fail=False))

    def _cards_for(
        self,
        rule_id: str,
        mode: GenerationMode,
        *,
        fail: bool = True,
    ) -> list[LexemeCard]:
        cards = self.cards_by_rule.get(rule_id, [])
        if mode is GenerationMode.POSITIVE:
            result = [
                card
                for card in cards
                if card.safe_contexts
                and any(forms.get("correct") != forms.get("wrong") for forms in card.forms.values())
                and self.specs[card.rule_id].model_role != "hard_negative_only"
            ]
        elif mode is GenerationMode.HARD_NEGATIVE:
            result = [card for card in cards if card.hard_negative_contexts]
        else:
            result = []
        if fail and not result:
            raise ValueError(f"No orthography cards for rule {rule_id!r} in mode {mode.value!r}.")
        return result

    def _rules_with_cards(self, mode: GenerationMode) -> tuple[str, ...]:
        return tuple(rule_id for rule_id in sorted(self.cards_by_rule) if self._cards_for(rule_id, mode, fail=False))

    def _metadata(
        self,
        card: LexemeCard,
        context: dict[str, Any],
        construction_id: str,
        source: str,
        target: str,
        form_key: str,
        *,
        expected_edits: int,
    ) -> dict[str, Any]:
        spec = self.specs[card.rule_id]
        site_type = card.site_type or spec.site_type
        morph_features = _morph_features(card, form_key)
        return {
            "layer": "morpheme",
            "production": True,
            "uses_construction_bank": True,
            "construction_id": construction_id,
            "construction_family": "orthography_morphemic",
            "uses_safety_clauses": False,
            "safety_clauses": [],
            "sub_rule_id": card.sub_rule_id or card.rule_id,
            "site_type": site_type,
            "orthography_site": {
                "site_type": site_type,
                "sub_rule_id": card.sub_rule_id or card.rule_id,
                "correct_site": card.correct_site,
                "wrong_site": card.wrong_site,
                "stress_position": card.stress_position,
                "derivational_base": card.derivational_base,
                "exception_group": card.exception_group,
            },
            "lexeme_card_id": card.lexeme_card_id,
            "orthography_rule_spec": spec.to_dict(),
            "expected_edit_count": expected_edits,
            "expected_token_edit_count": expected_edits,
            "expected_gap_edit_count": 0,
            "correct_form": target,
            "wrong_form": source,
            "morph_features": morph_features,
            "replacement": {"source": source, "target": target},
            "context_class": str(context.get("context_class") or ""),
        }


def _coerce_mode(mode: GenerationMode | str) -> GenerationMode:
    if isinstance(mode, GenerationMode):
        return mode
    return GenerationMode(str(mode))


def _choose_context(contexts: list[dict[str, Any]], rng: RandomSource) -> dict[str, Any]:
    if not contexts:
        return {}
    return dict(rng.choice(tuple(contexts)))


def _morph_features(card: LexemeCard, form_key: str) -> dict[str, str]:
    features = dict(card.morph_features)
    raw_form = card.forms.get(form_key, {})
    raw_features = raw_form.get("morph_features") if isinstance(raw_form, dict) else None
    if isinstance(raw_features, dict):
        features.update({str(key): str(value) for key, value in raw_features.items()})
    features.setdefault("form_key", form_key)
    features.setdefault("pos", card.pos)
    if card.gender:
        features.setdefault("gender", card.gender)
    return features


__all__ = ["OrthographicScenarioCompiler"]
