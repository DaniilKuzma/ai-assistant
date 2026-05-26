from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.orthography_gen.rule_specs import DEFAULT_ORTHOGRAPHY_DIR


@dataclass(frozen=True)
class LexemeCard:
    rule_id: str
    sub_rule_id: str
    correct_lemma: str
    wrong_lemma: str
    pos: str
    gender: str | None = None
    animacy: str | None = None
    semantic_class: str | None = None
    stress_position: str | None = None
    derivational_base: str | None = None
    site_type: str | None = None
    morph_features: dict[str, str] = field(default_factory=dict)
    correct_site: str = ""
    wrong_site: str = ""
    forms: dict[str, dict[str, Any]] = field(default_factory=dict)
    safe_contexts: list[dict[str, Any]] = field(default_factory=list)
    hard_negative_contexts: list[dict[str, Any]] = field(default_factory=list)
    exception_group: str | None = None
    explanation_id: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LexemeCard":
        card = cls(
            rule_id=_required_str(data, "rule_id"),
            sub_rule_id=str(data.get("sub_rule_id") or data.get("rule_id") or "").strip(),
            correct_lemma=_required_str(data, "correct_lemma"),
            wrong_lemma=_required_str(data, "wrong_lemma"),
            pos=_required_str(data, "pos"),
            gender=_optional_str(data.get("gender")),
            animacy=_optional_str(data.get("animacy")),
            semantic_class=_optional_str(data.get("semantic_class")),
            stress_position=_optional_str(data.get("stress_position")),
            derivational_base=_optional_str(data.get("derivational_base")),
            site_type=_optional_str(data.get("site_type")),
            morph_features=_string_mapping(data.get("morph_features")),
            correct_site=str(data.get("correct_site") or ""),
            wrong_site=str(data.get("wrong_site") or ""),
            forms=_forms_from_mapping(data.get("forms")),
            safe_contexts=_context_list(data.get("safe_contexts")),
            hard_negative_contexts=_context_list(data.get("hard_negative_contexts")),
            exception_group=_optional_str(data.get("exception_group")),
            explanation_id=str(data.get("explanation_id") or data.get("rule_id") or ""),
        )
        card.validate()
        return card

    @property
    def lexeme_card_id(self) -> str:
        payload = {
            "rule_id": self.rule_id,
            "sub_rule_id": self.sub_rule_id,
            "correct_lemma": self.correct_lemma,
            "wrong_lemma": self.wrong_lemma,
            "pos": self.pos,
            "correct_site": self.correct_site,
            "wrong_site": self.wrong_site,
            "site_type": self.site_type,
            "stress_position": self.stress_position,
            "exception_group": self.exception_group,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"{self.rule_id}:{digest}"

    def validate(self) -> None:
        if not self.forms:
            raise ValueError(f"LexemeCard {self.rule_id!r}:{self.correct_lemma!r} must define forms.")
        for key, forms in self.forms.items():
            if "correct" not in forms:
                raise ValueError(f"LexemeCard form {key!r} must define a correct form.")
            if "wrong" not in forms:
                raise ValueError(f"LexemeCard form {key!r} must define a wrong form.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "correct_lemma": self.correct_lemma,
            "wrong_lemma": self.wrong_lemma,
            "pos": self.pos,
            "gender": self.gender,
            "animacy": self.animacy,
            "semantic_class": self.semantic_class,
            "stress_position": self.stress_position,
            "derivational_base": self.derivational_base,
            "site_type": self.site_type,
            "morph_features": dict(self.morph_features),
            "correct_site": self.correct_site,
            "wrong_site": self.wrong_site,
            "forms": {key: dict(value) for key, value in self.forms.items()},
            "safe_contexts": [dict(item) for item in self.safe_contexts],
            "hard_negative_contexts": [dict(item) for item in self.hard_negative_contexts],
            "exception_group": self.exception_group,
            "explanation_id": self.explanation_id,
            "lexeme_card_id": self.lexeme_card_id,
        }


def load_lexeme_cards(path: str | Path = DEFAULT_ORTHOGRAPHY_DIR) -> list[LexemeCard]:
    cards: list[LexemeCard] = []
    for yaml_path in _yaml_paths(path):
        with yaml_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        raw_cards = raw.get("lexeme_cards", []) if isinstance(raw, Mapping) else []
        if raw_cards is None:
            continue
        if not isinstance(raw_cards, list):
            raise ValueError(f"lexeme_cards must be a list: {yaml_path}")
        cards.extend(LexemeCard.from_mapping(item) for item in raw_cards)
    return cards


def _yaml_paths(path: str | Path) -> tuple[Path, ...]:
    base = Path(path)
    if base.is_file():
        return (base,)
    if not base.exists():
        raise FileNotFoundError(f"Orthography lexicon directory does not exist: {base}")
    return tuple(sorted(base.glob("*.yaml")))


def _forms_from_mapping(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for key, raw_forms in value.items():
        if not isinstance(raw_forms, Mapping):
            raise ValueError(f"Invalid forms mapping for {key!r}.")
        result[str(key)] = dict(raw_forms)
        result[str(key)]["correct"] = str(raw_forms.get("correct") or "")
        result[str(key)]["wrong"] = str(raw_forms.get("wrong") or "")
    return result


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items() if str(key).strip()}


def _context_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("LexemeCard contexts must be a list.")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"LexemeCard is missing {key!r}.")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["LexemeCard", "load_lexeme_cards"]
