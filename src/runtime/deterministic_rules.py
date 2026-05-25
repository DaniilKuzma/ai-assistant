from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from src.runtime.tokenization import tokenize_runtime_words
from src.schema import RuntimeEdit, WordToken


SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?:;])")
SPACE_AFTER_PUNCT_RE = re.compile(r"([,;:])(?![,;:!?…])(?=\S)")
MULTISPACE_RE = re.compile(r"[ \t\r\f\v]{2,}")

HYPHEN_PAIRS: dict[tuple[str, str], tuple[str, str]] = {
    ("кто", "то"): ("hyphen_particles", "hyphen"),
    ("кто", "либо"): ("hyphen_particles", "hyphen"),
    ("кто", "нибудь"): ("hyphen_particles", "hyphen"),
    ("кое", "кто"): ("hyphen_koe", "hyphen"),
    ("по", "русски"): ("hyphen_po_adverb", "hyphen"),
}

NE_VERB_FORMS = frozenset(
    {
        "знал",
        "знала",
        "знало",
        "знали",
        "знает",
        "проверил",
        "проверила",
        "проверило",
        "проверили",
        "проверяет",
        "решил",
        "решила",
        "решило",
        "решили",
        "сказал",
        "сказала",
        "сказало",
        "сказали",
        "исправил",
        "исправила",
        "исправило",
        "исправили",
    }
)
NE_EXCEPTIONS = frozenset({"ненавидел", "ненавидела", "ненавидели", "негодовал", "недомогал", "недоумевал"})


@dataclass(frozen=True)
class DeterministicRuleConfig:
    enabled: bool = True
    final_punctuation: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "DeterministicRuleConfig":
        runtime = config.get("runtime", {}) if isinstance(config, Mapping) else {}
        if not isinstance(runtime, Mapping):
            runtime = {}
        return cls(
            enabled=bool(runtime.get("deterministic_first", True)),
            final_punctuation=bool(runtime.get("deterministic_final_punctuation", False)),
        )


class DeterministicRuleEngine:
    def __init__(self, config: DeterministicRuleConfig | None = None) -> None:
        self.config = config or DeterministicRuleConfig()
        self._ne_forms = _build_ne_verb_forms()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "DeterministicRuleEngine":
        return cls(DeterministicRuleConfig.from_config(config))

    def propose_edits(self, text: str) -> list[RuntimeEdit]:
        if not self.config.enabled:
            return []

        normalized = normalize_runtime_spacing(text)
        if normalized != text:
            return [
                RuntimeEdit(
                    start=0,
                    end=len(text),
                    source=text,
                    replacement=normalized,
                    edit_type="punctuation",
                    rule_id="spacing_normalization",
                    confidence=1.0,
                )
            ]

        tokens = tokenize_runtime_words(text)
        edits = [*self._hyphen_edits(text, tokens), *self._ne_verb_edits(text, tokens)]
        if self.config.final_punctuation:
            final_edit = self._final_punctuation_edit(text)
            if final_edit is not None:
                edits.append(final_edit)
        return edits

    def _hyphen_edits(self, text: str, tokens: list[WordToken]) -> list[RuntimeEdit]:
        edits: list[RuntimeEdit] = []
        index = 0
        while index + 1 < len(tokens):
            first = tokens[index]
            second = tokens[index + 1]
            rule = HYPHEN_PAIRS.get((first.text.lower(), second.text.lower()))
            if rule is None:
                index += 1
                continue
            rule_id, edit_type = rule
            start, end = first.start, second.end
            source = text[start:end]
            edits.append(
                RuntimeEdit(
                    start=start,
                    end=end,
                    source=source,
                    replacement=f"{first.text}-{second.text}",
                    edit_type=edit_type,
                    rule_id=rule_id,
                    confidence=1.0,
                )
            )
            index += 2
        return edits

    def _ne_verb_edits(self, text: str, tokens: list[WordToken]) -> list[RuntimeEdit]:
        edits: list[RuntimeEdit] = []
        for token in tokens:
            lowered = token.text.lower()
            if lowered in NE_EXCEPTIONS or not lowered.startswith("не") or len(lowered) <= 2:
                continue
            verb_form = lowered[2:]
            if verb_form not in self._ne_forms:
                continue
            source = text[token.start : token.end]
            replacement = f"{source[:2]} {source[2:]}"
            edits.append(
                RuntimeEdit(
                    start=token.start,
                    end=token.end,
                    source=source,
                    replacement=replacement,
                    edit_type="split_join",
                    rule_id="ne_verb",
                    confidence=1.0,
                )
            )
        return edits

    def _final_punctuation_edit(self, text: str) -> RuntimeEdit | None:
        stripped = text.rstrip()
        if not stripped or stripped[-1] in ".!?…:;":
            return None
        if not re.search(r"[А-Яа-яЁё]$", stripped):
            return None
        return RuntimeEdit(
            start=len(stripped),
            end=len(stripped),
            source="",
            replacement=".",
            edit_type="punctuation",
            rule_id="final_punctuation",
            confidence=1.0,
        )


def normalize_runtime_spacing(text: str) -> str:
    normalized = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    normalized = SPACE_AFTER_PUNCT_RE.sub(r"\1 ", normalized)
    normalized = re.sub(r"\s*—\s*", " — ", normalized)
    normalized = re.sub(r"([«(])\s+", r"\1", normalized)
    normalized = re.sub(r"\s+([»)])", r"\1", normalized)
    normalized = MULTISPACE_RE.sub(" ", normalized)
    return normalized.strip()


def _build_ne_verb_forms() -> frozenset[str]:
    forms = set(NE_VERB_FORMS)
    try:
        from src.grammar_gen import Lexicon, MorphologyEngine

        lexicon = Lexicon.default()
        morphology = MorphologyEngine(use_pymorphy=False)
        exceptions = lexicon.exceptions.get("ne_verb", frozenset())
        for verb in lexicon.verbs:
            if not verb.allow_ne or verb.lemma in exceptions:
                continue
            forms.add(morphology.infinitive(verb.lemma).lower())
            forms.add(morphology.inflect_verb_present_3sg(verb.lemma).lower())
            for gender in ("masc", "femn", "neut"):
                forms.add(morphology.inflect_verb_past(verb.lemma, gender).lower())
            forms.add(morphology.inflect_verb_past(verb.lemma, "masc", "plur").lower())
    except Exception:
        pass
    return frozenset(form for form in forms if form and not form.startswith("не"))


__all__ = ["DeterministicRuleConfig", "DeterministicRuleEngine", "normalize_runtime_spacing"]
