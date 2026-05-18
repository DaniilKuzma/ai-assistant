from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.candidates.frequent_errors import (
    CONTEXT_DEPENDENT_WHITELIST,
    HYPHEN_WHITELIST,
    SPLIT_JOIN_WHITELIST,
    WRONG_TO_CORRECT,
)
from src.candidates.morphology import has_pos, is_known_word, normal_forms
from src.preprocessing.tokenizer import tokenize_words
from src.rules.base import RuleCandidate, RuleContext, RuleCorruption, RuleEdit, RuleMode, RuleSpec


VERB_POSES = frozenset({"VERB", "INFN"})
TSYA_CONTEXT_POSES = frozenset({"VERB", "INFN"})
NE_JOINED_NORMAL_FORMS = frozenset(
    {
        "ненавидеть",
        "негодовать",
        "нездоровиться",
        "недоумевать",
        "недосмотреть",
        "недооценить",
        "недосыпать",
        "недополучить",
        "недоставать",
    }
)

SPELLING_PATTERNS = {
    "жы": "жи",
    "шы": "ши",
    "чя": "ча",
    "щя": "ща",
    "чю": "чу",
    "щю": "щу",
    "цы": "ци",
    "жо": "же",
    "шо": "ше",
    "чо": "че",
    "що": "ще",
}

CY_EXCEPTIONS = ("цыган", "цыпл", "цыц", "цык", "цып")
HARD_SIGN_PREFIXES = (
    "сверх",
    "меж",
    "контр",
    "пан",
    "двух",
    "трех",
    "пред",
    "под",
    "над",
    "раз",
    "без",
    "из",
    "об",
    "от",
    "в",
    "с",
    "ад",
    "ин",
    "кон",
    "суб",
)
HARD_SIGN_VOWELS = frozenset({"е", "ю", "я"})
VOICELESS = frozenset("кпстфхцчшщ")
VOICED_OR_SONORANT_OR_VOWEL = frozenset("бвгджзлмнраеёиоуыэюя")
PREFIX_Z_S_PAIRS = (
    ("без", "бес"),
    ("раз", "рас"),
    ("из", "ис"),
    ("воз", "вос"),
    ("низ", "нис"),
    ("вз", "вс"),
)
HYPHEN_PARTICLE_BASES = frozenset(
    {
        "кто",
        "что",
        "где",
        "куда",
        "откуда",
        "когда",
        "как",
        "сколько",
        "почему",
        "зачем",
        "какой",
        "какая",
        "какое",
        "какие",
        "чей",
        "чья",
        "чье",
        "чьи",
    }
)
HYPHEN_PARTICLES = frozenset({"то", "либо", "нибудь"})
KOE_KOY_PREFIXES = frozenset({"кое", "кой"})
KOE_KOY_BASES = HYPHEN_PARTICLE_BASES | frozenset({"там", "тут", "здесь"})
PO_ADVERB_SUFFIXES = ("ому", "ему", "ски", "цки", "ьи")
POL_VOWELS = frozenset("аеёиоуыэюя")
POL_NOUN_POSES = frozenset({"NOUN"})
SCORING_REQUIRED_RULE_IDS = frozenset(
    {
        "ne_verb",
        "tsya_soft_delete",
        "tsya_soft_insert",
        "context_pair",
        "hyphen_whitelist",
        "hyphen_particles",
        "hyphen_koe_koy",
        "hyphen_po_adverbs",
        "pol_polu_compounds",
    }
)

DETERMINISTIC_FREQUENT_ERROR_KEYS = frozenset(
    {
        "жызнь",
        "жывой",
        "жывотное",
        "машына",
        "шырина",
        "чясто",
        "чяща",
        "щястье",
        "чюдо",
        "чювство",
        "щюка",
        "зделать",
        "зделал",
        "безполезный",
        "безплатный",
        "безпокойный",
        "безконечный",
        "безшумный",
        "бесвкусный",
        "бесграмотный",
        "подезд",
        "подезде",
        "подезду",
        "подездом",
        "обьект",
        "обявление",
        "сьезд",
        "вюга",
        "цыфра",
        "цырк",
        "цытата",
        "шол",
        "пришол",
        "нашол",
        "произошол",
        "жолтый",
        "чорный",
        "дешовый",
    }
)


@dataclass(frozen=True)
class FrequentErrorRule:
    spec: RuleSpec = RuleSpec(
        id="frequent_errors",
        group="dictionary_spelling",
        scope="token",
        edit_type="spelling",
        mode="deterministic",
        confidence=0.95,
        requires=("dictionary",),
        description="Whitelist fallback for frequent dictionary spelling and split/join errors.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        lower = token.lower()
        if lower not in DETERMINISTIC_FREQUENT_ERROR_KEYS:
            return []
        replacement = WRONG_TO_CORRECT.get(lower)
        if not replacement:
            return []
        return [_candidate(self.spec, replacement, requires_model=False)]

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)


@dataclass(frozen=True)
class FrequentSplitJoinRule:
    spec: RuleSpec = RuleSpec(
        id="frequent_split_join",
        group="split_join",
        scope="token",
        edit_type="split_join",
        mode="candidate_only",
        confidence=0.95,
        requires=("dictionary", "model"),
        description="Whitelist frequent split/join candidates that require scorer confirmation.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        lower = token.lower()
        replacement = SPLIT_JOIN_WHITELIST.get(lower)
        if not replacement:
            return []
        return [_candidate(self.spec, replacement, requires_model=True)]

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)


@dataclass(frozen=True)
class FrequentModelRequiredRule:
    spec: RuleSpec = RuleSpec(
        id="frequent_dictionary_model_required",
        group="dictionary_model_required",
        scope="token",
        edit_type="spelling",
        mode="model_required",
        confidence=0.95,
        requires=("dictionary", "model"),
        description="Frequent dictionary repairs that are unsafe to apply without model scoring.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        lower = token.lower()
        if lower in DETERMINISTIC_FREQUENT_ERROR_KEYS or lower in SPLIT_JOIN_WHITELIST:
            return []
        replacement = WRONG_TO_CORRECT.get(lower)
        if not replacement:
            return []
        return [_candidate(self.spec, replacement, requires_model=True)]

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)


@dataclass(frozen=True)
class NeVerbRule:
    spec: RuleSpec = RuleSpec(
        id="ne_verb",
        group="ne",
        scope="token",
        edit_type="split_join",
        mode="candidate_only",
        confidence=0.97,
        requires=("morphology",),
        description="Separate particle 'не' from verbs and infinitives when the joined form is not an exception.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        if not word.startswith("не") or len(word) <= 4:
            return []
        rest = word[2:]
        if word.startswith("недо") and is_known_word(word):
            return []
        if normal_forms(word) & NE_JOINED_NORMAL_FORMS:
            return []
        if not has_pos(rest, VERB_POSES):
            return []
        return [_candidate(self.spec, f"не {rest}", requires_model=is_known_word(word))]

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        word_tokens = _word_tokens(text, tokens)
        corruptions: list[RuleCorruption] = []
        for index, word in _iter_token_positions(word_tokens, token_index):
            if word.text.lower() != "не" or index + 1 >= len(word_tokens):
                continue
            next_word = word_tokens[index + 1]
            start, end = int(word.start), int(next_word.end)
            if _span_overlaps_protected(start, end, protected):
                continue
            if not has_pos(next_word.text, VERB_POSES):
                continue
            clean = text[start:end]
            dirty = "не" + next_word.text.lower()
            if not _same_rule_repairs(self, dirty, clean):
                continue
            corruptions.append(
                RuleCorruption(
                    start=start,
                    end=end,
                    replacement=_match_case(clean, dirty),
                    error_type=self.spec.edit_type,
                    group="ne_verb",
                    rule_id=self.spec.id,
                )
            )
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class SpellingPatternRule:
    wrong: str
    correct: str

    @property
    def spec(self) -> RuleSpec:
        return RuleSpec(
            id=f"pattern_{self.wrong}_{self.correct}",
            group=_pattern_group(self.wrong, self.correct),
            scope="token",
            edit_type="spelling",
            mode="deterministic",
            confidence=0.94,
            requires=("dictionary",),
            description="General Russian orthographic letter-combination repair.",
        )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        if self.wrong not in word:
            return []
        replacement = word.replace(self.wrong, self.correct, 1)
        candidate = _candidate_for_known_replacement(self.spec, word, replacement)
        return [candidate] if candidate else []

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower().replace("ё", "е")
            if self.correct not in word:
                continue
            dirty = word.replace(self.correct, self.wrong, 1)
            if not _same_rule_repairs(self, dirty, word):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class CyExceptionRule:
    spec: RuleSpec = RuleSpec(
        id="cy_exception",
        group="ci",
        scope="token",
        edit_type="spelling",
        mode="deterministic",
        confidence=0.94,
        requires=("dictionary",),
        description="Restore 'ы' after ц for common exception stems.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        if not word.startswith("ци"):
            return []
        replacement = "цы" + word[2:]
        if not replacement.startswith(CY_EXCEPTIONS):
            return []
        candidate = _candidate_for_known_replacement(self.spec, word, replacement)
        return [candidate] if candidate else []

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower()
            if not word.startswith(CY_EXCEPTIONS):
                continue
            dirty = "ци" + word[2:]
            if not _same_rule_repairs(self, dirty, word):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class SoftToHardSignRule:
    spec: RuleSpec = RuleSpec(
        id="soft_to_hard_sign",
        group="hard_sign",
        scope="token",
        edit_type="spelling",
        mode="deterministic",
        confidence=0.95,
        requires=("dictionary",),
        description="Replace an erroneous soft sign with hard sign after a prefix.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        if "ь" not in word:
            return []
        replacement = word.replace("ь", "ъ", 1)
        candidate = _candidate_for_known_replacement(self.spec, word, replacement)
        if _has_hard_sign_after_prefix(replacement) and candidate:
            return [candidate]
        return []

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower()
            if "ъ" not in word or not _has_hard_sign_after_prefix(word):
                continue
            dirty = word.replace("ъ", "ь", 1)
            if not _same_rule_repairs(self, dirty, word):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class MissingHardSignRule:
    spec: RuleSpec = RuleSpec(
        id="missing_hard_sign",
        group="hard_sign",
        scope="token",
        edit_type="spelling",
        mode="deterministic",
        confidence=0.95,
        requires=("dictionary",),
        description="Insert hard sign after a prefix before е, ю, я.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        candidates: list[RuleCandidate] = []
        for index, char in enumerate(word):
            if char not in HARD_SIGN_VOWELS:
                continue
            replacement = word[:index] + "ъ" + word[index:]
            candidate = _candidate_for_known_replacement(self.spec, word, replacement)
            if _has_hard_sign_after_prefix(replacement) and candidate:
                candidates.append(candidate)
        return candidates

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower()
            if "ъ" not in word or not _has_hard_sign_after_prefix(word):
                continue
            dirty = word.replace("ъ", "", 1)
            if not _same_rule_repairs(self, dirty, word):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class SdelatPrefixRule:
    spec: RuleSpec = RuleSpec(
        id="sdelat_prefix",
        group="prefix_z_s",
        scope="token",
        edit_type="spelling",
        mode="deterministic",
        confidence=0.95,
        requires=("dictionary",),
        description="Correct impossible initial зд- prefix in forms such as сделать.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        if not word.startswith("зд"):
            return []
        candidate = _candidate_for_known_replacement(self.spec, word, "с" + word[1:])
        return [candidate] if candidate else []

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower()
            if not word.startswith("сд"):
                continue
            dirty = "з" + word[1:]
            if not _same_rule_repairs(self, dirty, word):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class PrefixZToSRule:
    spec: RuleSpec = RuleSpec(
        id="prefix_z_to_s",
        group="prefix_z_s",
        scope="token",
        edit_type="spelling",
        mode="deterministic",
        confidence=0.95,
        requires=("dictionary",),
        description="Use с-final prefix before voiceless consonants.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        candidates: list[RuleCandidate] = []
        for z_prefix, s_prefix in PREFIX_Z_S_PAIRS:
            if not word.startswith(z_prefix):
                continue
            rest = word[len(z_prefix) :]
            if rest[:1] not in VOICELESS:
                continue
            candidate = _candidate_for_known_replacement(self.spec, word, s_prefix + rest)
            if candidate:
                candidates.append(candidate)
        return candidates

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower()
            for z_prefix, s_prefix in PREFIX_Z_S_PAIRS:
                if not word.startswith(s_prefix):
                    continue
                rest = word[len(s_prefix) :]
                if rest[:1] not in VOICELESS:
                    continue
                dirty = z_prefix + rest
                if _same_rule_repairs(self, dirty, word):
                    corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class PrefixSToZRule:
    spec: RuleSpec = RuleSpec(
        id="prefix_s_to_z",
        group="prefix_z_s",
        scope="token",
        edit_type="spelling",
        mode="deterministic",
        confidence=0.95,
        requires=("dictionary",),
        description="Use з-final prefix before voiced consonants, sonorants, and vowels.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        candidates: list[RuleCandidate] = []
        for z_prefix, s_prefix in PREFIX_Z_S_PAIRS:
            if not word.startswith(s_prefix):
                continue
            rest = word[len(s_prefix) :]
            if rest[:1] not in VOICED_OR_SONORANT_OR_VOWEL:
                continue
            candidate = _candidate_for_known_replacement(self.spec, word, z_prefix + rest)
            if candidate:
                candidates.append(candidate)
        return candidates

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower()
            for z_prefix, s_prefix in PREFIX_Z_S_PAIRS:
                if not word.startswith(z_prefix):
                    continue
                rest = word[len(z_prefix) :]
                if rest[:1] not in VOICED_OR_SONORANT_OR_VOWEL:
                    continue
                dirty = s_prefix + rest
                if _same_rule_repairs(self, dirty, word):
                    corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class TsyaSoftDeleteRule:
    spec: RuleSpec = RuleSpec(
        id="tsya_soft_delete",
        group="tsya",
        scope="token",
        edit_type="spelling",
        mode="model_required",
        confidence=0.9,
        requires=("morphology",),
        description="Generate -ться -> -тся candidate for verb forms.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        if not word.endswith("ться"):
            return []
        replacement = word[: -len("ться")] + "тся"
        if is_known_word(replacement) and has_pos(replacement, TSYA_CONTEXT_POSES):
            return [_candidate(self.spec, replacement, requires_model=is_known_word(word))]
        return []

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower()
            if not word.endswith("тся"):
                continue
            dirty = word[: -len("тся")] + "ться"
            if not _same_rule_repairs(self, dirty, word):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class TsyaSoftInsertRule:
    spec: RuleSpec = RuleSpec(
        id="tsya_soft_insert",
        group="tsya",
        scope="token",
        edit_type="spelling",
        mode="model_required",
        confidence=0.9,
        requires=("morphology",),
        description="Generate -тся -> -ться candidate for verb forms.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        if not word.endswith("тся"):
            return []
        replacement = word[: -len("тся")] + "ться"
        if is_known_word(replacement) and has_pos(replacement, TSYA_CONTEXT_POSES):
            return [_candidate(self.spec, replacement, requires_model=is_known_word(word))]
        return []

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower()
            if not word.endswith("ться"):
                continue
            dirty = word[: -len("ться")] + "тся"
            if not _same_rule_repairs(self, dirty, word):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class ContextPairRule:
    spec: RuleSpec = RuleSpec(
        id="context_pair",
        group="context_split_join",
        scope="span",
        edit_type="split_join",
        mode="candidate_only",
        confidence=0.0,
        requires=("model",),
        description="Context-dependent split/join whitelist pairs that need model confirmation.",
    )

    def generate_span(self, text: str, tokens: tuple[object, ...], token_index: int) -> Iterable[RuleEdit]:
        candidates: list[RuleEdit] = []
        for source, replacement in CONTEXT_DEPENDENT_WHITELIST.items():
            source_words = source.split()
            end_index = token_index + len(source_words)
            if end_index > len(tokens):
                continue
            span_words = tokens[token_index:end_index]
            if [getattr(word, "text", "").lower() for word in span_words] != source_words:
                continue
            start = int(getattr(span_words[0], "start"))
            end = int(getattr(span_words[-1], "end"))
            if text[start:end].lower().split() != source_words:
                continue
            candidates.append(
                RuleEdit(
                    source=text[start:end],
                    replacement=_match_case(text[start:end], replacement),
                    edit_type=self.spec.edit_type,
                    start=start,
                    end=end,
                    confidence=self.spec.confidence,
                    requires_model=True,
                    rule_id=self.spec.id,
                    mode=self.spec.mode,
                )
            )
        return candidates


@dataclass(frozen=True)
class HyphenParticleRule:
    spec: RuleSpec = RuleSpec(
        id="hyphen_particles",
        group="hyphen",
        scope="span",
        edit_type="hyphen",
        mode="candidate_only",
        confidence=0.95,
        requires=("model",),
        description="Generate hyphen candidates for pronominal words with -то, -либо, -нибудь.",
    )

    def generate_span(self, text: str, tokens: tuple[object, ...], token_index: int) -> Iterable[RuleEdit]:
        word_tokens = _word_tokens(text, tokens if tokens else None)
        if token_index < 0 or token_index + 1 >= len(word_tokens):
            return []
        first = word_tokens[token_index]
        second = word_tokens[token_index + 1]
        base = str(getattr(first, "text", "")).lower()
        particle = str(getattr(second, "text", "")).lower()
        if base not in HYPHEN_PARTICLE_BASES or particle not in HYPHEN_PARTICLES:
            return []
        return [_span_candidate(self.spec, text, first, second, f"{base}-{particle}")]

    def dirty_to_candidate(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
    ) -> Iterable[RuleEdit]:
        return _span_candidates_for_rule(self, text, tokens, token_index)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            parts = str(token.text).lower().split("-")
            if len(parts) != 2 or parts[0] not in HYPHEN_PARTICLE_BASES or parts[1] not in HYPHEN_PARTICLES:
                continue
            dirty = f"{parts[0]} {parts[1]}"
            if not _span_rule_repairs(self, text, token, dirty):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class HyphenKoeKoyRule:
    spec: RuleSpec = RuleSpec(
        id="hyphen_koe_koy",
        group="hyphen",
        scope="span",
        edit_type="hyphen",
        mode="candidate_only",
        confidence=0.95,
        requires=("model",),
        description="Generate hyphen candidates for кое-/кой- pronominal and adverbial words.",
    )

    def generate_span(self, text: str, tokens: tuple[object, ...], token_index: int) -> Iterable[RuleEdit]:
        word_tokens = _word_tokens(text, tokens if tokens else None)
        if token_index < 0 or token_index + 1 >= len(word_tokens):
            return []
        first = word_tokens[token_index]
        second = word_tokens[token_index + 1]
        prefix = str(getattr(first, "text", "")).lower()
        base = str(getattr(second, "text", "")).lower()
        if prefix not in KOE_KOY_PREFIXES or base not in KOE_KOY_BASES:
            return []
        return [_span_candidate(self.spec, text, first, second, f"{prefix}-{base}")]

    def dirty_to_candidate(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
    ) -> Iterable[RuleEdit]:
        return _span_candidates_for_rule(self, text, tokens, token_index)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            parts = str(token.text).lower().split("-")
            if len(parts) != 2 or parts[0] not in KOE_KOY_PREFIXES or parts[1] not in KOE_KOY_BASES:
                continue
            dirty = f"{parts[0]} {parts[1]}"
            if not _span_rule_repairs(self, text, token, dirty):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class HyphenPoAdverbRule:
    spec: RuleSpec = RuleSpec(
        id="hyphen_po_adverbs",
        group="hyphen",
        scope="span",
        edit_type="hyphen",
        mode="candidate_only",
        confidence=0.95,
        requires=("model",),
        description="Generate hyphen candidates for adverbs with по- and -ому/-ему/-ски/-цки/-ьи suffixes.",
    )

    def generate_span(self, text: str, tokens: tuple[object, ...], token_index: int) -> Iterable[RuleEdit]:
        word_tokens = _word_tokens(text, tokens if tokens else None)
        if token_index < 0 or token_index + 1 >= len(word_tokens):
            return []
        first = word_tokens[token_index]
        second = word_tokens[token_index + 1]
        prefix = str(getattr(first, "text", "")).lower()
        base = str(getattr(second, "text", "")).lower()
        if prefix != "по" or not _is_po_adverb_base(base):
            return []
        return [_span_candidate(self.spec, text, first, second, f"по-{base}")]

    def dirty_to_candidate(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
    ) -> Iterable[RuleEdit]:
        return _span_candidates_for_rule(self, text, tokens, token_index)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            parts = str(token.text).lower().split("-")
            if len(parts) != 2 or parts[0] != "по" or not _is_po_adverb_base(parts[1]):
                continue
            dirty = f"по {parts[1]}"
            if not _span_rule_repairs(self, text, token, dirty):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class PolPoluCompoundRule:
    spec: RuleSpec = RuleSpec(
        id="pol_polu_compounds",
        group="hyphen",
        scope="span",
        edit_type="hyphen",
        mode="model_required",
        confidence=0.9,
        requires=("dictionary", "morphology", "model"),
        description="Generate bounded pol-/polu- compound candidates that require model scoring.",
    )

    def generate_span(self, text: str, tokens: tuple[object, ...], token_index: int) -> Iterable[RuleEdit]:
        word_tokens = _word_tokens(text, tokens if tokens else None)
        if token_index < 0 or token_index + 1 >= len(word_tokens):
            return []
        first = word_tokens[token_index]
        second = word_tokens[token_index + 1]
        prefix = str(getattr(first, "text", "")).lower()
        base_source = str(getattr(second, "text", ""))
        base = base_source.lower()
        if prefix == "полу":
            joined = f"полу{base}"
            if is_known_word(joined):
                return [_span_candidate(self.spec, text, first, second, joined)]
            return []
        if prefix != "пол" or not _is_pol_base_candidate(base_source):
            return []
        if _requires_pol_hyphen(base_source):
            return [_span_candidate(self.spec, text, first, second, f"пол-{base}")]
        joined = f"пол{base}"
        if is_known_word(joined):
            return [_span_candidate(self.spec, text, first, second, joined)]
        return []

    def dirty_to_candidate(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
    ) -> Iterable[RuleEdit]:
        return _span_candidates_for_rule(self, text, tokens, token_index)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = str(token.text)
            lower = word.lower()
            dirty = ""
            if lower.startswith("пол-") and _is_pol_base_candidate(word[len("пол-") :]):
                dirty = "пол " + lower[len("пол-") :]
            elif lower.startswith("полу") and len(lower) > len("полу") and is_known_word(lower[len("полу") :]):
                dirty = "полу " + lower[len("полу") :]
            elif lower.startswith("пол") and len(lower) > len("пол") and _is_joined_pol_corruption_base(lower[len("пол") :]):
                dirty = "пол " + lower[len("пол") :]
            if not dirty or not _span_rule_repairs(self, text, token, dirty):
                continue
            corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class HyphenWhitelistRule:
    spec: RuleSpec = RuleSpec(
        id="hyphen_whitelist",
        group="hyphen",
        scope="span",
        edit_type="hyphen",
        mode="candidate_only",
        confidence=0.95,
        requires=("dictionary", "model"),
        description="Whitelist hyphen insertion/removal candidates that require scorer confirmation.",
    )

    def generate_span(self, text: str, tokens: tuple[object, ...], token_index: int) -> Iterable[RuleEdit]:
        candidates: list[RuleEdit] = []
        word_tokens = _word_tokens(text, tokens if tokens else None)
        for source, replacement in HYPHEN_WHITELIST.items():
            if source == replacement:
                continue
            source_words = source.split()
            end_index = token_index + len(source_words)
            if token_index < 0 or end_index > len(word_tokens):
                continue
            span_words = word_tokens[token_index:end_index]
            if [getattr(word, "text", "").lower() for word in span_words] != source_words:
                continue
            start = int(getattr(span_words[0], "start"))
            end = int(getattr(span_words[-1], "end"))
            if text[start:end].lower().split() != source_words:
                continue
            candidates.append(
                RuleEdit(
                    source=text[start:end],
                    replacement=_match_case(text[start:end], replacement),
                    edit_type=self.spec.edit_type,
                    start=start,
                    end=end,
                    confidence=self.spec.confidence,
                    requires_model=True,
                    rule_id=self.spec.id,
                    mode=self.spec.mode,
                )
            )
        return candidates


@dataclass(frozen=True)
class SentenceStartCaseRule:
    spec: RuleSpec = RuleSpec(
        id="sentence_start_case",
        group="capitalization",
        scope="token",
        edit_type="case",
        mode="deterministic",
        confidence=0.9,
        requires=("none",),
        description="Uppercase the first token at sentence start in the lightweight fallback.",
    )

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        if not context or context.token_index != 0 or not token[:1].islower():
            return []
        return [_candidate(self.spec, token[:1].upper() + token[1:], requires_model=False)]


def orthography_rules() -> tuple[object, ...]:
    return ORTHOGRAPHY_RULES


def token_rules() -> tuple[object, ...]:
    return tuple(rule for rule in ORTHOGRAPHY_RULES if rule.spec.scope == "token")


def span_rules() -> tuple[object, ...]:
    return tuple(rule for rule in ORTHOGRAPHY_RULES if rule.spec.scope == "span")


def _span_candidates_for_rule(
    rule: object,
    text: str,
    tokens: tuple[Any, ...] | None,
    token_index: int | None,
) -> Iterable[RuleEdit]:
    word_tokens = _word_tokens(text, tokens if tokens else None)
    indexes = range(len(word_tokens)) if token_index is None else (token_index,)
    candidates: list[RuleEdit] = []
    for index in indexes:
        if 0 <= index < len(word_tokens):
            candidates.extend(rule.generate_span(text, word_tokens, index))
    return candidates


def _span_candidate(spec: RuleSpec, text: str, first: Any, last: Any, replacement: str) -> RuleEdit:
    start = int(first.start)
    end = int(last.end)
    source = text[start:end]
    return RuleEdit(
        source=source,
        replacement=_match_case(source, replacement),
        edit_type=spec.edit_type,
        start=start,
        end=end,
        confidence=spec.confidence,
        requires_model=True,
        rule_id=spec.id,
        mode=spec.mode,
    )


def _span_rule_repairs(rule: object, clean_text: str, clean_token: Any, dirty_replacement: str) -> bool:
    start = int(clean_token.start)
    end = int(clean_token.end)
    clean = clean_text[start:end]
    dirty = _match_case(clean, dirty_replacement)
    dirty_text = clean_text[:start] + dirty + clean_text[end:]
    for candidate in _span_candidates_for_rule(rule, dirty_text, None, None):
        if candidate.rule_id == rule.spec.id and candidate.replacement.lower() == clean.lower():
            return True
    return False


def _word_tokens(text: str, tokens: tuple[Any, ...] | None) -> tuple[Any, ...]:
    return tokens if tokens is not None else tuple(tokenize_words(text))


def _iter_token_positions(tokens: tuple[Any, ...], token_index: int | None) -> Iterable[tuple[int, Any]]:
    if token_index is None:
        return tuple(enumerate(tokens))
    if 0 <= token_index < len(tokens):
        return ((token_index, tokens[token_index]),)
    return ()


def _same_rule_repairs(rule: object, dirty: str, clean: str) -> bool:
    generator = getattr(rule, "generate_candidates", getattr(rule, "generate", None))
    if not generator:
        return False
    for candidate in generator(dirty, RuleContext()):
        if candidate.rule_id == rule.spec.id and candidate.replacement.lower() == clean.lower():
            return True
    return False


def _token_corruption(spec: RuleSpec, token: Any, dirty: str, error_type: str, group: str) -> RuleCorruption:
    return RuleCorruption(
        start=int(token.start),
        end=int(token.end),
        replacement=_match_case(str(token.text), dirty),
        error_type=error_type,
        group=group,
        rule_id=spec.id,
    )


def _span_overlaps_protected(start: int, end: int, protected: tuple[tuple[int, int], ...]) -> bool:
    return any(start < protected_end and protected_start < end for protected_start, protected_end in protected)


def _candidate(
    spec: RuleSpec,
    replacement: str,
    *,
    edit_type: str | None = None,
    requires_model: bool,
    mode: RuleMode | None = None,
) -> RuleCandidate:
    candidate_mode = mode or spec.mode
    return RuleCandidate(
        replacement=replacement,
        edit_type=edit_type or spec.edit_type,
        confidence=spec.confidence,
        requires_model=requires_model or candidate_mode in {"candidate_only", "model_required"},
        rule_id=spec.id,
        mode=candidate_mode,
    )


def _candidate_for_known_replacement(spec: RuleSpec, source: str, replacement: str) -> RuleCandidate | None:
    if source == replacement or not is_known_word(replacement):
        return None
    return _candidate(spec, replacement, requires_model=is_known_word(source))


def _has_hard_sign_after_prefix(word: str) -> bool:
    if "ъ" not in word:
        return False
    index = word.find("ъ")
    before = word[:index]
    after = word[index + 1 : index + 2]
    return after in HARD_SIGN_VOWELS and any(before.endswith(prefix) for prefix in HARD_SIGN_PREFIXES)


def _is_po_adverb_base(word: str) -> bool:
    return len(word) > 2 and word.endswith(PO_ADVERB_SUFFIXES)


def _is_pol_base_candidate(word: str) -> bool:
    return bool(word) and has_pos(word.lower(), POL_NOUN_POSES)


def _requires_pol_hyphen(word: str) -> bool:
    lower = word.lower()
    return bool(lower) and (lower[0] in POL_VOWELS or lower.startswith("л") or word[:1].isupper())


def _is_joined_pol_corruption_base(rest: str) -> bool:
    return bool(rest) and rest[0] not in POL_VOWELS and not rest.startswith("л") and is_known_word(rest)


def _pattern_group(wrong: str, correct: str) -> str:
    if wrong == "цы" or correct == "цы":
        return "ci"
    if wrong in {"жо", "шо", "чо", "що"}:
        return "hissing_o_e"
    return "combo"


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


ORTHOGRAPHY_RULES: tuple[object, ...] = (
    FrequentErrorRule(),
    NeVerbRule(),
    FrequentSplitJoinRule(),
    FrequentModelRequiredRule(),
    *(SpellingPatternRule(wrong, correct) for wrong, correct in SPELLING_PATTERNS.items()),
    CyExceptionRule(),
    SoftToHardSignRule(),
    MissingHardSignRule(),
    SdelatPrefixRule(),
    PrefixZToSRule(),
    PrefixSToZRule(),
    TsyaSoftDeleteRule(),
    TsyaSoftInsertRule(),
    ContextPairRule(),
    HyphenParticleRule(),
    HyphenKoeKoyRule(),
    HyphenPoAdverbRule(),
    PolPoluCompoundRule(),
    HyphenWhitelistRule(),
    SentenceStartCaseRule(),
)
