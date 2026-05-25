from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from src.schema.lexical_resources import (
    HYPHEN_WHITELIST,
    WRONG_TO_CORRECT,
)
from src.runtime.morphology import has_pos, is_known_word, normal_forms, parses
from src.preprocessing.tokenizer import tokenize_words
from src.rules.base import RuleCandidate, RuleContext, RuleCorruption, RuleEdit, RuleMode, RuleSpec
from src.rules.syntax_orthography import syntax_orthography_rules


VERB_POSES = frozenset({"VERB", "INFN"})
TSYA_CONTEXT_POSES = frozenset({"VERB", "INFN"})
NE_ADJECTIVE_POSES = frozenset({"ADJF", "ADJS"})
NE_PARTICIPLE_POSES = frozenset({"PRTF", "PRTS"})
NE_ADVERB_POSES = frozenset({"ADVB"})
N_NN_ADJECTIVE_POSES = frozenset({"ADJF", "ADJS"})
N_NN_PARTICIPLE_POSES = frozenset({"PRTF", "PRTS"})
CONTEXT_REQUIRES = ("syntax", "model")
NE_REQUIRES = ("morphology", "syntax", "model")
N_NN_REQUIRES = ("morphology", "syntax", "dictionary", "model")
PREFIX_PRE_PRI_REQUIRES = ("dictionary", "model")
NI_STABLE_EXPRESSION_PAIRS = (
    ("не разу", "ни разу"),
    ("ни разу", "не разу"),
    ("не в коем случае", "ни в коем случае"),
    ("ни в коем случае", "не в коем случае"),
    ("во что бы то не стало", "во что бы то ни стало"),
    ("во что бы то ни стало", "во что бы то не стало"),
)
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
SENTENCE_START_PREFIX_CHARS = frozenset(" \t\r\n\"'«„“([{—")
SENTENCE_START_ABBREVIATIONS = frozenset({"г", "см", "ул", "стр", "рис", "тыс", "млн", "млрд", "руб", "коп"})
SENTENCE_TERMINATORS = frozenset(".!?")
LATIN_COMPANY_ABBREVIATION_RE = re.compile(r"\b(?:co|inc|ltd|corp)\.\s*$", re.IGNORECASE)
RUSSIAN_GRAPHIC_ABBREVIATION_RE = re.compile(r"(?:\b(?:г|см|ул|стр|рис|тыс|млн|млрд|руб|коп)\.|(?:т\.д|т\.п|и\.о)\.)\s*$", re.IGNORECASE)
INITIAL_ABBREVIATIONS = {
    "сша": "США",
    "рф": "РФ",
    "нбб": "НББ",
    "ооо": "ООО",
    "ао": "АО",
    "ип": "ИП",
}
NER_CAPITALIZATION_TYPES = frozenset({"PER", "LOC", "ORG"})
SCORING_REQUIRED_RULE_IDS = frozenset(
    {
        "frequent_error_exact",
        "dictionary_fuzzy",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "swapped_letters_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
        "yo_e_candidate",
        "ne_verb",
        "ne_adjective",
        "ne_participle",
        "ne_adverb",
        "ne_short_form",
        "ne_predicative",
        "ni_stable_expression",
        "ni_particle_context",
        "n_nn_adjective",
        "n_nn_participle",
        "n_nn_deverbal_adjective",
        "n_nn_short_form",
        "prefix_pre_pri",
        "tsya_soft_delete",
        "tsya_soft_insert",
        "context_pair",
        "context_tak_zhe",
        "context_to_zhe",
        "context_chto_by",
        "context_za_to",
        "context_vsledstvie",
        "context_nesmotrya",
        "hyphen_whitelist",
        "hyphen_particles",
        "hyphen_koe_koy",
        "hyphen_po_adverbs",
        "pol_polu_compounds",
        "capitalization_sentence_start",
        "capitalization_ner",
        "abbreviation_case_protection",
    }
)


@dataclass(frozen=True)
class DictionaryCandidateMetadataRule:
    rule_id: str
    group: str
    description: str

    @property
    def spec(self) -> RuleSpec:
        return RuleSpec(
            id=self.rule_id,
            group=self.group,
            scope="token",
            edit_type="spelling",
            mode="model_required",
            confidence=0.0,
            requires=("dictionary", "model"),
            description=self.description,
        )


@dataclass(frozen=True)
class FrequentErrorRule:
    spec: RuleSpec = RuleSpec(
        id="frequent_error_exact",
        group="dictionary_spelling",
        scope="token",
        edit_type="spelling",
        mode="candidate_only",
        confidence=0.95,
        requires=("dictionary", "model"),
        description="Exact whitelist fallback for frequent dictionary spelling and split/join errors.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        lower = token.lower()
        replacement = WRONG_TO_CORRECT.get(lower)
        if not replacement:
            return []
        edit_type = "split_join" if " " in replacement else "spelling"
        return [_candidate(self.spec, replacement, edit_type=edit_type, requires_model=True)]

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
        mode="model_required",
        confidence=0.35,
        requires=NE_REQUIRES,
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
class NePartOfSpeechRule:
    rule_id: str
    poses: frozenset[str]
    description: str

    @property
    def spec(self) -> RuleSpec:
        return RuleSpec(
            id=self.rule_id,
            group="ne",
            scope="span",
            edit_type="split_join",
            mode="model_required",
            confidence=0.35,
            requires=NE_REQUIRES,
            description=self.description,
        )

    def generate_span(self, text: str, tokens: tuple[object, ...], token_index: int) -> Iterable[RuleEdit]:
        word_tokens = _word_tokens(text, tokens if tokens else None)
        if token_index < 0 or token_index >= len(word_tokens):
            return []

        token = word_tokens[token_index]
        word = str(getattr(token, "text", "")).lower()
        candidates: list[RuleEdit] = []

        if word.startswith("не") and len(word) > 4 and is_known_word(word):
            base = word[2:]
            if _has_known_pos(base, self.poses):
                candidates.append(_span_candidate(self.spec, text, token, token, f"не {base}"))

        if word == "не" and token_index + 1 < len(word_tokens):
            next_token = word_tokens[token_index + 1]
            base = str(getattr(next_token, "text", "")).lower()
            joined = f"не{base}"
            if _has_known_pos(base, self.poses) and is_known_word(joined):
                candidates.append(_span_candidate(self.spec, text, token, next_token, joined))

        return candidates

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        word_tokens = _word_tokens(text, tokens)
        corruptions: list[RuleCorruption] = []
        for index, token in _iter_token_positions(word_tokens, token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = str(getattr(token, "text", "")).lower()
            if word.startswith("не") and len(word) > 4 and is_known_word(word):
                base = word[2:]
                dirty = f"не {base}"
                if _has_known_pos(base, self.poses) and _span_rule_repairs(self, text, token, dirty):
                    corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.edit_type, "ne_pos"))
                continue
            if word != "не" or index + 1 >= len(word_tokens):
                continue
            next_token = word_tokens[index + 1]
            start, end = int(token.start), int(next_token.end)
            if _span_overlaps_protected(start, end, protected):
                continue
            base = str(getattr(next_token, "text", "")).lower()
            dirty = f"не{base}"
            if (
                _has_known_pos(base, self.poses)
                and is_known_word(dirty)
                and _span_rule_repairs_span(self, text, start, end, dirty)
            ):
                corruptions.append(
                    RuleCorruption(
                        start=start,
                        end=end,
                        replacement=_match_case(text[start:end], dirty),
                        error_type=self.spec.edit_type,
                        group="ne_pos",
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
class NiStableExpressionRule:
    spec: RuleSpec = RuleSpec(
        id="ni_stable_expression",
        group="ne_ni",
        scope="span",
        edit_type="split_join",
        mode="model_required",
        confidence=0.0,
        requires=NE_REQUIRES,
        description="Generate bounded не/ни candidates for stable constructions.",
    )

    def generate_span(self, text: str, tokens: tuple[object, ...], token_index: int) -> Iterable[RuleEdit]:
        word_tokens = _word_tokens(text, tokens if tokens else None)
        candidates: list[RuleEdit] = []
        for source, replacement in NI_STABLE_EXPRESSION_PAIRS:
            candidate = _phrase_span_candidate(self.spec, text, word_tokens, token_index, source, replacement)
            if candidate:
                candidates.append(candidate)
        return candidates

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        del tokens, token_index
        return _phrase_corruptions_for_pairs(self, text, NI_STABLE_EXPRESSION_PAIRS, "ne_ni", protected)

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class NnRule:
    rule_id: str
    description: str

    @property
    def spec(self) -> RuleSpec:
        return RuleSpec(
            id=self.rule_id,
            group="n_nn",
            scope="token",
            edit_type="spelling",
            mode="model_required",
            confidence=0.35,
            requires=N_NN_REQUIRES,
            description=self.description,
        )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        del context
        word = token.lower()
        candidates: list[RuleCandidate] = []
        for replacement in _n_nn_variants(word):
            if not is_known_word(replacement) or not self._matches(word, replacement):
                continue
            candidates.append(_candidate(self.spec, replacement, requires_model=True))
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
            word = str(getattr(token, "text", "")).lower()
            for dirty in _n_nn_variants(word):
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

    def _matches(self, source: str, replacement: str) -> bool:
        if self.rule_id == "n_nn_short_form":
            return _is_short_n_nn_form(replacement)
        if self.rule_id == "n_nn_participle":
            return _has_known_pos(replacement, N_NN_PARTICIPLE_POSES) and not _is_short_n_nn_form(replacement)
        if self.rule_id == "n_nn_deverbal_adjective":
            return _has_known_pos(source, N_NN_PARTICIPLE_POSES) and _has_known_pos(replacement, N_NN_ADJECTIVE_POSES)
        if self.rule_id == "n_nn_adjective":
            return _has_known_pos(replacement, N_NN_ADJECTIVE_POSES) and not _has_known_pos(
                replacement,
                N_NN_PARTICIPLE_POSES,
            )
        return False


@dataclass(frozen=True)
class PrefixPrePriRule:
    spec: RuleSpec = RuleSpec(
        id="prefix_pre_pri",
        group="prefix_pre_pri",
        scope="token",
        edit_type="spelling",
        mode="model_required",
        confidence=0.35,
        requires=PREFIX_PRE_PRI_REQUIRES,
        description="Generate dictionary-backed candidates for initial при-/пре- alternations.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        del context
        word = token.lower()
        replacement = _toggle_pre_pri(word)
        if not replacement or not is_known_word(replacement):
            return []
        return [_candidate(self.spec, replacement, requires_model=True)]

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
            word = str(getattr(token, "text", "")).lower()
            dirty = _toggle_pre_pri(word)
            if dirty and _same_rule_repairs(self, dirty, word):
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
        requires=NE_REQUIRES,
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
        requires=NE_REQUIRES,
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
    rule_id: str
    pairs: tuple[tuple[str, str], ...]

    @property
    def spec(self) -> RuleSpec:
        return RuleSpec(
            id=self.rule_id,
            group="context_split_join",
            scope="span",
            edit_type="split_join",
            mode="model_required",
            confidence=0.0,
            requires=CONTEXT_REQUIRES,
            description="Context-dependent split/join candidates that need model confirmation.",
        )

    def generate_span(self, text: str, tokens: tuple[object, ...], token_index: int) -> Iterable[RuleEdit]:
        candidates: list[RuleEdit] = []
        word_tokens = _word_tokens(text, tokens if tokens else None)
        for source, replacement in self.pairs:
            candidate = _phrase_span_candidate(self.spec, text, word_tokens, token_index, source, replacement)
            if candidate:
                candidates.append(candidate)
        return candidates

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        del tokens, token_index
        return _phrase_corruptions_for_pairs(self, text, self.pairs, "context_pairs", protected)

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


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
        id="capitalization_sentence_start",
        group="capitalization",
        scope="token",
        edit_type="case",
        mode="candidate_only",
        confidence=0.9,
        requires=("model",),
        description="Generate a bounded uppercase candidate for real sentence starts.",
    )

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        if not context or context.token_index < 0:
            return []
        token_info = context.tokens[context.token_index] if context.token_index < len(context.tokens) else None
        if token_info is None:
            return []
        start = int(getattr(token_info, "start", -1))
        end = int(getattr(token_info, "end", -1))
        if not is_safe_sentence_start_case(context.text, start, end, token, context.protected_spans):
            return []
        return [_candidate(self.spec, token[:1].upper() + token[1:], requires_model=True)]


@dataclass(frozen=True)
class CapitalizationNerRule:
    spec: RuleSpec = RuleSpec(
        id="capitalization_ner",
        group="capitalization",
        scope="token",
        edit_type="case",
        mode="model_required",
        confidence=0.35,
        requires=("ner", "syntax", "model"),
        description="Generate NER-backed capitalization candidates for proper names.",
    )

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        if not context or context.token_index < 0:
            return []
        token_info = context.tokens[context.token_index] if context.token_index < len(context.tokens) else None
        if token_info is None:
            return []
        start = int(getattr(token_info, "start", -1))
        end = int(getattr(token_info, "end", -1))
        if not _is_safe_capitalization_source(context.text, start, end, token, context.protected_spans):
            return []
        syntax_token = _matching_syntax_token(context.syntax_tokens, start, end)
        if syntax_token is None or str(getattr(syntax_token, "ner", "") or "") not in NER_CAPITALIZATION_TYPES:
            return []
        if str(getattr(syntax_token, "pos", "") or "").upper() != "PROPN":
            return []
        return [_candidate(self.spec, token[:1].upper() + token[1:], requires_model=True)]


@dataclass(frozen=True)
class AbbreviationCaseProtectionRule:
    spec: RuleSpec = RuleSpec(
        id="abbreviation_case_protection",
        group="abbreviations",
        scope="token",
        edit_type="case",
        mode="model_required",
        confidence=0.5,
        requires=("dictionary", "model"),
        description="Generate case-only candidates for known initial abbreviations without punctuation edits.",
    )

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        if not context or context.token_index < 0:
            return []
        token_info = context.tokens[context.token_index] if context.token_index < len(context.tokens) else None
        if token_info is None:
            return []
        start = int(getattr(token_info, "start", -1))
        end = int(getattr(token_info, "end", -1))
        if _span_overlaps_protected(start, end, context.protected_spans):
            return []
        replacement = INITIAL_ABBREVIATIONS.get(token.lower())
        if not replacement or token == replacement:
            return []
        return [_candidate(self.spec, replacement, requires_model=True)]


def is_safe_sentence_start_case(
    text: str,
    start: int,
    end: int,
    token: str,
    protected_spans: tuple[tuple[int, int], ...] = (),
) -> bool:
    if not _is_safe_capitalization_source(text, start, end, token, protected_spans):
        return False

    previous = _previous_nonspace_index(text, start)
    if previous is None:
        return True
    if text[previous] not in SENTENCE_TERMINATORS:
        return False
    if _position_inside_spans(previous, protected_spans):
        return False
    prefix = text[max(0, previous - 48) : previous + 1]
    if LATIN_COMPANY_ABBREVIATION_RE.search(prefix) or RUSSIAN_GRAPHIC_ABBREVIATION_RE.search(prefix):
        return False
    return True


def _is_safe_capitalization_source(
    text: str,
    start: int,
    end: int,
    token: str,
    protected_spans: tuple[tuple[int, int], ...] = (),
) -> bool:
    if start < 0 or end < start or end > len(text) or not token[:1].islower():
        return False
    if _span_overlaps_protected(start, end, protected_spans):
        return False
    if token.isupper() or _looks_like_abbreviation_token(text, start, end, token):
        return False
    if token.lower() in INITIAL_ABBREVIATIONS:
        return False
    if _looks_like_technical_token(token):
        return False
    if start > 0 and (text[start - 1].isdigit() or text[start - 1] == "-"):
        return False
    return True


def _looks_like_abbreviation_token(text: str, start: int, end: int, token: str) -> bool:
    del start
    return token.lower() in SENTENCE_START_ABBREVIATIONS and end < len(text) and text[end : end + 1] == "."


def _previous_nonspace_index(text: str, position: int) -> int | None:
    index = position - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    return index if index >= 0 else None


def _position_inside_spans(position: int, spans: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= position < end for start, end in spans)


def _looks_like_technical_token(token: str) -> bool:
    return any(char.isdigit() for char in token) and any(char.isalpha() for char in token)


def _matching_syntax_token(syntax_tokens: tuple[Any, ...], start: int, end: int) -> Any | None:
    return next(
        (
            token
            for token in syntax_tokens
            if int(getattr(token, "start", -1)) == start and int(getattr(token, "end", -1)) == end
        ),
        None,
    )


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
        group=spec.group,
        requires=spec.requires,
    )


def _phrase_span_candidate(
    spec: RuleSpec,
    text: str,
    tokens: tuple[Any, ...],
    token_index: int,
    source: str,
    replacement: str,
) -> RuleEdit | None:
    source_words = source.split()
    end_index = token_index + len(source_words)
    if token_index < 0 or end_index > len(tokens):
        return None
    span_words = tokens[token_index:end_index]
    if [str(getattr(word, "text", "")).lower() for word in span_words] != source_words:
        return None
    start = int(getattr(span_words[0], "start"))
    end = int(getattr(span_words[-1], "end"))
    if text[start:end].lower().split() != source_words:
        return None
    return RuleEdit(
        source=text[start:end],
        replacement=_match_case(text[start:end], replacement),
        edit_type=spec.edit_type,
        start=start,
        end=end,
        confidence=spec.confidence,
        requires_model=True,
        rule_id=spec.id,
        mode=spec.mode,
        group=spec.group,
        requires=spec.requires,
    )


def _span_rule_repairs(rule: object, clean_text: str, clean_token: Any, dirty_replacement: str) -> bool:
    start = int(clean_token.start)
    end = int(clean_token.end)
    return _span_rule_repairs_span(rule, clean_text, start, end, dirty_replacement)


def _span_rule_repairs_span(rule: object, clean_text: str, start: int, end: int, dirty_replacement: str) -> bool:
    clean = clean_text[start:end]
    dirty = _match_case(clean, dirty_replacement)
    dirty_text = clean_text[:start] + dirty + clean_text[end:]
    for candidate in _span_candidates_for_rule(rule, dirty_text, None, None):
        if candidate.rule_id == rule.spec.id and candidate.replacement.lower() == clean.lower():
            return True
    return False


def _phrase_corruptions_for_pairs(
    rule: object,
    text: str,
    pairs: tuple[tuple[str, str], ...],
    group: str,
    protected: tuple[tuple[int, int], ...],
) -> list[RuleCorruption]:
    corruptions: list[RuleCorruption] = []
    lower = text.lower()
    seen: set[tuple[int, int, str]] = set()
    for dirty, clean in pairs:
        start = lower.find(clean)
        if start < 0:
            continue
        end = start + len(clean)
        if _span_overlaps_protected(start, end, protected):
            continue
        key = (start, end, dirty)
        if key in seen or not _span_rule_repairs_span(rule, text, start, end, dirty):
            continue
        seen.add(key)
        corruptions.append(
            RuleCorruption(
                start=start,
                end=end,
                replacement=_match_case(text[start:end], dirty),
                error_type=rule.spec.edit_type,
                group=group,
                rule_id=rule.spec.id,
            )
        )
    return corruptions


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
        group=spec.group,
        requires=spec.requires,
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


def _n_nn_variants(word: str) -> tuple[str, ...]:
    variants: list[str] = []
    seen: set[str] = set()
    for index, char in enumerate(word):
        if char != "н":
            continue
        if word[index : index + 2] == "нн":
            variant = word[:index] + "н" + word[index + 2 :]
        elif (index == 0 or word[index - 1] != "н") and (index + 1 == len(word) or word[index + 1] != "н"):
            variant = word[:index] + "нн" + word[index + 1 :]
        else:
            continue
        if variant != word and variant not in seen:
            seen.add(variant)
            variants.append(variant)
    return tuple(variants)


def _has_known_pos(word: str, poses: frozenset[str]) -> bool:
    return any(getattr(parse, "is_known", False) and getattr(parse.tag, "POS", None) in poses for parse in parses(word))


def _is_short_n_nn_form(word: str) -> bool:
    return _has_known_pos(word, frozenset({"ADJS", "PRTS"}))


def _toggle_pre_pri(word: str) -> str:
    if word.startswith("при") and len(word) > 4:
        return "пре" + word[3:]
    if word.startswith("пре") and len(word) > 4:
        return "при" + word[3:]
    return ""


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
    DictionaryCandidateMetadataRule(
        "dictionary_fuzzy",
        "dictionary_model_required",
        "Lexicon-backed fuzzy spelling candidates that require model scoring.",
    ),
    DictionaryCandidateMetadataRule(
        "double_consonant_candidate",
        "double_consonants",
        "Lexicon-backed one/double consonant candidates that require model scoring.",
    ),
    DictionaryCandidateMetadataRule(
        "keyboard_typo_candidate",
        "typos",
        "Lexicon-backed keyboard-neighbor typo candidates that require model scoring.",
    ),
    DictionaryCandidateMetadataRule(
        "swapped_letters_candidate",
        "typos",
        "Lexicon-backed adjacent-letter swap candidates that require model scoring.",
    ),
    DictionaryCandidateMetadataRule(
        "missing_letter_candidate",
        "typos",
        "Lexicon-backed missing-letter candidates that require model scoring.",
    ),
    DictionaryCandidateMetadataRule(
        "extra_letter_candidate",
        "typos",
        "Lexicon-backed extra-letter candidates that require model scoring.",
    ),
    DictionaryCandidateMetadataRule(
        "yo_e_candidate",
        "dictionary_model_required",
        "Opt-in lexicon-backed е/ё candidates that require model scoring.",
    ),
    *syntax_orthography_rules(),
    PrefixPrePriRule(),
    FrequentErrorRule(),
    *(SpellingPatternRule(wrong, correct) for wrong, correct in SPELLING_PATTERNS.items()),
    CyExceptionRule(),
    SoftToHardSignRule(),
    MissingHardSignRule(),
    SdelatPrefixRule(),
    PrefixZToSRule(),
    PrefixSToZRule(),
    HyphenParticleRule(),
    HyphenKoeKoyRule(),
    HyphenPoAdverbRule(),
    PolPoluCompoundRule(),
    HyphenWhitelistRule(),
    AbbreviationCaseProtectionRule(),
    CapitalizationNerRule(),
    SentenceStartCaseRule(),
)
