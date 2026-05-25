from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.candidates.frequent_errors import CONTEXT_DEPENDENT_WHITELIST, WRONG_TO_CORRECT
from src.candidates.morphology import parses
from src.preprocessing.protected_spans import ProtectedSpan, find_protected_spans
from src.preprocessing.tokenizer import tokenize_words
from src.rules.orthography import SCORING_REQUIRED_RULE_IDS, VERB_POSES, is_safe_sentence_start_case
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import is_allowed_edit_type


TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE: dict[str, tuple[str, str]] = {
    "context_chto_by": ("что бы", "чтобы"),
    "context_tak_zhe": ("так же", "также"),
    "context_to_zhe": ("то же", "тоже"),
    "context_za_to": ("за то", "зато"),
    "context_nesmotrya": ("не смотря", "несмотря"),
    "context_vsledstvie": ("в следствие", "вследствие"),
}


@dataclass
class ValidationResult:
    source: str
    target: str
    edits: list[Edit]

    @property
    def accepted_edits(self) -> list[Edit]:
        return [edit for edit in self.edits if edit.status == "accepted"]

    @property
    def rejected_edits(self) -> list[Edit]:
        return [edit for edit in self.edits if edit.status == "rejected"]

    def apply_accepted(self) -> str:
        if not self.rejected_edits:
            return self.target

        text = self.source
        positioned_edits: list[Edit] = []
        fallback_edits: list[Edit] = []
        for edit in self.accepted_edits:
            if _is_positioned_edit(edit):
                positioned_edits.append(edit)
            else:
                fallback_edits.append(edit)

        for edit in sorted(positioned_edits, key=lambda item: (item.start, item.end), reverse=True):
            text = _apply_positioned_edit(text, edit)

        for edit in fallback_edits:
            if edit.edit_type == "punctuation_insert" and edit.replacement == ",":
                text = _insert_comma_before_common_subordinator(text)
        return text


class StrictValidator:
    def __init__(self, context_pair_threshold: float = 0.98, tsya_threshold: float | None = 0.98) -> None:
        self.diff_analyzer = DiffAnalyzer()
        self.context_pair_threshold = context_pair_threshold
        self.tsya_threshold = context_pair_threshold if tsya_threshold is None else tsya_threshold

    def pre_validate(self, text: str) -> list[ProtectedSpan]:
        return find_protected_spans(text)

    def validate(
        self,
        source: str,
        target: str,
        trusted_edits: list[Any] | None = None,
        *,
        allow_training_context_pairs: bool = False,
        expected_rule_id: str = "",
    ) -> ValidationResult:
        edits = self.diff_analyzer.analyze(source, target)
        protected = self.pre_validate(source)
        trusted = trusted_edits or []
        trusted_context_keys = _trusted_context_pair_keys(trusted, self.context_pair_threshold)
        training_context_keys = (
            _trusted_training_context_pair_keys(source, trusted, expected_rule_id)
            if allow_training_context_pairs
            else set()
        )
        trusted_strict_edits = _trusted_strict_edits(source, trusted)
        trusted_tsya_keys = _trusted_tsya_keys(source, trusted, self.tsya_threshold)
        validated: list[Edit] = []

        for edit in trusted_strict_edits:
            guard_reason = _guard_rejection_reason(source, target, edit, protected, trusted_tsya_keys)
            if guard_reason:
                validated.append(edit.with_status("rejected", guard_reason))
            else:
                validated.append(edit.with_status("accepted", "trusted bounded candidate"))

        for edit in edits:
            if _is_explained_by_trusted_edit(edit, trusted_strict_edits):
                continue
            guard_reason = _guard_rejection_reason(source, target, edit, protected, trusted_tsya_keys)
            if guard_reason:
                validated.append(edit.with_status("rejected", guard_reason))
                continue
            if _is_context_dependent_edit(edit):
                if _has_trusted_context_pair_key(edit, training_context_keys):
                    validated.append(edit.with_status("accepted", "trusted training context pair"))
                elif _has_trusted_context_pair_key(edit, trusted_context_keys):
                    if _passes_context_pair_guard(source, edit):
                        validated.append(edit.with_status("accepted", "trusted high-confidence context pair"))
                    else:
                        validated.append(edit.with_status("rejected", "context_pair_unsafe"))
                else:
                    validated.append(edit.with_status("rejected", "context-dependent pair requires trusted model confidence"))
            elif edit.edit_type == "final_punctuation":
                validated.append(edit.with_status("rejected", "requires_trusted_candidate"))
            elif _is_scoring_required_rule_edit(edit):
                validated.append(edit.with_status("rejected", "requires_trusted_candidate"))
            elif is_allowed_edit_type(edit.edit_type):
                validated.append(edit.with_status("accepted", "allowed strict-scope edit"))
            else:
                validated.append(edit.with_status("rejected", "outside strict spelling/punctuation scope"))

        return ValidationResult(source=source, target=target, edits=validated)


PERCENT_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?%")
NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,:/-]\d+)*")
DECIMAL_RE = re.compile(r"(?<![\w])\d+[.,]\d+(?![\w])")
INITIAL_ABBREVIATION_RE = re.compile(r"\b(?:УФСБ|РИА|США|РФ|НББ|ООО|АО|ИП)\b")
ABBREVIATION_RE = re.compile(
    r"(?:\b\d{4}\s+[гГ]\.|\b(?:см|т\.д|т\.п|ул|стр|рис|г)\.|№\s*\d+)",
    re.IGNORECASE,
)
PUNCTUATION_NOISE_RE = re.compile(
    r"(?:…[.!?…]+|[.!?]+…|[!?]\.|(?<!\.)\.\.(?!\.)|\.{4,}|[!?]{2,}|([,;:])\s*\1)"
)
UNSAFE_FINAL_DOT_TAIL_RE = re.compile(r"(?::\)|:\(|;\)|[,;:!?…])$")
LATIN_COMPANY_ABBREVIATION_RE = re.compile(r"\b(?:co|inc|ltd|corp)\.\s*$", re.IGNORECASE)
LATIN_RE = re.compile(r"[A-Za-z]")
RUSSIAN_LETTER_RE = re.compile(r"[А-Яа-яЁё]")
QUOTE_NORMALIZATION_RULE_IDS = frozenset({"quote_open", "quote_close"})
N_NN_RULE_IDS = frozenset({"n_nn_adjective", "n_nn_participle", "n_nn_deverbal_adjective", "n_nn_short_form"})
PROTECTED_N_NN_CLEAN_FORMS = frozenset({("намерены", "намеренны")})
NE_SPLIT_JOIN_RULE_IDS = frozenset({"ne_verb", "ne_adjective", "ne_adverb", "ne_participle", "ne_short_form", "ne_predicative"})
NI_RULE_IDS = frozenset({"ni_stable_expression", "ni_particle_context"})
DIRECT_SPEECH_RULE_IDS = frozenset({"direct_speech_colon", "direct_speech_quotes", "direct_speech_dash"})
SYNTAX_PUNCTUATION_RULE_IDS = frozenset(
    {
        "comma_subordinate",
        "comma_conjunction",
        "introductory_comma",
        "address_comma",
        "homogeneous_comma",
        "detached_adverbial_comma",
        "detached_participial_comma",
        "apposition_comma",
        "clarification_comma",
        "comparative_turnover_comma",
        "subject_predicate_dash",
        "enumeration_colon",
        "enumeration_dash",
        "explanation_colon",
        "consequence_dash",
        "asyndetic_dash",
        "semicolon",
        "direct_speech_colon",
        "direct_speech_quotes",
        "direct_speech_dash",
    }
)
ASYNDETIC_RULE_IDS = frozenset({"asyndetic_dash", "consequence_dash", "explanation_colon", "semicolon"})
DISCOURSE_DASH_BLOCKERS = frozenset({"получается", "значит"})
CAPITALIZATION_NER_MIN_CONFIDENCE = 0.999999
PROTECTED_ACRONYMS = frozenset({"УФСБ", "РИА", "США", "РФ", "НББ", "ООО", "АО", "ИП"})
ORG_LIKE_SUFFIXES = ("телеком", "банк", "газ", "нефть", "медиа", "инвест", "строй", "транс")
MORPH_GUARDED_POSES = frozenset({"NOUN", "ADJF", "ADJS", "PRTF", "PRTS"})
VERB_LIKE_POSES = frozenset({"VERB", "INFN", "GRND", "PRTF", "PRTS"})
N_NN_ADJECTIVE_POSES = frozenset({"ADJF", "ADJS"})
N_NN_PARTICIPLE_POSES = frozenset({"PRTF", "PRTS"})
PROPER_LIKE_GRAMMEMES = frozenset({"Name", "Surn", "Patr", "Orgn", "Geox"})
SAFE_SINGLE_CHAR_SPELLING_SUBSTITUTIONS = frozenset(
    {
        ("ь", "ъ"),
        ("е", "и"),
        ("ы", "и"),
        ("я", "е"),
        ("ё", "е"),
        ("е", "ё"),
    }
)
RISKY_KNOWN_SOURCE_LEXICAL_RULE_IDS = frozenset(
    {
        "missing_hard_sign",
        "soft_to_hard_sign",
        "pattern_шо_ше",
        "pattern_жо_же",
        "pattern_чо_че",
        "pattern_цы_ци",
        "dictionary_fuzzy",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "swapped_letters_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
    }
)
SAFE_HYPHEN_PO_ADVERB_BASES = frozenset({"русски", "новому", "старому"})
ALWAYS_UNSAFE_LEXICAL_RULE_IDS = frozenset({"prefix_pre_pri"})
ABBREVIATION_SENTENCE_START_PREFIXES = frozenset({"г", "см", "ул", "стр", "рис", "тыс", "млн", "млрд", "руб", "коп"})
SPEECH_VERB_PATTERN = r"(говорит|написал[аи]?|написали|ответил[аи]?|ответили|сказал[аи]?|сказали|сообщил[аи]?|сообщили|спросил[аи]?|спросили)"


def _guard_rejection_reason(
    source: str,
    target: str,
    edit: Edit,
    protected: list[ProtectedSpan],
    trusted_tsya_keys: set[tuple[int, int, str, str]],
) -> str:
    if _breaks_protected_span(edit, protected, kinds={"url", "email"}):
        return "protected_span"
    number_reason = _breaks_number_or_percent(source, target, edit)
    if number_reason:
        return number_reason
    if _breaks_abbreviation(source, target, edit):
        return "breaks_abbreviation"
    if _breaks_protected_span(edit, protected, kinds={"technical_id"}):
        return "protected_span"
    if _is_unsafe_sentence_start_capitalization(source, edit):
        return "abbreviation_sentence_start_capitalization"
    case_reason = _case_rejection_reason(source, edit, protected)
    if case_reason:
        return case_reason
    if _is_low_confidence_ner_capitalization(edit):
        return "ner_capitalization_requires_confirmed_span"
    if _is_straight_quote_normalization(edit):
        return "quote_normalization_requires_policy"
    final_reason = _final_punctuation_rejection_reason(source, edit)
    if final_reason:
        return final_reason
    n_nn_reason = _n_nn_rejection_reason(edit)
    if n_nn_reason:
        return n_nn_reason
    lexical_reason = _lexical_spelling_rejection_reason(source, edit)
    if lexical_reason:
        return lexical_reason
    ne_ni_reason = _ne_ni_rejection_reason(source, edit)
    if ne_ni_reason:
        return ne_ni_reason
    if _is_unsafe_hyphen_po_adverb(edit):
        return "unsafe_hyphen_po_adverb"
    if _is_unsafe_colon_candidate(source, edit):
        return "unsafe_colon_candidate"
    syntax_reason = _syntax_punctuation_rejection_reason(source, target, edit)
    if syntax_reason:
        return syntax_reason
    if _creates_direct_speech_dash_inside_quotes(target, edit):
        return "unsafe_direct_speech_span"
    if _is_unsafe_direct_speech_punctuation(source, edit):
        return "unsafe_direct_speech_span"
    if _is_unsafe_discourse_dash(source, edit):
        return "unsafe_discourse_dash"
    if _is_tsya_pair(edit.source.lower(), edit.replacement.lower()):
        if _is_dangerous_tsya_edit(source, edit):
            return "tsya_unsafe"
        if not _is_trusted_tsya_edit(edit, trusted_tsya_keys):
            return "requires_trusted_candidate"
    elif _is_dangerous_tsya_edit(source, edit):
        return "tsya_unsafe"
    if _breaks_protected_span(edit, protected):
        return "protected_span"
    if _creates_duplicate_punctuation(source, target, edit):
        return "duplicate_punctuation"
    if _creates_repeated_punctuation_noise(source, target, edit):
        return "punctuation_noise"
    pair_reason = _unbalanced_pair_rejection_reason(source, target, edit)
    if pair_reason:
        return pair_reason
    return ""


def _touches_protected(edit: Edit, protected: list[ProtectedSpan]) -> bool:
    if edit.start < 0 or edit.end < 0:
        return False
    return any(edit.start < span.end and span.start < edit.end for span in protected)


def _breaks_protected_span(edit: Edit, protected: list[ProtectedSpan], *, kinds: set[str] | None = None) -> bool:
    return any(
        (kinds is None or span.kind in kinds) and _edit_touches_span(edit, span.start, span.end)
        for span in protected
    )


def _case_rejection_reason(source: str, edit: Edit, protected: list[ProtectedSpan]) -> str:
    if edit.edit_type != "case_change":
        return ""
    if edit.rule_id in {"capitalization_ner", "abbreviation_case_protection"}:
        return ""
    token = _word_token_covering(source, edit.start)
    if token is None:
        return "outside_sentence_start_case"
    protected_spans = tuple((span.start, span.end) for span in protected)
    if is_safe_sentence_start_case(source, token.start, token.end, token.text, protected_spans):
        return ""
    return "outside_sentence_start_case"


def _breaks_number_or_percent(source: str, target: str, edit: Edit) -> str:
    percent_spans = _regex_spans(PERCENT_RE, source)
    if any(_edit_touches_span(edit, start, end) for start, end in percent_spans) or _creates_percent_punctuation(target):
        return "breaks_percent"
    if _breaks_decimal_number(source, target, edit):
        return "breaks_number"
    number_spans = _regex_spans(NUMBER_RE, source)
    if any(_edit_touches_span(edit, start, end) for start, end in number_spans):
        return "breaks_number"
    return ""


def _breaks_decimal_number(source: str, target: str, edit: Edit) -> bool:
    del target
    return any(_edit_touches_span(edit, start, end) for start, end in _regex_spans(DECIMAL_RE, source))


def _breaks_abbreviation(source: str, target: str, edit: Edit) -> bool:
    del target
    spans = [*_regex_spans(INITIAL_ABBREVIATION_RE, source), *_regex_spans(ABBREVIATION_RE, source)]
    return any(_edit_touches_span(edit, start, end, include_insert_end=True) for start, end in spans)


def _creates_repeated_punctuation_noise(source: str, target: str, edit: Edit) -> bool:
    del edit
    return not _has_punctuation_noise(source) and _has_punctuation_noise(target)


def _creates_duplicate_punctuation(source: str, target: str, edit: Edit) -> bool:
    del edit
    return not re.search(r"([,;:])\s*\1", source) and bool(re.search(r"([,;:])\s*\1", target))


def _creates_unbalanced_pairs(source: str, target: str, edit: Edit) -> bool:
    del edit
    return _paired_punctuation_imbalance_score(target) > _paired_punctuation_imbalance_score(source)


def _unbalanced_pair_rejection_reason(source: str, target: str, edit: Edit) -> str:
    del edit
    if _ordered_pair_imbalance(target, "«", "»") > _ordered_pair_imbalance(source, "«", "»") or target.count('"') % 2 > source.count('"') % 2:
        return "unbalanced_quote"
    for open_char, close_char in (("(", ")"), ("[", "]")):
        if _ordered_pair_imbalance(target, open_char, close_char) > _ordered_pair_imbalance(source, open_char, close_char):
            return "unbalanced_bracket"
    return ""


def _paired_punctuation_imbalance_score(text: str) -> int:
    score = text.count('"') % 2
    for open_char, close_char in (("«", "»"), ("(", ")"), ("[", "]")):
        score += _ordered_pair_imbalance(text, open_char, close_char)
    return score


def _ordered_pair_imbalance(text: str, open_char: str, close_char: str) -> int:
    balance = 0
    unmatched_close = 0
    for char in text:
        if char == open_char:
            balance += 1
        elif char == close_char:
            if balance:
                balance -= 1
            else:
                unmatched_close += 1
    return balance + unmatched_close


def _is_low_confidence_ner_capitalization(edit: Edit) -> bool:
    return edit.rule_id == "capitalization_ner" and float(edit.confidence or 0.0) < CAPITALIZATION_NER_MIN_CONFIDENCE


def _is_straight_quote_normalization(edit: Edit) -> bool:
    return edit.source == '"' and edit.replacement in {"«", "»"}


def _n_nn_rejection_reason(edit: Edit) -> str:
    if edit.rule_id not in N_NN_RULE_IDS:
        return ""
    if (edit.source.lower(), edit.replacement.lower()) in PROTECTED_N_NN_CLEAN_FORMS:
        return "n_nn_unsafe"
    if not _n_nn_morphology_compatible(edit):
        return "morph_incompatible"
    if _is_protected_acronym_or_all_caps(edit.source):
        return "n_nn_unsafe"
    return ""


def _n_nn_morphology_compatible(edit: Edit) -> bool:
    replacement = edit.replacement.lower()
    source = edit.source.lower()
    if not _is_known_correct_word(replacement):
        return False
    if edit.rule_id == "n_nn_short_form":
        return _has_known_pos(replacement, frozenset({"ADJS", "PRTS"}))
    if edit.rule_id == "n_nn_participle":
        return _has_known_pos(replacement, N_NN_PARTICIPLE_POSES)
    if edit.rule_id == "n_nn_deverbal_adjective":
        return _has_known_pos(source, N_NN_PARTICIPLE_POSES) and _has_known_pos(replacement, N_NN_ADJECTIVE_POSES)
    if edit.rule_id == "n_nn_adjective":
        return _has_known_pos(replacement, N_NN_ADJECTIVE_POSES)
    return False


def _has_known_pos(word: str, poses: frozenset[str]) -> bool:
    return any(getattr(parse, "is_known", False) and getattr(parse.tag, "POS", None) in poses for parse in parses(word))


def _lexical_spelling_rejection_reason(source_text: str, edit: Edit) -> str:
    if edit.edit_type != "spelling_replace":
        return ""
    if edit.rule_id in ALWAYS_UNSAFE_LEXICAL_RULE_IDS:
        return "unsafe_fuzzy_spelling_candidate"
    if edit.rule_id not in RISKY_KNOWN_SOURCE_LEXICAL_RULE_IDS:
        return ""
    if edit.rule_id == "swapped_letters_candidate" and not _is_adjacent_swap(edit.source.lower(), edit.replacement.lower()):
        return "unsafe_fuzzy_spelling_candidate"
    if _is_protected_acronym_or_all_caps(edit.source):
        return "protected_lexical_guard"
    if _is_mixed_latin_token(edit.source) or _is_mixed_latin_token(edit.replacement):
        return "protected_lexical_guard"
    if _is_mixed_case_technical_token(edit.source):
        return "protected_lexical_guard"
    if _is_capitalized_proper_like_lexical_change(source_text, edit):
        return "protected_lexical_guard"
    if edit.rule_id == "swapped_letters_candidate":
        swapped_reason = _swapped_letters_rejection_reason(edit)
        if swapped_reason:
            return swapped_reason
    if edit.rule_id == "double_consonant_candidate":
        double_reason = _double_consonant_rejection_reason(edit)
        if double_reason:
            return double_reason
    if edit.rule_id == "extra_letter_candidate":
        extra_reason = _extra_letter_rejection_reason(edit)
        if extra_reason:
            return extra_reason
    if edit.rule_id == "missing_letter_candidate":
        missing_reason = _missing_letter_rejection_reason(edit)
        if missing_reason:
            return missing_reason
    if edit.rule_id == "pattern_чо_че" and _is_known_correct_word(edit.source):
        return "known_source_lexical_guard"
    if edit.rule_id == "dictionary_fuzzy" and not _is_known_correct_word(edit.replacement):
        return "unsafe_fuzzy_spelling_candidate"
    if _is_known_correct_word(edit.source) and _is_known_correct_word(edit.replacement):
        return "known_source_lexical_guard"
    if edit.rule_id == "dictionary_fuzzy" and _is_risky_same_length_dictionary_rewrite(edit):
        return "known_source_lexical_guard"
    if _morphology_incompatible_lexical_replacement(edit.source, edit.replacement):
        return "morphology_agreement_guard"
    if edit.rule_id == "dictionary_fuzzy" and _edit_distance(edit.source.lower(), edit.replacement.lower()) > 1:
        return "unsafe_fuzzy_spelling_candidate"
    return ""


def _is_known_correct_word(word: str) -> bool:
    if not word or " " in word:
        return False
    return any(getattr(parse, "is_known", False) for parse in parses(word.lower()))


def _is_protected_acronym_or_all_caps(word: str) -> bool:
    letters = "".join(char for char in word if char.isalpha())
    if not letters:
        return False
    return word in PROTECTED_ACRONYMS or (len(letters) >= 2 and letters.upper() == letters)


def _is_mixed_latin_token(word: str) -> bool:
    return bool(LATIN_RE.search(word) and RUSSIAN_LETTER_RE.search(word))


def _is_mixed_case_technical_token(word: str) -> bool:
    letters = [char for char in word if char.isalpha()]
    if len(letters) < 2:
        return False
    has_lower = any(char.islower() for char in letters)
    has_upper = any(char.isupper() for char in letters)
    if not (has_lower and has_upper):
        return False
    if word.istitle():
        return False
    return True


def _is_capitalized_proper_like_lexical_change(source_text: str, edit: Edit) -> bool:
    if not edit.source[:1].isupper():
        return False
    source_lower = edit.source.lower()
    if source_lower.endswith(ORG_LIKE_SUFFIXES):
        return True
    if _looks_like_safe_sentence_initial_dictionary_typo(source_text, edit):
        return False
    if _capitalized_token_is_mid_sentence(source_text, edit) or _has_proper_like_parse(edit.source):
        return True
    if _source_has_common_morphology(edit.source):
        return False
    return edit.replacement[:1].isupper() and _is_known_correct_word(edit.replacement)


def _looks_like_safe_sentence_initial_dictionary_typo(source_text: str, edit: Edit) -> bool:
    if edit.rule_id != "dictionary_fuzzy":
        return _is_sentence_initial_common_word_typo(source_text, edit)
    if edit.start != 0:
        return False
    if not _is_known_correct_word(edit.replacement):
        return False
    if _edit_distance(edit.source.lower(), edit.replacement.lower()) > 1:
        return False
    diff = _single_char_difference(edit.source.lower(), edit.replacement.lower())
    if diff is not None and diff not in SAFE_SINGLE_CHAR_SPELLING_SUBSTITUTIONS:
        if _has_proper_like_parse(edit.source) or _has_known_proper_like_parse(edit.replacement):
            return False
    return not _is_risky_same_length_dictionary_rewrite(edit)


def _capitalized_token_is_mid_sentence(source_text: str, edit: Edit) -> bool:
    previous = _previous_nonspace_index(source_text, edit.start)
    if previous is None:
        return False
    return source_text[previous] not in ".!?\n"


def _is_sentence_initial_common_word_typo(source_text: str, edit: Edit) -> bool:
    if edit.start != 0:
        return False
    if not _source_has_common_morphology(edit.source):
        return False
    return _morphology_compatible_lexical_replacement(edit.source, edit.replacement)


def _source_has_common_morphology(word: str) -> bool:
    return any(getattr(parse.tag, "POS", None) in MORPH_GUARDED_POSES for parse in parses(word.lower()))


def _has_proper_like_parse(word: str) -> bool:
    return any(
        PROPER_LIKE_GRAMMEMES.intersection(set(getattr(parse.tag, "grammemes", frozenset())))
        for parse in parses(word.lower())
    )


def _has_known_proper_like_parse(word: str) -> bool:
    return any(
        getattr(parse, "is_known", False)
        and PROPER_LIKE_GRAMMEMES.intersection(set(getattr(parse.tag, "grammemes", frozenset())))
        for parse in parses(word.lower())
    )


def _swapped_letters_rejection_reason(edit: Edit) -> str:
    if _is_known_correct_word(edit.source):
        return "known_source_lexical_guard"
    if not _is_known_correct_word(edit.replacement):
        return "unsafe_fuzzy_spelling_candidate"
    if _probable_clean_source_to_unrelated_known_replacement(edit.source, edit.replacement):
        return "known_source_lexical_guard"
    return ""


def _double_consonant_rejection_reason(edit: Edit) -> str:
    source = edit.source.lower()
    replacement = edit.replacement.lower()
    if _removes_duplicate_consonant(source, replacement):
        return "unsafe_double_consonant_candidate"
    if len(replacement) == len(source) + 1 and _removes_duplicate_consonant(replacement, source):
        return ""
    if _is_known_correct_word(edit.source):
        return "known_source_lexical_guard"
    if not _is_known_correct_word(edit.replacement):
        return "unsafe_fuzzy_spelling_candidate"
    return ""


def _extra_letter_rejection_reason(edit: Edit) -> str:
    source = edit.source.lower()
    replacement = edit.replacement.lower()
    deletion_index = _single_deletion_index(source, replacement)
    if deletion_index is None:
        return "unsafe_fuzzy_spelling_candidate"
    if deletion_index == 0:
        return "known_source_lexical_guard"
    if _is_known_correct_word(edit.source):
        return "known_source_lexical_guard"
    if not _is_known_correct_word(edit.replacement):
        return "unsafe_fuzzy_spelling_candidate"
    if not _deleted_char_is_duplicate(source, deletion_index) and _best_parse_score(source) >= 0.20:
        return "known_source_lexical_guard"
    return ""


def _missing_letter_rejection_reason(edit: Edit) -> str:
    source = edit.source.lower()
    replacement = edit.replacement.lower()
    insertion_index = _single_deletion_index(replacement, source)
    if insertion_index is None:
        return "unsafe_fuzzy_spelling_candidate"
    if insertion_index >= len(source):
        return "known_source_lexical_guard"
    if _is_known_correct_word(edit.source):
        return "known_source_lexical_guard"
    if not _is_known_correct_word(edit.replacement):
        return "unsafe_fuzzy_spelling_candidate"
    return ""


def _single_deletion_index(longer: str, shorter: str) -> int | None:
    if len(longer) != len(shorter) + 1:
        return None
    for index in range(len(longer)):
        if longer[:index] + longer[index + 1 :] == shorter:
            return index
    return None


def _deleted_char_is_duplicate(word: str, index: int) -> bool:
    return (index > 0 and word[index] == word[index - 1]) or (
        index + 1 < len(word) and word[index] == word[index + 1]
    )


def _removes_duplicate_consonant(source: str, replacement: str) -> bool:
    deletion_index = _single_deletion_index(source, replacement)
    if deletion_index is None:
        return False
    return source[deletion_index] in "бвгджзйклмнпрстфхцчшщ" and _deleted_char_is_duplicate(source, deletion_index)


def _probable_clean_source_to_unrelated_known_replacement(source: str, replacement: str) -> bool:
    replacement_poses = _known_coarse_pos_set(replacement)
    if not replacement_poses:
        return False
    source_best = _best_parse_score(source)
    if source_best <= 0.0:
        return False
    compatible_score = _best_compatible_pos_score(source, replacement_poses)
    if compatible_score >= source_best * 0.80:
        return False
    return True


def _is_risky_same_length_dictionary_rewrite(edit: Edit) -> bool:
    source = edit.source.lower()
    replacement = edit.replacement.lower()
    if len(source) < 8 or len(source) != len(replacement):
        return False
    if _edit_distance(source, replacement) != 1:
        return False
    if not _is_known_correct_word(edit.replacement):
        return False
    diff = _single_char_difference(source, replacement)
    if diff is None or diff in SAFE_SINGLE_CHAR_SPELLING_SUBSTITUTIONS:
        return False
    return _best_parse_score(source) >= 0.50


def _single_char_difference(source: str, replacement: str) -> tuple[str, str] | None:
    differences = [(left, right) for left, right in zip(source, replacement) if left != right]
    if len(differences) != 1:
        return None
    return differences[0]


def _is_adjacent_swap(source: str, replacement: str) -> bool:
    if len(source) != len(replacement) or source == replacement:
        return False
    differences = [index for index, (left, right) in enumerate(zip(source, replacement)) if left != right]
    if len(differences) != 2:
        return False
    first, second = differences
    return second == first + 1 and source[first] == replacement[second] and source[second] == replacement[first]


def _known_coarse_pos_set(word: str) -> frozenset[str]:
    return frozenset(
        _coarse_lexical_pos(str(getattr(parse.tag, "POS", "") or ""))
        for parse in parses(word.lower())
        if getattr(parse, "is_known", False) and str(getattr(parse.tag, "POS", "") or "")
    )


def _best_parse_score(word: str) -> float:
    scores = [float(getattr(parse, "score", 0.0)) for parse in parses(word.lower())]
    return max(scores) if scores else 0.0


def _best_compatible_pos_score(word: str, replacement_poses: frozenset[str]) -> float:
    scores = [
        float(getattr(parse, "score", 0.0))
        for parse in parses(word.lower())
        if _coarse_lexical_pos(str(getattr(parse.tag, "POS", "") or "")) in replacement_poses
    ]
    return max(scores) if scores else 0.0


def _coarse_lexical_pos(pos: str) -> str:
    if pos in VERB_LIKE_POSES:
        return "VERB"
    return _coarse_morph_pos(pos)


def _morphology_incompatible_lexical_replacement(source: str, replacement: str) -> bool:
    source_infos = _morphology_infos(source)
    replacement_infos = _morphology_infos(replacement)
    if not source_infos or not replacement_infos:
        return False
    return not any(_morphology_infos_compatible(left, right) for left in source_infos for right in replacement_infos)


def _morphology_compatible_lexical_replacement(source: str, replacement: str) -> bool:
    source_infos = _morphology_infos(source)
    replacement_infos = _morphology_infos(replacement)
    if not source_infos or not replacement_infos:
        return False
    return any(_morphology_infos_compatible(left, right) for left in source_infos for right in replacement_infos)


def _morphology_infos(word: str) -> tuple[dict[str, str], ...]:
    infos: list[dict[str, str]] = []
    for parse in parses(word.lower()):
        pos = str(getattr(parse.tag, "POS", "") or "")
        if pos not in MORPH_GUARDED_POSES:
            continue
        infos.append(
            {
                "pos": _coarse_morph_pos(pos),
                "case": str(getattr(parse.tag, "case", "") or ""),
                "number": str(getattr(parse.tag, "number", "") or ""),
                "gender": str(getattr(parse.tag, "gender", "") or ""),
            }
        )
    return tuple(infos)


def _coarse_morph_pos(pos: str) -> str:
    if pos in {"ADJF", "ADJS", "PRTF", "PRTS"}:
        return "ADJ"
    return pos


def _morphology_infos_compatible(left: dict[str, str], right: dict[str, str]) -> bool:
    if left["pos"] != right["pos"]:
        return False
    for feature in ("case", "number"):
        if left[feature] and right[feature] and left[feature] != right[feature]:
            return False
    if left["number"] != "plur" and right["number"] != "plur":
        if left["gender"] and right["gender"] and left["gender"] != right["gender"]:
            return False
    return True


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _final_punctuation_rejection_reason(source_text: str, edit: Edit) -> str:
    if edit.edit_type != "final_punctuation" or edit.replacement != ".":
        return ""
    stripped = source_text.rstrip()
    if not stripped:
        return ""
    if UNSAFE_FINAL_DOT_TAIL_RE.search(stripped):
        return "unsafe_final_punctuation"
    return ""


def _is_unsafe_sentence_start_capitalization(source_text: str, edit: Edit) -> bool:
    if edit.rule_id != "capitalization_sentence_start":
        return False
    previous = _previous_nonspace_index(source_text, edit.start)
    if previous is not None and source_text[previous] == "…":
        return True
    if previous is None or source_text[previous] != ".":
        return False
    prefix = source_text[: previous + 1].lower()
    if LATIN_COMPANY_ABBREVIATION_RE.search(prefix):
        return True
    if LATIN_RE.search(prefix[-48:]):
        return True
    if _previous_period_touches_protected_abbreviation(source_text, previous):
        return True
    if re.search(r"(?:^|\s)(?:и\.о|[а-яё])\.\s*$", prefix):
        return True
    match = re.search(r"([а-яё]+)\.\s*$", prefix)
    if match and match.group(1) in ABBREVIATION_SENTENCE_START_PREFIXES:
        return True
    return len(edit.source) <= 2


def _previous_period_touches_protected_abbreviation(source_text: str, period_index: int) -> bool:
    protected = find_protected_spans(source_text)
    return any(span.kind == "abbreviation" and span.end == period_index + 1 for span in protected)


def _previous_nonspace_index(text: str, position: int) -> int | None:
    index = position - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    return index if index >= 0 else None


def _ne_ni_rejection_reason(source_text: str, edit: Edit) -> str:
    if edit.rule_id in NI_RULE_IDS:
        if edit.source.lower().startswith("ни") and edit.replacement.lower().startswith("не"):
            return "ne_ni_unsafe"
        return ""
    if edit.rule_id not in NE_SPLIT_JOIN_RULE_IDS:
        return ""
    source = edit.source.lower()
    replacement = edit.replacement.lower()
    if not (source.startswith("не") or source.startswith("не ") or replacement.startswith("не") or replacement.startswith("не ")):
        return ""
    if edit.rule_id == "ne_verb":
        return "" if _looks_like_ne_verb_split(replacement) else "morph_incompatible"
    if (source, replacement) == ("не случайно", "неслучайно"):
        next_word = _next_word_after(source_text, edit.end)
        return "ne_ni_unsafe" if next_word else ""
    if " " in replacement:
        return "" if _has_ne_contrast_marker(source_text, edit) else "ne_ni_unsafe"
    return ""


def _looks_like_ne_verb_split(replacement: str) -> bool:
    parts = replacement.lower().split()
    if len(parts) != 2 or parts[0] != "не":
        return False
    return _looks_like_finite_verb(parts[1])


def _looks_like_finite_verb(word: str) -> bool:
    known_parses = [parse for parse in parses(word) if getattr(parse, "is_known", False)]
    if not known_parses:
        return False
    max_score = max(float(getattr(parse, "score", 0.0)) for parse in known_parses)
    verb_scores = [
        float(getattr(parse, "score", 0.0))
        for parse in known_parses
        if getattr(parse.tag, "POS", None) in VERB_POSES
    ]
    if not verb_scores:
        return False
    best_parse = max(known_parses, key=lambda parse: float(getattr(parse, "score", 0.0)))
    if getattr(best_parse.tag, "POS", None) in VERB_POSES:
        return True
    return max(verb_scores) >= max_score * 0.75


def _has_ne_contrast_marker(source_text: str, edit: Edit) -> bool:
    right_context = source_text[edit.end : edit.end + 48].lower()
    return bool(re.match(r"\s*,?\s*(а|но)\b", right_context))


def _is_unsafe_hyphen_po_adverb(edit: Edit) -> bool:
    if edit.rule_id != "hyphen_po_adverbs":
        return False
    parts = edit.source.lower().split()
    if len(parts) != 2 or parts[0] != "по":
        return True
    base = parts[1]
    if base not in SAFE_HYPHEN_PO_ADVERB_BASES:
        return True
    return edit.replacement.lower() != f"по-{base}"


def _is_unsafe_colon_candidate(source_text: str, edit: Edit) -> bool:
    if edit.rule_id == "explanation_colon":
        return True
    if edit.rule_id != "enumeration_colon":
        return False
    suffix = source_text[edit.end : edit.end + 96]
    return not bool(re.search(r"^\s+[^.!?]{1,64},", suffix))


def _is_unsafe_direct_speech_punctuation(source_text: str, edit: Edit) -> bool:
    return edit.rule_id in DIRECT_SPEECH_RULE_IDS and not _passes_direct_speech_punctuation_guard(source_text, edit)


def _syntax_punctuation_rejection_reason(source_text: str, target_text: str, edit: Edit) -> str:
    del target_text
    if edit.rule_id in SYNTAX_PUNCTUATION_RULE_IDS and float(edit.confidence or 0.0) < 0.5:
        return "syntax_low_confidence"
    if edit.rule_id == "comparative_turnover_comma" and _is_unsafe_comparative_as(source_text, edit):
        return "unsafe_comparative_as"
    if edit.rule_id in ASYNDETIC_RULE_IDS and not _has_asyndetic_clause_evidence(source_text, edit):
        return "unsafe_asyndetic"
    return ""


def _is_unsafe_comparative_as(source_text: str, edit: Edit) -> bool:
    suffix = source_text[edit.end : edit.end + 48].lower()
    if not re.match(r"\s+как\s+[а-яё-]+", suffix):
        return False
    previous = _previous_word_before(source_text, edit.start)
    return previous in {"работает", "работал", "работала", "известен", "известна", "используется", "служит"}


def _has_asyndetic_clause_evidence(source_text: str, edit: Edit) -> bool:
    prefix = source_text[: edit.start]
    suffix = source_text[edit.end :]
    if any(marker in suffix.lower().split()[:2] for marker in {"и", "а", "но", "что", "если", "когда"}):
        return False
    return _contains_finite_predicate(prefix) and _contains_finite_predicate(suffix)


def _contains_finite_predicate(text: str) -> bool:
    words = re.findall(r"[А-Яа-яЁё-]+", text.lower())
    return any(
        re.search(
            r"(л|ла|ло|ли|ет|ит|ют|ут|ат|ят|ется|ится|ются|утся|ался|алась|ались|или|ила|ило|али)$",
            word,
        )
        for word in words
    )


def _passes_direct_speech_punctuation_guard(source_text: str, edit: Edit) -> bool:
    prefix = source_text[max(0, edit.start - 48) : edit.start].lower()
    suffix = source_text[edit.end : edit.end + 48].lower()
    if edit.rule_id == "direct_speech_quotes":
        if any(char in source_text for char in {'"', "«", "»"}):
            return False
        return bool(re.search(rf"\b{SPEECH_VERB_PATTERN}\b[^,;:—.!?\n]*$", prefix))
    if edit.rule_id == "direct_speech_dash":
        return "»" in prefix and bool(re.search(rf"^\s*{SPEECH_VERB_PATTERN}\b", suffix))
    if re.search(rf"[,—-]\s*{SPEECH_VERB_PATTERN}\s+[а-яёa-z]\.?\s*$", prefix):
        return False
    return bool(re.search(rf"\b{SPEECH_VERB_PATTERN}\b[^,;:—.!?\n]*$", prefix))


def _creates_direct_speech_dash_inside_quotes(target_text: str, edit: Edit) -> bool:
    if edit.rule_id != "direct_speech_dash" or edit.replacement != "—":
        return False
    return bool(re.search(rf"«[^»\n]*—\s*»\s+{SPEECH_VERB_PATTERN}\b", target_text, flags=re.IGNORECASE))


def _is_unsafe_discourse_dash(source_text: str, edit: Edit) -> bool:
    if edit.replacement != "—":
        return False
    if _has_nearby_dash(source_text, edit.start):
        return True
    previous = _previous_word_before(source_text, edit.start)
    return previous in DISCOURSE_DASH_BLOCKERS


def _has_nearby_dash(text: str, position: int) -> bool:
    window = text[max(0, position - 3) : min(len(text), position + 4)]
    return any(char in window for char in {"-", "–", "—"})


def _previous_word_before(text: str, position: int) -> str:
    previous = ""
    for word in tokenize_words(text):
        if word.end <= position:
            previous = word.text.lower()
            continue
        break
    return previous


def _is_dangerous_tsya_edit(source_text: str, edit: Edit) -> bool:
    source = edit.source.lower()
    replacement = edit.replacement.lower()
    if WRONG_TO_CORRECT.get(source) == replacement and edit.rule_id == "frequent_error_exact":
        return False
    if not _is_tsya_pair(source, replacement):
        return False
    expects_infinitive = _previous_word_requires_infinitive(source_text, edit.start)
    if source.endswith("ться") and replacement.endswith("тся"):
        return expects_infinitive
    if source.endswith("тся") and replacement.endswith("ться"):
        return not expects_infinitive
    return False


def _is_tsya_pair(source: str, replacement: str) -> bool:
    return (
        source.endswith("ться")
        and replacement == source[: -len("ться")] + "тся"
    ) or (
        source.endswith("тся")
        and replacement == source[: -len("тся")] + "ться"
    )


def _previous_word_requires_infinitive(text: str, position: int) -> bool:
    previous = ""
    for word in tokenize_words(text):
        if word.end <= position:
            previous = word.text.lower()
            continue
        break
    return previous in INFINITIVE_TRIGGER_WORDS


INFINITIVE_TRIGGER_WORDS = frozenset(
    {
        "будем",
        "будет",
        "будете",
        "будешь",
        "буду",
        "будут",
        "должен",
        "должна",
        "должно",
        "должны",
        "можем",
        "может",
        "можете",
        "можешь",
        "мог",
        "могла",
        "могли",
        "могло",
        "могу",
        "могут",
        "надо",
        "нужно",
        "стал",
        "стала",
        "стали",
        "стало",
        "хотел",
        "хотела",
        "хотели",
        "хотело",
        "хотеть",
        "хотим",
        "хотите",
        "хотят",
        "хочет",
        "хочешь",
        "хочу",
    }
)


def _is_trusted_tsya_edit(edit: Edit, trusted_keys: set[tuple[int, int, str, str]]) -> bool:
    return (edit.start, edit.end, edit.source.lower(), edit.replacement.lower()) in trusted_keys


def _trusted_tsya_keys(source_text: str, trusted_edits: list[Any], threshold: float) -> set[tuple[int, int, str, str]]:
    keys: set[tuple[int, int, str, str]] = set()
    for edit in trusted_edits:
        source = str(getattr(edit, "source", "")).lower()
        replacement = str(getattr(edit, "replacement", "")).lower()
        confidence = float(getattr(edit, "confidence", 0.0))
        start = int(getattr(edit, "start", -1))
        end = int(getattr(edit, "end", -1))
        if confidence < threshold or not bool(getattr(edit, "requires_model", False)):
            continue
        if not _is_tsya_pair(source, replacement):
            continue
        if not _source_span_matches(source_text, source, start, end):
            continue
        keys.add((start, end, source, replacement))
    return keys


def _regex_spans(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _edit_touches_span(edit: Edit, start: int, end: int, *, include_insert_end: bool = False) -> bool:
    if edit.start < 0 or edit.end < 0:
        return False
    if edit.start == edit.end:
        if include_insert_end and edit.start == end:
            return True
        return start < edit.start < end
    return edit.start < end and start < edit.end


def _creates_percent_punctuation(text: str) -> bool:
    return bool(re.search(r"\d\s*[,.;:]\s*%", text))


def _has_punctuation_noise(text: str) -> bool:
    return bool(PUNCTUATION_NOISE_RE.search(text))


def _insert_comma_before_common_subordinator(text: str) -> str:
    for marker in (" что ", " чтобы ", " когда ", " если "):
        if marker in text and "," + marker[:-1] not in text:
            return text.replace(marker, "," + marker, 1)
    return text


def _is_context_dependent_edit(edit: Edit) -> bool:
    return CONTEXT_DEPENDENT_WHITELIST.get(edit.source.lower()) == edit.replacement.lower()


def _is_scoring_required_rule_edit(edit: Edit) -> bool:
    return edit.rule_id in SCORING_REQUIRED_RULE_IDS


def _trusted_strict_edits(source_text: str, trusted_edits: list[Any]) -> list[Edit]:
    edits: list[Edit] = []
    for item in trusted_edits:
        source = str(getattr(item, "source", ""))
        replacement = str(getattr(item, "replacement", ""))
        candidate_type = str(getattr(item, "edit_type", ""))
        start = int(getattr(item, "start", -1))
        end = int(getattr(item, "end", -1))
        confidence = float(getattr(item, "confidence", 0.0))
        rule_id = str(getattr(item, "rule_id", ""))
        if confidence <= 0 or source == replacement:
            continue
        if not source and candidate_type not in {"punctuation_insert", "final_punctuation"}:
            continue
        if _is_context_dependent_edit(Edit(source, replacement, "unknown", start, end, confidence=confidence)):
            continue
        if not _source_span_matches(source_text, source, start, end):
            continue
        edit_type = _trusted_candidate_edit_type(source, replacement, candidate_type)
        if edit_type is None:
            continue
        edits.append(Edit(source, replacement, edit_type, start, end, confidence=confidence, rule_id=rule_id))
    return _deduplicate_trusted_edits(edits)


def _source_span_matches(source_text: str, source: str, start: int, end: int) -> bool:
    if start < 0 or end < start or end > len(source_text):
        return False
    if not source:
        return start == end
    return source_text[start:end].lower() == source.lower()


def _trusted_candidate_edit_type(source: str, replacement: str, candidate_type: str) -> str | None:
    if candidate_type == "spelling":
        return "spelling_replace"
    if candidate_type == "split_join":
        if " " in replacement and " " not in source:
            return "split_word"
        if " " in source and " " not in replacement:
            return "join_words"
        return None
    if candidate_type == "hyphen":
        return "hyphen_change"
    if candidate_type == "case":
        return "case_change"
    if candidate_type in {
        "punctuation_insert",
        "punctuation_delete",
        "punctuation_replace",
        "final_punctuation",
    }:
        return candidate_type
    return None


def _is_explained_by_trusted_edit(edit: Edit, trusted_edits: list[Edit]) -> bool:
    for trusted in trusted_edits:
        if (
            edit.source.lower() == trusted.source.lower()
            and edit.replacement.lower() == trusted.replacement.lower()
            and edit.edit_type in {"unknown", trusted.edit_type}
        ):
            return True
        if edit.edit_type != "punctuation_insert" and _positioned_overlap(edit, trusted):
            return True
    if edit.edit_type == "unknown" and trusted_edits:
        return _unknown_edit_explained_by_trusted_edits(edit, trusted_edits)
    return False


def _unknown_edit_explained_by_trusted_edits(edit: Edit, trusted_edits: list[Edit]) -> bool:
    projected = edit.source.lower()
    for trusted in trusted_edits:
        projected = projected.replace(trusted.source.lower(), trusted.replacement.lower(), 1)
    return projected == edit.replacement.lower()


def _positioned_overlap(edit: Edit, trusted: Edit) -> bool:
    if edit.start < 0 or edit.end < 0 or trusted.start < 0 or trusted.end < 0:
        return False
    return edit.start < trusted.end and trusted.start < edit.end


def _deduplicate_trusted_edits(edits: list[Edit]) -> list[Edit]:
    seen: set[tuple[int, int, str, str, str]] = set()
    result: list[Edit] = []
    for edit in edits:
        key = (edit.start, edit.end, edit.source.lower(), edit.replacement.lower(), edit.edit_type)
        if key in seen:
            continue
        seen.add(key)
        result.append(edit)
    return result


def _trusted_context_pair_keys(trusted_edits: list[Any], threshold: float) -> set[tuple[int, int, str, str]]:
    keys: set[tuple[int, int, str, str]] = set()
    for edit in trusted_edits:
        source = str(getattr(edit, "source", "")).lower()
        replacement = str(getattr(edit, "replacement", "")).lower()
        confidence = float(getattr(edit, "confidence", 0.0))
        if confidence < threshold:
            continue
        if CONTEXT_DEPENDENT_WHITELIST.get(source) != replacement:
            continue
        keys.add((int(getattr(edit, "start", -1)), int(getattr(edit, "end", -1)), source, replacement))
    return keys


def _trusted_training_context_pair_keys(
    source_text: str,
    trusted_edits: list[Any],
    expected_rule_id: str,
) -> set[tuple[int, int, str, str]]:
    expected = str(expected_rule_id or "").strip()
    allowed = TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE.get(expected)
    if allowed is None:
        return set()
    allowed_source, allowed_replacement = allowed
    keys: set[tuple[int, int, str, str]] = set()
    for edit in trusted_edits:
        rule_id = str(getattr(edit, "rule_id", "") or "").strip()
        source = str(getattr(edit, "source", "") or "")
        replacement = str(getattr(edit, "replacement", "") or "")
        start = int(getattr(edit, "start", -1))
        end = int(getattr(edit, "end", -1))
        if rule_id != expected:
            continue
        if source.lower() != allowed_source or replacement.lower() != allowed_replacement:
            continue
        if not _source_span_matches(source_text, source, start, end):
            continue
        keys.add((start, end, source.lower(), replacement.lower()))
    return keys


def _has_trusted_context_pair_key(edit: Edit, trusted_keys: set[tuple[int, int, str, str]]) -> bool:
    key = (edit.start, edit.end, edit.source.lower(), edit.replacement.lower())
    return key in trusted_keys


def _is_trusted_context_pair(source_text: str, edit: Edit, trusted_keys: set[tuple[int, int, str, str]]) -> bool:
    return _has_trusted_context_pair_key(edit, trusted_keys) and _passes_context_pair_guard(source_text, edit)


def _passes_context_pair_guard(source_text: str, edit: Edit) -> bool:
    source = edit.source.lower()
    replacement = edit.replacement.lower()
    next_word = _next_word_after(source_text, edit.end)
    if source == "также" and replacement == "так же":
        return next_word == "как"
    if source == "так же" and replacement == "также":
        return next_word != "как"
    if source == "тоже" and replacement == "то же":
        return next_word == "что"
    if source == "то же" and replacement == "тоже":
        return next_word != "что"
    if source == "чтобы" and replacement == "что бы":
        return not _looks_like_infinitive(next_word)
    if source == "что бы" and replacement == "чтобы":
        return _looks_like_infinitive(next_word)
    if source == "зато" and replacement == "за то":
        return _next_word_after(source_text, edit.end) in {"решение", "дело", "задание", "окно", "место"}
    if source == "за то" and replacement == "зато":
        return False
    if source == "вследствие" and replacement == "в следствие":
        return False
    if source == "в следствие" and replacement == "вследствие":
        return _next_word_after(source_text, edit.end) not in {"по", "делу"}
    if source in {"несмотря", "несмотря на"} and replacement.startswith("не смотря"):
        return False
    if source in {"не смотря", "не смотря на"} and replacement.startswith("несмотря"):
        return _next_word_after(source_text, edit.end) != "экран"
    return True


def _next_word_after(text: str, position: int) -> str:
    for word in tokenize_words(text):
        if word.start >= position:
            return word.text.lower()
    return ""


def _word_token_covering(text: str, position: int) -> Any | None:
    if position < 0:
        return None
    for word in tokenize_words(text):
        if word.start <= position < word.end:
            return word
    return None


def _looks_like_infinitive(word: str) -> bool:
    return word.endswith(("ть", "ться", "ти", "чь"))


def _is_positioned_edit(edit: Edit) -> bool:
    if edit.start < 0 or edit.end < edit.start:
        return False
    return edit.edit_type in {
        "final_punctuation",
        "split_word",
        "join_words",
        "hyphen_change",
        "spelling_replace",
        "case_change",
        "punctuation_insert",
        "punctuation_delete",
        "punctuation_replace",
    }


def _apply_positioned_edit(text: str, edit: Edit) -> str:
    start = max(0, min(edit.start, len(text)))
    end = max(start, min(edit.end, len(text)))
    if edit.edit_type == "final_punctuation":
        return text[:start] + edit.replacement + text[end:]
    if edit.edit_type in {"split_word", "join_words", "hyphen_change", "spelling_replace", "case_change"}:
        return text[:start] + edit.replacement + text[end:]
    if edit.edit_type == "punctuation_insert":
        return text[:start] + edit.replacement + text[start:]
    if edit.edit_type == "punctuation_delete":
        return text[:start] + text[end:]
    if edit.edit_type == "punctuation_replace":
        return text[:start] + edit.replacement + text[end:]
    return text
