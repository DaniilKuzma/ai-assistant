from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Sequence

from src.preprocessing.tokenizer import Token
from src.rules.base import RuleMode, RuleSpec


PUNCTUATION_CHARS = set(",.!?:;—…\"'()«»[]")
PUNCTUATION_LABELS = {
    ",": "COMMA",
    ".": "DOT",
    "?": "QUESTION",
    "!": "EXCLAMATION",
    ":": "COLON",
    "—": "DASH",
    ";": "SEMICOLON",
    "…": "ELLIPSIS",
    "«": "QUOTE_OPEN",
    "»": "QUOTE_CLOSE",
    "(": "BRACKET_OPEN",
    ")": "BRACKET_CLOSE",
    "[": "BRACKET_OPEN",
    "]": "BRACKET_CLOSE",
}
CLOSING_FINAL_WRAPPERS = frozenset("\"'»”)]}")

SUBORDINATE_MARKERS = {
    "что": "chto",
    "чтобы": "chtoby",
    "если": "esli",
    "когда": "kogda",
    "поскольку": "poskolku",
    "где": "gde",
    "куда": "kuda",
    "откуда": "otkuda",
}
COMPLEX_SUBORDINATE_MARKERS = {
    ("потому", "что"): "potomu_chto",
    ("так", "как"): "tak_kak",
}
COMPLEX_SUBORDINATE_CONTINUATIONS = frozenset({"потому", "так"})
CONJUNCTION_MARKERS = {"а": "a", "но": "no", "однако": "odnako", "зато": "zato"}
INTRODUCTORY_MARKERS = {
    ("конечно",): "konechno",
    ("возможно",): "vozmozhno",
    ("вероятно",): "veroyatno",
    ("например",): "naprimer",
    ("во-первых",): "vo_pervyh",
    ("по-видимому",): "po_vidimomu",
    ("следовательно",): "sledovatelno",
    ("таким", "образом"): "takim_obrazom",
    ("к", "счастью"): "k_schastyu",
}
ADDRESS_OPENINGS = {
    ("коллеги",): "colleagues",
    ("иван",): "ivan",
    ("мария",): "maria",
    ("друзья",): "friends",
    ("уважаемые", "коллеги"): "uvazhaemye_kollegi",
}
ADDRESS_IMPERATIVE_HINTS = frozenset(
    {
        "добавь",
        "добавьте",
        "исправь",
        "исправьте",
        "напиши",
        "напишите",
        "открой",
        "откройте",
        "отправь",
        "отправьте",
        "посмотри",
        "посмотрите",
        "проверь",
        "проверьте",
        "скажи",
        "скажите",
        "сделай",
        "сделайте",
        "укажи",
        "укажите",
        "уточни",
        "уточните",
    }
)
ADDRESS_HORTATIVE_HINTS = frozenset({"проверим", "исправим", "посмотрим", "отправим", "сделаем", "укажем"})
REPEATED_HOMOGENEOUS_CONJUNCTIONS = frozenset({"и", "ни"})
COMPARATIVE_MARKERS = {"словно": "slovno", "будто": "budto"}
COMPLEX_COMPARATIVE_MARKERS = {("как", "будто"): "kak_budto"}
CLARIFICATION_MARKERS = {
    ("а", "именно"): "a_imenno",
    ("то", "есть"): "to_est",
    ("например",): "naprimer",
    ("очевидно",): "ochevidno",
}
GERUND_WORDS = frozenset({"проверив", "получив", "закончив", "сделав", "прочитав", "продолжив"})
GERUND_FALLBACK_SUFFIXES = ("вшись", "ившись", "авшись", "явшись", "вши", "ивши", "авши", "явши", "ив", "ав", "яв")
PREPOSITION_MARKERS = frozenset(
    {
        "без",
        "в",
        "во",
        "для",
        "до",
        "за",
        "из",
        "к",
        "ко",
        "между",
        "на",
        "над",
        "о",
        "об",
        "от",
        "по",
        "под",
        "при",
        "про",
        "с",
        "со",
        "у",
        "через",
    }
)
PARTICIPLE_SUFFIXES = (
    "вший",
    "вшая",
    "вшее",
    "вшие",
    "енный",
    "енная",
    "енное",
    "енные",
    "ённый",
    "ённая",
    "ённое",
    "ённые",
    "анный",
    "анная",
    "анное",
    "анные",
    "тый",
    "тая",
    "тое",
    "тые",
)
SPEECH_VERBS = frozenset(
    {
        "говорит",
        "написал",
        "написала",
        "написали",
        "ответил",
        "ответила",
        "ответили",
        "сказал",
        "сказала",
        "сказали",
        "сообщил",
        "сообщила",
        "сообщили",
        "спросил",
        "спросила",
        "спросили",
    }
)
DIRECT_SPEECH_BLOCKERS = frozenset(SUBORDINATE_MARKERS) | {"будто", "словно"}
ENUMERATION_COLON_MARKERS = frozenset({"следующее", "следующие"})
ENUMERATION_DASH_MARKERS = frozenset({"всё", "все"})
EXPLANATION_COLON_MARKERS = frozenset({"одно"})
CONSEQUENCE_DASH_PATTERNS = frozenset({("начался", "дождь")})
ASYNDETIC_DASH_PATTERNS = frozenset({("солнце", "село")})
SEMICOLON_PATTERNS = frozenset({("документ", "готов", "отчет", "отправлен")})
DASH_DISCOURSE_MARKERS = frozenset({"получается", "значит"})
PUNCTUATION_NOISE_RE = re.compile(
    r"(?:…[.!?…]+|[.!?]+…|[!?]\.|(?<!\.)\.\.(?!\.)|\.{4,}|[!?]{2,}|([,;:])\s*\1)"
)


@dataclass(frozen=True)
class PunctuationGapCandidate:
    source: str
    replacement: str
    edit_type: str
    start: int
    end: int
    confidence: float
    requires_model: bool
    rule_id: str
    mode: RuleMode
    action: str
    label: str
    gap_index: int
    requires: tuple[str, ...]
    group: str = ""
    edit_domain: str = "punctuation"
    syntax_family: str = ""
    subtype: str = ""
    trigger_text: str = ""
    trigger_lemma: str = ""
    trigger_pos: str = ""
    gap_kind: str = ""
    evidence: str = ""
    confidence_source: str = "syntax_rule"
    implementation_group: str = ""
    constraint_group: str = ""
    bundle_id: str = ""
    metadata: dict[str, Any] | None = None


def generate_syntax_punctuation_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
    *,
    syntax_tokens: Sequence[object] | None = None,
    syntax_provider: Callable[[str], Sequence[object]] | None = None,
) -> list[PunctuationGapCandidate]:
    if not words:
        return []

    candidates: list[PunctuationGapCandidate] = []
    candidates.extend(_punctuation_noise_candidates(text, words, protected, spec_by_id))
    candidates.extend(_subordinate_comma_candidates(text, words, protected, spec_by_id))
    candidates.extend(_conjunction_comma_candidates(text, words, protected, spec_by_id))
    candidates.extend(_introductory_comma_candidates(text, words, protected, spec_by_id))
    candidates.extend(_address_comma_candidates(text, words, protected, spec_by_id, syntax_tokens, syntax_provider))
    candidates.extend(_homogeneous_comma_candidates(text, words, protected, spec_by_id))
    candidates.extend(_detached_adverbial_comma_candidates(text, words, protected, spec_by_id))
    candidates.extend(_detached_participial_comma_candidates(text, words, protected, spec_by_id))
    candidates.extend(_apposition_comma_candidates(text, words, protected, spec_by_id))
    candidates.extend(_clarification_comma_candidates(text, words, protected, spec_by_id))
    candidates.extend(_comparative_turnover_comma_candidates(text, words, protected, spec_by_id))
    candidates.extend(_subject_predicate_dash_candidates(text, words, protected, spec_by_id))
    candidates.extend(_colon_dash_semicolon_candidates(text, words, protected, spec_by_id))
    candidates.extend(_direct_speech_candidates(text, words, protected, spec_by_id))
    candidates.extend(_quote_bracket_balance_candidates(text, words, protected, spec_by_id))
    return candidates


def _subordinate_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    spec = spec_by_id("comma_subordinate")
    lowered = [word.text.lower() for word in words]
    candidates: list[PunctuationGapCandidate] = []
    for index, word in enumerate(words):
        marker_subtype = SUBORDINATE_MARKERS.get(lowered[index])
        if marker_subtype:
            if index > 0 and lowered[index - 1] in COMPLEX_SUBORDINATE_CONTINUATIONS:
                continue
            if _is_adjacent_conjunction_with_correlative_to(lowered, index):
                continue
            candidate = _comma_before_word(
                text,
                words,
                index,
                protected,
                spec,
                "subordinate_clause_comma",
                marker_subtype,
                word.text,
                "subordinate marker before finite-like clause",
            )
            if candidate is not None:
                candidates.append(candidate)
        for marker, subtype in COMPLEX_SUBORDINATE_MARKERS.items():
            marker_end = index + len(marker)
            if tuple(lowered[index:marker_end]) != marker:
                continue
            trigger = text[words[index].start : words[marker_end - 1].end]
            candidate = _comma_before_word(
                text,
                words,
                index,
                protected,
                spec,
                "subordinate_clause_comma",
                subtype,
                trigger,
                "complex subordinate marker",
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _conjunction_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    spec = spec_by_id("comma_conjunction")
    lowered = [word.text.lower() for word in words]
    candidates: list[PunctuationGapCandidate] = []
    for index, word in enumerate(words):
        subtype = CONJUNCTION_MARKERS.get(lowered[index])
        if not subtype:
            continue
        if lowered[index] == "а" and index + 1 < len(words) and lowered[index + 1] == "именно":
            continue
        if lowered[index] == "а" and _looks_like_single_letter_initial(text, word):
            continue
        if index <= 0 or index + 1 >= len(words):
            continue
        candidate = _comma_before_word(
            text,
            words,
            index,
            protected,
            spec,
            "coordinating_conjunction_comma",
            subtype,
            word.text,
            "adversative coordinate conjunction",
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _introductory_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    spec = spec_by_id("introductory_comma")
    lowered = [word.text.lower() for word in words]
    candidates: list[PunctuationGapCandidate] = []
    for index in range(len(words)):
        for marker, subtype in INTRODUCTORY_MARKERS.items():
            marker_end = index + len(marker)
            if tuple(lowered[index:marker_end]) != marker or marker_end >= len(words):
                continue
            if index != 0 and text[words[index].start - 1 : words[index].start] not in ".!?…":
                continue
            start = words[marker_end - 1].end
            end = words[marker_end].start
            if not _is_safe_punctuation_gap(text, start, end, protected):
                continue
            trigger = text[words[index].start : words[marker_end - 1].end]
            candidates.append(
                _candidate(
                    source="",
                    replacement=",",
                    edit_type="punctuation_insert",
                    start=start,
                    end=start,
                    spec=spec,
                    action="INSERT",
                    label="COMMA",
                    gap_index=marker_end - 1,
                    syntax_family="introductory_words",
                    subtype=subtype,
                    trigger_text=trigger,
                    gap_kind="after_introductory_marker",
                    evidence=f"sentence-initial introductory marker: {trigger}",
                    implementation_group="syntax_parenthetical_commas",
                )
            )
    return candidates


def _address_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
    syntax_tokens: Sequence[object] | None = None,
    syntax_provider: Callable[[str], Sequence[object]] | None = None,
) -> list[PunctuationGapCandidate]:
    if len(words) < 2:
        return []
    lowered = [word.text.lower() for word in words]
    spec = spec_by_id("address_comma")
    for opening, subtype in ADDRESS_OPENINGS.items():
        marker_end = len(opening)
        if tuple(lowered[:marker_end]) != opening or marker_end >= len(words):
            continue
        if lowered[marker_end] not in ADDRESS_IMPERATIVE_HINTS | ADDRESS_HORTATIVE_HINTS:
            continue
        return _comma_after_address(text, words, marker_end - 1, protected, spec, subtype)

    if _syntax_indicates_initial_address(text, words, syntax_tokens, syntax_provider):
        return _comma_after_address(text, words, 0, protected, spec, "named_person")
    return []


def _homogeneous_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    spec = spec_by_id("homogeneous_comma")
    lowered = [word.text.lower() for word in words]
    candidates: list[PunctuationGapCandidate] = []
    for index in range(2, len(words) - 1):
        if lowered[index] in REPEATED_HOMOGENEOUS_CONJUNCTIONS and lowered[index - 2] == lowered[index]:
            candidate = _comma_before_word(
                text,
                words,
                index,
                protected,
                spec,
                "homogeneous_members",
                f"repeated_{lowered[index]}",
                words[index].text,
                "repeated homogeneous conjunction",
            )
            if candidate is not None:
                candidates.append(candidate)
    for index in range(3, len(words) - 2):
        if lowered[index + 1] != "и":
            continue
        if not _looks_like_finite_verb(lowered[index - 2]):
            continue
        if not (_looks_like_nominal_item(words[index - 1]) and _looks_like_nominal_item(words[index]) and _looks_like_nominal_item(words[index + 2])):
            continue
        candidate = _comma_after_word(
            text,
            words,
            index - 1,
            protected,
            spec,
            "homogeneous_members",
            "simple_noun_series",
            text[words[index - 1].start : words[index + 2].end],
            "three-item noun series with final и",
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _detached_adverbial_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    if len(words) < 3:
        return []
    lowered = [word.text.lower() for word in words]
    spec = spec_by_id("detached_adverbial_comma")
    if lowered[:2] == ["не", "глядя"] and len(words) >= 4:
        candidate = _comma_after_word(text, words, 1, protected, spec, "detached_adverbial_phrases", "ne_glyadya", "не глядя", "fixed adverbial participle")
        return [candidate] if candidate is not None else []
    if lowered[:2] == ["несмотря", "на"] and len(words) >= 4:
        end_index = min(3, len(words) - 2)
        candidate = _comma_after_word(text, words, end_index, protected, spec, "detached_adverbial_phrases", "nesmotrya_na", "несмотря на", "prepositional adverbial turnover")
        return [candidate] if candidate is not None else []
    if not _looks_like_sentence_initial_gerund(words):
        return []
    comma_after_index = _sentence_initial_gerund_phrase_end_index(words)
    candidate = _comma_after_word(
        text,
        words,
        comma_after_index,
        protected,
        spec,
        "detached_adverbial_phrases",
        "gerund",
        words[0].text,
        "sentence-initial adverbial participle phrase",
    )
    return [candidate] if candidate is not None else []


def _detached_participial_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    if len(words) < 4:
        return []
    spec = spec_by_id("detached_participial_comma")
    candidates: list[PunctuationGapCandidate] = []
    lowered = [word.text.lower() for word in words]
    for index in range(1, len(words) - 2):
        if not _looks_like_participle(lowered[index]):
            continue
        if not _looks_like_nominal_item(words[index - 1]):
            continue
        phrase_end = index
        for candidate_end in range(index + 1, min(len(words), index + 4)):
            if _looks_like_finite_verb(lowered[candidate_end]):
                break
            phrase_end = candidate_end
        if phrase_end == index or phrase_end + 1 >= len(words) or not _looks_like_finite_verb(lowered[phrase_end + 1]):
            continue
        before = _comma_before_word(
            text,
            words,
            index,
            protected,
            spec,
            "detached_participial_phrases",
            "post_noun_participial",
            words[index].text,
            "participle phrase after noun with finite predicate boundary",
        )
        after = _comma_after_word(
            text,
            words,
            phrase_end,
            protected,
            spec,
            "detached_participial_phrases",
            "post_noun_participial_close",
            text[words[index].start : words[phrase_end].end],
            "closing comma after participial phrase",
        )
        if before is not None:
            candidates.append(before)
        if after is not None:
            candidates.append(after)
    return candidates


def _apposition_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    if len(words) < 4 or not words[0].text[:1].isupper() or words[0].text.lower() in {"он", "она", "они", "мы", "вы", "ты", "я"}:
        return []
    comma_position = text.find(",", words[1].start, min(len(text), words[3].end + 2))
    if comma_position < 0 or comma_position < words[2].end:
        return []
    if not all(word.text[:1].islower() for word in words[1:3]):
        return []
    spec = spec_by_id("apposition_comma")
    candidate = _comma_after_word(
        text,
        words,
        0,
        protected,
        spec,
        "detached_applications",
        "paired_apposition_repair",
        text[words[1].start:comma_position],
        "existing right apposition comma gives left boundary",
    )
    return [candidate] if candidate is not None else []


def _clarification_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    spec = spec_by_id("clarification_comma")
    lowered = [word.text.lower() for word in words]
    candidates: list[PunctuationGapCandidate] = []
    for index in range(1, len(words) - 1):
        for marker, subtype in CLARIFICATION_MARKERS.items():
            marker_end = index + len(marker)
            if tuple(lowered[index:marker_end]) != marker:
                continue
            trigger = text[words[index].start : words[marker_end - 1].end]
            candidate = _comma_before_word(
                text,
                words,
                index,
                protected,
                spec,
                "clarification_members",
                subtype,
                trigger,
                "explicit clarification marker",
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _comparative_turnover_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    spec = spec_by_id("comparative_turnover_comma")
    lowered = [word.text.lower() for word in words]
    candidates: list[PunctuationGapCandidate] = []
    for index, word in enumerate(words):
        subtype = COMPARATIVE_MARKERS.get(lowered[index])
        if subtype and index + 1 < len(words):
            candidate = _comma_before_word(text, words, index, protected, spec, "comparative_turnovers", subtype, word.text, "explicit comparative marker")
            if candidate is not None:
                candidates.append(candidate)
        for marker, complex_subtype in COMPLEX_COMPARATIVE_MARKERS.items():
            marker_end = index + len(marker)
            if tuple(lowered[index:marker_end]) != marker or marker_end >= len(words):
                continue
            trigger = text[words[index].start : words[marker_end - 1].end]
            candidate = _comma_before_word(
                text,
                words,
                index,
                protected,
                spec,
                "comparative_turnovers",
                complex_subtype,
                trigger,
                "complex comparative marker",
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _subject_predicate_dash_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    if len(words) < 2:
        return []
    spec = spec_by_id("subject_predicate_dash")
    lowered = [word.text.lower() for word in words]
    candidates: list[PunctuationGapCandidate] = []
    for index, word in enumerate(words):
        if lowered[index] != "это" or index <= 0 or index + 1 >= len(words):
            continue
        if _subject_dash_start_index(words, index) is None or not _is_safe_punctuation_gap(text, words[index - 1].end, word.start, protected):
            continue
        candidates.append(
            _candidate(
                source="",
                replacement="—",
                edit_type="punctuation_insert",
                start=word.start,
                end=word.start,
                spec=spec,
                action="INSERT",
                label="DASH",
                gap_index=index - 1,
                syntax_family="subject_predicate_dash",
                subtype="eto_pattern",
                trigger_text=word.text,
                gap_kind="before_eto",
                evidence="explicit это predicative marker",
                implementation_group="syntax_predicative_dash",
            )
        )
    if _looks_like_nominal_predicate_sentence(words, lowered) and _is_safe_punctuation_gap(text, words[0].end, words[1].start, protected):
        candidates.append(
            _candidate(
                source="",
                replacement="—",
                edit_type="punctuation_insert",
                start=words[1].start,
                end=words[1].start,
                spec=spec,
                action="INSERT",
                label="DASH",
                gap_index=0,
                syntax_family="subject_predicate_dash",
                subtype="noun_predicate",
                trigger_text=text[words[0].start : words[1].end],
                gap_kind="between_nominal_subject_predicate",
                evidence="capitalized nominal subject plus noun predicate without finite verb",
                implementation_group="syntax_predicative_dash",
            )
        )
    return candidates


def _colon_dash_semicolon_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    if _has_punctuation_noise(text) or len(words) < 3:
        return []
    candidates: list[PunctuationGapCandidate] = []
    candidates.extend(_marker_after_word_candidates(text, words, protected, spec_by_id, ENUMERATION_COLON_MARKERS, "enumeration_colon", ":", "homogeneous_members", "summary_word_before_list"))
    candidates.extend(_marker_before_word_candidates(text, words, protected, spec_by_id, ENUMERATION_DASH_MARKERS, "enumeration_dash", "—", "enumeration_colon_dash", "summary_word_after_list"))
    candidates.extend(_marker_after_word_candidates(text, words, protected, spec_by_id, EXPLANATION_COLON_MARKERS, "explanation_colon", ":", "asyndetic_complex_sentence", "explanation_marker"))
    candidates.extend(_fixed_prefix_pattern_candidate(text, words, protected, spec_by_id, CONSEQUENCE_DASH_PATTERNS, "consequence_dash", "—", "asyndetic_complex_sentence", "consequence"))
    candidates.extend(_fixed_prefix_pattern_candidate(text, words, protected, spec_by_id, ASYNDETIC_DASH_PATTERNS, "asyndetic_dash", "—", "asyndetic_complex_sentence", "two_clause_dash"))
    candidates.extend(_semicolon_candidates(text, words, protected, spec_by_id))
    return candidates


def _direct_speech_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    if _has_punctuation_noise(text) or len(words) < 4:
        return []
    candidates: list[PunctuationGapCandidate] = []
    candidates.extend(_direct_speech_after_author_candidates(text, words, protected, spec_by_id))
    candidates.extend(_direct_speech_dash_candidates(text, words, protected, spec_by_id))
    return candidates


def _quote_bracket_balance_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    if _has_punctuation_noise(text):
        return []
    candidates: list[PunctuationGapCandidate] = []
    close_position = _sentence_content_end(text)
    if not _safe_punctuation_insert_position(text, close_position, protected):
        return candidates
    gap_index = _gap_index_for_position(words, close_position)
    pair_specs = [
        ("«", "»", "quote_pair_balance", "QUOTE_CLOSE", "quote_close_missing"),
        ("(", ")", "bracket_pair_balance", "BRACKET_CLOSE", "bracket_close_missing"),
        ("[", "]", "bracket_pair_balance", "BRACKET_CLOSE", "bracket_close_missing"),
    ]
    for open_char, close_char, rule_id, label, subtype in pair_specs:
        if text.count(open_char) == text.count(close_char) + 1 and not _has_unclosed_pair_suffix(text, close_char, close_position):
            candidates.append(
                _candidate(
                    source="",
                    replacement=close_char,
                    edit_type="punctuation_insert",
                    start=close_position,
                    end=close_position,
                    spec=spec_by_id(rule_id),
                    action="INSERT",
                    label=label,
                    gap_index=gap_index,
                    syntax_family="quote_bracket_balance",
                    subtype=subtype,
                    trigger_text=open_char,
                    gap_kind="pair_close_repair",
                    evidence=f"one-sided {open_char}{close_char} imbalance",
                    implementation_group="syntax_quote_bracket_balance",
                    constraint_group="balanced_pairs",
                )
            )
    quote_open_position = _opening_pair_insert_position(text, words, "»", protected)
    if quote_open_position is not None and text.count("»") == text.count("«") + 1:
        candidates.append(
            _candidate(
                source="",
                replacement="«",
                edit_type="punctuation_insert",
                start=quote_open_position,
                end=quote_open_position,
                spec=spec_by_id("quote_pair_balance"),
                action="INSERT",
                label="QUOTE_OPEN",
                gap_index=_gap_index_for_position(words, quote_open_position),
                syntax_family="quote_bracket_balance",
                subtype="quote_open_missing",
                trigger_text="»",
                gap_kind="pair_open_repair",
                evidence="one-sided closing quote",
                implementation_group="syntax_quote_bracket_balance",
                constraint_group="balanced_pairs",
            )
        )
    for close_char, open_char in ((")", "("), ("]", "[")):
        open_position = _opening_pair_insert_position(text, words, close_char, protected)
        if open_position is not None and text.count(close_char) == text.count(open_char) + 1:
            candidates.append(
                _candidate(
                    source="",
                    replacement=open_char,
                    edit_type="punctuation_insert",
                    start=open_position,
                    end=open_position,
                    spec=spec_by_id("bracket_pair_balance"),
                    action="INSERT",
                    label="BRACKET_OPEN",
                    gap_index=_gap_index_for_position(words, open_position),
                    syntax_family="quote_bracket_balance",
                    subtype="bracket_open_missing",
                    trigger_text=close_char,
                    gap_kind="pair_open_repair",
                    evidence="one-sided closing bracket",
                    implementation_group="syntax_quote_bracket_balance",
                    constraint_group="balanced_pairs",
                )
            )
    return candidates


def _punctuation_noise_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    spec = spec_by_id("punctuation_delete_replace")
    candidates: list[PunctuationGapCandidate] = []
    for match in re.finditer(r"([,;:])\s*\1", text):
        delete_start = match.start() + 1
        delete_end = match.end()
        if _edit_touches_spans(delete_start, delete_end, protected):
            continue
        candidates.append(
            _candidate(
                source=text[delete_start:delete_end],
                replacement="",
                edit_type="punctuation_delete",
                start=delete_start,
                end=delete_end,
                spec=spec,
                action="DELETE",
                label="NONE",
                gap_index=_gap_index_for_position(words, delete_start),
                syntax_family="punctuation_combinations",
                subtype=f"duplicate_{PUNCTUATION_LABELS.get(match.group(1), 'punctuation').lower()}",
                trigger_text=match.group(0),
                gap_kind="duplicate_punctuation",
                evidence="duplicate comma/colon/semicolon noise",
                implementation_group="syntax_punctuation_combinations",
                constraint_group="punctuation_noise",
                confidence_source="syntax_rule",
            )
        )
    for match in re.finditer(r"(?<!\.)\.\.(?!\.)", text):
        delete_start = match.start() + 1
        delete_end = match.end()
        if _edit_touches_spans(delete_start, delete_end, protected):
            continue
        candidates.append(
            _candidate(
                source=text[delete_start:delete_end],
                replacement="",
                edit_type="punctuation_delete",
                start=delete_start,
                end=delete_end,
                spec=spec,
                action="DELETE",
                label="NONE",
                gap_index=_gap_index_for_position(words, delete_start),
                syntax_family="punctuation_combinations",
                subtype="duplicate_dot",
                trigger_text=match.group(0),
                gap_kind="duplicate_punctuation",
                evidence="double dot noise",
                implementation_group="syntax_punctuation_combinations",
                constraint_group="punctuation_noise",
            )
        )
    for match in re.finditer(r"([!?])\1+", text):
        delete_start = match.start() + 1
        delete_end = match.end()
        if _edit_touches_spans(delete_start, delete_end, protected):
            continue
        candidates.append(
            _candidate(
                source=text[delete_start:delete_end],
                replacement="",
                edit_type="punctuation_delete",
                start=delete_start,
                end=delete_end,
                spec=spec,
                action="DELETE",
                label="NONE",
                gap_index=_gap_index_for_position(words, delete_start),
                syntax_family="punctuation_combinations",
                subtype="duplicate_expressive_mark",
                trigger_text=match.group(0),
                gap_kind="duplicate_punctuation",
                evidence="repeated question/exclamation noise",
                implementation_group="syntax_punctuation_combinations",
                constraint_group="punctuation_noise",
            )
        )
    for pattern, replacement, subtype in (
        (r"[!?]\.", lambda value: value[0], "terminal_dot_after_expressive_mark"),
        (r"…\.", lambda _value: "…", "dot_after_ellipsis"),
        (r"\.{4,}", lambda _value: "...", "long_dot_run"),
    ):
        for match in re.finditer(pattern, text):
            if _edit_touches_spans(match.start(), match.end(), protected):
                continue
            repl = replacement(match.group(0))
            candidates.append(
                _candidate(
                    source=match.group(0),
                    replacement=repl,
                    edit_type="punctuation_replace",
                    start=match.start(),
                    end=match.end(),
                    spec=spec,
                    action="REPLACE",
                    label=PUNCTUATION_LABELS.get(repl, "NONE"),
                    gap_index=_gap_index_for_position(words, match.start()),
                    syntax_family="punctuation_combinations",
                    subtype=subtype,
                    trigger_text=match.group(0),
                    gap_kind="punctuation_noise",
                    evidence="obvious invalid punctuation combination",
                    implementation_group="syntax_punctuation_combinations",
                    constraint_group="punctuation_noise",
                    confidence_source="syntax_rule",
                )
            )
    return candidates


def _comma_before_word(
    text: str,
    words: list[Token],
    word_index: int,
    protected: tuple[tuple[int, int], ...],
    spec: RuleSpec,
    syntax_family: str,
    subtype: str,
    trigger_text: str,
    evidence: str,
) -> PunctuationGapCandidate | None:
    if word_index <= 0:
        return None
    previous = words[word_index - 1]
    current = words[word_index]
    if not _is_safe_punctuation_gap(text, previous.end, current.start, protected):
        return None
    return _candidate(
        source="",
        replacement=",",
        edit_type="punctuation_insert",
        start=previous.end,
        end=previous.end,
        spec=spec,
        action="INSERT",
        label="COMMA",
        gap_index=word_index - 1,
        syntax_family=syntax_family,
        subtype=subtype,
        trigger_text=trigger_text,
        gap_kind="before_trigger",
        evidence=evidence,
        implementation_group=_implementation_group(syntax_family),
    )


def _comma_after_word(
    text: str,
    words: list[Token],
    word_index: int,
    protected: tuple[tuple[int, int], ...],
    spec: RuleSpec,
    syntax_family: str,
    subtype: str,
    trigger_text: str,
    evidence: str,
) -> PunctuationGapCandidate | None:
    if word_index < 0 or word_index + 1 >= len(words):
        return None
    word = words[word_index]
    next_word = words[word_index + 1]
    if not _is_safe_punctuation_gap(text, word.end, next_word.start, protected):
        return None
    return _candidate(
        source="",
        replacement=",",
        edit_type="punctuation_insert",
        start=word.end,
        end=word.end,
        spec=spec,
        action="INSERT",
        label="COMMA",
        gap_index=word_index,
        syntax_family=syntax_family,
        subtype=subtype,
        trigger_text=trigger_text,
        gap_kind="after_trigger",
        evidence=evidence,
        implementation_group=_implementation_group(syntax_family),
    )


def _comma_after_address(
    text: str,
    words: list[Token],
    word_index: int,
    protected: tuple[tuple[int, int], ...],
    spec: RuleSpec,
    subtype: str,
) -> list[PunctuationGapCandidate]:
    candidate = _comma_after_word(
        text,
        words,
        word_index,
        protected,
        spec,
        "address_comma",
        subtype,
        text[words[0].start : words[word_index].end],
        "sentence-initial address plus imperative/hortative verb",
    )
    return [candidate] if candidate is not None else []


def _candidate(
    *,
    source: str,
    replacement: str,
    edit_type: str,
    start: int,
    end: int,
    spec: RuleSpec,
    action: str,
    label: str,
    gap_index: int,
    syntax_family: str,
    subtype: str,
    trigger_text: str,
    gap_kind: str,
    evidence: str,
    implementation_group: str,
    constraint_group: str = "",
    bundle_id: str = "",
    confidence_source: str = "syntax_rule",
) -> PunctuationGapCandidate:
    metadata = {
        "edit_domain": "punctuation",
        "syntax_family": syntax_family,
        "subtype": subtype,
        "trigger_text": trigger_text,
        "trigger_lemma": trigger_text.lower(),
        "trigger_pos": "",
        "gap_kind": gap_kind,
        "evidence": evidence,
        "confidence_source": confidence_source,
        "implementation_group": implementation_group,
        "constraint_group": constraint_group,
        "bundle_id": bundle_id,
    }
    return PunctuationGapCandidate(
        source=source,
        replacement=replacement,
        edit_type=edit_type,
        start=start,
        end=end,
        confidence=spec.confidence,
        requires_model=spec.mode in {"candidate_only", "model_required"} or "model" in spec.requires,
        rule_id=spec.id,
        mode=spec.mode,
        action=action,
        label=label,
        gap_index=gap_index,
        requires=spec.requires,
        group=spec.group,
        edit_domain="punctuation",
        syntax_family=syntax_family,
        subtype=subtype,
        trigger_text=trigger_text,
        trigger_lemma=trigger_text.lower(),
        trigger_pos="",
        gap_kind=gap_kind,
        evidence=evidence,
        confidence_source=confidence_source,
        implementation_group=implementation_group,
        constraint_group=constraint_group,
        bundle_id=bundle_id,
        metadata=metadata,
    )


def _marker_after_word_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
    markers: frozenset[str],
    rule_id: str,
    replacement: str,
    syntax_family: str,
    subtype: str,
) -> list[PunctuationGapCandidate]:
    spec = spec_by_id(rule_id)
    candidates: list[PunctuationGapCandidate] = []
    for index, word in enumerate(words[:-1]):
        if word.text.lower() not in markers:
            continue
        if not _is_safe_punctuation_gap(text, word.end, words[index + 1].start, protected):
            continue
        candidates.append(
            _candidate(
                source="",
                replacement=replacement,
                edit_type="punctuation_insert",
                start=word.end,
                end=word.end,
                spec=spec,
                action="INSERT",
                label=PUNCTUATION_LABELS[replacement],
                gap_index=index,
                syntax_family=syntax_family,
                subtype=subtype,
                trigger_text=word.text,
                gap_kind="after_summary_marker",
                evidence="bounded marker before list or clause",
                implementation_group=_implementation_group(syntax_family),
                constraint_group="clause_boundary" if replacement in {"—", ";", ":"} else "",
            )
        )
    return candidates


def _marker_before_word_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
    markers: frozenset[str],
    rule_id: str,
    replacement: str,
    syntax_family: str,
    subtype: str,
) -> list[PunctuationGapCandidate]:
    spec = spec_by_id(rule_id)
    candidates: list[PunctuationGapCandidate] = []
    lowered = [word.text.lower() for word in words]
    for index in range(2, len(words)):
        if lowered[index] not in markers:
            continue
        candidate = _insert_before_word(text, words, index, protected, spec, replacement, syntax_family, subtype, words[index].text, "summary word after list")
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _fixed_prefix_pattern_candidate(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
    patterns: frozenset[tuple[str, str]],
    rule_id: str,
    replacement: str,
    syntax_family: str,
    subtype: str,
) -> list[PunctuationGapCandidate]:
    if len(words) < 4:
        return []
    lowered = tuple(word.text.lower() for word in words)
    if lowered[:2] not in patterns or not _is_safe_punctuation_gap(text, words[1].end, words[2].start, protected):
        return []
    return [
        _candidate(
            source="",
            replacement=replacement,
            edit_type="punctuation_insert",
            start=words[1].end,
            end=words[1].end,
            spec=spec_by_id(rule_id),
            action="INSERT",
            label=PUNCTUATION_LABELS[replacement],
            gap_index=1,
            syntax_family=syntax_family,
            subtype=subtype,
            trigger_text=text[words[0].start : words[1].end],
            gap_kind="between_clauses",
            evidence="fixed two-clause bounded pattern with finite predicates",
            implementation_group=_implementation_group(syntax_family),
            constraint_group="asyndetic_clause_boundary",
        )
    ]


def _semicolon_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    if len(words) < 4:
        return []
    lowered = tuple(word.text.lower() for word in words)
    if lowered[:4] not in SEMICOLON_PATTERNS or not _is_safe_punctuation_gap(text, words[1].end, words[2].start, protected):
        return []
    return [
        _candidate(
            source="",
            replacement=";",
            edit_type="punctuation_insert",
            start=words[1].end,
            end=words[1].end,
            spec=spec_by_id("semicolon"),
            action="INSERT",
            label="SEMICOLON",
            gap_index=1,
            syntax_family="asyndetic_complex_sentence",
            subtype="semicolon_two_clauses",
            trigger_text=text[words[0].start : words[3].end],
            gap_kind="between_clauses",
            evidence="bounded two-clause semicolon pattern",
            implementation_group="syntax_asyndetic_complex",
            constraint_group="asyndetic_clause_boundary",
        )
    ]


def _insert_before_word(
    text: str,
    words: list[Token],
    word_index: int,
    protected: tuple[tuple[int, int], ...],
    spec: RuleSpec,
    replacement: str,
    syntax_family: str,
    subtype: str,
    trigger_text: str,
    evidence: str,
) -> PunctuationGapCandidate | None:
    if word_index <= 0:
        return None
    if not _is_safe_punctuation_gap(text, words[word_index - 1].end, words[word_index].start, protected):
        return None
    return _candidate(
        source="",
        replacement=replacement,
        edit_type="punctuation_insert",
        start=words[word_index - 1].end,
        end=words[word_index - 1].end,
        spec=spec,
        action="INSERT",
        label=PUNCTUATION_LABELS[replacement],
        gap_index=word_index - 1,
        syntax_family=syntax_family,
        subtype=subtype,
        trigger_text=trigger_text,
        gap_kind="before_trigger",
        evidence=evidence,
        implementation_group=_implementation_group(syntax_family),
        constraint_group="list_boundary",
    )


def _direct_speech_after_author_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    colon_spec = spec_by_id("direct_speech_colon")
    quotes_spec = spec_by_id("direct_speech_quotes")
    lowered = [word.text.lower() for word in words]
    candidates: list[PunctuationGapCandidate] = []
    for index, word in enumerate(words[:-2]):
        if lowered[index] not in SPEECH_VERBS or lowered[index + 1] in DIRECT_SPEECH_BLOCKERS:
            continue
        if not _is_safe_punctuation_gap(text, word.end, words[index + 1].start, protected):
            continue
        speech_end_index = _last_word_index_before(words, _sentence_content_end(text))
        if speech_end_index is None or speech_end_index - index < 2:
            continue
        bundle_id = f"direct_speech:{word.start}:{words[speech_end_index].end}"
        candidates.append(
            _candidate(
                source="",
                replacement=":",
                edit_type="punctuation_insert",
                start=word.end,
                end=word.end,
                spec=colon_spec,
                action="INSERT",
                label="COLON",
                gap_index=index,
                syntax_family="direct_speech_syntax",
                subtype="author_before_speech",
                trigger_text=word.text,
                gap_kind="after_author_words",
                evidence="speech verb followed by unquoted direct speech span",
                implementation_group="syntax_direct_speech",
                constraint_group="direct_speech_bundle",
                bundle_id=bundle_id,
            )
        )
        if not _has_quote_between(text, words[index + 1].start, words[speech_end_index].end):
            candidates.append(
                _candidate(
                    source="",
                    replacement="«",
                    edit_type="punctuation_insert",
                    start=words[index + 1].start,
                    end=words[index + 1].start,
                    spec=quotes_spec,
                    action="INSERT",
                    label="QUOTE_OPEN",
                    gap_index=index,
                    syntax_family="direct_speech_syntax",
                    subtype="direct_speech_quote_open",
                    trigger_text=word.text,
                    gap_kind="quote_open",
                    evidence="direct speech quote opening paired with speech verb",
                    implementation_group="syntax_direct_speech",
                    constraint_group="direct_speech_bundle",
                    bundle_id=bundle_id,
                )
            )
            candidates.append(
                _candidate(
                    source="",
                    replacement="»",
                    edit_type="punctuation_insert",
                    start=words[speech_end_index].end,
                    end=words[speech_end_index].end,
                    spec=quotes_spec,
                    action="INSERT",
                    label="QUOTE_CLOSE",
                    gap_index=speech_end_index,
                    syntax_family="direct_speech_syntax",
                    subtype="direct_speech_quote_close",
                    trigger_text=word.text,
                    gap_kind="quote_close",
                    evidence="direct speech quote closing paired with speech verb",
                    implementation_group="syntax_direct_speech",
                    constraint_group="direct_speech_bundle",
                    bundle_id=bundle_id,
                )
            )
    return candidates


def _direct_speech_dash_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    spec_by_id: Callable[[str], RuleSpec],
) -> list[PunctuationGapCandidate]:
    spec = spec_by_id("direct_speech_dash")
    speech_verbs = "|".join(sorted(SPEECH_VERBS, key=len, reverse=True))
    candidates: list[PunctuationGapCandidate] = []
    for match in re.finditer(rf"»\s+({speech_verbs})\b", text, flags=re.IGNORECASE):
        position = match.start() + 1
        verb_start = match.start(1)
        if not _is_safe_punctuation_gap(text, position, verb_start, protected):
            continue
        candidates.append(
            _candidate(
                source="",
                replacement="—",
                edit_type="punctuation_insert",
                start=verb_start,
                end=verb_start,
                spec=spec,
                action="INSERT",
                label="DASH",
                gap_index=_gap_index_for_position(words, position),
                syntax_family="direct_speech_syntax",
                subtype="quote_before_author",
                trigger_text=match.group(1),
                gap_kind="after_closing_quote",
                evidence="closing quote followed by speech verb",
                implementation_group="syntax_direct_speech",
                constraint_group="direct_speech_span",
            )
        )
    return candidates


def _is_adjacent_conjunction_with_correlative_to(lowered: list[str], index: int) -> bool:
    return lowered[index] == "если" and index > 0 and lowered[index - 1] == "что" and "то" in lowered[index + 1 :]


def _syntax_indicates_initial_address(
    text: str,
    words: list[Token],
    syntax_tokens: Sequence[object] | None,
    syntax_provider: Callable[[str], Sequence[object]] | None,
) -> bool:
    parsed = tuple(syntax_tokens) if syntax_tokens is not None else tuple(_parse_syntax_safely(text, syntax_provider))
    if len(parsed) < 2:
        return False
    first, second = parsed[0], parsed[1]
    if int(getattr(first, "start", -1)) != words[0].start or int(getattr(first, "end", -1)) != words[0].end:
        return False
    first_pos = str(getattr(first, "pos", ""))
    first_ner = str(getattr(first, "ner", "") or "")
    second_pos = str(getattr(second, "pos", ""))
    second_feats = getattr(second, "feats", {}) or {}
    if first_pos != "PROPN" and first_ner != "PER":
        return False
    return second_pos == "VERB" and str(second_feats.get("Mood", "")) == "Imp"


def _parse_syntax_safely(text: str, syntax_provider: Callable[[str], Sequence[object]] | None = None) -> list[object]:
    if syntax_provider is not None:
        try:
            return list(syntax_provider(text))
        except Exception:
            return []
    return []


def _looks_like_sentence_initial_gerund(words: list[Token]) -> bool:
    if not words or words[0].start != 0:
        return False
    first = words[0].text.lower()
    return first in GERUND_WORDS or (len(first) >= 6 and first.endswith(GERUND_FALLBACK_SUFFIXES))


def _sentence_initial_gerund_phrase_end_index(words: list[Token]) -> int:
    end_index = 1
    if len(words) > 2 and words[2].text.lower() in PREPOSITION_MARKERS:
        end_index = 2
        for index in range(3, len(words)):
            if words[index].text.lower() in {"а", "но", "и", "что", "если", "когда"}:
                break
            if words[index].text[:1].isupper():
                break
            end_index = index
            if index - 2 >= 4:
                break
    return end_index


def _looks_like_participle(word: str) -> bool:
    return any(word.endswith(suffix) for suffix in PARTICIPLE_SUFFIXES)


def _looks_like_finite_verb(word: str) -> bool:
    if word in {"вырос", "выросла", "выросло", "выросли", "упал", "упала", "упало", "упали", "стал", "стала", "стало", "стали"}:
        return True
    return bool(
        re.search(
            r"(л|ла|ло|ли|ет|ит|ют|ут|ат|ят|ется|ится|ются|утся|ется|ался|алась|ались|или|ила|ило|али)$",
            word,
        )
    )


def _looks_like_nominal_item(word: Token) -> bool:
    lowered = word.text.lower()
    return bool(re.search(r"[а-яё]{4,}$", lowered)) and not _looks_like_finite_verb(lowered)


def _subject_dash_start_index(words: list[Token], marker_index: int) -> int | None:
    first_index = max(0, marker_index - 3)
    for candidate_index in range(marker_index - 1, first_index - 1, -1):
        token = words[candidate_index]
        lowered = token.text.lower()
        if lowered in DASH_DISCOURSE_MARKERS:
            return None
        if not token.text[:1].isupper():
            continue
        if all(words[inner_index].text.lower() not in DASH_DISCOURSE_MARKERS for inner_index in range(candidate_index, marker_index)):
            return candidate_index
    return None


def _looks_like_nominal_predicate_sentence(words: list[Token], lowered: list[str]) -> bool:
    if len(words) not in {3, 4}:
        return False
    if not words[0].text[:1].isupper() or words[0].text.lower() in DASH_DISCOURSE_MARKERS:
        return False
    if any(word in CONJUNCTION_MARKERS or word in SUBORDINATE_MARKERS for word in lowered):
        return False
    if any(_looks_like_finite_verb(word) for word in lowered):
        return False
    return _looks_like_nominal_item(words[1])


def _looks_like_single_letter_initial(text: str, word: Token) -> bool:
    return len(word.text) == 1 and word.text[:1].isupper() and text[word.end : word.end + 1] == "."


def _is_safe_punctuation_gap(text: str, start: int, end: int, protected: tuple[tuple[int, int], ...]) -> bool:
    if start < 0 or end < start:
        return False
    if _has_punctuation_noise(text):
        return False
    if _gap_touches_spans(start, end, protected):
        return False
    if any(char in PUNCTUATION_CHARS for char in text[start:end]):
        return False
    return True


def _safe_punctuation_insert_position(text: str, position: int, protected: tuple[tuple[int, int], ...]) -> bool:
    if position < 0 or position > len(text):
        return False
    if _position_inside_spans(position, protected):
        return False
    return True


def _position_inside_spans(position: int, spans: tuple[tuple[int, int], ...]) -> bool:
    return any(start < position < end for start, end in spans)


def _gap_touches_spans(start: int, end: int, spans: tuple[tuple[int, int], ...]) -> bool:
    if start == end:
        return any(span_start < start < span_end for span_start, span_end in spans)
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def _edit_touches_spans(start: int, end: int, spans: tuple[tuple[int, int], ...]) -> bool:
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def _has_punctuation_noise(text: str) -> bool:
    return bool(PUNCTUATION_NOISE_RE.search(text))


def _sentence_content_end(text: str) -> int:
    position = len(text.rstrip())
    while position > 0 and text[position - 1] in CLOSING_FINAL_WRAPPERS:
        position -= 1
        while position > 0 and text[position - 1].isspace():
            position -= 1
    if position > 0 and text[position - 1] in ".!?…":
        return position - 1
    return position


def _last_word_index_before(words: list[Token], position: int) -> int | None:
    for index in range(len(words) - 1, -1, -1):
        if words[index].end <= position:
            return index
    return None


def _has_quote_between(text: str, start: int, end: int) -> bool:
    return any(char in text[start:end] for char in "\"«»")


def _has_unclosed_pair_suffix(text: str, close_char: str, position: int) -> bool:
    return close_char in text[position:]


def _opening_pair_insert_position(
    text: str,
    words: list[Token],
    close_char: str,
    protected: tuple[tuple[int, int], ...],
) -> int | None:
    close_position = text.find(close_char)
    if close_position <= 0:
        return None
    for word in reversed(words):
        if word.start < close_position and _safe_punctuation_insert_position(text, word.start, protected):
            return word.start
    return None


def _gap_index_for_position(words: list[Token], position: int) -> int:
    previous = -1
    for index, word in enumerate(words):
        if word.start >= position:
            return max(previous, 0)
        previous = index
    return max(previous, 0)


def _implementation_group(syntax_family: str) -> str:
    return {
        "subordinate_clause_comma": "syntax_clause_commas",
        "coordinating_conjunction_comma": "syntax_conjunction_commas",
        "introductory_words": "syntax_parenthetical_commas",
        "address_comma": "syntax_address_commas",
        "homogeneous_members": "syntax_homogeneous_members",
        "detached_adverbial_phrases": "syntax_detached_adverbials",
        "detached_participial_phrases": "syntax_detached_participials",
        "detached_applications": "syntax_detached_applications",
        "clarification_members": "syntax_clarification_members",
        "comparative_turnovers": "syntax_comparative_turnovers",
        "subject_predicate_dash": "syntax_predicative_dash",
        "asyndetic_complex_sentence": "syntax_asyndetic_complex",
        "enumeration_colon_dash": "syntax_enumeration_colon_dash",
        "direct_speech_syntax": "syntax_direct_speech",
        "quote_bracket_balance": "syntax_quote_bracket_balance",
        "punctuation_combinations": "syntax_punctuation_combinations",
    }.get(syntax_family, f"syntax_{syntax_family}")
