from __future__ import annotations

from dataclasses import dataclass
import difflib
import re

from src.preprocessing.protected_spans import find_protected_spans
from src.preprocessing.tokenizer import Token, tokenize_words
from src.rules.base import RuleEdit, RuleMode, RuleSpec


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
SUBORDINATE_MARKERS = frozenset({"что", "чтобы", "если", "когда", "поскольку", "где", "куда", "откуда"})
COMPLEX_SUBORDINATE_MARKERS = (("потому", "что"), ("так", "как"))
COMPLEX_SUBORDINATE_CONTINUATIONS = frozenset({"потому", "так"})
CONJUNCTION_MARKERS = frozenset({"а", "но", "однако", "зато"})
INTRODUCTORY_WORDS = frozenset({"конечно", "например", "во-первых", "кажется", "возможно", "следовательно"})
ADDRESS_OPENINGS = frozenset({"коллеги", "иван", "мария"})
ADDRESS_IMPERATIVE_HINTS = frozenset(
    {
        "добавь",
        "исправь",
        "напиши",
        "открой",
        "отправь",
        "посмотри",
        "проверь",
        "скажи",
        "сделай",
        "укажи",
    }
)
ADDRESS_HORTATIVE_HINTS = frozenset({"проверим", "исправим", "посмотрим", "отправим", "сделаем", "укажем"})
REPEATED_HOMOGENEOUS_CONJUNCTIONS = frozenset({"и", "ни"})
COMPARATIVE_MARKERS = frozenset({"словно", "будто"})
COMPLEX_COMPARATIVE_MARKERS = (("как", "будто"),)
GERUND_FALLBACK_SUFFIXES = ("вшись", "в")
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
PROTECTED_PUNCTUATION_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?%|(?<![\w])\d+(?:[.,:/-]\d+)*")
PUNCTUATION_NOISE_RE = re.compile(r"(?:…[.!?…]+|[.!?]+…|[!?]\.|\.{2,}|[!?]{2,}|([,;:])\s*\1)")
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
DIRECT_SPEECH_BLOCKERS = SUBORDINATE_MARKERS | {"будто", "словно"}
ENUMERATION_COLON_MARKERS = frozenset({"следующее", "следующие"})
EXPLANATION_COLON_MARKERS = frozenset({"одно"})
CONSEQUENCE_DASH_PATTERNS = frozenset({("начался", "дождь")})
ASYNDETIC_DASH_PATTERNS = frozenset({("солнце", "село")})
SEMICOLON_PATTERNS = frozenset({("документ", "готов", "отчет", "отправлен")})
DASH_DISCOURSE_MARKERS = frozenset({"получается", "значит"})


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


@dataclass(frozen=True)
class RegexPunctuationRule:
    spec: RuleSpec
    pattern: str
    replacement: str
    flags: int = 0

    def apply(self, text: str) -> tuple[str, tuple[RuleEdit, ...]]:
        updated = re.sub(self.pattern, self.replacement, text, flags=self.flags)
        return updated, tuple(_punctuation_diff_edits(text, updated, self.spec))


@dataclass(frozen=True)
class FunctionPunctuationRule:
    spec: RuleSpec
    transform: object

    def apply(self, text: str) -> tuple[str, tuple[RuleEdit, ...]]:
        updated = self.transform(text)
        return updated, tuple(_punctuation_diff_edits(text, updated, self.spec))


@dataclass(frozen=True)
class FinalPunctuationRule:
    spec: RuleSpec = RuleSpec(
        id="final_punctuation_default",
        group="final_punctuation",
        scope="punctuation_gap",
        edit_type="final_punctuation",
        mode="model_required",
        confidence=0.9,
        requires=("model",),
        description="Generate a final full stop candidate for model scoring when sentence-final punctuation is absent.",
    )

    def apply(self, text: str) -> tuple[str, tuple[RuleEdit, ...]]:
        return text, ()


def apply_punctuation_rules(
    text: str,
    allowed_modes: set[RuleMode] | None = None,
) -> tuple[str, list[RuleEdit]]:
    current = text
    edits: list[RuleEdit] = []
    for rule in PUNCTUATION_RULES:
        if allowed_modes is not None and rule.spec.mode not in allowed_modes:
            continue
        current, rule_edits = rule.apply(current)
        edits.extend(rule_edits)
    return current, edits


def generate_punctuation_candidates(
    text: str,
    allowed_modes: set[RuleMode] | None = None,
) -> list[PunctuationGapCandidate]:
    words = tokenize_words(text)
    if not words:
        return []

    protected = _protected_spans(text)
    candidates: list[PunctuationGapCandidate] = []
    seen: set[tuple[int, int, str, str, str]] = set()

    for candidate in _final_punctuation_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _subordinate_comma_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _conjunction_comma_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _introductory_comma_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _address_comma_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _homogeneous_comma_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _detached_adverbial_comma_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _comparative_turnover_comma_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _subject_predicate_dash_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _direct_speech_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _quote_bracket_balance_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)
    for candidate in _colon_dash_semicolon_candidates(text, words, protected):
        _append_punctuation_candidate(candidates, seen, candidate)

    for rule in PUNCTUATION_RULES:
        if rule.spec.id in {
            "final_punctuation_default",
            "comma_subordinate",
            "comma_conjunction",
            "introductory_comma",
            "address_comma",
            "homogeneous_comma",
            "detached_adverbial_comma",
            "comparative_turnover_comma",
            "subject_predicate_dash",
            "direct_speech_colon",
            "direct_speech_dash",
            "direct_speech_quotes",
            "quote_open",
            "quote_close",
            "quote_pair_balance",
            "bracket_pair_balance",
            "enumeration_colon",
            "explanation_colon",
            "consequence_dash",
            "asyndetic_dash",
            "semicolon",
        }:
            continue
        if allowed_modes is not None and rule.spec.mode not in allowed_modes:
            continue
        _updated, edits = rule.apply(text)
        for edit in edits:
            candidate = _candidate_from_edit(text, words, protected, edit, rule.spec)
            if candidate is not None:
                _append_punctuation_candidate(candidates, seen, candidate)

    if allowed_modes is None:
        return candidates
    return [candidate for candidate in candidates if candidate.mode in allowed_modes]


def punctuation_rules() -> tuple[object, ...]:
    return PUNCTUATION_RULES


def _remove_obvious_extra_punctuation(text: str) -> str:
    text = re.sub(r"([,;:])\s*\1+", r"\1", text)
    text = re.sub(r",\s*([.!?…])", r"\1", text)
    text = re.sub(r"([«(])\s*([,;:])\s*", r"\1", text)
    text = re.sub(r"\s*([,;:])\s*([»)])", r"\2", text)
    return text


def _normalize_simple_direct_speech_quotes(text: str) -> str:
    speech_verbs = r"сказал[аи]?|спросил[аи]?|ответил[аи]?|написал[аи]?"
    return re.sub(
        rf"\b({speech_verbs})\s*:?\s*\"([^\"\n]+)\"",
        r"\1: «\2»",
        text,
        flags=re.IGNORECASE,
    )


def _add_simple_direct_speech_colon(text: str) -> str:
    speech_verbs = r"сказал[аи]?|спросил[аи]?|ответил[аи]?|написал[аи]?"
    return re.sub(rf"\b({speech_verbs})\s+(«[^»]+»)", r"\1: \2", text, flags=re.IGNORECASE)


def _add_obvious_subject_predicate_dash(text: str) -> str:
    return re.sub(r"^([А-ЯЁ][а-яё]+)\s+это\s+", r"\1 — это ", text)


def _add_simple_enumeration_colon(text: str) -> str:
    return re.sub(r"\b(следующее)\s+(?=[а-яёА-ЯЁ])", r"\1: ", text, count=1)


def _identity(text: str) -> str:
    return text


def _final_punctuation_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    spec = _spec_by_id("final_punctuation_default")
    stripped = text.rstrip()
    if not stripped or _has_sentence_final_punctuation(stripped):
        return []
    position = len(stripped)
    if _position_inside_spans(position, protected):
        return []
    return [
        _candidate(
            source="",
            replacement=".",
            edit_type="final_punctuation",
            start=position,
            end=position,
            spec=spec,
            action="INSERT",
            label="DOT",
            gap_index=len(words) - 1,
        )
    ]


def _has_sentence_final_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in CLOSING_FINAL_WRAPPERS:
        stripped = stripped[:-1].rstrip()
    return bool(stripped and stripped[-1] in ".!?…")


def _subordinate_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    spec = _spec_by_id("comma_subordinate")
    candidates: list[PunctuationGapCandidate] = []
    lowered = [word.text.lower() for word in words]

    for index, word in enumerate(words):
        if lowered[index] in SUBORDINATE_MARKERS:
            if index > 0 and lowered[index - 1] in COMPLEX_SUBORDINATE_CONTINUATIONS:
                continue
            if _is_adjacent_conjunction_with_correlative_to(lowered, index):
                continue
            candidate = _comma_before_word(text, words, index, protected, spec)
            if candidate is not None:
                candidates.append(candidate)

        for marker in COMPLEX_SUBORDINATE_MARKERS:
            marker_end = index + len(marker)
            if tuple(lowered[index:marker_end]) != marker:
                continue
            candidate = _comma_before_word(text, words, index, protected, spec)
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _conjunction_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    spec = _spec_by_id("comma_conjunction")
    candidates: list[PunctuationGapCandidate] = []
    lowered = [word.text.lower() for word in words]

    for index, word in enumerate(words):
        if lowered[index] not in CONJUNCTION_MARKERS:
            continue
        if lowered[index] == "а" and _looks_like_single_letter_initial(text, word):
            continue
        if index <= 0 or index + 1 >= len(words):
            continue
        candidate = _comma_before_word(text, words, index, protected, spec)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _introductory_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    spec = _spec_by_id("introductory_comma")
    candidates: list[PunctuationGapCandidate] = []
    for index, word in enumerate(words):
        if word.text.lower() not in INTRODUCTORY_WORDS:
            continue
        if index + 1 >= len(words) or not _is_safe_punctuation_gap(text, word.end, words[index + 1].start, protected):
            continue
        candidates.append(
            _candidate(
                source="",
                replacement=",",
                edit_type="punctuation_insert",
                start=word.end,
                end=word.end,
                spec=spec,
                action="INSERT",
                label="COMMA",
                gap_index=index,
            )
        )
    return candidates


def _address_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    if len(words) < 2:
        return []
    if not _looks_like_possible_address_opening(words):
        return []
    if _looks_like_lexical_address_opening(words):
        return _address_candidate_after_first_word(text, words, protected)

    syntax_tokens = _parse_syntax_safely(text)
    if len(syntax_tokens) >= 2:
        first, second = syntax_tokens[0], syntax_tokens[1]
        if (
            first.start == words[0].start
            and first.end == words[0].end
            and _looks_like_explicit_address(first, second)
        ):
            return _address_candidate_after_first_word(text, words, protected)

    return []


def _address_candidate_after_first_word(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    if len(words) < 2 or not _is_safe_punctuation_gap(text, words[0].end, words[1].start, protected):
        return []

    spec = _spec_by_id("address_comma")
    return [
        _candidate(
            source="",
            replacement=",",
            edit_type="punctuation_insert",
            start=words[0].end,
            end=words[0].end,
            spec=spec,
            action="INSERT",
            label="COMMA",
            gap_index=0,
        )
    ]


def _homogeneous_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    spec = _spec_by_id("homogeneous_comma")
    candidates: list[PunctuationGapCandidate] = []
    lowered = [word.text.lower() for word in words]
    for index in range(2, len(words) - 1):
        if lowered[index] not in REPEATED_HOMOGENEOUS_CONJUNCTIONS:
            continue
        if lowered[index - 2] != lowered[index]:
            continue
        candidate = _comma_before_word(text, words, index, protected, spec)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _detached_adverbial_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    if len(words) < 3 or not _looks_like_sentence_initial_gerund(text, words):
        return []
    spec = _spec_by_id("detached_adverbial_comma")
    comma_after_index = _sentence_initial_gerund_phrase_end_index(words)
    return _comma_after_word(text, words, comma_after_index, protected, spec)


def _is_adjacent_conjunction_with_correlative_to(lowered: list[str], index: int) -> bool:
    return (
        lowered[index] == "если"
        and index > 0
        and lowered[index - 1] == "что"
        and "то" in lowered[index + 1 :]
    )


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


def _comparative_turnover_comma_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    spec = _spec_by_id("comparative_turnover_comma")
    candidates: list[PunctuationGapCandidate] = []
    lowered = [word.text.lower() for word in words]

    for index, word in enumerate(words):
        if lowered[index] in COMPARATIVE_MARKERS:
            if index + 1 >= len(words):
                continue
            candidate = _comma_before_word(text, words, index, protected, spec)
            if candidate is not None:
                candidates.append(candidate)

        for marker in COMPLEX_COMPARATIVE_MARKERS:
            marker_end = index + len(marker)
            if tuple(lowered[index:marker_end]) != marker:
                continue
            if marker_end >= len(words):
                continue
            candidate = _comma_before_word(text, words, index, protected, spec)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _subject_predicate_dash_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    if len(words) < 3:
        return []
    lowered = [word.text.lower() for word in words]
    candidates: list[PunctuationGapCandidate] = []
    spec = _spec_by_id("subject_predicate_dash")
    for index, word in enumerate(words):
        if lowered[index] != "это" or index <= 0 or index + 1 >= len(words):
            continue
        subject_start_index = _subject_dash_start_index(words, index)
        if subject_start_index is None:
            continue
        if not _is_safe_punctuation_gap(text, words[index - 1].end, word.start, protected):
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
            )
        )
    return candidates


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


def _direct_speech_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    if _has_punctuation_noise(text) or len(words) < 4:
        return []

    candidates: list[PunctuationGapCandidate] = []
    candidates.extend(_direct_speech_after_author_candidates(text, words, protected))
    candidates.extend(_direct_speech_dash_candidates(text, words, protected))
    return candidates


def _direct_speech_after_author_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    colon_spec = _spec_by_id("direct_speech_colon")
    quotes_spec = _spec_by_id("direct_speech_quotes")
    candidates: list[PunctuationGapCandidate] = []
    lowered = [word.text.lower() for word in words]

    for index, word in enumerate(words[:-2]):
        if lowered[index] not in SPEECH_VERBS:
            continue
        next_word = words[index + 1]
        if lowered[index + 1] in DIRECT_SPEECH_BLOCKERS:
            continue
        if not _is_safe_punctuation_gap(text, word.end, next_word.start, protected):
            continue
        speech_end_index = _last_word_index_before(words, _sentence_content_end(text))
        if speech_end_index is None or speech_end_index - index < 2:
            continue

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
            )
        )
        if not _has_quote_between(text, next_word.start, words[speech_end_index].end):
            candidates.append(
                _candidate(
                    source="",
                    replacement="«",
                    edit_type="punctuation_insert",
                    start=next_word.start,
                    end=next_word.start,
                    spec=quotes_spec,
                    action="INSERT",
                    label="QUOTE_OPEN",
                    gap_index=index,
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
                )
            )
    return candidates


def _direct_speech_dash_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    spec = _spec_by_id("direct_speech_dash")
    candidates: list[PunctuationGapCandidate] = []
    speech_verbs = "|".join(sorted(SPEECH_VERBS, key=len, reverse=True))
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
            )
        )
    return candidates


def _quote_bracket_balance_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    if _has_punctuation_noise(text):
        return []

    candidates: list[PunctuationGapCandidate] = []
    candidates.extend(_straight_quote_candidates(text, words, protected))
    close_position = _sentence_content_end(text)
    if not _safe_punctuation_insert_position(text, close_position, protected):
        return candidates
    gap_index = _gap_index_for_position(words, close_position)

    if text.count("«") == text.count("»") + 1 and not _has_unclosed_pair_suffix(text, "»", close_position):
        spec = _spec_by_id("quote_pair_balance")
        candidates.append(
            _candidate(
                source="",
                replacement="»",
                edit_type="punctuation_insert",
                start=close_position,
                end=close_position,
                spec=spec,
                action="INSERT",
                label="QUOTE_CLOSE",
                gap_index=gap_index,
            )
        )

    if text.count("(") == text.count(")") + 1 and not _has_unclosed_pair_suffix(text, ")", close_position):
        spec = _spec_by_id("bracket_pair_balance")
        candidates.append(
            _candidate(
                source="",
                replacement=")",
                edit_type="punctuation_insert",
                start=close_position,
                end=close_position,
                spec=spec,
                action="INSERT",
                label="BRACKET_CLOSE",
                gap_index=gap_index,
            )
        )

    if text.count("[") == text.count("]") + 1 and not _has_unclosed_pair_suffix(text, "]", close_position):
        spec = _spec_by_id("bracket_pair_balance")
        candidates.append(
            _candidate(
                source="",
                replacement="]",
                edit_type="punctuation_insert",
                start=close_position,
                end=close_position,
                spec=spec,
                action="INSERT",
                label="BRACKET_CLOSE",
                gap_index=gap_index,
            )
        )
    return candidates


def _straight_quote_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    quote_positions = [index for index, char in enumerate(text) if char == '"']
    if len(quote_positions) < 2 or len(quote_positions) % 2:
        return []

    open_spec = _spec_by_id("quote_open")
    close_spec = _spec_by_id("quote_close")
    candidates: list[PunctuationGapCandidate] = []
    for pair_index in range(0, len(quote_positions), 2):
        open_position = quote_positions[pair_index]
        close_position = quote_positions[pair_index + 1]
        if _edit_touches_spans(open_position, open_position + 1, protected):
            continue
        if _edit_touches_spans(close_position, close_position + 1, protected):
            continue
        candidates.append(
            _candidate(
                source='"',
                replacement="«",
                edit_type="punctuation_replace",
                start=open_position,
                end=open_position + 1,
                spec=open_spec,
                action="REPLACE",
                label="QUOTE_OPEN",
                gap_index=_gap_index_for_position(words, open_position),
            )
        )
        candidates.append(
            _candidate(
                source='"',
                replacement="»",
                edit_type="punctuation_replace",
                start=close_position,
                end=close_position + 1,
                spec=close_spec,
                action="REPLACE",
                label="QUOTE_CLOSE",
                gap_index=_gap_index_for_position(words, close_position),
            )
        )
    return candidates


def _colon_dash_semicolon_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    if _has_punctuation_noise(text) or len(words) < 3:
        return []

    candidates: list[PunctuationGapCandidate] = []
    candidates.extend(_marker_after_word_candidates(text, words, protected, ENUMERATION_COLON_MARKERS, "enumeration_colon", ":"))
    candidates.extend(_marker_after_word_candidates(text, words, protected, EXPLANATION_COLON_MARKERS, "explanation_colon", ":"))
    candidates.extend(_fixed_prefix_pattern_candidate(text, words, protected, CONSEQUENCE_DASH_PATTERNS, "consequence_dash", "—"))
    candidates.extend(_fixed_prefix_pattern_candidate(text, words, protected, ASYNDETIC_DASH_PATTERNS, "asyndetic_dash", "—"))
    candidates.extend(_semicolon_candidates(text, words, protected))
    return candidates


def _marker_after_word_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    markers: frozenset[str],
    rule_id: str,
    replacement: str,
) -> list[PunctuationGapCandidate]:
    spec = _spec_by_id(rule_id)
    label = PUNCTUATION_LABELS[replacement]
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
                label=label,
                gap_index=index,
            )
        )
    return candidates


def _fixed_prefix_pattern_candidate(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    patterns: frozenset[tuple[str, str]],
    rule_id: str,
    replacement: str,
) -> list[PunctuationGapCandidate]:
    if len(words) < 4:
        return []
    lowered = tuple(word.text.lower() for word in words)
    if lowered[:2] not in patterns:
        return []
    if not _is_safe_punctuation_gap(text, words[1].end, words[2].start, protected):
        return []
    spec = _spec_by_id(rule_id)
    return [
        _candidate(
            source="",
            replacement=replacement,
            edit_type="punctuation_insert",
            start=words[1].end,
            end=words[1].end,
            spec=spec,
            action="INSERT",
            label=PUNCTUATION_LABELS[replacement],
            gap_index=1,
        )
    ]


def _semicolon_candidates(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
) -> list[PunctuationGapCandidate]:
    if len(words) < 4:
        return []
    lowered = tuple(word.text.lower() for word in words)
    if lowered[:4] not in SEMICOLON_PATTERNS:
        return []
    if not _is_safe_punctuation_gap(text, words[1].end, words[2].start, protected):
        return []
    spec = _spec_by_id("semicolon")
    return [
        _candidate(
            source="",
            replacement=";",
            edit_type="punctuation_insert",
            start=words[1].end,
            end=words[1].end,
            spec=spec,
            action="INSERT",
            label="SEMICOLON",
            gap_index=1,
        )
    ]


def _looks_like_possible_address_opening(words: list[Token]) -> bool:
    first = words[0].text
    second = words[1].text.lower()
    if first.lower() in ADDRESS_OPENINGS:
        return second in ADDRESS_IMPERATIVE_HINTS or second in ADDRESS_HORTATIVE_HINTS
    if not first[:1].isupper() or first.lower() in {"я", "мы", "он", "она", "они", "вы", "ты"}:
        return False
    return second in ADDRESS_IMPERATIVE_HINTS


def _looks_like_lexical_address_opening(words: list[Token]) -> bool:
    if len(words) < 2:
        return False
    first = words[0].text.lower()
    second = words[1].text.lower()
    return first in ADDRESS_OPENINGS and (second in ADDRESS_IMPERATIVE_HINTS or second in ADDRESS_HORTATIVE_HINTS)


def _looks_like_explicit_address(first: object, second: object) -> bool:
    first_pos = str(getattr(first, "pos", ""))
    first_ner = str(getattr(first, "ner", "") or "")
    second_pos = str(getattr(second, "pos", ""))
    second_feats = getattr(second, "feats", {}) or {}
    if first_pos != "PROPN" and first_ner != "PER":
        return False
    return second_pos == "VERB" and str(second_feats.get("Mood", "")) == "Imp"


def _looks_like_single_letter_initial(text: str, word: Token) -> bool:
    return len(word.text) == 1 and word.text[:1].isupper() and text[word.end : word.end + 1] == "."


def _looks_like_sentence_initial_gerund(text: str, words: list[Token]) -> bool:
    if words[0].start != 0:
        return False
    first = words[0].text.lower()
    if len(first) >= 6 and first.endswith(GERUND_FALLBACK_SUFFIXES):
        return True
    return False


def _syntax_token_matches_word(syntax_token: object, word: Token) -> bool:
    return int(getattr(syntax_token, "start", -1)) == word.start and int(getattr(syntax_token, "end", -1)) == word.end


def _parse_syntax_safely(text: str) -> list[object]:
    try:
        from src.nlp.syntax import parse_syntax
    except Exception:
        return []
    try:
        return list(parse_syntax(text))
    except Exception:
        return []


def _comma_before_word(
    text: str,
    words: list[Token],
    word_index: int,
    protected: tuple[tuple[int, int], ...],
    spec: RuleSpec,
) -> PunctuationGapCandidate | None:
    if word_index <= 0:
        return None
    previous = words[word_index - 1]
    current = words[word_index]
    position = previous.end
    if not _is_safe_punctuation_gap(text, previous.end, current.start, protected):
        return None
    return _candidate(
        source="",
        replacement=",",
        edit_type="punctuation_insert",
        start=position,
        end=position,
        spec=spec,
        action="INSERT",
        label="COMMA",
        gap_index=word_index - 1,
    )


def _comma_after_word(
    text: str,
    words: list[Token],
    word_index: int,
    protected: tuple[tuple[int, int], ...],
    spec: RuleSpec,
) -> list[PunctuationGapCandidate]:
    if word_index < 0 or word_index + 1 >= len(words):
        return []
    word = words[word_index]
    next_word = words[word_index + 1]
    if not _is_safe_punctuation_gap(text, word.end, next_word.start, protected):
        return []
    return [
        _candidate(
            source="",
            replacement=",",
            edit_type="punctuation_insert",
            start=word.end,
            end=word.end,
            spec=spec,
            action="INSERT",
            label="COMMA",
            gap_index=word_index,
        )
    ]


def _candidate_from_edit(
    text: str,
    words: list[Token],
    protected: tuple[tuple[int, int], ...],
    edit: RuleEdit,
    spec: RuleSpec,
) -> PunctuationGapCandidate | None:
    if _edit_touches_spans(edit.start, edit.end, protected):
        return None
    action = _action_for_edit(edit)
    label = _label_for_edit(edit)
    if not action:
        return None
    if edit.edit_type == "punctuation_insert" and _has_existing_punctuation_at(text, edit.start, edit.replacement):
        return None
    return _candidate(
        source=edit.source,
        replacement=edit.replacement,
        edit_type=edit.edit_type,
        start=edit.start,
        end=edit.end,
        spec=spec,
        action=action,
        label=label,
        gap_index=_gap_index_for_position(words, edit.start),
    )


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
) -> PunctuationGapCandidate:
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
    )


def _append_punctuation_candidate(
    candidates: list[PunctuationGapCandidate],
    seen: set[tuple[int, int, str, str, str]],
    candidate: PunctuationGapCandidate,
) -> None:
    key = (candidate.start, candidate.end, candidate.replacement, candidate.edit_type, candidate.action)
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def _action_for_edit(edit: RuleEdit) -> str:
    if edit.edit_type in {"punctuation_insert", "final_punctuation"} and edit.replacement:
        return "INSERT" if not edit.source else "REPLACE"
    if edit.edit_type == "punctuation_delete":
        return "DELETE"
    if edit.edit_type == "punctuation_replace":
        return "REPLACE"
    return ""


def _label_for_edit(edit: RuleEdit) -> str:
    if edit.edit_type == "punctuation_delete":
        return "NONE"
    return PUNCTUATION_LABELS.get(edit.replacement, "NONE")


def _spec_by_id(rule_id: str) -> RuleSpec:
    for rule in PUNCTUATION_RULES:
        if rule.spec.id == rule_id:
            return rule.spec
    raise KeyError(rule_id)


def _protected_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans = [(span.start, span.end) for span in find_protected_spans(text)]
    spans.extend((match.start(), match.end()) for match in PROTECTED_PUNCTUATION_RE.finditer(text))
    return tuple(sorted(set(spans)))


def _is_safe_punctuation_gap(
    text: str,
    start: int,
    end: int,
    protected: tuple[tuple[int, int], ...],
) -> bool:
    if start < 0 or end < start:
        return False
    if _has_punctuation_noise(text):
        return False
    if _gap_touches_spans(start, end, protected):
        return False
    if _has_any_punctuation_between(text, start, end):
        return False
    return True


def _position_inside_spans(position: int, spans: tuple[tuple[int, int], ...]) -> bool:
    return any(start < position < end for start, end in spans)


def _gap_touches_spans(start: int, end: int, spans: tuple[tuple[int, int], ...]) -> bool:
    if start == end:
        return _position_inside_spans(start, spans)
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def _edit_touches_spans(start: int, end: int, spans: tuple[tuple[int, int], ...]) -> bool:
    if start < 0 or end < start:
        return False
    if start == end:
        return _position_inside_spans(start, spans)
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def _safe_punctuation_insert_position(
    text: str,
    position: int,
    protected: tuple[tuple[int, int], ...],
) -> bool:
    if position < 0 or position > len(text):
        return False
    if _position_inside_spans(position, protected):
        return False
    return not (position < len(text) and text[position] in "«»()[]")


def _sentence_content_end(text: str) -> int:
    end = len(text.rstrip())
    while end > 0 and text[end - 1] in ".!?…":
        end -= 1
    while end > 0 and text[end - 1].isspace():
        end -= 1
    return end


def _last_word_index_before(words: list[Token], position: int) -> int | None:
    for index in range(len(words) - 1, -1, -1):
        if words[index].end <= position:
            return index
    return None


def _has_quote_between(text: str, start: int, end: int) -> bool:
    return any(char in "\"'«»“”„" for char in text[max(0, start) : max(start, end)])


def _has_unclosed_pair_suffix(text: str, close_char: str, position: int) -> bool:
    return position < len(text) and text[position] == close_char


def _has_punctuation_between(text: str, start: int, end: int, chars: str) -> bool:
    return any(char in chars for char in text[max(0, start) : max(start, end)])


def _has_any_punctuation_between(text: str, start: int, end: int) -> bool:
    return any(char in PUNCTUATION_CHARS for char in text[max(0, start) : max(start, end)])


def _has_punctuation_noise(text: str) -> bool:
    return bool(PUNCTUATION_NOISE_RE.search(text))


def _has_existing_punctuation_at(text: str, position: int, replacement: str) -> bool:
    if not replacement:
        return False
    index = position
    while index < len(text) and text[index].isspace():
        index += 1
    return index < len(text) and text[index] == replacement


def _next_word_start(words: list[Token], index: int) -> int:
    if index + 1 >= len(words):
        return words[index].end
    return words[index + 1].start


def _gap_index_for_position(words: list[Token], position: int) -> int:
    gap = 0
    for index, word in enumerate(words):
        if word.end <= position:
            gap = index
        elif word.start > position:
            break
    return max(0, min(gap, len(words) - 1))


def _punctuation_diff_edits(source: str, target: str, spec: RuleSpec) -> list[RuleEdit]:
    edits: list[RuleEdit] = []
    matcher = difflib.SequenceMatcher(a=source, b=target)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            edits.extend(_insert_edits(i1, target[j1:j2], spec))
        elif tag == "delete":
            edits.extend(_delete_edits(source, i1, i2, spec))
        else:
            edits.extend(_replace_edits(source, target, i1, i2, j1, j2, spec))
    return edits


def _insert_edits(position: int, inserted: str, spec: RuleSpec) -> list[RuleEdit]:
    return [
        _edit("", char, "punctuation_insert", position + offset, position + offset, spec)
        for offset, char in enumerate(inserted)
        if char in PUNCTUATION_CHARS
    ]


def _delete_edits(source: str, start: int, end: int, spec: RuleSpec) -> list[RuleEdit]:
    return [
        _edit(char, "", "punctuation_delete", index, index + 1, spec)
        for index, char in enumerate(source[start:end], start=start)
        if char in PUNCTUATION_CHARS
    ]


def _replace_edits(source: str, target: str, i1: int, i2: int, j1: int, j2: int, spec: RuleSpec) -> list[RuleEdit]:
    source_punct = [(index, char) for index, char in enumerate(source[i1:i2], start=i1) if char in PUNCTUATION_CHARS]
    target_punct = [(index, char) for index, char in enumerate(target[j1:j2], start=j1) if char in PUNCTUATION_CHARS]

    if not source_punct:
        return _insert_edits(i1, target[j1:j2], spec)
    if not target_punct:
        return _delete_edits(source, i1, i2, spec)

    edits: list[RuleEdit] = []
    paired_count = min(len(source_punct), len(target_punct))
    for pair_index in range(paired_count):
        source_index, source_char = source_punct[pair_index]
        _target_index, target_char = target_punct[pair_index]
        if source_char != target_char:
            edits.append(_edit(source_char, target_char, "punctuation_replace", source_index, source_index + 1, spec))

    for source_index, source_char in source_punct[paired_count:]:
        edits.append(_edit(source_char, "", "punctuation_delete", source_index, source_index + 1, spec))

    insert_position = source_punct[-1][0] + 1
    for _target_index, target_char in target_punct[paired_count:]:
        edits.append(_edit("", target_char, "punctuation_insert", insert_position, insert_position, spec))
    return edits


def _edit(source: str, replacement: str, edit_type: str, start: int, end: int, spec: RuleSpec) -> RuleEdit:
    return RuleEdit(
        source=source,
        replacement=replacement,
        edit_type=edit_type,
        start=start,
        end=end,
        confidence=spec.confidence,
        requires_model=spec.mode in {"candidate_only", "model_required"},
        rule_id=spec.id,
        mode=spec.mode,
        group=spec.group,
        requires=spec.requires,
    )


PUNCTUATION_RULES: tuple[object, ...] = (
    FunctionPunctuationRule(
        RuleSpec(
            id="punctuation_delete_replace",
            group="delete_replace",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="deterministic",
            confidence=0.85,
            requires=("none",),
            description="Remove obvious duplicate punctuation and punctuation inside paired marks.",
        ),
        _remove_obvious_extra_punctuation,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="direct_speech_quotes",
            group="direct_speech",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.86,
            requires=("syntax", "model"),
            description="Generate bounded direct speech quote candidates for model scoring.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="direct_speech_colon",
            group="direct_speech",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.86,
            requires=("syntax", "model"),
            description="Generate bounded colon-before-direct-speech candidates for model scoring.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="direct_speech_dash",
            group="direct_speech",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.84,
            requires=("syntax", "model"),
            description="Generate bounded dash candidates around already quoted direct speech.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="quote_open",
            group="quotes_brackets",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.82,
            requires=("model",),
            description="Generate opening quote candidates for model scoring.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="quote_close",
            group="quotes_brackets",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.82,
            requires=("model",),
            description="Generate closing quote candidates for model scoring.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="quote_pair_balance",
            group="quotes_brackets",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.82,
            requires=("model",),
            description="Generate quote pair-balance candidates only for one-sided imbalance.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="bracket_pair_balance",
            group="quotes_brackets",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.82,
            requires=("model",),
            description="Generate bracket pair-balance candidates only for one-sided imbalance.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="subject_predicate_dash",
            group="dash",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.86,
            requires=("syntax", "model"),
            description="Generate subject-predicate dash edits for model scoring.",
        ),
        _add_obvious_subject_predicate_dash,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="enumeration_colon",
            group="colon",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.86,
            requires=("syntax", "model"),
            description="Generate enumeration colon edits for model scoring.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="explanation_colon",
            group="colon",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.82,
            requires=("syntax", "model"),
            description="Generate bounded explanation colon candidates for model scoring.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="consequence_dash",
            group="dash",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.82,
            requires=("syntax", "model"),
            description="Generate bounded consequence dash candidates for model scoring.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="asyndetic_dash",
            group="dash",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.82,
            requires=("syntax", "model"),
            description="Generate bounded asyndetic dash candidates for model scoring.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="semicolon",
            group="semicolon",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.82,
            requires=("syntax", "model"),
            description="Generate bounded semicolon candidates for model scoring.",
        ),
        _identity,
    ),
    RegexPunctuationRule(
        RuleSpec(
            id="comma_subordinate",
            group="comma_subordinate",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.86,
            requires=("syntax", "model"),
            description="Generate subordinate comma edits for model scoring.",
        ),
        r"\b(не знаю|думаю|считаю) что\b",
        r"\1, что",
        re.IGNORECASE,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="comma_conjunction",
            group="comma_conjunction",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.84,
            requires=("syntax", "model"),
            description="Generate bounded comma candidates before explicit adversative conjunctions for model scoring.",
        ),
        _identity,
    ),
    RegexPunctuationRule(
        RuleSpec(
            id="introductory_comma",
            group="introductory",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.86,
            requires=("syntax", "model"),
            description="Generate introductory comma edits for model scoring.",
        ),
        r"\b(во-первых|во-вторых|в-третьих)\s+(?!,)",
        r"\1, ",
        re.IGNORECASE,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="address_comma",
            group="addresses",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.82,
            requires=("syntax", "model"),
            description="Generate explicit address comma candidates when syntax features are available.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="homogeneous_comma",
            group="homogeneous_members",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.82,
            requires=("syntax", "model"),
            description="Generate bounded comma candidates for repeated conjunction homogeneous-member patterns.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="detached_adverbial_comma",
            group="detached_members",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.8,
            requires=("syntax", "model"),
            description="Generate bounded comma candidates for clear sentence-initial gerundial turnovers.",
        ),
        _identity,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="comparative_turnover_comma",
            group="comparative_turnovers",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.8,
            requires=("syntax", "model"),
            description="Generate bounded comma candidates for explicit comparative markers excluding bare как.",
        ),
        _identity,
    ),
    FinalPunctuationRule(),
)
