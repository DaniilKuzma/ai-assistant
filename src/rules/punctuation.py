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
PROTECTED_PUNCTUATION_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?%|(?<![\w])\d+(?:[.,:/-]\d+)*")
PUNCTUATION_NOISE_RE = re.compile(r"(?:…[.!?…]+|[.!?]+…|\.{2,}|[!?]{2,}|([,;:])\s*\1)")


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

    syntax_tokens = _parse_syntax_safely(text)
    if len(syntax_tokens) >= 2:
        first, second = syntax_tokens[0], syntax_tokens[1]
        if (
            first.start == words[0].start
            and first.end == words[0].end
            and _looks_like_explicit_address(first, second)
        ):
            return _address_candidate_after_first_word(text, words, protected)

    if _looks_like_lexical_address_opening(words):
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
    return _comma_after_word(text, words, 1, protected, spec)


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
        if not words[index - 1].text[:1].isupper():
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


def _looks_like_sentence_initial_gerund(text: str, words: list[Token]) -> bool:
    if words[0].start != 0:
        return False
    syntax_tokens = _parse_syntax_safely(text)
    if syntax_tokens and _syntax_token_matches_word(syntax_tokens[0], words[0]):
        feats = getattr(syntax_tokens[0], "feats", {}) or {}
        if str(feats.get("VerbForm", "")) == "Conv":
            return True

    first = words[0].text.lower()
    return len(first) >= 6 and first.endswith(GERUND_FALLBACK_SUFFIXES)


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
            requires=("model",),
            description="Generate simple direct speech quote normalization edits for model scoring.",
        ),
        _normalize_simple_direct_speech_quotes,
    ),
    FunctionPunctuationRule(
        RuleSpec(
            id="direct_speech_colon",
            group="direct_speech",
            scope="punctuation_gap",
            edit_type="punctuation",
            mode="model_required",
            confidence=0.86,
            requires=("model",),
            description="Generate colon-before-direct-speech edits for model scoring.",
        ),
        _add_simple_direct_speech_colon,
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
        _add_simple_enumeration_colon,
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
