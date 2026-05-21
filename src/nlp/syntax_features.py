from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.nlp.syntax_types import ClauseSpan, PhraseSpan, SyntaxAnalysis, SyntaxToken
from src.preprocessing.protected_spans import find_protected_spans
from src.preprocessing.tokenizer import PUNCTUATION, tokenize_words


SUBORDINATE_MARKERS = frozenset({"что", "чтобы", "если", "когда", "поскольку", "где", "куда", "откуда"})
COMPLEX_SUBORDINATE_MARKERS = (("потому", "что"), ("так", "как"))
COORDINATING_MARKERS = frozenset({"а", "но", "однако", "зато"})
INTRODUCTORY_WORDS = frozenset({"конечно", "например", "во-первых", "кажется", "возможно", "следовательно"})
ADDRESS_OPENINGS = frozenset({"коллеги", "иван", "мария"})
ADDRESS_IMPERATIVE_HINTS = frozenset({"добавь", "исправь", "напиши", "открой", "отправь", "посмотри", "проверь", "скажи", "сделай", "укажи"})
ADDRESS_HORTATIVE_HINTS = frozenset({"проверим", "исправим", "посмотрим", "отправим", "сделаем", "укажем"})
REPEATED_HOMOGENEOUS_CONJUNCTIONS = frozenset({"и", "ни"})
COMPARATIVE_MARKERS = frozenset({"словно", "будто", "чем"})
COMPLEX_COMPARATIVE_MARKERS = (("как", "будто"),)
CLARIFICATION_MARKERS = (("то", "есть"), ("а", "именно"))
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
ADVERBIAL_PARTICIPLE_SUFFIXES = ("вшись", "ившись", "авшись", "явшись", "вши", "ивши", "авши", "явши", "ив", "ав", "яв")
PARTICIPLE_SUFFIXES = (
    "вший",
    "вшая",
    "вшее",
    "вшие",
    "вшего",
    "вшей",
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
PAIR_CHARS = {"«": "»", '"': '"', "'": "'", "(": ")", "[": "]"}
PUNCTUATION_CHARS = set(",.!?:;—…\"'()«»[]")


@dataclass(frozen=True)
class _FeatureToken:
    id: int
    text: str
    start: int
    end: int
    lemma: str = ""
    pos: str = ""
    feats: dict[str, Any] | None = None
    head_id: int | None = None
    dep_rel: str = ""
    sentence_id: int = 0

    @property
    def is_punctuation(self) -> bool:
        return self.text in PUNCTUATION


class SyntaxFeatureExtractor:
    """Extract syntax-support spans and evidence without producing edits."""

    def __init__(self, analysis: SyntaxAnalysis | None = None) -> None:
        self.analysis = analysis
        self._protected_spans = analysis.protected_spans if analysis is not None else []

    def find_sentence_boundaries(self, text: str) -> list[tuple[int, int]]:
        protected = [(span.start, span.end) for span in find_protected_spans(text)]
        boundaries: list[tuple[int, int]] = []
        start = 0
        for index, char in enumerate(text):
            if char not in ".!?…":
                continue
            if _position_inside(index, protected):
                continue
            end = index + 1
            stripped_start = _skip_space_forward(text, start)
            if stripped_start < end:
                boundaries.append((stripped_start, end))
            start = _skip_space_forward(text, end)
        if start < len(text.strip()) or (start < len(text) and text[start:].strip()):
            stripped_start = _skip_space_forward(text, start)
            stripped_end = len(text.rstrip())
            if stripped_start < stripped_end:
                boundaries.append((stripped_start, stripped_end))
        return boundaries or ([(0, len(text))] if text else [])

    def find_clauses(self, analysis: SyntaxAnalysis | None = None) -> list[ClauseSpan]:
        analysis = self._analysis(analysis)
        tokens = self._tokens(analysis)
        if not tokens:
            return []

        clauses: list[ClauseSpan] = []
        subordinate_ids = {marker["token_ids"][0] for marker in self.find_subordinate_conjunctions(analysis) if marker["token_ids"]}
        coordinate_ids = {marker["token_ids"][0] for marker in self.find_coordinating_conjunctions(analysis) if marker["token_ids"]}
        split_ids = sorted(subordinate_ids | coordinate_ids)

        if split_ids:
            first_split = split_ids[0]
            prefix = [token for token in tokens if token.id < first_split and not _is_punctuation_token(token)]
            if prefix:
                clauses.append(_clause_from_tokens(prefix, "main", None, "", 0.65))
            for split_id in split_ids:
                marker = _token_by_id(tokens, split_id)
                suffix = [token for token in tokens if token.id >= split_id and not _is_terminal_punctuation(token)]
                kind = "subordinate" if split_id in subordinate_ids else "coordinate"
                if suffix:
                    clauses.append(_clause_from_tokens(suffix, kind, marker.id if marker else None, marker.text if marker else "", 0.78))
        else:
            sentence_tokens = [token for token in tokens if not _is_terminal_punctuation(token)]
            if sentence_tokens:
                clauses.append(_clause_from_tokens(sentence_tokens, "unknown", None, "", 0.45))
        return _dedupe_clauses(clauses)

    def find_subordinate_conjunctions(self, analysis: SyntaxAnalysis | None = None) -> list[dict[str, Any]]:
        tokens = self._word_tokens(analysis)
        lowered = [token.text.lower() for token in tokens]
        markers: list[dict[str, Any]] = []
        for index, token in enumerate(tokens):
            for marker in COMPLEX_SUBORDINATE_MARKERS:
                end_index = index + len(marker)
                if tuple(lowered[index:end_index]) == marker:
                    marker_tokens = tokens[index:end_index]
                    markers.append(_marker_dict(marker_tokens, " ".join(marker), "subordinate"))
            if lowered[index] in SUBORDINATE_MARKERS:
                markers.append(_marker_dict([token], lowered[index], "subordinate"))
        return _dedupe_markers(markers)

    def find_coordinating_conjunctions(self, analysis: SyntaxAnalysis | None = None) -> list[dict[str, Any]]:
        markers: list[dict[str, Any]] = []
        for token in self._word_tokens(analysis):
            lowered = token.text.lower()
            if lowered in COORDINATING_MARKERS:
                markers.append(_marker_dict([token], lowered, "coordinate"))
        return markers

    def find_introductory_candidates(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        phrases: list[PhraseSpan] = []
        for token in self._word_tokens(analysis):
            if token.text.lower() in INTRODUCTORY_WORDS:
                phrases.append(_phrase([token], "clarification", 0.58, f"introductory:{token.text}"))
        return phrases

    def find_address_candidates(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        words = self._word_tokens(analysis)
        if len(words) < 2:
            return []
        first, second = words[0], words[1]
        first_lower = first.text.lower()
        second_lower = second.text.lower()
        if first_lower in ADDRESS_OPENINGS and second_lower in ADDRESS_IMPERATIVE_HINTS | ADDRESS_HORTATIVE_HINTS:
            return [_phrase([first], "clarification", 0.68, f"address:{first.text}")]
        if first.text[:1].isupper() and second_lower in ADDRESS_IMPERATIVE_HINTS:
            return [_phrase([first], "clarification", 0.55, f"address_candidate:{first.text}")]
        return []

    def find_homogeneous_series(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        words = self._word_tokens(analysis)
        lowered = [word.text.lower() for word in words]
        phrases: list[PhraseSpan] = []
        for index in range(2, len(words) - 1):
            if lowered[index] in REPEATED_HOMOGENEOUS_CONJUNCTIONS and lowered[index - 2] == lowered[index]:
                phrases.append(_phrase(words[index - 2 : index + 2], "homogeneous_series", 0.7, "repeated conjunction"))
        return phrases

    def find_adverbial_participle_phrases(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        tokens = self._tokens(analysis)
        phrases: list[PhraseSpan] = []
        for token in tokens:
            if not _looks_like_adverbial_participle(token):
                continue
            phrase_tokens = _tokens_until_punctuation(tokens, token.id, ",;:—")
            if phrase_tokens:
                phrases.append(_phrase(phrase_tokens, "adverbial_participle", 0.74, _span_text(analysis.text if analysis else "", phrase_tokens)))
        return _dedupe_phrases(phrases)

    def find_participial_phrases(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        analysis = self._analysis(analysis)
        tokens = self._tokens(analysis)
        phrases: list[PhraseSpan] = []
        for token in tokens:
            if not _looks_like_participle(token):
                continue
            phrase_tokens = _comma_bounded_tokens(analysis.text, tokens, token.id) or _tokens_until_punctuation(tokens, token.id, ",;:—")
            if phrase_tokens:
                phrases.append(_phrase(phrase_tokens, "participial", 0.72, _span_text(analysis.text, phrase_tokens)))
        return _dedupe_phrases(phrases)

    def find_appositions(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        analysis = self._analysis(analysis)
        phrases: list[PhraseSpan] = []
        for match in re.finditer(r",\s*([А-ЯЁа-яё][^,]{1,50})\s*,", analysis.text):
            start, end = match.start(1), match.end(1)
            phrases.append(PhraseSpan(start, end, _token_ids_in_span(self._tokens(analysis), start, end), "apposition", 0.42, match.group(1)))
        return phrases

    def find_clarifications(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        words = self._word_tokens(analysis)
        lowered = [word.text.lower() for word in words]
        phrases: list[PhraseSpan] = []
        for index, word in enumerate(words):
            if lowered[index] == "например":
                phrases.append(_phrase([word], "clarification", 0.62, "marker:например"))
            for marker in CLARIFICATION_MARKERS:
                end_index = index + len(marker)
                if tuple(lowered[index:end_index]) == marker:
                    phrases.append(_phrase(words[index:end_index], "clarification", 0.66, f"marker:{' '.join(marker)}"))
        return phrases

    def find_comparative_turnovers(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        words = self._word_tokens(analysis)
        lowered = [word.text.lower() for word in words]
        phrases: list[PhraseSpan] = []
        for index, word in enumerate(words):
            if lowered[index] in COMPARATIVE_MARKERS:
                phrases.append(_phrase(words[index : min(len(words), index + 4)], "comparative", 0.62, f"marker:{word.text}"))
            for marker in COMPLEX_COMPARATIVE_MARKERS:
                end_index = index + len(marker)
                if tuple(lowered[index:end_index]) == marker:
                    phrases.append(_phrase(words[index : min(len(words), index + 4)], "comparative", 0.68, f"marker:{' '.join(marker)}"))
        return _dedupe_phrases(phrases)

    def find_subject_predicate_dash_candidates(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        analysis = self._analysis(analysis)
        tokens = self._tokens(analysis)
        phrases: list[PhraseSpan] = []
        for match in re.finditer(r"\s[—-]\s", analysis.text):
            left = _nearest_token_before(tokens, match.start())
            right = _nearest_token_after(tokens, match.end())
            if left is not None and right is not None:
                phrases.append(PhraseSpan(left.start, right.end, [left.id, right.id], "subject_predicate", 0.7, "dash between nominal parts"))
        words = self._word_tokens(analysis)
        for index, word in enumerate(words):
            if word.text.lower() == "это" and 0 < index < len(words) - 1:
                phrases.append(_phrase(words[index - 1 : index + 2], "subject_predicate", 0.58, "marker:это"))
        return _dedupe_phrases(phrases)

    def find_asyndetic_candidates(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        analysis = self._analysis(analysis)
        tokens = self._tokens(analysis)
        phrases: list[PhraseSpan] = []
        for match in re.finditer(r"\s[—:;]\s", analysis.text):
            left = _nearest_token_before(tokens, match.start())
            right = _nearest_token_after(tokens, match.end())
            if left is not None and right is not None:
                phrases.append(PhraseSpan(left.start, right.end, [left.id, right.id], "subject_predicate", 0.45, "asyndetic punctuation boundary"))
        return phrases

    def find_direct_speech_spans(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        analysis = self._analysis(analysis)
        phrases: list[PhraseSpan] = []
        speech_verbs = "|".join(sorted(SPEECH_VERBS, key=len, reverse=True))
        pattern = rf"[«\"][^»\"]+[»\"]\s*,?\s*[—-]\s*(?:{speech_verbs})\b[^.!?…]*"
        for match in re.finditer(pattern, analysis.text, flags=re.IGNORECASE):
            phrases.append(PhraseSpan(match.start(), match.end(), _token_ids_in_span(self._tokens(analysis), match.start(), match.end()), "direct_speech", 0.78, match.group(0)))
        after_author = rf"\b(?:{speech_verbs})\s*:\s*[«\"][^»\"]+[»\"]"
        for match in re.finditer(after_author, analysis.text, flags=re.IGNORECASE):
            phrases.append(PhraseSpan(match.start(), match.end(), _token_ids_in_span(self._tokens(analysis), match.start(), match.end()), "direct_speech", 0.72, match.group(0)))
        return _dedupe_phrases(phrases)

    def find_quote_spans(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        analysis = self._analysis(analysis)
        return _paired_spans(analysis, "«", "»", "quote_span") + _paired_spans(analysis, '"', '"', "quote_span")

    def find_bracket_spans(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        analysis = self._analysis(analysis)
        return _paired_spans(analysis, "(", ")", "bracket_span") + _paired_spans(analysis, "[", "]", "bracket_span")

    def is_safe_gap_for_punctuation(self, text: str, start: int, end: int) -> bool:
        if start < 0 or end < start or end > len(text):
            return False
        if self.is_inside_protected_span(start, end):
            return False
        return not any(char in PUNCTUATION_CHARS for char in text[start:end])

    def is_inside_protected_span(self, start: int, end: int) -> bool:
        if start == end:
            return any(span_start < start < span_end for span_start, span_end in self._protected_spans)
        return any(start < span_end and span_start < end for span_start, span_end in self._protected_spans)

    def collect_phrase_spans(self, analysis: SyntaxAnalysis | None = None) -> list[PhraseSpan]:
        analysis = self._analysis(analysis)
        phrases: list[PhraseSpan] = []
        phrases.extend(self.find_introductory_candidates(analysis))
        phrases.extend(self.find_address_candidates(analysis))
        phrases.extend(self.find_homogeneous_series(analysis))
        phrases.extend(self.find_adverbial_participle_phrases(analysis))
        phrases.extend(self.find_participial_phrases(analysis))
        phrases.extend(self.find_appositions(analysis))
        phrases.extend(self.find_clarifications(analysis))
        phrases.extend(self.find_comparative_turnovers(analysis))
        phrases.extend(self.find_subject_predicate_dash_candidates(analysis))
        phrases.extend(self.find_asyndetic_candidates(analysis))
        phrases.extend(self.find_direct_speech_spans(analysis))
        phrases.extend(self.find_quote_spans(analysis))
        phrases.extend(self.find_bracket_spans(analysis))
        return _dedupe_phrases(phrases)

    def _analysis(self, analysis: SyntaxAnalysis | None) -> SyntaxAnalysis:
        resolved = analysis or self.analysis
        if resolved is None:
            raise ValueError("SyntaxAnalysis is required")
        return resolved

    def _tokens(self, analysis: SyntaxAnalysis | None) -> list[SyntaxToken | _FeatureToken]:
        resolved = self._analysis(analysis)
        if resolved.tokens:
            return list(resolved.tokens)
        return [
            _FeatureToken(id=index, text=token.text, start=token.start, end=token.end, sentence_id=0)
            for index, token in enumerate(tokenize_words(resolved.text))
        ]

    def _word_tokens(self, analysis: SyntaxAnalysis | None) -> list[SyntaxToken | _FeatureToken]:
        return [token for token in self._tokens(analysis) if not _is_punctuation_token(token)]


def _clause_from_tokens(
    tokens: list[SyntaxToken | _FeatureToken],
    kind: str,
    trigger_token_id: int | None,
    trigger_text: str,
    confidence: float,
) -> ClauseSpan:
    return ClauseSpan(
        start=min(token.start for token in tokens),
        end=max(token.end for token in tokens),
        token_ids=[token.id for token in tokens],
        kind=kind,  # type: ignore[arg-type]
        trigger_token_id=trigger_token_id,
        trigger_text=trigger_text,
        confidence=confidence,
    )


def _marker_dict(tokens: list[SyntaxToken | _FeatureToken], text: str, kind: str) -> dict[str, Any]:
    return {
        "start": min(token.start for token in tokens),
        "end": max(token.end for token in tokens),
        "token_ids": [token.id for token in tokens],
        "text": text,
        "kind": kind,
        "confidence": 0.75 if len(tokens) > 1 else 0.65,
    }


def _phrase(tokens: list[SyntaxToken | _FeatureToken], kind: str, confidence: float, evidence: str) -> PhraseSpan:
    return PhraseSpan(
        start=min(token.start for token in tokens),
        end=max(token.end for token in tokens),
        token_ids=[token.id for token in tokens],
        kind=kind,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=evidence,
    )


def _paired_spans(analysis: SyntaxAnalysis, open_char: str, close_char: str, kind: str) -> list[PhraseSpan]:
    spans: list[PhraseSpan] = []
    stack: list[int] = []
    if open_char == close_char:
        indexes = [match.start() for match in re.finditer(re.escape(open_char), analysis.text)]
        for left, right in zip(indexes[0::2], indexes[1::2], strict=False):
            spans.append(PhraseSpan(left, right + 1, _token_ids_in_span(_tokens_for_pairing(analysis), left, right + 1), kind, 0.75, analysis.text[left : right + 1]))  # type: ignore[arg-type]
        return spans

    for index, char in enumerate(analysis.text):
        if char == open_char:
            stack.append(index)
        elif char == close_char and stack:
            start = stack.pop()
            spans.append(PhraseSpan(start, index + 1, _token_ids_in_span(_tokens_for_pairing(analysis), start, index + 1), kind, 0.82, analysis.text[start : index + 1]))  # type: ignore[arg-type]
    return spans


def _tokens_for_pairing(analysis: SyntaxAnalysis) -> list[SyntaxToken | _FeatureToken]:
    if analysis.tokens:
        return list(analysis.tokens)
    return [_FeatureToken(id=index, text=token.text, start=token.start, end=token.end) for index, token in enumerate(tokenize_words(analysis.text))]


def _token_ids_in_span(tokens: list[SyntaxToken | _FeatureToken], start: int, end: int) -> list[int]:
    return [token.id for token in tokens if start <= token.start and token.end <= end]


def _token_by_id(tokens: list[SyntaxToken | _FeatureToken], token_id: int) -> SyntaxToken | _FeatureToken | None:
    return next((token for token in tokens if token.id == token_id), None)


def _nearest_token_before(tokens: list[SyntaxToken | _FeatureToken], position: int) -> SyntaxToken | _FeatureToken | None:
    return next((token for token in reversed(tokens) if not _is_punctuation_token(token) and token.end <= position), None)


def _nearest_token_after(tokens: list[SyntaxToken | _FeatureToken], position: int) -> SyntaxToken | _FeatureToken | None:
    return next((token for token in tokens if not _is_punctuation_token(token) and token.start >= position), None)


def _tokens_until_punctuation(tokens: list[SyntaxToken | _FeatureToken], start_id: int, punctuation: str) -> list[SyntaxToken | _FeatureToken]:
    collected: list[SyntaxToken | _FeatureToken] = []
    for token in tokens:
        if token.id < start_id:
            continue
        if token.id > start_id and token.text in punctuation:
            break
        if not _is_punctuation_token(token):
            collected.append(token)
    return collected[:6]


def _comma_bounded_tokens(
    text: str,
    tokens: list[SyntaxToken | _FeatureToken],
    token_id: int,
) -> list[SyntaxToken | _FeatureToken]:
    token = _token_by_id(tokens, token_id)
    if token is None:
        return []
    left_comma = text.rfind(",", 0, token.start)
    right_comma = text.find(",", token.end)
    if left_comma < 0 or right_comma < 0:
        return []
    return [item for item in tokens if left_comma < item.start and item.end < right_comma and not _is_punctuation_token(item)]


def _looks_like_adverbial_participle(token: SyntaxToken | _FeatureToken) -> bool:
    feats = getattr(token, "feats", None) or {}
    lower = token.text.lower()
    if str(feats.get("VerbForm", "")) == "Conv":
        return True
    return len(lower) >= 6 and lower.endswith(ADVERBIAL_PARTICIPLE_SUFFIXES)


def _looks_like_participle(token: SyntaxToken | _FeatureToken) -> bool:
    feats = getattr(token, "feats", None) or {}
    lower = token.text.lower()
    if str(feats.get("VerbForm", "")) == "Part":
        return True
    return len(lower) >= 7 and lower.endswith(PARTICIPLE_SUFFIXES)


def _is_punctuation_token(token: SyntaxToken | _FeatureToken) -> bool:
    return bool(getattr(token, "is_punctuation", False)) or token.text in PUNCTUATION_CHARS


def _is_terminal_punctuation(token: SyntaxToken | _FeatureToken) -> bool:
    return token.text in ".!?…"


def _span_text(text: str, tokens: list[SyntaxToken | _FeatureToken]) -> str:
    if not text:
        return " ".join(token.text for token in tokens)
    return text[min(token.start for token in tokens) : max(token.end for token in tokens)]


def _dedupe_markers(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for marker in markers:
        key = (int(marker["start"]), int(marker["end"]), str(marker["kind"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(marker)
    return result


def _dedupe_phrases(phrases: list[PhraseSpan]) -> list[PhraseSpan]:
    result: list[PhraseSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for phrase in phrases:
        key = (phrase.start, phrase.end, phrase.kind)
        if key in seen:
            continue
        seen.add(key)
        result.append(phrase)
    return result


def _dedupe_clauses(clauses: list[ClauseSpan]) -> list[ClauseSpan]:
    result: list[ClauseSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for clause in clauses:
        key = (clause.start, clause.end, clause.kind)
        if key in seen:
            continue
        seen.add(key)
        result.append(clause)
    return result


def _skip_space_forward(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _position_inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)
