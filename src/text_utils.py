"""Text tokenization and reconstruction helpers for the hybrid corrector."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Sequence


WORD_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*$")
TOKEN_RE = re.compile(
    r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*"
    r"|\d+(?:[.,:/-]\d+)*"
    r"|[^\w\s]",
    re.UNICODE,
)

PUNCT_CHARS = set('.,!?;:-"()[]{}—–«»')
SIMPLE_PUNCT_CHARS = set(",.!?:;")
STRUCTURAL_PUNCT_CHARS = set('"()[]{}—–«»№/\\')
SENTENCE_END = {".", "!", "?"}
TECHNICAL_ABBREVIATIONS = {
    "ч",
    "ст",
    "п",
    "пп",
    "рис",
    "ул",
    "г",
    "им",
    "т",
    "д",
    "млн",
    "тыс",
    "руб",
    "стр",
    "см",
}
COMMON_DOMAIN_ZONES = {
    "ru",
    "com",
    "net",
    "org",
    "su",
    "рф",
    "by",
    "kz",
    "ua",
    "edu",
    "gov",
    "io",
}
PROTECTED_SPACING_RE = re.compile(
    r"https?://\S+"
    r"|www\.\S+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w.-]+)+"
    r"|\S*/\S*"
    r"|\b[A-Za-z][A-Za-z0-9._/-]*\b"
    r"|\b\d+(?:[.,:/-]\d+)+\b"
    r"|№\s*\d+",
    re.UNICODE,
)


@dataclass(frozen=True)
class TextToken:
    text: str
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class WordSlot:
    word: str
    punct_after: str = ""
    start: int = 0
    end: int = 0


def token_kind(token: str) -> str:
    if WORD_RE.match(token):
        return "word"
    if any(ch.isdigit() for ch in token):
        return "number"
    if all(ch in PUNCT_CHARS for ch in token):
        return "punct"
    return "other"


def tokenize(text: str) -> List[TextToken]:
    return [
        TextToken(m.group(0), m.start(), m.end(), token_kind(m.group(0)))
        for m in TOKEN_RE.finditer(str(text))
    ]


def is_word(token: str) -> bool:
    return bool(WORD_RE.match(str(token)))


def normalize_word(word: str) -> str:
    return str(word).replace("Ё", "Е").replace("ё", "е").lower()


def word_tokens(text: str, include_numbers: bool = False) -> List[str]:
    kinds = {"word", "number"} if include_numbers else {"word"}
    return [t.text for t in tokenize(text) if t.kind in kinds]


def extract_word_slots(text: str) -> List[WordSlot]:
    """Extract word/number tokens and attach following punctuation to each one."""
    tokens = tokenize(text)
    slots: List[WordSlot] = []
    last_word_idx: int | None = None

    for token in tokens:
        if token.kind in {"word", "number"}:
            slots.append(WordSlot(token.text, "", token.start, token.end))
            last_word_idx = len(slots) - 1
            continue

        if token.kind == "punct" and last_word_idx is not None:
            current = slots[last_word_idx]
            slots[last_word_idx] = WordSlot(
                current.word,
                current.punct_after + token.text,
                current.start,
                current.end,
            )

    return slots


def choose_punctuation_label(punct: str) -> str:
    """Collapse punctuation after a token to one conservative trainable label."""
    punct = str(punct)
    if any(ch in STRUCTURAL_PUNCT_CHARS for ch in punct):
        return ""
    if "?" in punct:
        return "?"
    if "!" in punct:
        return "!"
    if "." in punct or "…" in punct:
        return "."
    if "," in punct:
        return ","
    if ":" in punct:
        return ":"
    if ";" in punct:
        return ";"
    return ""


def choose_punctuation_label_for_gap(
    punct: str,
    *,
    word: str = "",
    gap: str = "",
    next_word: str = "",
    is_final: bool = False,
) -> str:
    """Return a trainable punctuation label unless the gap is protected."""
    if is_protected_punctuation_gap(gap or punct, word=word, next_word=next_word, is_final=is_final):
        return ""
    return choose_punctuation_label(punct)


def punctuation_labels_for_slots(text: str, slots: Sequence[WordSlot]) -> List[str]:
    labels: List[str] = []
    for i, slot in enumerate(slots):
        next_start = slots[i + 1].start if i + 1 < len(slots) else len(text)
        gap = str(text)[slot.end:next_start]
        next_word = slots[i + 1].word if i + 1 < len(slots) else ""
        labels.append(
            choose_punctuation_label_for_gap(
                slot.punct_after,
                word=slot.word,
                gap=gap,
                next_word=next_word,
                is_final=i + 1 >= len(slots),
            )
        )
    return labels


def is_simple_punctuation_gap(gap: str) -> bool:
    """Return True when a gap can be safely rewritten by the punctuation head."""
    gap = str(gap)
    if any(ch in STRUCTURAL_PUNCT_CHARS for ch in gap):
        return False
    return all(ch.isspace() or ch in SIMPLE_PUNCT_CHARS for ch in gap)


def is_trainable_punctuation_gap(
    gap: str,
    *,
    word: str = "",
    next_word: str = "",
    is_final: bool = False,
) -> bool:
    return is_simple_punctuation_gap(gap) and not is_protected_punctuation_gap(
        gap,
        word=word,
        next_word=next_word,
        is_final=is_final,
    )


def is_protected_punctuation_gap(
    gap: str,
    *,
    word: str = "",
    next_word: str = "",
    is_final: bool = False,
) -> bool:
    """Return True for punctuation that should be preserved byte-for-byte."""
    gap = str(gap)
    if not gap:
        return False
    if any(ch in STRUCTURAL_PUNCT_CHARS for ch in gap):
        return True

    word_s = str(word or "")
    next_s = str(next_word or "")
    norm = normalize_word(word_s).strip(".")
    next_norm = normalize_word(next_s).strip(".")
    has_dot = "." in gap
    has_colon = ":" in gap

    if has_dot and norm in TECHNICAL_ABBREVIATIONS:
        return True
    if has_dot and len(word_s) == 1 and (word_s.isupper() or _has_latin(word_s)):
        return True
    if has_dot and (_has_latin(word_s) or _has_latin(next_s)):
        return True
    if has_dot and next_norm in COMMON_DOMAIN_ZONES:
        return True
    if has_dot and any(ch.isdigit() for ch in next_s):
        return True
    if has_colon and any(ch.isdigit() for ch in word_s + next_s):
        return True
    if any(ch.isdigit() for ch in word_s) and any(ch.isdigit() for ch in next_s):
        return True
    return False


def _has_latin(text: str) -> bool:
    return any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in str(text))


def render_simple_gap(label: str, is_final: bool) -> str:
    label = str(label or "")
    if label and label not in SIMPLE_PUNCT_CHARS:
        label = ""
    if is_final:
        return label
    return f"{label} " if label else " "


def normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=[A-Za-zА-Яа-яЁё0-9])", r"\1 ", text)
    text = re.sub(r"\s*([—–])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deterministic_spacing_correction(text: str) -> tuple[str, bool]:
    """Fix only high-precision whitespace issues in plain Cyrillic context."""
    current = str(text)
    protected = _protected_spacing_spans(current)
    current = _collapse_safe_spaces(current, protected)
    protected = _protected_spacing_spans(current)
    current = _remove_safe_space_before_punctuation(current, protected)
    protected = _protected_spacing_spans(current)
    current = _add_safe_space_after_punctuation(current, protected)
    protected = _protected_spacing_spans(current)
    current = _normalize_safe_dash_spacing(current, protected)
    return current, current != str(text)


def _protected_spacing_spans(text: str) -> List[tuple[int, int]]:
    return [match.span() for match in PROTECTED_SPACING_RE.finditer(str(text))]


def _index_in_spans(index: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in spans)


def _span_touches_protected(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _is_cyrillic_letter(ch: str) -> bool:
    return bool(re.match(r"[А-Яа-яЁё]", str(ch or "")))


def _is_latin_letter(ch: str) -> bool:
    return bool(re.match(r"[A-Za-z]", str(ch or "")))


def _safe_spacing_neighbors(text: str, prev_idx: int, next_idx: int, spans: Sequence[tuple[int, int]]) -> bool:
    if prev_idx < 0 or next_idx >= len(text):
        return False
    if _index_in_spans(prev_idx, spans) or _index_in_spans(next_idx, spans):
        return False
    prev_ch = text[prev_idx]
    next_ch = text[next_idx]
    if prev_ch in STRUCTURAL_PUNCT_CHARS or next_ch in STRUCTURAL_PUNCT_CHARS:
        return False
    if _is_latin_letter(prev_ch) or _is_latin_letter(next_ch):
        return False
    if prev_ch.isdigit() or next_ch.isdigit():
        return False
    return True


def _collapse_safe_spaces(text: str, spans: Sequence[tuple[int, int]]) -> str:
    def replace(match: re.Match) -> str:
        start, end = match.span()
        prev_idx = start - 1
        next_idx = end
        if not _safe_spacing_neighbors(text, prev_idx, next_idx, spans):
            return match.group(0)
        prev_ch = text[prev_idx]
        next_ch = text[next_idx]
        if (_is_cyrillic_letter(prev_ch) or prev_ch in SIMPLE_PUNCT_CHARS) and _is_cyrillic_letter(next_ch):
            return " "
        return match.group(0)

    return re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", replace, text)


def _remove_safe_space_before_punctuation(text: str, spans: Sequence[tuple[int, int]]) -> str:
    def replace(match: re.Match) -> str:
        start, end = match.span()
        punct = match.group(1)
        punct_idx = end - 1
        prev_idx = start - 1
        next_idx = end if end < len(text) else -1
        if prev_idx < 0 or _span_touches_protected(start, end, spans) or _index_in_spans(prev_idx, spans):
            return match.group(0)
        prev_ch = text[prev_idx]
        next_ch = text[next_idx] if next_idx >= 0 else ""
        if prev_ch.isdigit() or next_ch.isdigit() or _is_latin_letter(prev_ch) or _is_latin_letter(next_ch):
            return match.group(0)
        if prev_ch in STRUCTURAL_PUNCT_CHARS:
            return match.group(0)
        if punct == "." and _looks_like_abbreviation_dot(text, punct_idx, next_idx):
            return match.group(0)
        return punct

    return re.sub(r"[ \t]+([,.;:!?])", replace, text)


def _add_safe_space_after_punctuation(text: str, spans: Sequence[tuple[int, int]]) -> str:
    def replace(match: re.Match) -> str:
        punct = match.group(1)
        punct_idx = match.start(1)
        next_idx = match.end(1)
        prev_idx = punct_idx - 1
        if next_idx >= len(text) or not _safe_spacing_neighbors(text, prev_idx, next_idx, spans):
            return punct
        if not _is_cyrillic_letter(text[next_idx]):
            return punct
        if punct == "." and _looks_like_abbreviation_dot(text, punct_idx, next_idx):
            return punct
        return punct + " "

    return re.sub(r"([,.;:!?])(?=\S)", replace, text)


def _normalize_safe_dash_spacing(text: str, spans: Sequence[tuple[int, int]]) -> str:
    def replace(match: re.Match) -> str:
        start, end = match.span()
        if _span_touches_protected(start, end, spans):
            return match.group(0)
        return f" {match.group(1)} "

    return re.sub(r"(?<=[А-Яа-яЁё])[ \t]*([—–])[ \t]*(?=[А-Яа-яЁё])", replace, text)


def _looks_like_abbreviation_dot(text: str, punct_idx: int, next_idx: int) -> bool:
    if punct_idx <= 0 or next_idx >= len(text):
        return False
    prev_match = re.search(r"[A-Za-zА-Яа-яЁё]+$", text[:punct_idx])
    next_match = re.match(r"[A-Za-zА-Яа-яЁё]+", text[next_idx:])
    if not prev_match or not next_match:
        return False
    return len(prev_match.group(0)) == 1 and len(next_match.group(0)) == 1


def apply_case_like(candidate: str, source: str) -> str:
    if not candidate:
        return candidate
    if source.isupper():
        return candidate.upper()
    if source[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate


def rebuild_from_slots(words: Iterable[str], puncts: Iterable[str]) -> str:
    chunks: List[str] = []
    for word, punct in zip(words, puncts):
        if not word:
            continue
        chunks.append(str(word) + str(punct or ""))
    return normalize_spacing(" ".join(chunks))


def rebuild_preserving_layout(
    original: str,
    slots: Sequence[WordSlot],
    words: Sequence[str],
    puncts: Sequence[str] | None = None,
    *,
    punctuation_mode: str = "conservative",
    source_puncts: Sequence[str] | None = None,
    rewrite_punct_indices: Iterable[int] | None = None,
) -> str:
    """Replace token text while preserving protected punctuation and spacing.

    In conservative mode only simple gaps made from whitespace and ``.,!?;:``
    may be rewritten by punctuation predictions. Quotes, dashes, brackets,
    slashes, numbers markers and other structural punctuation stay untouched.
    """
    original = str(original)
    if not slots:
        return original

    chunks: List[str] = []
    cursor = 0
    limit = min(len(slots), len(words))
    rewrite_set = set(rewrite_punct_indices or [])

    for i in range(limit):
        slot = slots[i]
        next_start = slots[i + 1].start if i + 1 < len(slots) else len(original)

        chunks.append(original[cursor:slot.start])
        output_word = str(words[i] or "")
        chunks.append(output_word)

        gap = original[slot.end:next_start]
        predicted = puncts[i] if puncts is not None and i < len(puncts) else ""
        source_label = source_puncts[i] if source_puncts is not None and i < len(source_puncts) else choose_punctuation_label_for_gap(
            slot.punct_after,
            word=slot.word,
            gap=gap,
            next_word=slots[i + 1].word if i + 1 < len(slots) else "",
            is_final=i + 1 >= len(slots),
        )
        if (
            puncts is not None
            and punctuation_mode == "conservative"
            and (i in rewrite_set or predicted != source_label)
            and is_trainable_punctuation_gap(
                gap,
                word=slot.word,
                next_word=slots[i + 1].word if i + 1 < len(slots) else "",
                is_final=i + 1 >= len(slots),
            )
        ):
            chunks.append(render_simple_gap(predicted, is_final=i + 1 >= len(slots)))
        else:
            chunks.append(gap)

        cursor = next_start

    if limit < len(slots):
        chunks.append(original[cursor:])

    rebuilt = "".join(chunks)
    if punctuation_mode == "legacy":
        return normalize_spacing(rebuilt)
    return rebuilt


def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like chunks while preserving final punctuation."""
    text = str(text)
    if not text.strip():
        return [text]

    parts = re.split(r"([.!?]+(?:\s+|$))", text)
    sentences: List[str] = []
    for i in range(0, len(parts), 2):
        body = parts[i]
        tail = parts[i + 1] if i + 1 < len(parts) else ""
        piece = body + tail
        if piece:
            sentences.append(piece)
    return sentences or [text]
