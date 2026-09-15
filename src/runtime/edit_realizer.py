from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from collections.abc import Sequence

import yaml

from src.runtime.orthographic_lexicon import OrthographicCorrectionLexicon
from src.schema import RuntimeEdit, WordToken


PUNCTUATION_BY_LABEL = {
    "COMMA": ",",
    "DASH": "—",
    "COLON": ":",
    "SEMICOLON": ";",
    "DOT": ".",
    "QUESTION": "?",
    "EXCLAMATION": "!",
    "ELLIPSIS": "…",
    "COMMA_DASH": ", —",
}
PUNCTUATION_CHARS = set(",;:\u2014-.!?\u2026")
DELETE_PUNCTUATION_CHARS = PUNCTUATION_CHARS
MAX_LEXICON_SPAN_TOKENS = 4
OPEN_QUOTE_DELETE_CHARS = frozenset({"«", '"', "“", "„"})
OPEN_QUOTE_NORMALIZE_CHARS = frozenset({'"', "“", "„"})
CLOSE_QUOTE_DELETE_CHARS = frozenset({"»", '"', "”"})
CLOSE_QUOTE_NORMALIZE_CHARS = frozenset({'"', "”"})
CLOSING_WRAPPER_CHARS = frozenset({"»", '"', "”", ")", "]"})
WRAPPER_AWARE_GAP_LABELS = frozenset({"COMMA", "DOT", "QUESTION", "EXCLAMATION", "ELLIPSIS", "COMMA_DASH"})

TSYA_TO_TTSYA_REPLACEMENTS = {
    "\u043e\u0448\u0438\u0431\u0430\u0435\u0442\u0441\u044f": "\u043e\u0448\u0438\u0431\u0430\u0442\u044c\u0441\u044f",
    "\u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0435\u0442\u0441\u044f": "\u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0442\u044c\u0441\u044f",
}
TTSYA_TO_TSYA_REPLACEMENTS = {value: key for key, value in TSYA_TO_TTSYA_REPLACEMENTS.items()}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


LABEL_RULE_EDIT_TYPE: dict[str, tuple[str, str]] = {
    "DELETE": ("delete", "spelling"),
    "LOWERCASE": ("casing", "casing"),
    "UPPERCASE": ("casing", "casing"),
    "CAPITALIZE": ("casing", "casing"),
    "SPLIT_NE_VERB": ("ne_verb", "split_join"),
    "MERGE_TAK_ZHE_TO_TAKZHE": ("takzhe_tak_zhe", "split_join"),
    "SPLIT_TAKZHE_TO_TAK_ZHE": ("takzhe_tak_zhe", "split_join"),
    "MERGE_TO_ZHE_TO_TOZHE": ("tozhe_to_zhe", "split_join"),
    "SPLIT_TOZHE_TO_TO_ZHE": ("tozhe_to_zhe", "split_join"),
    "MERGE_ZA_TO_TO_ZATO": ("zato_za_to", "split_join"),
    "SPLIT_ZATO_TO_ZA_TO": ("zato_za_to", "split_join"),
    "HYPHENATE_PARTICLE_TO": ("hyphen_particles", "hyphen"),
    "HYPHENATE_PARTICLE_LIBO": ("hyphen_particles", "hyphen"),
    "HYPHENATE_PARTICLE_NIBUD": ("hyphen_particles", "hyphen"),
    "HYPHENATE_KOE": ("hyphen_koe", "hyphen"),
    "HYPHENATE_PO_ADVERB": ("hyphen_po_adverb", "hyphen"),
    "FIX_TSYA_TO_TTSYA": ("tsya_ttsya", "spelling"),
    "FIX_TTSYA_TO_TSYA": ("tsya_ttsya", "spelling"),
}

BOUNDARY_LABEL_RULE_EDIT_TYPE: dict[str, tuple[str, str]] = {
    "INSERT_OPEN_QUOTE": ("quotation_dialogue", "punctuation"),
    "DELETE_OPEN_QUOTE": ("quotation_dialogue", "punctuation"),
    "NORMALIZE_OPEN_QUOTE": ("quotation_dialogue", "punctuation"),
    "INSERT_CLOSE_QUOTE": ("quotation_dialogue", "punctuation"),
    "DELETE_CLOSE_QUOTE": ("quotation_dialogue", "punctuation"),
    "NORMALIZE_CLOSE_QUOTE": ("quotation_dialogue", "punctuation"),
    "INSERT_OPEN_BRACKET": ("quotation_dialogue", "punctuation"),
    "DELETE_OPEN_BRACKET": ("quotation_dialogue", "punctuation"),
    "INSERT_CLOSE_BRACKET": ("quotation_dialogue", "punctuation"),
    "DELETE_CLOSE_BRACKET": ("quotation_dialogue", "punctuation"),
}


def apply_token_edit_labels(
    text: str,
    tokens: Sequence[WordToken],
    labels: Sequence[str],
    confidences: Sequence[float],
    threshold: float,
    *,
    rule_ids: Sequence[str] | None = None,
    orthographic_lexicon: OrthographicCorrectionLexicon | None = None,
) -> tuple[str, list[RuntimeEdit]]:
    edits: list[RuntimeEdit] = []
    consumed: set[int] = set()
    count = min(len(tokens), len(labels), len(confidences))

    for index in range(count):
        if index in consumed:
            continue
        label = str(labels[index])
        if label in {"KEEP", "SKIP_MERGED"}:
            continue
        confidence = float(confidences[index])
        if confidence < threshold:
            continue

        edit = _token_edit_for_label(
            text,
            tokens,
            index,
            label,
            confidence,
            _rule_id_for(index, rule_ids, label),
            orthographic_lexicon,
        )
        if edit is None:
            continue
        edits.append(edit)
        consumed.update(_consumed_indexes(tokens, index, label, edit))

    return apply_runtime_edits(text, edits), edits


def apply_boundary_and_token_edit_labels(
    text: str,
    tokens: Sequence[WordToken],
    labels: Sequence[str],
    confidences: Sequence[float],
    threshold: float,
    *,
    boundary_before_labels: Sequence[str] | None = None,
    boundary_after_labels: Sequence[str] | None = None,
    boundary_before_confidences: Sequence[float] | None = None,
    boundary_after_confidences: Sequence[float] | None = None,
    rule_ids: Sequence[str] | None = None,
    orthographic_lexicon: OrthographicCorrectionLexicon | None = None,
) -> tuple[str, list[RuntimeEdit]]:
    edits: list[RuntimeEdit] = []
    consumed: set[int] = set()
    count = min(len(tokens), len(labels), len(confidences))
    before_labels = _padded_labels(boundary_before_labels, len(tokens), "NONE")
    after_labels = _padded_labels(boundary_after_labels, len(tokens), "NONE")
    before_confidences = _padded_confidences(boundary_before_confidences, len(tokens), 1.0)
    after_confidences = _padded_confidences(boundary_after_confidences, len(tokens), 1.0)

    for index in range(len(tokens)):
        before_edit = _boundary_before_edit_for_label(
            text,
            tokens[index],
            before_labels[index],
            float(before_confidences[index]),
            threshold,
            _boundary_rule_id_for(index, rule_ids, before_labels[index]),
        )
        if before_edit is not None:
            edits.append(before_edit)

        if index < count and index not in consumed:
            label = str(labels[index])
            if label not in {"KEEP", "SKIP_MERGED"}:
                confidence = float(confidences[index])
                if confidence >= threshold:
                    edit = _token_edit_for_label(
                        text,
                        tokens,
                        index,
                        label,
                        confidence,
                        _rule_id_for(index, rule_ids, label),
                        orthographic_lexicon,
                    )
                    if edit is not None:
                        edits.append(edit)
                        consumed.update(_consumed_indexes(tokens, index, label, edit))

        after_edit = _boundary_after_edit_for_label(
            text,
            tokens[index],
            after_labels[index],
            float(after_confidences[index]),
            threshold,
            _boundary_rule_id_for(index, rule_ids, after_labels[index]),
        )
        if after_edit is not None:
            edits.append(after_edit)

    return apply_runtime_edits(text, edits), edits


def apply_boundary_labels(
    text: str,
    tokens: Sequence[WordToken],
    boundary_before_labels: Sequence[str],
    boundary_after_labels: Sequence[str],
    confidences: Sequence[float],
    threshold: float,
    *,
    rule_ids: Sequence[str] | None = None,
) -> tuple[str, list[RuntimeEdit]]:
    return apply_boundary_and_token_edit_labels(
        text,
        tokens,
        ["KEEP"] * len(tokens),
        [1.0] * len(tokens),
        threshold,
        boundary_before_labels=boundary_before_labels,
        boundary_after_labels=boundary_after_labels,
        boundary_before_confidences=confidences,
        boundary_after_confidences=confidences,
        rule_ids=rule_ids,
    )


def apply_gap_labels(
    text: str,
    tokens: Sequence[WordToken],
    gap_labels: Sequence[str],
    confidences: Sequence[float],
    threshold: float,
    *,
    rule_ids: Sequence[str] | None = None,
) -> tuple[str, list[RuntimeEdit]]:
    edits: list[RuntimeEdit] = []
    count = min(len(tokens), len(gap_labels), len(confidences))
    for index in range(count):
        label = str(gap_labels[index])
        if label == "NONE":
            continue
        confidence = float(confidences[index])
        if confidence < threshold:
            continue
        if label == "DELETE_PUNCTUATION":
            edit = _delete_punctuation_edit(
                text,
                tokens[index],
                confidence,
                _gap_rule_id_for(index, rule_ids, label),
            )
            if edit is not None:
                edits.append(edit)
            continue
        punctuation = PUNCTUATION_BY_LABEL.get(label)
        if punctuation is None:
            continue
        edit = _gap_edit_for_label(
            text,
            tokens[index],
            label,
            punctuation,
            confidence,
            _gap_rule_id_for(index, rule_ids, label),
        )
        if edit is not None:
            edits.append(edit)

    return apply_runtime_edits(text, edits), edits


def apply_runtime_edits(text: str, edits: Sequence[RuntimeEdit]) -> str:
    result = text
    for edit in sorted(enumerate(edits), key=lambda item: (item[1].start, item[1].end, item[0]), reverse=True):
        item = edit[1]
        start = max(0, min(item.start, len(result)))
        end = max(start, min(item.end, len(result)))
        result = result[:start] + item.replacement + result[end:]
    return result


def token_edit_type_for_label(label: str) -> str:
    return LABEL_RULE_EDIT_TYPE.get(label, ("none", "spelling"))[1]


def gap_edit_type_for_label(label: str) -> str:
    return "punctuation" if label != "NONE" else "none"


def boundary_edit_type_for_label(label: str) -> str:
    return BOUNDARY_LABEL_RULE_EDIT_TYPE.get(label, ("none", "none"))[1]


def _padded_labels(labels: Sequence[str] | None, count: int, default: str) -> list[str]:
    result = [str(label) for label in (labels or [])[:count]]
    if len(result) < count:
        result.extend([default] * (count - len(result)))
    return result


def _padded_confidences(confidences: Sequence[float] | None, count: int, default: float) -> list[float]:
    result = [float(value) for value in (confidences or [])[:count]]
    if len(result) < count:
        result.extend([default] * (count - len(result)))
    return result


def _token_edit_for_label(
    text: str,
    tokens: Sequence[WordToken],
    index: int,
    label: str,
    confidence: float,
    rule_id: str,
    orthographic_lexicon: OrthographicCorrectionLexicon | None,
) -> RuntimeEdit | None:
    token = tokens[index]
    source = text[token.start : token.end]
    start = token.start
    end = token.end
    replacement: str | None = None

    if label == "DELETE":
        replacement = ""
    elif label == "LOWERCASE":
        replacement = source.lower()
    elif label == "UPPERCASE":
        replacement = source.upper()
    elif label == "CAPITALIZE":
        replacement = source[:1].upper() + source[1:].lower()
    elif label == "SPLIT_NE_VERB":
        if not source.lower().startswith("не") or len(source) <= 2:
            return None
        replacement = f"{source[:2]} {source[2:]}"
    elif label in {
        "MERGE_TAK_ZHE_TO_TAKZHE",
        "MERGE_TO_ZHE_TO_TOZHE",
        "MERGE_ZA_TO_TO_ZATO",
        "HYPHENATE_PARTICLE_TO",
        "HYPHENATE_PARTICLE_LIBO",
        "HYPHENATE_PARTICLE_NIBUD",
        "HYPHENATE_KOE",
        "HYPHENATE_PO_ADVERB",
    }:
        if index + 1 >= len(tokens):
            return None
        next_token = tokens[index + 1]
        start, end = token.start, next_token.end
        source = text[start:end]
        if label.startswith("MERGE_"):
            replacement = tokens[index].text + tokens[index + 1].text
        else:
            replacement = f"{tokens[index].text}-{tokens[index + 1].text}"
    elif label == "SPLIT_TAKZHE_TO_TAK_ZHE":
        replacement = _split_exact(source, "также", "так же")
    elif label == "SPLIT_TOZHE_TO_TO_ZHE":
        replacement = _split_exact(source, "тоже", "то же")
    elif label == "SPLIT_ZATO_TO_ZA_TO":
        replacement = _split_exact(source, "зато", "за то")
    elif label == "FIX_TSYA_TO_TTSYA":
        replacement = _fix_tsya_to_ttsya(source)
    elif label == "FIX_TTSYA_TO_TSYA":
        replacement = _fix_ttsya_to_tsya(source)
    elif label == "DICT_REPLACE":
        if orthographic_lexicon is None:
            return None
        entries = [
            *orthographic_lexicon.lookup(source, rule_id=rule_id, operation="dict_replace"),
            *orthographic_lexicon.lookup(source, rule_id=rule_id, operation="replace"),
        ]
        targets = {entry.target for entry in entries}
        if len(targets) != 1:
            return None
        replacement = _match_case(source, next(iter(targets)))
    elif label == "SPAN_REPLACE_BY_LEXICON":
        return _span_replace_by_lexicon_edit(
            text,
            tokens,
            index,
            confidence,
            rule_id,
            orthographic_lexicon,
        )

    if replacement is None or replacement == source:
        return None
    edit_type = token_edit_type_for_label(label)
    return RuntimeEdit(
        start=start,
        end=end,
        source=source,
        replacement=replacement,
        edit_type=edit_type,
        rule_id=rule_id,
        confidence=confidence,
    )


def _boundary_before_edit_for_label(
    text: str,
    token: WordToken,
    label: str,
    confidence: float,
    threshold: float,
    rule_id: str,
) -> RuntimeEdit | None:
    if label == "NONE" or confidence < threshold:
        return None
    position = token.start
    if label == "INSERT_OPEN_QUOTE":
        return RuntimeEdit(position, position, "", "«", "punctuation", rule_id, confidence)
    if label == "INSERT_OPEN_BRACKET":
        return RuntimeEdit(position, position, "", "(", "punctuation", rule_id, confidence)
    if position <= 0:
        return None
    previous = text[position - 1]
    if label == "DELETE_OPEN_QUOTE" and previous in OPEN_QUOTE_DELETE_CHARS:
        return RuntimeEdit(position - 1, position, previous, "", "punctuation", rule_id, confidence)
    if label == "DELETE_OPEN_BRACKET" and previous == "(":
        return RuntimeEdit(position - 1, position, previous, "", "punctuation", rule_id, confidence)
    if label == "NORMALIZE_OPEN_QUOTE":
        if previous == "«":
            return None
        if previous in OPEN_QUOTE_NORMALIZE_CHARS:
            return RuntimeEdit(position - 1, position, previous, "«", "punctuation", rule_id, confidence)
    return None


def _boundary_after_edit_for_label(
    text: str,
    token: WordToken,
    label: str,
    confidence: float,
    threshold: float,
    rule_id: str,
) -> RuntimeEdit | None:
    if label == "NONE" or confidence < threshold:
        return None
    position = token.end
    if label == "INSERT_CLOSE_QUOTE":
        return RuntimeEdit(position, position, "", "»", "punctuation", rule_id, confidence)
    if label == "INSERT_CLOSE_BRACKET":
        return RuntimeEdit(position, position, "", ")", "punctuation", rule_id, confidence)
    if position >= len(text):
        return None
    following = text[position]
    if label == "DELETE_CLOSE_QUOTE" and following in CLOSE_QUOTE_DELETE_CHARS:
        return RuntimeEdit(position, position + 1, following, "", "punctuation", rule_id, confidence)
    if label == "DELETE_CLOSE_BRACKET" and following == ")":
        return RuntimeEdit(position, position + 1, following, "", "punctuation", rule_id, confidence)
    if label == "NORMALIZE_CLOSE_QUOTE":
        if following == "»":
            return None
        if following in CLOSE_QUOTE_NORMALIZE_CHARS:
            return RuntimeEdit(position, position + 1, following, "»", "punctuation", rule_id, confidence)
    return None


def _span_replace_by_lexicon_edit(
    text: str,
    tokens: Sequence[WordToken],
    index: int,
    confidence: float,
    rule_id: str,
    orthographic_lexicon: OrthographicCorrectionLexicon | None,
) -> RuntimeEdit | None:
    if orthographic_lexicon is None or index >= len(tokens):
        return None

    max_stop = min(len(tokens), index + MAX_LEXICON_SPAN_TOKENS)
    for stop in range(max_stop, index, -1):
        start = tokens[index].start
        end = tokens[stop - 1].end
        source = text[start:end]
        entries = [
            entry
            for entry in orthographic_lexicon.lookup(source, rule_id=rule_id)
            if _entry_allowed_for_span_replace(entry, text)
        ]
        if not entries:
            continue
        targets = {entry.target for entry in entries}
        if len(targets) != 1:
            return None
        entry = entries[0]
        replacement = _match_case(source, entry.target)
        if not replacement or replacement == source:
            return None
        return RuntimeEdit(
            start=start,
            end=end,
            source=source,
            replacement=replacement,
            edit_type=_edit_type_for_lexicon_operation(entry.operation, source=source, replacement=replacement),
            rule_id=rule_id,
            confidence=confidence,
        )
    return None


def _gap_edit_for_label(
    text: str,
    token: WordToken,
    label: str,
    punctuation: str,
    confidence: float,
    rule_id: str,
) -> RuntimeEdit | None:
    anchor = _gap_anchor(text, token, label)
    scan = anchor
    while scan < len(text) and text[scan].isspace():
        scan += 1

    existing = _punctuation_span_at(text, scan)
    if existing is not None:
        start, end, source = existing
        if source == punctuation:
            return None
        return RuntimeEdit(start, end, source, punctuation, "punctuation", rule_id, confidence)

    replacement = f" {punctuation}" if label == "DASH" else punctuation
    return RuntimeEdit(anchor, anchor, "", replacement, "punctuation", rule_id, confidence)


def _delete_punctuation_edit(
    text: str,
    token: WordToken,
    confidence: float,
    rule_id: str,
) -> RuntimeEdit | None:
    anchor = _gap_anchor(text, token, "COMMA_DASH")
    scan = anchor
    while scan < len(text) and text[scan].isspace():
        scan += 1

    if text.startswith(", \u2014", scan):
        return RuntimeEdit(scan, scan + 3, text[scan : scan + 3], "", "punctuation", rule_id, confidence)

    if scan >= len(text) or text[scan] not in DELETE_PUNCTUATION_CHARS:
        return None
    if text[scan] in {"-", "\u2014"}:
        end = scan + 1
        while end < len(text) and text[end].isspace():
            end += 1
        return RuntimeEdit(anchor, end, text[anchor:end], " ", "punctuation", rule_id, confidence)
    return RuntimeEdit(scan, scan + 1, text[scan], "", "punctuation", rule_id, confidence)


def _gap_anchor(text: str, token: WordToken, label: str) -> int:
    if label in WRAPPER_AWARE_GAP_LABELS and token.end < len(text) and text[token.end] in CLOSING_WRAPPER_CHARS:
        return token.end + 1
    return token.end


def _punctuation_span_at(text: str, index: int) -> tuple[int, int, str] | None:
    if index >= len(text):
        return None
    if text.startswith(", \u2014", index):
        return (index, index + 3, text[index : index + 3])
    if text.startswith("...", index):
        return (index, index + 3, text[index : index + 3])
    if text[index] in PUNCTUATION_CHARS:
        return (index, index + 1, text[index])
    return None


def _rule_id_for(index: int, rule_ids: Sequence[str] | None, label: str) -> str:
    if rule_ids is not None and index < len(rule_ids) and rule_ids[index] not in {"", "none"}:
        return str(rule_ids[index])
    return LABEL_RULE_EDIT_TYPE.get(label, ("none", "spelling"))[0]


def _gap_rule_id_for(index: int, rule_ids: Sequence[str] | None, label: str) -> str:
    if rule_ids is not None and index < len(rule_ids) and rule_ids[index] not in {"", "none"}:
        return str(rule_ids[index])
    if label == "DASH":
        return "dash_subject_predicate"
    if label in {"DOT", "QUESTION", "EXCLAMATION", "ELLIPSIS"}:
        return "final_punctuation"
    if label == "COMMA":
        return "comma_subordinate"
    return "punctuation"


def _boundary_rule_id_for(index: int, rule_ids: Sequence[str] | None, label: str) -> str:
    if rule_ids is not None and index < len(rule_ids) and rule_ids[index] not in {"", "none"}:
        return str(rule_ids[index])
    return BOUNDARY_LABEL_RULE_EDIT_TYPE.get(label, ("none", "none"))[0]


def _consumed_indexes(tokens: Sequence[WordToken], index: int, label: str, edit: RuntimeEdit) -> set[int]:
    if label == "SPAN_REPLACE_BY_LEXICON":
        return {
            position
            for position in range(index, len(tokens))
            if tokens[position].start >= tokens[index].start and tokens[position].end <= edit.end
        }
    if label in {
        "MERGE_TAK_ZHE_TO_TAKZHE",
        "MERGE_TO_ZHE_TO_TOZHE",
        "MERGE_ZA_TO_TO_ZATO",
        "HYPHENATE_PARTICLE_TO",
        "HYPHENATE_PARTICLE_LIBO",
        "HYPHENATE_PARTICLE_NIBUD",
        "HYPHENATE_KOE",
        "HYPHENATE_PO_ADVERB",
    } and index + 1 < len(tokens):
        return {index, index + 1}
    return {index}


def _split_exact(source: str, lowered: str, replacement: str) -> str | None:
    if source.lower() != lowered:
        return None
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _fix_tsya_to_ttsya(source: str) -> str:
    mapped = _mapped_replacement(source, _tsya_to_ttsya_replacements())
    if mapped is not None:
        return mapped
    return re.sub(r"тся$", "ться", source, flags=re.IGNORECASE)


def _fix_ttsya_to_tsya(source: str) -> str:
    mapped = _mapped_replacement(source, _ttsya_to_tsya_replacements())
    if mapped is not None:
        return mapped
    return re.sub(r"ться$", "тся", source, flags=re.IGNORECASE)


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _edit_type_for_lexicon_operation(operation: str, *, source: str = "", replacement: str = "") -> str:
    if source and replacement:
        source_letters = _letters_only(source)
        replacement_letters = _letters_only(replacement)
        if source_letters and replacement_letters and source_letters != replacement_letters:
            return "spelling"
    if operation in {"split_join", "split", "merge", "join"}:
        return "split_join"
    if operation in {"hyphen", "hyphenate", "unhyphen", "unhyphenate", "dehyphen"}:
        return "hyphen"
    return "spelling"


def _entry_allowed_in_context(entry: object, text: str) -> bool:
    forbidden = getattr(entry, "forbidden_contexts", ()) or ()
    if not forbidden:
        return True
    lowered = text.casefold()
    return not any(str(item).casefold() in lowered for item in forbidden if str(item).strip())


def _entry_allowed_for_span_replace(entry: object, text: str) -> bool:
    operation = str(getattr(entry, "operation", "") or "")
    rule_id = str(getattr(entry, "rule_id", "") or "")
    if operation in {"dict_replace", "replace"} and not rule_id.startswith("semantic_"):
        return False
    return _entry_allowed_in_context(entry, text)


def _word_count(value: str) -> int:
    return len(re.findall(r"[А-Яа-яЁё]+", value))


def _letters_only(value: str) -> str:
    return "".join(re.findall(r"[А-Яа-яЁё]+", value.casefold())).replace("ё", "е")


def _mapped_replacement(source: str, replacements: dict[str, str]) -> str | None:
    replacement = replacements.get(source.lower())
    if replacement is None:
        return None
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


@lru_cache(maxsize=1)
def _tsya_to_ttsya_replacements() -> dict[str, str]:
    replacements = dict(TSYA_TO_TTSYA_REPLACEMENTS)
    replacements.update(_controlled_tsya_pairs())
    return replacements


@lru_cache(maxsize=1)
def _ttsya_to_tsya_replacements() -> dict[str, str]:
    replacements = dict(TTSYA_TO_TSYA_REPLACEMENTS)
    replacements.update({target: source for source, target in _controlled_tsya_pairs().items()})
    return replacements


@lru_cache(maxsize=1)
def _controlled_tsya_pairs() -> dict[str, str]:
    path = PROJECT_ROOT / "lexicon" / "constructions" / "tsya_ttsya.yaml"
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    patterns = raw.get("patterns", []) if isinstance(raw, dict) else []
    pairs: dict[str, str] = {}
    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue
        metadata = pattern.get("metadata", {})
        verbs = metadata.get("verbs", []) if isinstance(metadata, dict) else []
        if not isinstance(verbs, list):
            continue
        for item in verbs:
            if not isinstance(item, dict):
                continue
            infinitive = str(item.get("infinitive") or "").strip().lower()
            finite = str(item.get("finite") or "").strip().lower()
            if finite.endswith("тся") and infinitive.endswith("ться"):
                pairs[finite] = infinitive
    return pairs


__all__ = [
    "apply_boundary_and_token_edit_labels",
    "apply_boundary_labels",
    "apply_gap_labels",
    "apply_runtime_edits",
    "apply_token_edit_labels",
    "boundary_edit_type_for_label",
    "gap_edit_type_for_label",
    "token_edit_type_for_label",
]
