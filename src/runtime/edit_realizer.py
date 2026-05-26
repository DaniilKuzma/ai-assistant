from __future__ import annotations

from dataclasses import replace
import re
from collections.abc import Sequence

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
}
PUNCTUATION_CHARS = set(",;:\u2014-.!?\u2026")
DELETE_PUNCTUATION_CHARS = PUNCTUATION_CHARS
MAX_LEXICON_SPAN_TOKENS = 4

TSYA_TO_TTSYA_REPLACEMENTS = {
    "\u043e\u0448\u0438\u0431\u0430\u0435\u0442\u0441\u044f": "\u043e\u0448\u0438\u0431\u0430\u0442\u044c\u0441\u044f",
    "\u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0435\u0442\u0441\u044f": "\u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0442\u044c\u0441\u044f",
}
TTSYA_TO_TSYA_REPLACEMENTS = {value: key for key, value in TSYA_TO_TTSYA_REPLACEMENTS.items()}


LABEL_RULE_EDIT_TYPE: dict[str, tuple[str, str]] = {
    "DELETE": ("delete", "spelling"),
    "LOWERCASE": ("casing", "casing"),
    "UPPERCASE": ("casing", "casing"),
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
            if entry.operation not in {"dict_replace", "replace"}
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
            edit_type=_edit_type_for_lexicon_operation(entry.operation),
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
    scan = token.end
    while scan < len(text) and text[scan].isspace():
        scan += 1

    if scan < len(text) and text[scan] in PUNCTUATION_CHARS:
        if text[scan] == punctuation:
            return None
        return RuntimeEdit(scan, scan + 1, text[scan], punctuation, "punctuation", rule_id, confidence)

    replacement = f" {punctuation}" if label == "DASH" else punctuation
    return RuntimeEdit(token.end, token.end, "", replacement, "punctuation", rule_id, confidence)


def _delete_punctuation_edit(
    text: str,
    token: WordToken,
    confidence: float,
    rule_id: str,
) -> RuntimeEdit | None:
    scan = token.end
    while scan < len(text) and text[scan].isspace():
        scan += 1
    if scan >= len(text) or text[scan] not in DELETE_PUNCTUATION_CHARS:
        return None
    return RuntimeEdit(scan, scan + 1, text[scan], "", "punctuation", rule_id, confidence)


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
    mapped = _mapped_replacement(source, TSYA_TO_TTSYA_REPLACEMENTS)
    if mapped is not None:
        return mapped
    return re.sub(r"тся$", "ться", source, flags=re.IGNORECASE)


def _fix_ttsya_to_tsya(source: str) -> str:
    mapped = _mapped_replacement(source, TTSYA_TO_TSYA_REPLACEMENTS)
    if mapped is not None:
        return mapped
    return re.sub(r"ться$", "тся", source, flags=re.IGNORECASE)


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _edit_type_for_lexicon_operation(operation: str) -> str:
    if operation in {"split_join", "split", "merge", "join"}:
        return "split_join"
    if operation in {"hyphen", "hyphenate", "unhyphen", "unhyphenate", "dehyphen"}:
        return "hyphen"
    return "spelling"


def _mapped_replacement(source: str, replacements: dict[str, str]) -> str | None:
    replacement = replacements.get(source.lower())
    if replacement is None:
        return None
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


__all__ = [
    "apply_gap_labels",
    "apply_runtime_edits",
    "apply_token_edit_labels",
    "gap_edit_type_for_label",
    "token_edit_type_for_label",
]
