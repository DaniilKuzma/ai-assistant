from __future__ import annotations

from dataclasses import dataclass, replace
import difflib
import re

from src.schema.lexical_resources import CONTEXT_DEPENDENT_WHITELIST, HYPHEN_WHITELIST, SPLIT_JOIN_WHITELIST, WRONG_TO_CORRECT
from src.preprocessing.tokenizer import PUNCTUATION


FINAL_PUNCT = ".!?…"
PUNCT_RE = re.compile(r"[,.!?:;—…\"'()«»]")
SENTENCE_START_PREFIX_CHARS = set(" \t\r\n\"'«„“([{—-")


@dataclass(frozen=True)
class Edit:
    source: str
    replacement: str
    edit_type: str
    start: int = -1
    end: int = -1
    status: str = "proposed"
    reason: str = ""
    confidence: float = 1.0
    rule_id: str = ""

    def with_status(self, status: str, reason: str = "") -> "Edit":
        return replace(self, status=status, reason=reason)


class DiffAnalyzer:
    """Classify text differences into the strict edit taxonomy."""

    def analyze(self, source: str, target: str) -> list[Edit]:
        edits: list[Edit] = []
        working_source = source
        working_target = target

        final_edit = self._final_punctuation_edit(working_source, working_target)
        if final_edit:
            edits.append(final_edit)
            working_source = _strip_final_punctuation(working_source)
            working_target = _strip_final_punctuation(working_target)

        edits.extend(self._known_word_edits(source, target))
        edits.extend(self._case_edits(source, target))
        edits.extend(self._punctuation_edits(working_source, working_target))

        if not self._known_edits_explain(source, target, edits):
            edits.extend(self._unknown_word_replacements(source, target, edits))

        return _deduplicate(edits)

    def punctuation_edits(self, source: str, target: str) -> list[Edit]:
        edits: list[Edit] = []
        working_source = source
        working_target = target

        final_edit = self._final_punctuation_edit(working_source, working_target)
        if final_edit:
            edits.append(final_edit)
            working_source = _strip_final_punctuation(working_source)
            working_target = _strip_final_punctuation(working_target)

        edits.extend(self._punctuation_edits(working_source, working_target))
        return _deduplicate(edits)

    def _known_word_edits(
        self,
        source: str,
        target: str,
    ) -> list[Edit]:
        edits: list[Edit] = []
        source_lower = source.lower()
        target_lower = target.lower()

        for wrong, correct in WRONG_TO_CORRECT.items():
            if wrong in source_lower and correct in target_lower:
                start = source_lower.find(wrong)
                edit_type = "split_word" if " " in correct else "spelling_replace"
                rule_id = _orthography_rule_id(wrong, correct, edit_type)
                edits.append(
                    Edit(
                        source[start : start + len(wrong)],
                        correct,
                        edit_type,
                        start,
                        start + len(wrong),
                        confidence=0.95,
                        rule_id=rule_id,
                    )
                )

        for match in re.finditer(r"\bне([а-яё]+)\b", source_lower):
            wrong = match.group(0)
            correct = f"не {match.group(1)}"
            if correct not in target_lower:
                continue
            edits.append(
                Edit(
                    source[match.start() : match.end()],
                    correct,
                    "split_word",
                    match.start(),
                    match.end(),
                    confidence=0.95,
                    rule_id="ne_verb",
                )
            )

        for match in re.finditer(r"\b[а-яё]+\b", source_lower):
            replacement = _pattern_replacement(match.group(0))
            if replacement is None:
                continue
            correct, rule_id = replacement
            if correct not in target_lower:
                continue
            edits.append(
                Edit(
                    source[match.start() : match.end()],
                    correct,
                    "spelling_replace",
                    match.start(),
                    match.end(),
                    confidence=0.95,
                    rule_id=rule_id,
                )
            )

        for wrong, correct in SPLIT_JOIN_WHITELIST.items():
            if correct in source_lower and wrong in target_lower:
                start = source_lower.find(correct)
                edits.append(
                    Edit(
                        source[start : start + len(correct)],
                        wrong,
                        "join_words",
                        start,
                        start + len(correct),
                        confidence=0.95,
                        rule_id="frequent_error_exact",
                    )
                )

        for source_phrase, replacement in CONTEXT_DEPENDENT_WHITELIST.items():
            if source_phrase in source_lower and replacement in target_lower:
                start = source_lower.find(source_phrase)
                edit_type = _split_join_edit_type(source_phrase, replacement)
                edits.append(
                    Edit(
                        source[start : start + len(source_phrase)],
                        replacement,
                        edit_type,
                        start,
                        start + len(source_phrase),
                        confidence=0.95,
                        rule_id="context_pair",
                    )
                )

        for wrong, correct in HYPHEN_WHITELIST.items():
            if wrong == correct:
                continue
            if wrong in source_lower and correct in target_lower:
                start = source_lower.find(wrong)
                edits.append(
                    Edit(
                        source[start : start + len(wrong)],
                        correct,
                        "hyphen_change",
                        start,
                        start + len(wrong),
                        confidence=0.95,
                        rule_id=_hyphen_rule_id(wrong, correct),
                    )
                )

        return edits

    def _case_edits(self, source: str, target: str) -> list[Edit]:
        edits: list[Edit] = []
        for index, (source_char, target_char) in enumerate(zip(source, target, strict=False)):
            if not _is_russian_letter(source_char) or not _is_russian_letter(target_char):
                continue
            if source_char == target_char or source_char.lower() != target_char.lower():
                continue
            if target_char.isupper() and _is_sentence_start_case_position(source, index):
                edits.append(Edit(source_char, target_char, "case_change", index, index + 1, confidence=0.9))
            else:
                edits.append(Edit(source_char, target_char, "unknown", index, index + 1, confidence=0.0))
        return edits

    def _punctuation_edits(self, source: str, target: str) -> list[Edit]:
        edits: list[Edit] = []
        matcher = difflib.SequenceMatcher(a=source, b=target)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "insert":
                edits.extend(_punctuation_inserts(i1, target[j1:j2]))
            elif tag == "delete":
                edits.extend(_punctuation_deletes(source, i1, i2))
            else:
                edits.extend(_punctuation_replacements(source, target, i1, i2, j1, j2))

        return edits

    def _final_punctuation_edit(self, source: str, target: str) -> Edit | None:
        source_stripped = source.rstrip()
        target_stripped = target.rstrip()
        if not target_stripped:
            return None
        source_final = source_stripped[-1] if source_stripped and source_stripped[-1] in FINAL_PUNCT else ""
        target_final = target_stripped[-1] if target_stripped[-1] in FINAL_PUNCT else ""
        if source_final == target_final:
            return None
        if target_final:
            edit_type = "final_punctuation"
            source_value = source_final
            return Edit(source_value, target_final, edit_type, len(source_stripped) - len(source_value), len(source_stripped), confidence=0.9)
        return None

    def _known_edits_explain(self, source: str, target: str, edits: list[Edit]) -> bool:
        projected = _strip_final_punctuation(source)
        for edit in edits:
            if edit.edit_type in {"punctuation_insert", "punctuation_delete", "punctuation_replace", "final_punctuation"}:
                continue
            if edit.source and edit.source in projected:
                projected = projected.replace(edit.source, edit.replacement, 1)
        projected = _remove_punctuation(projected).lower()
        comparable_target = _remove_punctuation(_strip_final_punctuation(target)).lower()
        return projected == comparable_target

    def _unknown_word_replacements(self, source: str, target: str, known_edits: list[Edit]) -> list[Edit]:
        source_clean = _remove_punctuation(_strip_final_punctuation(source))
        target_clean = _remove_punctuation(_strip_final_punctuation(target))
        for edit in known_edits:
            if edit.edit_type in {"punctuation_insert", "punctuation_delete", "punctuation_replace", "final_punctuation"}:
                continue
            source_clean = source_clean.replace(edit.source, edit.replacement, 1)

        source_words = source_clean.split()
        target_words = target_clean.split()
        matcher = difflib.SequenceMatcher(a=source_words, b=target_words)
        edits: list[Edit] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            edits.append(Edit(" ".join(source_words[i1:i2]), " ".join(target_words[j1:j2]), "unknown", confidence=0.0))
        return edits


def _strip_final_punctuation(text: str) -> str:
    stripped = text.rstrip()
    if stripped and stripped[-1] in FINAL_PUNCT:
        return stripped[:-1]
    return text


def _remove_punctuation(text: str) -> str:
    return PUNCT_RE.sub("", text)


def _split_join_edit_type(source: str, replacement: str) -> str:
    if len(source.split()) > len(replacement.split()):
        return "join_words"
    return "split_word"


def _is_sentence_start_case_position(text: str, position: int) -> bool:
    return all(char in SENTENCE_START_PREFIX_CHARS for char in text[:position])


def _is_russian_letter(char: str) -> bool:
    return bool(re.fullmatch(r"[А-Яа-яЁё]", char))


def _orthography_rule_id(wrong: str, correct: str, edit_type: str) -> str:
    if edit_type == "split_word" and wrong.startswith("не") and correct == f"не {wrong[2:]}":
        return "ne_verb"

    pattern_rules = (
        ("чя", "ча", "pattern_чя_ча"),
        ("щя", "ща", "pattern_щя_ща"),
        ("чю", "чу", "pattern_чю_чу"),
        ("щю", "щу", "pattern_щю_щу"),
        ("жы", "жи", "pattern_жы_жи"),
        ("шы", "ши", "pattern_шы_ши"),
        ("цы", "ци", "pattern_цы_ци"),
    )
    for wrong_fragment, correct_fragment, rule_id in pattern_rules:
        if wrong_fragment in wrong and correct == wrong.replace(wrong_fragment, correct_fragment, 1):
            return rule_id
    return "frequent_error_exact"


def _pattern_replacement(word: str) -> tuple[str, str] | None:
    pattern_rules = (
        ("чя", "ча", "pattern_чя_ча"),
        ("щя", "ща", "pattern_щя_ща"),
        ("чю", "чу", "pattern_чю_чу"),
        ("щю", "щу", "pattern_щю_щу"),
        ("жы", "жи", "pattern_жы_жи"),
        ("шы", "ши", "pattern_шы_ши"),
        ("цы", "ци", "pattern_цы_ци"),
    )
    for wrong_fragment, correct_fragment, rule_id in pattern_rules:
        if wrong_fragment in word:
            return word.replace(wrong_fragment, correct_fragment, 1), rule_id
    return None


def _hyphen_rule_id(wrong: str, correct: str) -> str:
    if wrong.startswith("кое ") or correct.startswith("кое-"):
        return "hyphen_koe"
    if wrong.startswith("по ") or correct.startswith("по-"):
        return "hyphen_po_adverb"
    return "hyphen_particles"


def _punctuation_inserts(position: int, inserted: str) -> list[Edit]:
    edits: list[Edit] = []
    for offset, char in enumerate(inserted):
        if char in PUNCTUATION:
            edits.append(Edit("", char, "punctuation_insert", position + offset, position + offset, confidence=0.85))
    return edits


def _punctuation_deletes(source: str, start: int, end: int) -> list[Edit]:
    return [
        Edit(char, "", "punctuation_delete", index, index + 1, confidence=0.85)
        for index, char in enumerate(source[start:end], start=start)
        if char in PUNCTUATION
    ]


def _punctuation_replacements(source: str, target: str, i1: int, i2: int, j1: int, j2: int) -> list[Edit]:
    source_punct = [(index, char) for index, char in enumerate(source[i1:i2], start=i1) if char in PUNCTUATION]
    target_punct = [(index, char) for index, char in enumerate(target[j1:j2], start=j1) if char in PUNCTUATION]

    if not source_punct:
        return _punctuation_inserts(i1, target[j1:j2])
    if not target_punct:
        return _punctuation_deletes(source, i1, i2)

    edits: list[Edit] = []
    paired_count = min(len(source_punct), len(target_punct))
    for pair_index in range(paired_count):
        source_index, source_char = source_punct[pair_index]
        _target_index, target_char = target_punct[pair_index]
        if source_char != target_char:
            edits.append(
                Edit(source_char, target_char, "punctuation_replace", source_index, source_index + 1, confidence=0.85)
            )

    for source_index, source_char in source_punct[paired_count:]:
        edits.append(Edit(source_char, "", "punctuation_delete", source_index, source_index + 1, confidence=0.85))

    insert_position = source_punct[-1][0] + 1
    for _target_index, target_char in target_punct[paired_count:]:
        edits.append(Edit("", target_char, "punctuation_insert", insert_position, insert_position, confidence=0.85))

    return edits


def _deduplicate(edits: list[Edit]) -> list[Edit]:
    seen: set[tuple[str, str, str, int, int]] = set()
    result: list[Edit] = []
    result_index_by_key: dict[tuple[str, str, str, int, int], int] = {}
    for edit in edits:
        key = (edit.source, edit.replacement, edit.edit_type, edit.start, edit.end)
        if key in seen:
            existing_index = result_index_by_key[key]
            if (
                (not result[existing_index].rule_id and edit.rule_id)
                or (result[existing_index].rule_id == "context_pair" and edit.rule_id.startswith("context_"))
            ):
                result[existing_index] = edit
            continue
        seen.add(key)
        result_index_by_key[key] = len(result)
        result.append(edit)
    return result
