from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

from src.grammar_gen.ast import (
    Clause,
    ComplexSentence,
    DashSubjectPredicateSentence,
    HomogeneousSentence,
    IntroductorySentence,
    NounPhrase,
    SimpleSentence,
    VerbPhrase,
)
from src.grammar_gen.lexicon import Lexicon, NounEntry
from src.grammar_gen.semantic_safety import reject_semantic_nonsense, validate_clause_semantics
from src.grammar_gen.semantics import VerbFrame
from src.schema import GeneratedExample


LATIN_RE = re.compile(r"[A-Za-z]")
REPEATED_PUNCTUATION_RE = re.compile(r"([,!?;:])\1+|\.{2}(?!\.)|\.{4,}")
FINAL_PUNCTUATION_RE = re.compile(r"(\.\.\.|[.!?\u2026])$")
FINAL_PUNCTUATION_LABELS = {
    ".": "DOT",
    "?": "QUESTION",
    "!": "EXCLAMATION",
    "...": "ELLIPSIS",
    "\u2026": "ELLIPSIS",
}
VO_RE = re.compile(r"(^|\s)во\s+", re.IGNORECASE)
BAD_PAIR_REASONS = (
    ("bad_pair_devochka_poshel", re.compile(r"\bдевочка\s+пош[её]л\b", re.IGNORECASE)),
    ("bad_pair_devochka_proveril", re.compile(r"\bдевочка\s+проверил\b", re.IGNORECASE)),
    ("bad_pair_komissiya_reshil", re.compile(r"\bкомиссия\s+решил\b", re.IGNORECASE)),
    ("bad_pair_student_poshla", re.compile(r"\bстудент\s+пошла\b", re.IGNORECASE)),
    ("bad_pair_vo_ogorod", re.compile(r"\bво\s+огород\b", re.IGNORECASE)),
)
VO_WHITELIST = ("во дворе",)


def validate_surface(text: str) -> list[str]:
    reasons: list[str] = []
    if "{" in text or "}" in text:
        reasons.append("template_brace")
    if "  " in text:
        reasons.append("double_space")
    if LATIN_RE.search(text):
        reasons.append("latin_letters")
    if _has_bad_vo_phrase(text.lower()):
        reasons.append("bad_vo_phrase")
    if text and not text[0].isupper():
        reasons.append("sentence_start_not_uppercase")
    if not FINAL_PUNCTUATION_RE.search(text):
        reasons.append("missing_final_punctuation")
    if REPEATED_PUNCTUATION_RE.search(text):
        reasons.append("repeated_punctuation")

    lowered = text.lower()
    for reason, pattern in BAD_PAIR_REASONS:
        if pattern.search(lowered):
            reasons.append(reason)

    return _dedupe(reasons)


def validate_generated_pair(example: GeneratedExample) -> list[str]:
    source_reasons = validate_surface(example.source_text)
    target_reasons = validate_surface(example.target_text)
    allowed_source = allowed_source_surface_failures(example)
    real_source_reasons = [reason for reason in source_reasons if reason not in allowed_source]
    return _dedupe(real_source_reasons + target_reasons)


def allowed_source_surface_failures(example: GeneratedExample) -> set[str]:
    if (
        example.primary_rule_id == "final_punctuation"
        and example.mode == "positive"
        and example.metadata.get("expected_error") == "missing_final_punctuation"
        and validate_surface(example.source_text) == ["missing_final_punctuation"]
        and validate_surface(example.target_text) == []
        and _is_valid_final_punctuation_positive_source(example)
    ):
        return {"missing_final_punctuation"}
    return set()


def validate_ast_sentence(ast: Any, rendered_text: str) -> list[str]:
    reasons = validate_surface(rendered_text)
    for clause in _clauses_for_ast(ast):
        reasons.extend(_validate_clause_semantics(clause))
    reasons.extend(f"semantic_surface:{reason}" for reason in reject_semantic_nonsense(rendered_text))
    return _dedupe(reasons)


def assert_valid_or_raise(ast: Any, rendered_text: str) -> None:
    reasons = validate_ast_sentence(ast, rendered_text)
    if reasons:
        raise ValueError(f"Invalid generated sentence: {', '.join(reasons)}")


def _clauses_for_ast(ast: Any) -> tuple[Clause, ...]:
    if isinstance(ast, SimpleSentence):
        return (ast.clause,)
    if isinstance(ast, ComplexSentence):
        return (ast.main, ast.subordinate)
    if isinstance(ast, IntroductorySentence):
        return (ast.clause,)
    if isinstance(ast, HomogeneousSentence):
        return tuple(Clause(subject=ast.subject, predicate=predicate) for predicate in ast.predicates)
    if isinstance(ast, DashSubjectPredicateSentence):
        return ()
    return ()


def _validate_clause_semantics(clause: Clause) -> list[str]:
    lexicon = _default_lexicon()
    frame = _resolve_frame(lexicon, clause.predicate)
    subject = _resolve_noun(lexicon, clause.subject)
    object_entry = _semantic_object_entry(lexicon, frame, clause.predicate.object_np)

    payload = {"frame": frame, "subject": subject, "object_np": object_entry}
    reasons = validate_clause_semantics(payload)

    if frame is not None and subject is not None and clause.predicate.verb_lemma != frame.verb_lemma:
        reasons.append("verb_frame_mismatch")

    if frame is not None and clause.predicate.object_np is not None:
        reasons.extend(_validate_object_np_semantics(lexicon, frame, clause.predicate.object_np))

    return reasons


def _semantic_object_entry(
    lexicon: Lexicon,
    frame: VerbFrame | None,
    object_np: NounPhrase | None,
) -> NounEntry | None:
    if object_np is None:
        return None
    if frame is not None and object_np.preposition is not None and not frame.object_classes:
        return None
    return _resolve_noun(lexicon, object_np)


def _validate_object_np_semantics(lexicon: Lexicon, frame: VerbFrame, object_np: NounPhrase) -> list[str]:
    object_entry = _resolve_noun(lexicon, object_np)
    if object_entry is None:
        return ["unknown_object_np"]

    if object_np.preposition is None:
        return []

    if not lexicon.frames.validate_prep_slot(frame, object_np.preposition, object_entry, object_np.case):
        return ["invalid_prep_slot_semantics"]
    return []


def _resolve_frame(lexicon: Lexicon, predicate: VerbPhrase) -> VerbFrame | None:
    if predicate.frame_id:
        for frame in lexicon.frames.frames:
            if frame.frame_id == predicate.frame_id:
                return frame
    frames = lexicon.frames.frames_for_verb(predicate.verb_lemma)
    return frames[0] if frames else None


def _resolve_noun(lexicon: Lexicon, np: NounPhrase | None) -> NounEntry | None:
    if np is None:
        return None
    for noun in lexicon.nouns:
        if noun.lemma == np.noun_lemma:
            return noun
    return None


def _has_bad_vo_phrase(text: str) -> bool:
    for match in VO_RE.finditer(text):
        start = match.start() + len(match.group(1) or "")
        suffix = text[start:]
        if not any(_starts_with_whitelist_phrase(suffix, phrase) for phrase in VO_WHITELIST):
            return True
    return False


def _is_valid_final_punctuation_positive_source(example: GeneratedExample) -> bool:
    target_mark = _final_mark(example.target_text)
    if target_mark is None:
        return False
    if _final_mark(example.source_text) is not None:
        return False
    if example.source_text != example.target_text[: -len(target_mark)]:
        return False
    if not example.gap_labels:
        return False
    if example.gap_labels[-1] != FINAL_PUNCTUATION_LABELS[target_mark]:
        return False
    return all(label == "KEEP" for label in example.token_edit_labels)


def _final_mark(text: str) -> str | None:
    stripped = text.rstrip()
    if stripped.endswith("..."):
        return "..."
    if stripped.endswith("\u2026"):
        return "\u2026"
    if stripped and stripped[-1] in ".!?":
        return stripped[-1]
    return None


def _starts_with_whitelist_phrase(text: str, phrase: str) -> bool:
    if not text.startswith(phrase):
        return False
    if len(text) == len(phrase):
        return True
    return text[len(phrase)] in " \t\r\n,.!?;:"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@lru_cache(maxsize=1)
def _default_lexicon() -> Lexicon:
    return Lexicon.default()
