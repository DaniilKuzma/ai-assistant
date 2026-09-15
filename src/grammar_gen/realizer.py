from __future__ import annotations

from dataclasses import replace
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
from src.grammar_gen.lexicon import Lexicon
from src.grammar_gen.morphology import MorphologyEngine
from src.schema import WordToken


WORD_RE = re.compile(r"[А-Яа-яЁё-]+")
SPACE_RE = re.compile(r"\s+")
REPEATED_PUNCTUATION_RE = re.compile(r"([,!?;:])\1+")
FINAL_PUNCTUATION_RE = re.compile(r"[.!?…]+$")


class Realizer:
    def __init__(self, lexicon: Lexicon, morphology: MorphologyEngine) -> None:
        self.lexicon = lexicon
        self.morphology = morphology

    def render_np(self, np: NounPhrase) -> str:
        parts: list[str] = []
        if np.preposition:
            parts.append(np.preposition)
        parts.extend(
            self.morphology.inflect_adjective(adjective, np.gender, _adjective_case(np), np.number)
            for adjective in np.adjective_lemmas
        )
        parts.append(self.morphology.inflect_noun(np.noun_lemma, np.case, np.number))
        return _normalize_inner_spacing(" ".join(parts))

    def render_vp(self, vp: VerbPhrase, subject: NounPhrase) -> str:
        verb = self._render_verb(vp, subject)
        parts = list(vp.adverbs)
        if vp.negated:
            parts.append("не")
        parts.append(verb)

        if vp.object_np is not None:
            parts.append(self.render_np(_object_case_np(vp.object_np)))

        return _normalize_inner_spacing(" ".join(parts))

    def render_clause(self, clause: Clause) -> str:
        parts = [
            *clause.left_adverbials,
            self.render_np(clause.subject),
            self.render_vp(clause.predicate, clause.subject),
            *clause.right_adverbials,
        ]
        return _normalize_inner_spacing(" ".join(part for part in parts if part))

    def render_sentence(self, ast: Any) -> str:
        if isinstance(ast, SimpleSentence):
            text = self.render_clause(ast.clause)
            return _finish_sentence(text, ast.final_punctuation)

        if isinstance(ast, ComplexSentence):
            main = self.render_clause(ast.main)
            subordinate = self.render_clause(ast.subordinate)
            joiner = f", {ast.conjunction} " if ast.comma_before_conjunction else f" {ast.conjunction} "
            return _finish_sentence(f"{main}{joiner}{subordinate}", ast.final_punctuation)

        if isinstance(ast, IntroductorySentence):
            clause = self.render_clause(ast.clause)
            introductory = ast.introductory.strip()
            if ast.position == "medial":
                subject = self.render_np(ast.clause.subject)
                predicate = self.render_vp(ast.clause.predicate, ast.clause.subject)
                right = " ".join(ast.clause.right_adverbials)
                text = _normalize_inner_spacing(f"{subject}, {introductory}, {predicate} {right}")
            else:
                text = f"{introductory}, {clause}"
            return _finish_sentence(text, ast.final_punctuation)

        if isinstance(ast, HomogeneousSentence):
            text = self._render_homogeneous(ast)
            return _finish_sentence(text, ast.final_punctuation)

        if isinstance(ast, DashSubjectPredicateSentence):
            text = f"{self.render_np(ast.subject)} - {self.render_np(ast.predicate_nominal)}"
            return _finish_sentence(text, ast.final_punctuation)

        raise TypeError(f"Unsupported sentence AST: {type(ast).__name__}.")

    def tokenize_words_with_offsets(self, text: str) -> list[WordToken]:
        tokens: list[WordToken] = []
        for match in WORD_RE.finditer(text):
            token_text = match.group(0)
            analysis = self.morphology.analyze_token(token_text)
            tokens.append(
                WordToken(
                    text=token_text,
                    start=match.start(),
                    end=match.end(),
                    lemma=analysis.lemma,
                    pos=analysis.pos,
                    feats={},
                )
            )
        return tokens

    def _render_verb(self, vp: VerbPhrase, subject: NounPhrase) -> str:
        if vp.tense == "present":
            return self.morphology.inflect_verb_present(vp.verb_lemma, number=subject.number)
        if vp.tense == "past":
            return self.morphology.inflect_verb_past(vp.verb_lemma, subject.gender, subject.number)
        return self.morphology.infinitive(vp.verb_lemma)

    def _render_homogeneous(self, ast: HomogeneousSentence) -> str:
        subject = self.render_np(ast.subject)
        predicates = [self.render_vp(predicate, ast.subject) for predicate in ast.predicates]
        if not predicates:
            return subject
        if len(predicates) == 1:
            return f"{subject} {predicates[0]}"

        if ast.conjunction:
            predicate_text = f" {ast.conjunction} ".join(predicates)
        else:
            predicate_text = ", ".join(predicates)
        return f"{subject} {predicate_text}"


def _object_case_np(np: NounPhrase) -> NounPhrase:
    if np.preposition is not None:
        return np
    if np.case == "nomn":
        return replace(np, case="accs")
    return np


def _adjective_case(np: NounPhrase) -> str:
    if np.case != "accs" or np.animacy != "inanim":
        return np.case
    if np.number == "plur" or np.gender in {"masc", "neut"}:
        return "nomn"
    return np.case


def _finish_sentence(text: str, punctuation: str) -> str:
    text = _normalize_sentence_spacing(text)
    text = FINAL_PUNCTUATION_RE.sub("", text).rstrip(" ,;:")
    text = _capitalize_first(text)
    final = _normalize_final_punctuation(punctuation)
    return f"{text}{final}"


def _normalize_sentence_spacing(text: str) -> str:
    text = _normalize_inner_spacing(text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,;:])([^\s])", r"\1 \2", text)
    return REPEATED_PUNCTUATION_RE.sub(r"\1", text).strip()


def _normalize_inner_spacing(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def _capitalize_first(text: str) -> str:
    if not text:
        return text
    return f"{text[0].upper()}{text[1:]}"


def _normalize_final_punctuation(punctuation: str) -> str:
    if punctuation == "...":
        return punctuation
    if punctuation and punctuation[0] in ".!?":
        return punctuation[0]
    return "."
