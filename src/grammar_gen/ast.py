from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NounPhrase:
    noun_lemma: str
    gender: str
    animacy: str
    number: str = "sing"
    case: str = "nomn"
    adjective_lemmas: tuple[str, ...] = ()
    semantic_class: str = ""
    preposition: str | None = None


@dataclass(frozen=True)
class VerbPhrase:
    verb_lemma: str
    tense: str = "past"
    transitive: bool = True
    object_np: NounPhrase | None = None
    adverbs: tuple[str, ...] = ()
    negated: bool = False
    frame_id: str = ""


@dataclass(frozen=True)
class Clause:
    subject: NounPhrase
    predicate: VerbPhrase
    left_adverbials: tuple[str, ...] = ()
    right_adverbials: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimpleSentence:
    clause: Clause
    final_punctuation: str = "."


@dataclass(frozen=True)
class ComplexSentence:
    main: Clause
    conjunction: str
    subordinate: Clause
    comma_before_conjunction: bool = True
    final_punctuation: str = "."


@dataclass(frozen=True)
class IntroductorySentence:
    introductory: str
    clause: Clause
    position: str = "initial"
    final_punctuation: str = "."


@dataclass(frozen=True)
class HomogeneousSentence:
    subject: NounPhrase
    predicates: tuple[VerbPhrase, ...]
    conjunction: str | None = None
    final_punctuation: str = "."


@dataclass(frozen=True)
class DashSubjectPredicateSentence:
    subject: NounPhrase
    predicate_nominal: NounPhrase
    pair_id: str = ""
    final_punctuation: str = "."
