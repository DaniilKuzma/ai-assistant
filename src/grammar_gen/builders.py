from __future__ import annotations

import re

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
from src.grammar_gen.morphology import MorphologyEngine
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.semantics import PrepSlot, VerbFrame


CYRILLIC_PREPOSITION_RE = re.compile(r"^[А-Яа-яЁё-]+$")
SAFE_SUBORDINATE_FRAME_IDS = (
    "contain_info",
    "law_contains",
    "document_describes",
    "report_shows",
    "plan_includes",
    "message_contains",
    "book_contains",
    "request_contains",
    "data_indicates",
    "include_requirement",
)
DASH_NOMINAL_PAIRS = (
    ("report_document", "отчёт", "документ"),
    ("law_document", "закон", "документ"),
    ("error_problem", "ошибка", "проблема"),
    ("request_document", "заявка", "документ"),
    ("protocol_document", "протокол", "документ"),
    ("instruction_text", "инструкция", "текст"),
    ("meeting_event", "собрание", "событие"),
    ("plan_document", "план", "документ"),
)


class GrammarBuilder:
    def __init__(self, lexicon: Lexicon, morphology: MorphologyEngine, rng: RandomSource) -> None:
        self.lexicon = lexicon
        self.morphology = morphology
        self.rng = rng

    def random_subject(self, semantic_class: str | None = None) -> NounPhrase:
        return self._noun_phrase(self.lexicon.random_noun(self.rng, semantic_class=semantic_class))

    def random_object(self) -> NounPhrase:
        candidates = tuple(noun for noun in self.lexicon.nouns if noun.can_be_patient)
        return self._noun_phrase(self.rng.choice(candidates), case="accs")

    def random_clause(self, transitive: bool | None = None, allow_negation: bool = False) -> Clause:
        frame = self.lexicon.frames.random_frame(self.rng, allow_object=transitive)
        subject_entry = self.lexicon.random_subject_for_frame(frame, self.rng)
        subject = self._noun_phrase(subject_entry)
        object_np = self._object_for_frame(frame)
        adverbs = self._optional_adverbs()
        predicate = VerbPhrase(
            verb_lemma=frame.verb_lemma,
            transitive=bool(frame.object_classes),
            object_np=object_np,
            adverbs=adverbs,
            negated=allow_negation and frame.allow_ne and self.rng.chance(0.12),
            frame_id=frame.frame_id,
        )
        left_adverbials = self._optional_left_adverbials()
        right_adverbials = () if left_adverbials else self._optional_right_adverbials()
        return Clause(
            subject=subject,
            predicate=predicate,
            left_adverbials=left_adverbials,
            right_adverbials=right_adverbials,
        )

    def simple_sentence(self) -> SimpleSentence:
        return SimpleSentence(clause=self.random_clause())

    def complex_subordinate_sentence(self, conjunction: str = "что") -> ComplexSentence:
        if conjunction == "что":
            return self._that_complement_sentence()
        return ComplexSentence(
            main=self.random_clause(transitive=True),
            conjunction=conjunction,
            subordinate=self.random_clause(),
        )

    def introductory_sentence(self) -> IntroductorySentence:
        position = "medial" if self.rng.chance(0.25) else "initial"
        return IntroductorySentence(
            introductory=self.lexicon.random_introductory(self.rng).lemma,
            clause=self.random_clause(),
            position=position,
        )

    def homogeneous_sentence(self) -> HomogeneousSentence:
        subject_entry = self._subject_with_compatible_frames(min_count=2)
        subject = self._noun_phrase(subject_entry)
        frames = self._compatible_frames(subject_entry)
        first = self.rng.choice(frames)
        remaining = tuple(frame for frame in frames if frame.frame_id != first.frame_id)
        second = self.rng.choice(remaining or frames)
        predicates = (
            self._predicate_for_frame(first, allow_negation=False),
            self._predicate_for_frame(second, allow_negation=False),
        )
        conjunction = "и" if self.rng.chance(0.55) else None
        return HomogeneousSentence(subject=subject, predicates=predicates, conjunction=conjunction)

    def dash_subject_predicate_sentence(self) -> DashSubjectPredicateSentence:
        pair_id, subject_lemma, predicate_lemma = self.rng.choice(DASH_NOMINAL_PAIRS)
        subject = self._noun_phrase(self._noun_by_lemma(subject_lemma), allow_adjectives=False)
        predicate = self._noun_phrase(self._noun_by_lemma(predicate_lemma), allow_adjectives=False)
        return DashSubjectPredicateSentence(subject=subject, predicate_nominal=predicate, pair_id=pair_id)

    def _that_complement_sentence(self) -> ComplexSentence:
        main_frame = self.lexicon.frames.random_frame(
            self.rng,
            frame_family="subordinate_complement",
            allow_object=False,
        )
        return ComplexSentence(
            main=self._clause_for_frame(main_frame, allow_adverbs=False, allow_adverbials=False),
            conjunction="что",
            subordinate=self._safe_subordinate_clause(),
        )

    def _safe_subordinate_clause(self) -> Clause:
        frame = self._frame_by_id(self.rng.choice(SAFE_SUBORDINATE_FRAME_IDS))
        tense = "present" if frame.frame_family == "content" and self.rng.chance(0.65) else "past"
        return self._clause_for_frame(frame, tense=tense, allow_adverbs=False, allow_adverbials=False)

    def _clause_for_frame(
        self,
        frame: VerbFrame,
        *,
        tense: str = "past",
        allow_adverbs: bool = True,
        allow_adverbials: bool = True,
    ) -> Clause:
        if frame.frame_id in {"check_showed", "check_established"}:
            subject_entry = self._noun_by_lemma("проверка")
        else:
            subject_entry = self.lexicon.random_subject_for_frame(frame, self.rng)
        subject = self._noun_phrase(subject_entry)
        predicate = VerbPhrase(
            verb_lemma=frame.verb_lemma,
            tense=tense,
            transitive=bool(frame.object_classes),
            object_np=self._object_for_frame(frame),
            adverbs=self._optional_adverbs() if allow_adverbs else (),
            negated=False,
            frame_id=frame.frame_id,
        )
        left_adverbials = self._optional_left_adverbials() if allow_adverbials else ()
        right_adverbials = self._optional_right_adverbials() if allow_adverbials and not left_adverbials else ()
        return Clause(
            subject=subject,
            predicate=predicate,
            left_adverbials=left_adverbials,
            right_adverbials=right_adverbials,
        )

    def _predicate_for_frame(self, frame: VerbFrame, allow_negation: bool) -> VerbPhrase:
        return VerbPhrase(
            verb_lemma=frame.verb_lemma,
            transitive=bool(frame.object_classes),
            object_np=self._object_for_frame(frame),
            adverbs=self._optional_adverbs(),
            negated=allow_negation and frame.allow_ne and self.rng.chance(0.12),
            frame_id=frame.frame_id,
        )

    def _object_for_frame(self, frame: VerbFrame) -> NounPhrase | None:
        if frame.object_classes:
            return self._noun_phrase(self.lexicon.random_object_for_frame(frame, self.rng), case="accs")

        slot = self._optional_prep_slot(frame)
        if slot is None:
            return None
        candidates = tuple(
            noun
            for noun in self.lexicon.nouns
            if self.lexicon.frames.validate_prep_slot(frame, slot.preposition, noun, slot.case)
        )
        if not candidates:
            return None
        return self._noun_phrase(
            self.rng.choice(candidates),
            case=slot.case,
            preposition=slot.preposition,
        )

    def _optional_prep_slot(self, frame: VerbFrame) -> PrepSlot | None:
        slots = tuple(slot for slot in frame.prep_slots if CYRILLIC_PREPOSITION_RE.fullmatch(slot.preposition))
        if not slots or not self.rng.chance(0.75):
            return None
        return self.rng.choice(slots)

    def _optional_adverbs(self) -> tuple[str, ...]:
        if not self.rng.chance(0.25):
            return ()
        candidates = tuple(adverb for adverb in self.lexicon.adverbs if adverb.semantic_class == "manner")
        return (self.rng.choice(candidates or self.lexicon.adverbs).lemma,)

    def _optional_left_adverbials(self) -> tuple[str, ...]:
        if not self.rng.chance(0.12):
            return ()
        return (self._random_adverbial("time"),)

    def _optional_right_adverbials(self) -> tuple[str, ...]:
        if not self.rng.chance(0.10):
            return ()
        return (self._random_adverbial("time"),)

    def _random_adverbial(self, semantic_class: str) -> str:
        candidates = tuple(adverb for adverb in self.lexicon.adverbs if adverb.semantic_class == semantic_class)
        return self.rng.choice(candidates or self.lexicon.adverbs).lemma

    def _noun_phrase(
        self,
        noun: NounEntry,
        *,
        case: str = "nomn",
        preposition: str | None = None,
        allow_adjectives: bool = True,
    ) -> NounPhrase:
        number = "plur" if noun.gender == "plur" else "sing"
        return NounPhrase(
            noun_lemma=noun.lemma,
            gender=noun.gender,
            animacy=noun.animacy,
            number=number,
            case=case,
            adjective_lemmas=self._optional_adjectives(noun) if allow_adjectives else (),
            semantic_class=noun.semantic_class,
            preposition=preposition,
        )

    def _optional_adjectives(self, noun: NounEntry) -> tuple[str, ...]:
        if not self.rng.chance(0.30):
            return ()
        try:
            first = self.lexicon.random_adjective_for_noun(noun, self.rng).lemma
        except ValueError:
            return ()
        return (first,)

    def _subject_with_compatible_frames(self, min_count: int) -> NounEntry:
        candidates = tuple(
            noun for noun in self.lexicon.nouns if len(self._compatible_frames(noun)) >= min_count
        )
        if not candidates:
            return self.lexicon.random_subject_for_frame(self.lexicon.frames.random_frame(self.rng), self.rng)
        return self.rng.choice(candidates)

    def _compatible_frames(self, subject: NounEntry) -> tuple[VerbFrame, ...]:
        return tuple(
            frame
            for frame in self.lexicon.frames.frames
            if self.lexicon.frames.validate_subject(frame, subject)
        )

    def _frame_by_id(self, frame_id: str) -> VerbFrame:
        for frame in self.lexicon.frames.frames:
            if frame.frame_id == frame_id:
                return frame
        raise ValueError(f"Unknown semantic frame: {frame_id!r}.")

    def _noun_by_lemma(self, lemma: str) -> NounEntry:
        for noun in self.lexicon.nouns:
            if noun.lemma == lemma:
                return noun
        raise ValueError(f"Unknown noun lemma: {lemma!r}.")
