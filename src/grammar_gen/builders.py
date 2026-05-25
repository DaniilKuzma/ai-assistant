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
        return Clause(
            subject=subject,
            predicate=predicate,
            left_adverbials=self._optional_left_adverbials(),
            right_adverbials=self._optional_right_adverbials(),
        )

    def simple_sentence(self) -> SimpleSentence:
        return SimpleSentence(clause=self.random_clause())

    def complex_subordinate_sentence(self, conjunction: str = "что") -> ComplexSentence:
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
        subject_classes = ("document", "report", "text", "message", "law", "plan", "request")
        predicate_classes = ("document", "report", "text", "message", "rule", "fact", "information")
        subject = self._noun_phrase(self.lexicon.random_noun_for_classes(subject_classes, self.rng))
        predicate = self._noun_phrase(self.lexicon.random_noun_for_classes(predicate_classes, self.rng))
        return DashSubjectPredicateSentence(subject=subject, predicate_nominal=predicate)

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
        return (self.lexicon.random_adverb(self.rng).lemma,)

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
    ) -> NounPhrase:
        number = "plur" if noun.gender == "plur" else "sing"
        return NounPhrase(
            noun_lemma=noun.lemma,
            gender=noun.gender,
            animacy=noun.animacy,
            number=number,
            case=case,
            adjective_lemmas=self._optional_adjectives(),
            semantic_class=noun.semantic_class,
            preposition=preposition,
        )

    def _optional_adjectives(self) -> tuple[str, ...]:
        if not self.rng.chance(0.30):
            return ()
        first = self.lexicon.random_adjective(self.rng).lemma
        if self.rng.chance(0.08):
            second = self.lexicon.random_adjective(self.rng).lemma
            if second != first:
                return (first, second)
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
