from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.load_config import load_config
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.semantics import SemanticFrameLexicon, VerbFrame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Lexeme:
    lemma: str


@dataclass(frozen=True)
class NounEntry(Lexeme):
    gender: str
    animacy: str
    semantic_class: str
    agentive: bool
    can_be_patient: bool
    can_be_location: bool
    can_be_content_source: bool
    forms: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VerbEntry(Lexeme):
    transitive: bool
    semantic_class: str
    allow_object: bool
    allow_ne: bool
    forms: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AdjectiveEntry(Lexeme):
    semantic_class: str
    forms: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AdverbEntry(Lexeme):
    semantic_class: str


@dataclass(frozen=True)
class PrepositionEntry:
    preposition: str
    case: str
    allowed_semantic_class: str


@dataclass(frozen=True)
class IntroductoryEntry(Lexeme):
    semantic_class: str


@dataclass(frozen=True)
class Lexicon:
    nouns: tuple[NounEntry, ...]
    verbs: tuple[VerbEntry, ...]
    adjectives: tuple[AdjectiveEntry, ...]
    adverbs: tuple[AdverbEntry, ...]
    prepositions: tuple[PrepositionEntry, ...]
    introductory_words: tuple[IntroductoryEntry, ...]
    conjunctions: tuple[str, ...]
    exceptions: dict[str, frozenset[str]]
    frames: SemanticFrameLexicon

    @classmethod
    def default(cls) -> "Lexicon":
        lexicon_dir = PROJECT_ROOT / "lexicon"
        if DEFAULT_CONFIG_PATH.exists():
            config = load_config(DEFAULT_CONFIG_PATH)
            configured = config.get("paths", {}).get("lexicon_dir")
            if configured:
                candidate = Path(str(configured))
                lexicon_dir = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        return cls.from_dir(lexicon_dir)

    @classmethod
    def from_dir(cls, path: str | Path) -> "Lexicon":
        base_path = Path(path)
        required = (
            "nouns.csv",
            "verbs.csv",
            "adjectives.csv",
            "adverbs.csv",
            "prepositions.csv",
            "introductory_words.csv",
            "conjunctions.csv",
            "exceptions.csv",
        )
        if not base_path.exists() or any(not (base_path / name).exists() for name in required):
            return _fallback_lexicon()

        try:
            lexicon = cls(
                nouns=tuple(
                    NounEntry(
                        lemma=row["lemma"],
                        gender=row["gender"],
                        animacy=row["animacy"],
                        semantic_class=row["semantic_class"],
                        agentive=_parse_bool(row["agentive"]),
                        can_be_patient=_parse_bool(row["can_be_patient"]),
                        can_be_location=_parse_bool(row["can_be_location"]),
                        can_be_content_source=_parse_bool(row["can_be_content_source"]),
                        forms=_forms_from_row(row, NOUN_FORM_COLUMNS),
                    )
                    for row in _read_csv(base_path / "nouns.csv")
                ),
                verbs=tuple(
                    VerbEntry(
                        lemma=row["lemma"],
                        semantic_class=row["semantic_class"],
                        transitive=_parse_bool(row["transitive"]),
                        allow_object=_parse_bool(row["allow_object"]),
                        allow_ne=_parse_bool(row["allow_ne"]),
                        forms=_forms_from_row(row, VERB_FORM_COLUMNS),
                    )
                    for row in _read_csv(base_path / "verbs.csv")
                ),
                adjectives=tuple(
                    AdjectiveEntry(
                        lemma=row["lemma"],
                        semantic_class=row["semantic_class"],
                        forms=_forms_from_row(row, ADJECTIVE_FORM_COLUMNS),
                    )
                    for row in _read_csv(base_path / "adjectives.csv")
                ),
                adverbs=tuple(
                    AdverbEntry(lemma=row["lemma"], semantic_class=row["semantic_class"])
                    for row in _read_csv(base_path / "adverbs.csv")
                ),
                prepositions=tuple(
                    PrepositionEntry(
                        preposition=row["preposition"],
                        case=row["case"],
                        allowed_semantic_class=row["allowed_semantic_class"],
                    )
                    for row in _read_csv(base_path / "prepositions.csv")
                ),
                introductory_words=tuple(
                    IntroductoryEntry(lemma=row["lemma"], semantic_class=row["semantic_class"])
                    for row in _read_csv(base_path / "introductory_words.csv")
                ),
                conjunctions=tuple(row["conjunction"] for row in _read_csv(base_path / "conjunctions.csv")),
                exceptions=_read_exceptions(base_path / "exceptions.csv"),
                frames=SemanticFrameLexicon.from_dir(base_path),
            )
        except (AttributeError, KeyError, ValueError, OSError, csv.Error):
            return _fallback_lexicon()

        if not all(
            (
                lexicon.nouns,
                lexicon.verbs,
                lexicon.adjectives,
                lexicon.adverbs,
                lexicon.prepositions,
                lexicon.introductory_words,
                lexicon.conjunctions,
            )
        ):
            return _fallback_lexicon()
        return lexicon

    def random_noun(
        self,
        rng: RandomSource,
        semantic_class: str | None = None,
        animacy: str | None = None,
    ) -> NounEntry:
        candidates = [
            noun
            for noun in self.nouns
            if (semantic_class is None or noun.semantic_class == semantic_class)
            and (animacy is None or noun.animacy == animacy)
        ]
        if not candidates:
            raise ValueError("No noun entries match the requested filters.")
        return rng.choice(tuple(candidates))

    def random_noun_for_classes(self, classes: tuple[str, ...] | list[str] | set[str], rng: RandomSource) -> NounEntry:
        allowed = set(classes)
        candidates = [noun for noun in self.nouns if noun.semantic_class in allowed]
        if not candidates:
            raise ValueError("No noun entries match the requested semantic classes.")
        return rng.choice(tuple(candidates))

    def random_subject_for_frame(self, frame: VerbFrame, rng: RandomSource) -> NounEntry:
        candidates = [noun for noun in self.nouns if self.frames.validate_subject(frame, noun)]
        if not candidates:
            raise ValueError(f"No noun entries can fill subject slot for frame {frame.frame_id!r}.")
        return rng.choice(tuple(candidates))

    def random_object_for_frame(self, frame: VerbFrame, rng: RandomSource) -> NounEntry:
        if not frame.object_classes:
            raise ValueError(f"Frame {frame.frame_id!r} does not allow a direct object.")
        candidates = [noun for noun in self.nouns if self.frames.validate_object(frame, noun)]
        if not candidates:
            raise ValueError(f"No noun entries can fill object slot for frame {frame.frame_id!r}.")
        return rng.choice(tuple(candidates))

    def random_verb(
        self,
        rng: RandomSource,
        allow_ne: bool | None = None,
        transitive: bool | None = None,
    ) -> VerbEntry:
        candidates = [
            verb
            for verb in self.verbs
            if (allow_ne is None or verb.allow_ne is allow_ne)
            and (transitive is None or verb.transitive is transitive)
        ]
        if not candidates:
            raise ValueError("No verb entries match the requested filters.")
        return rng.choice(tuple(candidates))

    def random_adjective(self, rng: RandomSource) -> AdjectiveEntry:
        return rng.choice(self.adjectives)

    def random_adverb(self, rng: RandomSource) -> AdverbEntry:
        return rng.choice(self.adverbs)

    def random_introductory(self, rng: RandomSource) -> IntroductoryEntry:
        return rng.choice(self.introductory_words)

    def random_preposition_for(
        self,
        noun: NounEntry,
        rng: RandomSource,
        allowed_case: str | None = None,
    ) -> PrepositionEntry:
        candidates = [
            entry
            for entry in self.prepositions
            if _semantic_matches(entry.allowed_semantic_class, noun.semantic_class)
            and (allowed_case is None or _canonical_case(entry.case) == _canonical_case(allowed_case))
        ]
        if not candidates:
            raise ValueError("No preposition entries match the requested noun and case.")
        return rng.choice(tuple(candidates))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
            if any((value or "").strip() for value in row.values())
        ]


NOUN_FORM_COLUMNS = (
    "nom_sg",
    "gen_sg",
    "dat_sg",
    "acc_sg",
    "ins_sg",
    "loc_sg",
    "nom_pl",
    "acc_pl",
)
VERB_FORM_COLUMNS = (
    "past_masc",
    "past_fem",
    "past_neut",
    "past_plur",
    "present_3sg",
    "infinitive",
)
ADJECTIVE_FORM_COLUMNS = (
    "masc_nom",
    "fem_nom",
    "neut_nom",
    "plur_nom",
    "fem_acc",
    "masc_acc_inanim",
    "neut_acc",
)


def _forms_from_row(row: dict[str, str], columns: tuple[str, ...]) -> dict[str, str]:
    return {
        column: row[column]
        for column in columns
        if row.get(column)
    }


def _read_exceptions(path: Path) -> dict[str, frozenset[str]]:
    exceptions: dict[str, set[str]] = {}
    for row in _read_csv(path):
        exceptions.setdefault(row["rule_id"], set()).add(row["lemma"])
    return {rule_id: frozenset(lemmas) for rule_id, lemmas in exceptions.items()}


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected true or false, got {value!r}.")


def _canonical_case(case: str) -> str:
    return {
        "nom": "nomn",
        "nomn": "nomn",
        "gen": "gent",
        "gent": "gent",
        "dat": "datv",
        "datv": "datv",
        "acc": "accs",
        "accs": "accs",
        "ins": "ablt",
        "ablt": "ablt",
        "prep": "loct",
        "loct": "loct",
    }.get(case, case)


def _semantic_matches(allowed: str, actual: str) -> bool:
    if allowed == actual:
        return True
    if allowed == "entity" and actual in {"person", "organization"}:
        return True
    return False


def _fallback_lexicon() -> Lexicon:
    return Lexicon(
        nouns=(
            NounEntry("девочка", "fem", "anim", "person", True, True, False, False),
            NounEntry("студент", "masc", "anim", "person", True, True, False, False),
            NounEntry("комиссия", "fem", "inanim", "organization", True, True, False, False),
            NounEntry("дом", "masc", "inanim", "building", False, True, True, False),
            NounEntry("отчёт", "masc", "inanim", "report", False, True, False, True),
            NounEntry("продукты", "plur", "inanim", "food", False, True, False, False),
        ),
        verbs=(
            VerbEntry("пойти", False, "motion", False, True),
            VerbEntry("решить", True, "action", True, True),
            VerbEntry("проверить", True, "action", True, True),
            VerbEntry("ненавидеть", True, "emotion", True, False),
        ),
        adjectives=(
            AdjectiveEntry("умный", "quality"),
            AdjectiveEntry("новый", "quality"),
            AdjectiveEntry("важный", "quality"),
        ),
        adverbs=(
            AdverbEntry("быстро", "manner"),
            AdverbEntry("внимательно", "manner"),
            AdverbEntry("сегодня", "time"),
        ),
        prepositions=(
            PrepositionEntry("в", "loct", "place"),
            PrepositionEntry("к", "dat", "person"),
            PrepositionEntry("у", "gent", "person"),
            PrepositionEntry("после", "gent", "event"),
        ),
        introductory_words=(
            IntroductoryEntry("конечно", "certainty"),
            IntroductoryEntry("возможно", "uncertainty"),
            IntroductoryEntry("например", "example"),
        ),
        conjunctions=("что", "чтобы", "потому что", "если", "когда", "хотя", "но", "а"),
        exceptions={"ne_verb": frozenset({"ненавидеть", "негодовать", "недомогать", "недоумевать", "неймётся"})},
        frames=SemanticFrameLexicon.from_dir(Path()),
    )
