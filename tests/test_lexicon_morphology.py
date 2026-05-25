from __future__ import annotations

from pathlib import Path

import pytest

from src.grammar_gen import (
    Lexicon,
    MorphologyEngine,
    RandomSource,
    reject_bad_surface,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_lexicon_loads_curated_entries() -> None:
    lexicon = Lexicon.default()

    assert len(lexicon.nouns) >= 80
    assert len(lexicon.verbs) >= 80
    assert len(lexicon.adjectives) >= 40
    assert len(lexicon.adverbs) >= 30
    assert len(lexicon.prepositions) >= 20
    assert len(lexicon.introductory_words) >= 20


def test_required_curated_csv_files_exist() -> None:
    expected_files = {
        "nouns.csv",
        "verbs.csv",
        "adjectives.csv",
        "adverbs.csv",
        "prepositions.csv",
        "introductory_words.csv",
        "conjunctions.csv",
        "exceptions.csv",
    }

    assert expected_files <= {path.name for path in (ROOT / "lexicon").iterdir()}


@pytest.mark.parametrize(
    ("noun", "verb", "expected"),
    [
        ("девочка", "пойти", "девочка пошла"),
        ("студент", "пойти", "студент пошёл"),
        ("комиссия", "решить", "комиссия решила"),
    ],
)
def test_past_tense_verb_agrees_with_subject(noun: str, verb: str, expected: str) -> None:
    lexicon = Lexicon.default()
    engine = MorphologyEngine()
    subject = next(entry for entry in lexicon.nouns if entry.lemma == noun)

    surface = f"{subject.lemma} {engine.inflect_verb_past(verb, subject.gender)}"

    assert surface == expected


@pytest.mark.parametrize(
    ("noun", "adjective", "expected"),
    [
        ("девочка", "умный", "умная девочка"),
        ("студент", "умный", "умный студент"),
        ("здание", "умный", "умное здание"),
    ],
)
def test_adjective_agrees_with_noun_gender(noun: str, adjective: str, expected: str) -> None:
    lexicon = Lexicon.default()
    engine = MorphologyEngine()
    entry = next(item for item in lexicon.nouns if item.lemma == noun)

    surface = f"{engine.inflect_adjective(adjective, entry.gender, 'nomn')} {entry.lemma}"

    assert surface == expected


def test_bad_vo_ogorod_surface_is_rejected() -> None:
    assert "bad_vo_phrase" in reject_bad_surface("во огород")


def test_random_source_is_deterministic_by_seed() -> None:
    left = RandomSource(seed=17)
    right = RandomSource(seed=17)
    lexicon = Lexicon.default()

    left_values = [
        lexicon.random_noun(left).lemma,
        lexicon.random_verb(left).lemma,
        lexicon.random_adjective(left).lemma,
        left.randint(1, 100),
    ]
    right_values = [
        lexicon.random_noun(right).lemma,
        lexicon.random_verb(right).lemma,
        lexicon.random_adjective(right).lemma,
        right.randint(1, 100),
    ]

    assert left_values == right_values


def test_fallback_morphology_covers_required_forms() -> None:
    engine = MorphologyEngine(use_pymorphy=False)

    assert engine.inflect_verb_past("пойти", "fem") == "пошла"
    assert engine.inflect_verb_past("пойти", "masc") == "пошёл"
    assert engine.inflect_verb_past("решить", "fem") == "решила"
    assert engine.inflect_adjective("умный", "neut", "nomn") == "умное"

