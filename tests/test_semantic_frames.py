from __future__ import annotations

from src.grammar_gen import Lexicon, RandomSource
from src.grammar_gen.semantic_safety import reject_semantic_nonsense, validate_frame_fillers


def test_required_frame_subject_validation() -> None:
    lexicon = Lexicon.default()
    buy_goods = _frame(lexicon, "buy_goods")
    approve_decision = _frame(lexicon, "approve_decision")
    contain_info = _frame(lexicon, "contain_info")
    say_fact = _frame(lexicon, "say_fact")
    stand_place = _frame(lexicon, "stand_place")

    assert not lexicon.frames.validate_subject(buy_goods, _noun(lexicon, "дом"))
    assert lexicon.frames.validate_subject(buy_goods, _noun(lexicon, "студент"))
    assert lexicon.frames.validate_subject(approve_decision, _noun(lexicon, "комиссия"))
    assert lexicon.frames.validate_subject(contain_info, _noun(lexicon, "отчёт"))
    assert not lexicon.frames.validate_subject(say_fact, _noun(lexicon, "отчёт"))
    assert lexicon.frames.validate_subject(stand_place, _noun(lexicon, "дом"))


def test_buy_goods_subject_sampling_excludes_ineligible_classes() -> None:
    lexicon = Lexicon.default()
    rng = RandomSource(seed=11)
    frame = _frame(lexicon, "buy_goods")
    forbidden = {"building", "place", "document"}

    subjects = [lexicon.random_subject_for_frame(frame, rng) for _ in range(100)]

    assert {subject.semantic_class for subject in subjects}.isdisjoint(forbidden)


def test_buy_goods_object_sampling_uses_only_allowed_classes() -> None:
    lexicon = Lexicon.default()
    rng = RandomSource(seed=13)
    frame = _frame(lexicon, "buy_goods")
    allowed = {"food", "goods", "object", "property", "service"}

    objects = [lexicon.random_object_for_frame(frame, rng) for _ in range(100)]

    assert {item.semantic_class for item in objects} <= allowed


def test_random_frame_first_clauses_do_not_render_home_bought_food() -> None:
    lexicon = Lexicon.default()
    rng = RandomSource(seed=17)
    rendered = []

    for _ in range(1000):
        frame = lexicon.frames.random_frame(rng)
        subject = lexicon.random_subject_for_frame(frame, rng)
        obj = lexicon.random_object_for_frame(frame, rng) if frame.object_classes else None
        rendered.append(_render_clause(subject.lemma, frame.verb_lemma, obj.lemma if obj else None))

    assert "Дом купил продукты" not in rendered


def test_semantic_safety_rejects_manual_nonsense() -> None:
    lexicon = Lexicon.default()
    buy_goods = _frame(lexicon, "buy_goods")
    home = _noun(lexicon, "дом")
    products = _noun(lexicon, "продукты")

    assert validate_frame_fillers(buy_goods, home, products)
    assert reject_semantic_nonsense("Дом купил продукты.")
    assert reject_semantic_nonsense("Отчёт сказал факт.")


def _frame(lexicon: Lexicon, frame_id: str):
    return next(frame for frame in lexicon.frames.frames if frame.frame_id == frame_id)


def _noun(lexicon: Lexicon, lemma: str):
    return next(noun for noun in lexicon.nouns if noun.lemma == lemma)


def _render_clause(subject_lemma: str, verb_lemma: str, object_lemma: str | None) -> str:
    verb = _past_masculine_verb(verb_lemma)
    words = [subject_lemma.capitalize(), verb]
    if object_lemma:
        words.append(object_lemma)
    return " ".join(words)


def _past_masculine_verb(lemma: str) -> str:
    forms = {
        "купить": "купил",
        "сказать": "сказал",
        "объяснить": "объяснил",
        "прочитать": "прочитал",
        "решить": "решил",
    }
    if lemma in forms:
        return forms[lemma]
    if lemma.endswith("ить"):
        return f"{lemma[:-3]}ил"
    if lemma.endswith("ать"):
        return f"{lemma[:-3]}ал"
    if lemma.endswith("ять"):
        return f"{lemma[:-3]}ял"
    return lemma
