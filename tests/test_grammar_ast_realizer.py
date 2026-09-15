from __future__ import annotations

from dataclasses import replace

from src.grammar_gen import (
    Clause,
    GrammarBuilder,
    Lexicon,
    MorphologyEngine,
    NounPhrase,
    RandomSource,
    Realizer,
    SimpleSentence,
    VerbPhrase,
)
from src.grammar_gen.safety import validate_ast_sentence, validate_surface
from src.grammar_gen.semantic_safety import reject_semantic_nonsense


def test_generated_simple_sentences_pass_surface_validation() -> None:
    builder, realizer = _generator(seed=101)

    texts = [realizer.render_sentence(builder.simple_sentence()) for _ in range(100)]

    assert all(not validate_surface(text) for text in texts)
    assert all("{" not in text and "}" not in text for text in texts)
    assert all("во огород" not in text.lower() for text in texts)
    assert all("девочка пошёл" not in text.lower() for text in texts)


def test_generated_complex_sentences_pass_surface_validation() -> None:
    builder, realizer = _generator(seed=102)

    texts = [realizer.render_sentence(builder.complex_subordinate_sentence()) for _ in range(100)]

    assert all(not validate_surface(text) for text in texts)
    assert all(", что " in text for text in texts)


def test_generated_introductory_sentences_pass_surface_validation() -> None:
    builder, realizer = _generator(seed=103)

    texts = [realizer.render_sentence(builder.introductory_sentence()) for _ in range(100)]

    assert all(not validate_surface(text) for text in texts)
    assert all("," in text for text in texts)


def test_generated_homogeneous_sentences_pass_surface_validation() -> None:
    builder, realizer = _generator(seed=104)

    texts = [realizer.render_sentence(builder.homogeneous_sentence()) for _ in range(100)]

    assert all(not validate_surface(text) for text in texts)


def test_generated_dash_sentences_pass_surface_validation() -> None:
    builder, realizer = _generator(seed=105)

    texts = [realizer.render_sentence(builder.dash_subject_predicate_sentence()) for _ in range(50)]

    assert all(not validate_surface(text) for text in texts)
    assert all(" - " in text for text in texts)


def test_generation_is_deterministic_by_seed() -> None:
    first_builder, first_realizer = _generator(seed=201)
    second_builder, second_realizer = _generator(seed=201)

    first = [first_realizer.render_sentence(first_builder.simple_sentence()) for _ in range(10)]
    second = [second_realizer.render_sentence(second_builder.simple_sentence()) for _ in range(10)]

    assert first == second


def test_different_seeds_generate_different_sentences() -> None:
    first_builder, first_realizer = _generator(seed=301)
    second_builder, second_realizer = _generator(seed=302)

    first = [first_realizer.render_sentence(first_builder.simple_sentence()) for _ in range(10)]
    second = [second_realizer.render_sentence(second_builder.simple_sentence()) for _ in range(10)]

    assert first != second


def test_random_simple_sentences_pass_ast_semantic_safety() -> None:
    builder, realizer = _generator(seed=401)

    for _ in range(1000):
        ast = builder.simple_sentence()
        rendered = realizer.render_sentence(ast)
        assert not validate_ast_sentence(ast, rendered), rendered
        assert not reject_semantic_nonsense(rendered), rendered


def test_home_bought_products_cannot_be_generated_or_validated() -> None:
    builder, realizer = _generator(seed=501)
    generated = [realizer.render_sentence(builder.simple_sentence()) for _ in range(1000)]

    assert "Дом купил продукты." not in generated

    lexicon = Lexicon.default()
    buy_frame = _frame(lexicon, "buy_goods")
    bad_ast = SimpleSentence(
        Clause(
            subject=_np(lexicon, "дом"),
            predicate=VerbPhrase(
                verb_lemma=buy_frame.verb_lemma,
                object_np=replace(_np(lexicon, "продукты"), case="accs"),
                frame_id=buy_frame.frame_id,
            ),
        )
    )
    bad_text = realizer.render_sentence(bad_ast)

    assert bad_text == "Дом купил продукты."
    assert "invalid_subject_semantics" in validate_ast_sentence(bad_ast, bad_text)


def test_required_frame_backed_sentences_can_be_rendered() -> None:
    lexicon = Lexicon.default()
    realizer = Realizer(lexicon, MorphologyEngine())

    cases = [
        (
            "contain_info",
            _np(lexicon, "отчёт"),
            VerbPhrase(
                verb_lemma="содержать",
                tense="present",
                object_np=replace(_np(lexicon, "ошибка"), case="accs"),
                frame_id="contain_info",
            ),
            "Отчёт содержит ошибку.",
        ),
        (
            "stand_place",
            _np(lexicon, "дом"),
            VerbPhrase(
                verb_lemma="стоять",
                object_np=replace(_np(lexicon, "дорога"), case="gent", preposition="у"),
                transitive=False,
                frame_id="stand_place",
            ),
            "Дом стоял у дороги.",
        ),
        (
            "solve_task",
            _np(lexicon, "комиссия"),
            VerbPhrase(
                verb_lemma="решить",
                object_np=replace(_np(lexicon, "вопрос"), case="accs"),
                frame_id="solve_task",
            ),
            "Комиссия решила вопрос.",
        ),
    ]

    for frame_id, subject, predicate, expected in cases:
        assert predicate.frame_id == frame_id
        ast = SimpleSentence(Clause(subject=subject, predicate=predicate))
        rendered = realizer.render_sentence(ast)
        assert rendered == expected
        assert not validate_ast_sentence(ast, rendered)


def test_required_frame_backed_sentences_render_without_pymorphy() -> None:
    lexicon = Lexicon.default()
    realizer = Realizer(lexicon, MorphologyEngine(use_pymorphy=False))

    cases = [
        (
            _np(lexicon, "отчёт"),
            VerbPhrase(
                verb_lemma="содержать",
                tense="present",
                object_np=replace(_np(lexicon, "ошибка"), case="accs"),
                frame_id="contain_info",
            ),
            "Отчёт содержит ошибку.",
        ),
        (
            _np(lexicon, "просьба"),
            VerbPhrase(
                verb_lemma="содержать",
                object_np=replace(_np(lexicon, "информация"), case="accs"),
                frame_id="request_contains",
            ),
            "Просьба содержала информацию.",
        ),
        (
            _np(lexicon, "комиссия"),
            VerbPhrase(
                verb_lemma="решить",
                object_np=replace(_np(lexicon, "вопрос"), case="accs"),
                frame_id="solve_task",
            ),
            "Комиссия решила вопрос.",
        ),
        (
            _np(lexicon, "инженер"),
            VerbPhrase(
                verb_lemma="отправить",
                object_np=replace(_np(lexicon, "таблица"), case="accs", adjective_lemmas=("тихий",)),
                frame_id="send_message",
            ),
            "Инженер отправил тихую таблицу.",
        ),
    ]

    for subject, predicate, expected in cases:
        ast = SimpleSentence(Clause(subject=subject, predicate=predicate))
        rendered = realizer.render_sentence(ast)
        assert rendered == expected
        assert not validate_ast_sentence(ast, rendered)


def test_surface_semantic_guard_allows_frame_approved_content_and_location_forms() -> None:
    allowed = [
        "Отчёт показывал данные.",
        "Документ точно включал факт.",
        "Документ описывал причину.",
        "Документ лежал на странице.",
        "Здание стояло у дороги.",
    ]

    for text in allowed:
        assert not reject_semantic_nonsense(text), text


def test_inanimate_masculine_object_adjective_uses_nominative_like_accusative() -> None:
    lexicon = Lexicon.default()
    realizer = Realizer(lexicon, MorphologyEngine())
    ast = SimpleSentence(
        Clause(
            subject=_np(lexicon, "документ"),
            predicate=VerbPhrase(
                verb_lemma="включать",
                object_np=replace(_np(lexicon, "факт"), case="accs", adjective_lemmas=("чистый",)),
                frame_id="include_requirement",
            ),
        )
    )

    assert realizer.render_sentence(ast) == "Документ включал чистый факт."


def test_inanimate_masculine_object_noun_uses_curated_accusative_with_pymorphy() -> None:
    lexicon = Lexicon.default()
    realizer = Realizer(lexicon, MorphologyEngine())
    ast = SimpleSentence(
        Clause(
            subject=_np(lexicon, "больница"),
            predicate=VerbPhrase(
                verb_lemma="отменить",
                object_np=replace(_np(lexicon, "вебинар"), case="accs", adjective_lemmas=("школьный",)),
                frame_id="cancel_event",
            ),
        )
    )

    rendered = realizer.render_sentence(ast)

    assert rendered == "Больница отменила школьный вебинар."
    assert not validate_ast_sentence(ast, rendered)


def test_present_plural_predicate_agrees_with_plural_subject() -> None:
    lexicon = Lexicon.default()
    realizer = Realizer(lexicon, MorphologyEngine(use_pymorphy=False))
    ast = SimpleSentence(
        Clause(
            subject=_np(lexicon, "данные"),
            predicate=VerbPhrase(
                verb_lemma="показывать",
                tense="present",
                object_np=replace(_np(lexicon, "факт"), case="accs"),
                frame_id="data_indicates",
            ),
        )
    )

    rendered = realizer.render_sentence(ast)

    assert rendered == "Данные показывают факт."
    assert validate_ast_sentence(ast, rendered) == []


def test_present_singular_predicate_with_plural_subject_is_rejected() -> None:
    lexicon = Lexicon.default()
    ast = SimpleSentence(
        Clause(
            subject=_np(lexicon, "данные"),
            predicate=VerbPhrase(
                verb_lemma="показывать",
                tense="present",
                object_np=replace(_np(lexicon, "факт"), case="accs"),
                frame_id="data_indicates",
            ),
        )
    )

    assert "subject_verb_number_agreement" in validate_ast_sentence(ast, "Данные показывает факт.")


def _generator(seed: int) -> tuple[GrammarBuilder, Realizer]:
    lexicon = Lexicon.default()
    morphology = MorphologyEngine()
    return GrammarBuilder(lexicon, morphology, RandomSource(seed=seed)), Realizer(lexicon, morphology)


def _np(lexicon: Lexicon, lemma: str, *, case: str = "nomn") -> NounPhrase:
    entry = next(noun for noun in lexicon.nouns if noun.lemma == lemma)
    number = "plur" if entry.gender == "plur" else "sing"
    return NounPhrase(
        noun_lemma=entry.lemma,
        gender=entry.gender,
        animacy=entry.animacy,
        number=number,
        case=case,
        semantic_class=entry.semantic_class,
    )


def _frame(lexicon: Lexicon, frame_id: str):
    return next(frame for frame in lexicon.frames.frames if frame.frame_id == frame_id)
