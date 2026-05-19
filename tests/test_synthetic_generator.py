import pytest

from src.candidates.candidate_generator import CandidateGenerator
from src.data.synthetic_generator import SyntheticGenerator
from src.rules.registry import rule_by_id
from src.rules.synthetic import synthetic_transformations

EXPANDED_ORTHOGRAM_CASES = [
    ("хочется", "хочеться"),
    ("жизнь", "жызнь"),
    ("часто", "чясто"),
    ("чудо", "чюдо"),
    ("длинный", "длиный"),
    ("превосходный", "привосходный"),
    ("сделать", "зделать"),
    ("подъезд", "подезд"),
    ("цифра", "цыфра"),
    ("шел", "шол"),
    ("корова", "карова"),
    ("лестница", "лесница"),
    ("грамматика", "граматика"),
    ("пришел", "пришол"),
    ("нашел", "нашол"),
    ("произошел", "произошол"),
    ("подъезде", "подезде"),
    ("подъезду", "подезду"),
    ("подъездом", "подездом"),
]


def test_synthetic_generator_creates_allowed_error_example():
    generator = SyntheticGenerator(seed=7)

    example = generator.generate_from_clean("Я не знаю, что делать.")

    assert example.source != example.target
    assert example.target == "Я не знаю, что делать."
    assert set(example.error_types).issubset({"spelling", "punctuation", "split_join", "hyphen", "final_punctuation"})


def test_synthetic_transformations_carry_rule_ids():
    generator = SyntheticGenerator(seed=7)

    transformations = generator._available_transformations("Я не знаю, что делать.")

    assert transformations
    assert all(transformation.rule_id for transformation in transformations)


def test_ne_verb_rule_generates_corruption_and_candidate_with_same_rule_id():
    rule = rule_by_id("ne_verb")

    corruptions = list(rule.generate_corruptions("Я не думаю об этом."))
    corruption_values = {(item.apply("Я не думаю об этом."), item.rule_id) for item in corruptions}
    candidates = list(rule.generate_candidates("недумаю"))

    assert ("Я недумаю об этом.", "ne_verb") in corruption_values
    assert any(candidate.replacement == "не думаю" and candidate.rule_id == "ne_verb" for candidate in candidates)


def test_cha_shcha_rule_generates_corruption_and_candidate_with_same_rule_id():
    rule = rule_by_id("pattern_чя_ча")

    corruptions = list(rule.generate_corruptions("Я вижу чащу."))
    corruption_values = {(item.apply("Я вижу чащу."), item.rule_id) for item in corruptions}
    candidates = list(rule.generate_candidates("чящу"))

    assert ("Я вижу чящу.", "pattern_чя_ча") in corruption_values
    assert any(candidate.replacement == "чащу" and candidate.rule_id == "pattern_чя_ча" for candidate in candidates)


def test_hyphen_particle_rule_generates_corruption_and_candidate_with_same_rule_id():
    rule = rule_by_id("hyphen_particles")
    target = "Кто-то пришёл."

    corruptions = list(rule.generate_corruptions(target))
    corruption_values = {(item.apply(target), item.rule_id) for item in corruptions}
    candidates = list(rule.generate_span("Кто то пришёл.", tuple(), 0))

    assert ("Кто то пришёл.", "hyphen_particles") in corruption_values
    assert any(candidate.replacement.lower() == "кто-то" and candidate.rule_id == "hyphen_particles" for candidate in candidates)


def test_koe_koy_rule_generates_corruption_and_candidate_with_same_rule_id():
    rule = rule_by_id("hyphen_koe_koy")
    target = "Кое-где это указано."

    corruptions = list(rule.generate_corruptions(target))
    corruption_values = {(item.apply(target), item.rule_id) for item in corruptions}
    candidates = list(rule.generate_span("Кое где это указано.", tuple(), 0))

    assert ("Кое где это указано.", "hyphen_koe_koy") in corruption_values
    assert any(candidate.replacement.lower() == "кое-где" and candidate.rule_id == "hyphen_koe_koy" for candidate in candidates)


def test_po_adverb_rule_generates_corruption_and_candidate_with_same_rule_id():
    rule = rule_by_id("hyphen_po_adverbs")
    target = "Он говорит по-русски."

    corruptions = list(rule.generate_corruptions(target))
    corruption_values = {(item.apply(target), item.rule_id) for item in corruptions}
    candidates = list(rule.generate_span("Он говорит по русски.", tuple(), 2))

    assert ("Он говорит по русски.", "hyphen_po_adverbs") in corruption_values
    assert any(candidate.replacement.lower() == "по-русски" and candidate.rule_id == "hyphen_po_adverbs" for candidate in candidates)


def test_pol_polu_rule_generates_corruption_and_candidate_with_same_rule_id():
    rule = rule_by_id("pol_polu_compounds")
    target = "Пол-лимона и полуфинал готовы."

    corruptions = list(rule.generate_corruptions(target))
    corruption_values = {(item.apply(target), item.rule_id) for item in corruptions}
    hyphen_candidates = list(rule.generate_span("Пол лимона готов.", tuple(), 0))
    joined_candidates = list(rule.generate_span("Ждали полу финал.", tuple(), 1))

    assert ("Пол лимона и полуфинал готовы.", "pol_polu_compounds") in corruption_values
    assert ("Пол-лимона и полу финал готовы.", "pol_polu_compounds") in corruption_values
    assert any(candidate.replacement.lower() == "пол-лимона" and candidate.rule_id == "pol_polu_compounds" for candidate in hyphen_candidates)
    assert any(candidate.replacement.lower() == "полуфинал" and candidate.rule_id == "pol_polu_compounds" for candidate in joined_candidates)


@pytest.mark.parametrize(
    ("rule_id", "target", "expected_source_fragment", "expected_replacement"),
    [
        ("ne_adjective", "Некрасивый ответ удивил всех.", "Не красивый", "некрасивый"),
        ("ne_participle", "Непрочитанный текст лежал на столе.", "Не прочитанный", "непрочитанный"),
        ("ne_adverb", "Недолго музыка играла.", "Не долго", "недолго"),
        ("ni_stable_expression", "Ни разу не ошибся.", "Не разу", "ни разу"),
        ("n_nn_adjective", "Длинный путь занял день.", "Длиный", "длинный"),
        ("n_nn_participle", "Раненный солдат вернулся.", "Раненый", "раненный"),
        ("n_nn_deverbal_adjective", "Жареный картофель остыл.", "Жаренный", "жареный"),
        ("n_nn_short_form", "Ответ прочитан.", "прочитанн", "прочитан"),
        ("prefix_pre_pri", "Превосходный результат удивил всех.", "Привосходный", "превосходный"),
        ("context_tak_zhe", "Он сделал так же, как я.", "также", "так же"),
        ("context_to_zhe", "Он сделал то же упражнение.", "тоже", "то же"),
        ("context_chto_by", "Что бы ты ни сказал, решение принято.", "Чтобы", "что бы"),
        ("context_za_to", "Он отвечает за то решение.", "зато", "за то"),
        ("context_vsledstvie", "В следствие по делу добавили документ.", "Вследствие", "в следствие"),
        ("context_nesmotrya", "Он шел не смотря на экран.", "несмотря", "не смотря"),
    ],
)
def test_model_required_orthography_rules_generate_corruption_and_candidate_with_same_rule_id(
    rule_id,
    target,
    expected_source_fragment,
    expected_replacement,
):
    rule = rule_by_id(rule_id)

    corruptions = list(rule.generate_corruptions(target))
    dirty_sources = {(item.apply(target), item.rule_id) for item in corruptions}
    candidates = CandidateGenerator().generate(next(source for source, candidate_rule_id in dirty_sources if candidate_rule_id == rule_id))

    assert any(expected_source_fragment.lower() in source.lower() and candidate_rule_id == rule_id for source, candidate_rule_id in dirty_sources)
    assert any(
        candidate.replacement.lower() == expected_replacement and candidate.rule_id == rule_id
        for candidate in candidates
    )


def test_dictionary_fuzzy_synthetic_corruption_can_be_repaired_by_dictionary_candidate_with_same_rule_id():
    target = "Библиотека открыта."

    corruptions = [item for item in synthetic_transformations(target) if item.rule_id == "dictionary_fuzzy"]
    dirty = corruptions[0].apply(target)
    candidates = CandidateGenerator(dictionary_lexicon=["библиотека"], dictionary_min_score=85).generate(dirty)

    assert "Библеотека" in dirty
    assert any(candidate.replacement == "Библиотека" and candidate.rule_id == "dictionary_fuzzy" for candidate in candidates)


@pytest.mark.parametrize(
    ("target", "expected_group", "expected_rule_id"),
    [
        ("Коллеги, проверьте текст.", "address_comma", "address_comma"),
        ("Мы купили и чай, и кофе.", "homogeneous_members", "homogeneous_comma"),
        ("Закончив работу, мы ушли.", "detached_members", "detached_adverbial_comma"),
        ("Москва — это столица.", "subject_predicate_dash", "subject_predicate_dash"),
        ("Он сказал: «Проект готов».", "direct_speech", "direct_speech_colon"),
    ],
)
def test_synthetic_transformations_cover_requested_punctuation_groups(target, expected_group, expected_rule_id):
    transformations = synthetic_transformations(target)

    assert any(item.group == expected_group for item in transformations)
    assert any(item.rule_id == expected_rule_id for item in transformations)


def test_synthetic_generator_can_create_identity_examples():
    generator = SyntheticGenerator(seed=7)

    examples = generator.add_identity_examples(["Чистый текст.", "Еще один текст."], source_dataset="unit")

    assert all(example.source == example.target for example in examples)
    assert all(example.is_clean for example in examples)


def test_synthetic_generator_does_not_remove_decimal_commas():
    generator = SyntheticGenerator(seed=7)
    target = "Евро стоил 40,16 рубля, прибавив 4,25 копейки."

    variants = generator.generate_variants_from_clean(target, max_variants=8)

    assert variants
    assert all("40,16" in variant.source for variant in variants)
    assert all("4,25" in variant.source for variant in variants)
    assert all("рубля прибавив" not in variant.source for variant in variants)


def test_synthetic_generator_marks_expanded_orthogram_errors_as_spelling():
    generator = SyntheticGenerator(seed=7)

    variants = generator.generate_variants_from_clean("Жизнь прекрасна.", max_variants=8)

    assert any("жызнь" in variant.source.lower() and "spelling" in variant.error_types for variant in variants)


def test_synthetic_generator_covers_expanded_orthogram_classes():
    generator = SyntheticGenerator(seed=7)
    target = " ".join(correct for correct, _wrong in EXPANDED_ORTHOGRAM_CASES) + "."

    variants = generator.generate_variants_from_clean(target, max_variants=80)
    sources = "\n".join(variant.source.lower() for variant in variants)

    for _correct, wrong in EXPANDED_ORTHOGRAM_CASES:
        assert wrong in sources


def test_synthetic_generator_creates_diverse_punctuation_variants():
    generator = SyntheticGenerator(seed=7)
    target = 'Он сказал: «Привет», и добавил: (это важно).'

    variants = generator.generate_variants_from_clean(target, max_variants=30)
    sources = [variant.source for variant in variants]

    assert any("сказал «Привет»" in source for source in sources)
    assert not any("сказал, «Привет»" in source for source in sources)
    assert not any('"Привет"' in source for source in sources)
    assert not any("(это важно)" not in source and "это важно" in source for source in sources)
    assert not any("Привет»,, и" in source for source in sources)
    assert any(set(variant.error_types).issubset({"punctuation", "final_punctuation"}) for variant in variants)


def test_synthetic_generator_limits_errors_per_example():
    generator = SyntheticGenerator(seed=3, max_errors_per_sentence=3)

    example = generator.generate_from_clean('Он сказал: «Привет», и добавил: (это важно).')

    assert 1 <= len(example.error_types) <= 3
