from src.data.synthetic_generator import SyntheticGenerator

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
]


def test_synthetic_generator_creates_allowed_error_example():
    generator = SyntheticGenerator(seed=7)

    example = generator.generate_from_clean("Я не знаю, что делать.")

    assert example.source != example.target
    assert example.target == "Я не знаю, что делать."
    assert set(example.error_types).issubset({"spelling", "punctuation", "split_join", "hyphen", "final_punctuation"})


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
    assert any("рубля прибавив" in variant.source for variant in variants)


def test_synthetic_generator_marks_expanded_orthogram_errors_as_spelling():
    generator = SyntheticGenerator(seed=7)

    variants = generator.generate_variants_from_clean("Жизнь прекрасна.", max_variants=8)

    assert any("жызнь" in variant.source.lower() and "spelling" in variant.error_types for variant in variants)


def test_synthetic_generator_covers_expanded_orthogram_classes():
    generator = SyntheticGenerator(seed=7)
    target = " ".join(correct for correct, _wrong in EXPANDED_ORTHOGRAM_CASES) + "."

    variants = generator.generate_variants_from_clean(target, max_variants=40)
    sources = "\n".join(variant.source.lower() for variant in variants)

    for _correct, wrong in EXPANDED_ORTHOGRAM_CASES:
        assert wrong in sources


def test_synthetic_generator_creates_diverse_punctuation_variants():
    generator = SyntheticGenerator(seed=7)
    target = 'Он сказал: «Привет», и добавил: (это важно).'

    variants = generator.generate_variants_from_clean(target, max_variants=30)
    sources = [variant.source for variant in variants]

    assert any("сказал, «Привет»" in source for source in sources)
    assert any("сказал «Привет»" in source for source in sources)
    assert any('"Привет"' in source for source in sources)
    assert any("(это важно)" not in source and "это важно" in source for source in sources)
    assert any("Привет»,, и" in source for source in sources)
    assert all(set(variant.error_types).issubset({"punctuation", "final_punctuation"}) for variant in variants)


def test_synthetic_generator_limits_errors_per_example():
    generator = SyntheticGenerator(seed=3, max_errors_per_sentence=3)

    example = generator.generate_from_clean('Он сказал: «Привет», и добавил: (это важно).')

    assert 1 <= len(example.error_types) <= 3
