from src.candidates.candidate_generator import CandidateGenerator
from src.candidates.spelling_rules import spelling_candidates


SPELLING_CASES = [
    ("хочеться", "хочется"),
    ("жызнь", "жизнь"),
    ("чясто", "часто"),
    ("чюдо", "чудо"),
    ("длиный", "длинный"),
    ("привосходный", "превосходный"),
    ("зделать", "сделать"),
    ("подезд", "подъезд"),
    ("цыфра", "цифра"),
    ("шол", "шел"),
    ("карова", "корова"),
    ("лесница", "лестница"),
    ("граматика", "грамматика"),
    ("сдесь", "здесь"),
]


def test_candidate_generator_returns_keep_and_whitelist_split():
    generator = CandidateGenerator()
    candidates = generator.generate("Я незнаю что делать")

    values = {(candidate.source, candidate.replacement, candidate.edit_type) for candidate in candidates}

    assert ("незнаю", "незнаю", "keep") in values
    assert ("незнаю", "не знаю", "split_join") in values
    assert ("что", "что", "keep") in values


def test_candidate_generator_supports_hyphen_whitelist_pair():
    generator = CandidateGenerator()
    candidates = generator.generate("что то случилось")

    values = {(candidate.source, candidate.replacement, candidate.edit_type) for candidate in candidates}

    assert ("что то", "что-то", "hyphen") in values


def test_candidate_generator_only_sentence_start_case_candidate():
    generator = CandidateGenerator()
    candidates = generator.generate("сегодня хорошая погода")

    values = {(candidate.source, candidate.replacement, candidate.edit_type) for candidate in candidates}

    assert ("сегодня", "Сегодня", "case") in values
    assert ("хорошая", "Хорошая", "case") not in values


def test_spelling_rules_cover_required_orthogram_classes():
    for wrong, correct in SPELLING_CASES:
        assert correct in spelling_candidates(wrong)


def test_candidate_generator_emits_spelling_candidates_for_expanded_orthograms():
    generator = CandidateGenerator()
    text = " ".join(wrong for wrong, _correct in SPELLING_CASES)

    values = {(candidate.source.lower(), candidate.replacement.lower(), candidate.edit_type) for candidate in generator.generate(text)}

    for wrong, correct in SPELLING_CASES:
        assert (wrong, correct, "spelling") in values


def test_spelling_rules_do_not_restore_yo_or_random_typos():
    assert spelling_candidates("елка") == []
    assert spelling_candidates("молко") == []


def test_candidate_generator_emits_context_dependent_split_join_candidates_as_model_only():
    generator = CandidateGenerator()
    candidates = generator.generate("Также он сказал что бы мы остались, зато не смотря на дождь пришел.")

    values = {
        (candidate.source.lower(), candidate.replacement.lower(), candidate.edit_type, candidate.requires_model)
        for candidate in candidates
    }

    assert ("также", "так же", "split_join", True) in values
    assert ("что бы", "чтобы", "split_join", True) in values
    assert ("зато", "за то", "split_join", True) in values
    assert ("не смотря на", "несмотря на", "split_join", True) in values
