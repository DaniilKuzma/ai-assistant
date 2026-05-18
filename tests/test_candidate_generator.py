from src.candidates.candidate_generator import CandidateGenerator
from src.candidates.spelling_rules import spelling_candidates


VALID_RULE_MODES = {"deterministic", "candidate_only", "model_required"}


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

INFLECTED_SPELLING_CASES = [
    ("пришол", "пришел"),
    ("нашол", "нашел"),
    ("произошол", "произошел"),
    ("подезде", "подъезде"),
    ("подезду", "подъезду"),
    ("подездом", "подъездом"),
]


def test_candidate_generator_returns_keep_and_whitelist_split():
    generator = CandidateGenerator()
    candidates = generator.generate("Я незнаю что делать")

    values = {
        (candidate.source, candidate.replacement, candidate.edit_type, candidate.mode, candidate.requires_scoring)
        for candidate in candidates
    }

    assert ("незнаю", "незнаю", "keep", "deterministic", False) in values
    assert ("незнаю", "не знаю", "split_join", "candidate_only", True) in values
    assert ("что", "что", "keep", "deterministic", False) in values


def test_all_registry_rules_have_valid_modes():
    from src.rules.registry import all_rules

    modes = {rule.spec.id: getattr(rule.spec, "mode", "") for rule in all_rules()}

    assert modes
    assert set(modes.values()) <= VALID_RULE_MODES
    assert all(modes.values())


def test_candidate_generator_emits_deterministic_candidate_metadata():
    generator = CandidateGenerator()
    candidates = generator.generate("Жызнь прекрасна.")

    candidate = next(
        item for item in candidates if item.source == "Жызнь" and item.replacement == "Жизнь"
    )

    assert candidate.rule_id
    assert candidate.mode == "deterministic"
    assert candidate.requires_model is False
    assert candidate.requires_scoring is False


def test_candidate_generator_treats_voobschem_as_split_join():
    generator = CandidateGenerator()
    candidates = generator.generate("Вообщем, это важный пример.")

    values = {(candidate.source, candidate.replacement, candidate.edit_type) for candidate in candidates}

    assert ("Вообщем", "В общем", "split_join") in values


def test_candidate_generator_supports_hyphen_whitelist_pair():
    generator = CandidateGenerator()
    candidates = generator.generate("что то случилось")

    values = {
        (candidate.source, candidate.replacement, candidate.edit_type, candidate.mode, candidate.requires_scoring)
        for candidate in candidates
    }

    assert ("что то", "что-то", "hyphen", "candidate_only", True) in values


def test_candidate_generator_emits_rule_backed_hyphen_candidates_with_metadata():
    generator = CandidateGenerator()
    cases = [
        ("Кто то пришёл.", "кто-то", "hyphen_particles"),
        ("Где либо это указано.", "где-либо", "hyphen_particles"),
        ("Когда нибудь мы закончим.", "когда-нибудь", "hyphen_particles"),
        ("Кое кто ошибся.", "кое-кто", "hyphen_koe_koy"),
        ("Он говорит по русски.", "по-русски", "hyphen_po_adverbs"),
    ]

    for text, expected_replacement, expected_rule_id in cases:
        candidates = generator.generate(text)
        candidate = next(
            item
            for item in candidates
            if item.replacement.lower() == expected_replacement and item.rule_id == expected_rule_id
        )

        assert candidate.edit_type == "hyphen"
        assert candidate.mode == "candidate_only"
        assert candidate.confidence == 0.95
        assert candidate.requires_model is True
        assert candidate.requires_scoring is True


def test_candidate_generator_emits_model_required_pol_polu_candidates():
    generator = CandidateGenerator()
    candidates = generator.generate("Он купил пол лимона, пол часа и ждал полу финал.")

    values = {
        (
            candidate.source.lower(),
            candidate.replacement.lower(),
            candidate.edit_type,
            candidate.rule_id,
            candidate.mode,
            candidate.requires_model,
            candidate.requires_scoring,
        )
        for candidate in candidates
    }

    assert ("пол лимона", "пол-лимона", "hyphen", "pol_polu_compounds", "model_required", True, True) in values
    assert ("пол часа", "полчаса", "hyphen", "pol_polu_compounds", "model_required", True, True) in values
    assert ("полу финал", "полуфинал", "hyphen", "pol_polu_compounds", "model_required", True, True) in values


def test_candidate_generator_does_not_emit_hyphen_repairs_for_correct_hyphenated_words():
    generator = CandidateGenerator()
    candidates = generator.generate("Кто-то пришёл, кое-где написано по-русски.")

    values = {
        (candidate.source.lower(), candidate.replacement.lower(), candidate.rule_id)
        for candidate in candidates
        if candidate.edit_type == "hyphen"
    }

    assert not values


def test_candidate_generator_only_sentence_start_case_candidate():
    generator = CandidateGenerator()
    candidates = generator.generate("сегодня хорошая погода")

    values = {(candidate.source, candidate.replacement, candidate.edit_type) for candidate in candidates}

    assert ("сегодня", "Сегодня", "case") in values
    assert ("хорошая", "Хорошая", "case") not in values


def test_spelling_rules_cover_required_orthogram_classes():
    for wrong, correct in SPELLING_CASES:
        assert correct in spelling_candidates(wrong)


def test_spelling_rules_cover_common_inflected_forms_from_synthetic_data():
    for wrong, correct in INFLECTED_SPELLING_CASES:
        assert correct in spelling_candidates(wrong)


def test_candidate_generator_emits_spelling_candidates_for_expanded_orthograms():
    generator = CandidateGenerator()
    cases = [*SPELLING_CASES, *INFLECTED_SPELLING_CASES]
    text = " ".join(wrong for wrong, _correct in cases)

    values = {(candidate.source.lower(), candidate.replacement.lower(), candidate.edit_type) for candidate in generator.generate(text)}

    for wrong, correct in cases:
        assert (wrong, correct, "spelling") in values


def test_spelling_rules_do_not_restore_yo_or_random_typos():
    assert spelling_candidates("елка") == []
    assert spelling_candidates("молко") == []


def test_candidate_generator_emits_context_dependent_split_join_candidates_as_model_only():
    generator = CandidateGenerator()
    candidates = generator.generate("Также он тоже сказал что бы мы остались, зато не смотря на дождь пришел.")

    values = {
        (
            candidate.source.lower(),
            candidate.replacement.lower(),
            candidate.edit_type,
            candidate.mode,
            candidate.requires_model,
            candidate.requires_scoring,
        )
        for candidate in candidates
    }

    assert ("также", "так же", "split_join", "candidate_only", True, True) in values
    assert ("тоже", "то же", "split_join", "candidate_only", True, True) in values
    assert ("что бы", "чтобы", "split_join", "candidate_only", True, True) in values
    assert ("зато", "за то", "split_join", "candidate_only", True, True) in values
    assert ("не смотря на", "несмотря на", "split_join", "candidate_only", True, True) in values


def test_candidate_budget_keeps_real_edits_before_keep_candidates():
    from src.candidates.candidate_generator import CandidateGenerator
    from src.candidates.candidate_ranking import rank_candidates_for_budget

    text = " ".join([f"слово{i}" for i in range(30)]) + " недумаю"
    candidates = CandidateGenerator().generate(text)
    ranked = rank_candidates_for_budget(candidates, 16)

    assert any(candidate.replacement.lower() == "не думаю" for candidate in ranked)


def test_generated_candidates_keep_rule_id():
    from src.candidates.candidate_generator import CandidateGenerator

    candidates = CandidateGenerator().generate("Я недумаю и вижу чящу.")
    pairs = {(candidate.replacement.lower(), candidate.rule_id) for candidate in candidates}

    assert ("не думаю", "ne_verb") in pairs
    assert any(replacement == "чащу" and rule.startswith("pattern_") for replacement, rule in pairs)


def test_candidate_generator_emits_tsya_candidates_as_model_required_with_rule_id():
    from src.rules.registry import rule_by_id

    candidates = CandidateGenerator().generate("Они могут появится и появиться завтра.")
    values = {
        (
            candidate.source.lower(),
            candidate.replacement.lower(),
            candidate.mode,
            candidate.requires_model,
            candidate.requires_scoring,
            candidate.rule_id,
        )
        for candidate in candidates
    }

    assert rule_by_id("tsya_soft_insert").spec.mode == "model_required"
    assert rule_by_id("tsya_soft_delete").spec.mode == "model_required"
    assert ("появится", "появиться", "model_required", True, True, "tsya_soft_insert") in values
    assert ("появиться", "появится", "model_required", True, True, "tsya_soft_delete") in values


def test_candidate_generator_emits_injected_dictionary_candidate_metadata():
    generator = CandidateGenerator(dictionary_lexicon=["библиотека"])
    candidates = generator.generate("Библеотека открыта.")

    values = {
        (
            candidate.source,
            candidate.replacement,
            candidate.edit_type,
            candidate.mode,
            candidate.requires_model,
            candidate.requires_scoring,
            candidate.rule_id,
        )
        for candidate in candidates
    }

    assert ("Библеотека", "Библиотека", "spelling", "model_required", True, True, "dictionary_fuzzy") in values


def test_candidate_generator_skips_dictionary_candidates_for_known_words_and_protected_spans():
    generator = CandidateGenerator(dictionary_lexicon=["корова", "пример"])
    candidates = generator.generate("корова test@example.com https://example.com 12345")

    assert not any(candidate.rule_id == "dictionary_fuzzy" for candidate in candidates)
