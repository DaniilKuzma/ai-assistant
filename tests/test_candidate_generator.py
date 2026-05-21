from types import SimpleNamespace

from src.candidates.candidate_generator import CandidateGenerator
from src.candidates.spelling_rules import spelling_candidates
from src.inference.corrector import Corrector


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
    assert ("незнаю", "не знаю", "split_join", "model_required", True) in values
    assert ("что", "что", "keep", "deterministic", False) in values


def test_candidate_generator_emits_ne_verb_as_model_required_with_metadata():
    candidates = CandidateGenerator().generate("Он незнал ответа.")

    candidate = next(
        item
        for item in candidates
        if item.source.lower() == "незнал" and item.replacement.lower() == "не знал"
    )

    assert candidate.rule_id == "ne_verb"
    assert candidate.group == "ne"
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires_scoring is True
    assert candidate.requires == ("morphology", "syntax", "model")
    assert candidate.confidence == 0.35


def test_all_registry_rules_have_valid_modes():
    from src.rules.registry import all_rules

    modes = {rule.spec.id: getattr(rule.spec, "mode", "") for rule in all_rules()}

    assert modes
    assert set(modes.values()) <= VALID_RULE_MODES
    assert all(modes.values())


def test_candidate_generator_emits_frequent_error_exact_candidate_metadata():
    generator = CandidateGenerator()
    candidates = generator.generate("Жызнь прекрасна.")

    candidate = next(
        item for item in candidates if item.source == "Жызнь" and item.replacement == "Жизнь"
    )

    assert candidate.rule_id
    assert candidate.rule_id == "frequent_error_exact"
    assert candidate.mode == "candidate_only"
    assert candidate.requires_model is True
    assert candidate.requires_scoring is True


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

    values = {(candidate.source, candidate.replacement, candidate.edit_type, candidate.rule_id, candidate.mode) for candidate in candidates}

    assert ("сегодня", "Сегодня", "case", "capitalization_sentence_start", "candidate_only") in values
    assert not any(candidate.source == "хорошая" and candidate.edit_type == "case" for candidate in candidates)


def test_candidate_generator_does_not_capitalize_inside_numeric_hyphenated_token():
    source = "69-летний Прокопович..."
    candidates = CandidateGenerator().generate(source)

    assert not [
        candidate
        for candidate in candidates
        if candidate.rule_id == "capitalization_sentence_start" or candidate.edit_type == "case"
    ]
    assert Corrector().correct(source).corrected_text == source


def test_candidate_generator_keeps_existing_all_caps_abbreviations_unchanged():
    candidates = CandidateGenerator().generate("США и НББ согласовали документ.")

    assert not [
        candidate
        for candidate in candidates
        if candidate.source in {"США", "НББ"} and candidate.replacement in {"Сша", "Нбб"}
    ]
    assert Corrector().correct("США и НББ согласовали документ.").corrected_text == "США и НББ согласовали документ."


def test_candidate_generator_protects_graphical_abbreviation_before_city():
    source = "г. Москва готовит отчет."
    candidates = CandidateGenerator().generate(source)

    assert not [
        candidate
        for candidate in candidates
        if candidate.start < source.index("Москва") and candidate.edit_type != "keep"
    ]
    assert Corrector().correct(source).corrected_text == source


def test_candidate_generator_emits_lowercase_abbreviation_case_candidate_as_model_required():
    candidates = CandidateGenerator().generate("сша и нбб открыли офис.")
    values = {
        (
            candidate.source,
            candidate.replacement,
            candidate.rule_id,
            candidate.mode,
            candidate.requires_model,
            candidate.requires_scoring,
        )
        for candidate in candidates
    }

    assert ("сша", "США", "abbreviation_case_protection", "model_required", True, True) in values
    assert ("нбб", "НББ", "abbreviation_case_protection", "model_required", True, True) in values
    assert not any(candidate.replacement in {"Сша", "Нбб"} for candidate in candidates)


def test_candidate_generator_emits_lowercase_ner_candidate_as_model_required():
    def syntax_provider(_text):
        return [
            SimpleNamespace(text="иван", start=0, end=4, ner="PER", pos="PROPN"),
            SimpleNamespace(text="приехал", start=5, end=12, ner=None, pos="VERB"),
        ]

    candidates = CandidateGenerator(syntax_provider=syntax_provider).generate("иван приехал.")

    candidate = next(item for item in candidates if item.rule_id == "capitalization_ner")
    assert candidate.source == "иван"
    assert candidate.replacement == "Иван"
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires_scoring is True
    assert candidate.requires == ("ner", "syntax", "model")


def test_candidate_generator_requires_propn_for_ner_capitalization_candidate():
    def syntax_provider(_text):
        return [
            SimpleNamespace(text="налоговой", start=18, end=27, ner="ORG", pos="ADJF"),
            SimpleNamespace(text="службы", start=28, end=34, ner="ORG", pos="NOUN"),
        ]

    text = "глава федеральной налоговой службы России"
    candidates = CandidateGenerator(syntax_provider=syntax_provider).generate(text)

    assert not any(candidate.rule_id == "capitalization_ner" for candidate in candidates)


def test_candidate_generator_does_not_treat_single_letter_initial_as_conjunction():
    source = 'Это канал прямой коммуникации", - сказал А.Белоусов.'
    candidates = CandidateGenerator().generate(source)

    assert not [
        candidate
        for candidate in candidates
        if candidate.rule_id == "comma_conjunction" and candidate.start == source.index("А.Белоусов")
    ]


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


def test_candidate_generator_emits_context_dependent_split_join_candidates_as_model_required():
    generator = CandidateGenerator()
    candidates = []
    for text in [
        "Также он сделал так же, как я.",
        "Я тоже видел то же самое.",
        "Чтобы уйти, что бы ты ни сказал.",
        "Зато он ответил за то решение.",
        "Вследствие ошибки началось в следствие по делу.",
        "Несмотря на дождь, он шел не смотря на экран.",
    ]:
        candidates.extend(generator.generate(text))

    values = {
        (
            candidate.source.lower(),
            candidate.replacement.lower(),
            candidate.edit_type,
            candidate.rule_id,
            candidate.group,
            candidate.mode,
            candidate.requires_model,
            candidate.requires_scoring,
            candidate.requires,
        )
        for candidate in candidates
    }

    expected_requires = ("syntax", "morphology", "model")
    assert ("также", "так же", "split_join", "context_tak_zhe", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("так же", "также", "split_join", "context_tak_zhe", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("тоже", "то же", "split_join", "context_to_zhe", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("то же", "тоже", "split_join", "context_to_zhe", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("чтобы", "что бы", "split_join", "context_chto_by", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("что бы", "чтобы", "split_join", "context_chto_by", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("зато", "за то", "split_join", "context_za_to", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("за то", "зато", "split_join", "context_za_to", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("вследствие", "в следствие", "split_join", "context_vsledstvie", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("в следствие", "вследствие", "split_join", "context_vsledstvie", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("несмотря на", "не смотря на", "split_join", "context_nesmotrya", "context_split_join", "model_required", True, True, expected_requires) in values
    assert ("не смотря на", "несмотря на", "split_join", "context_nesmotrya", "context_split_join", "model_required", True, True, expected_requires) in values


def test_candidate_generator_emits_ne_ni_n_nn_and_prefix_candidates_as_model_required():
    candidates = CandidateGenerator().generate(
        "Некрасивый ответ, не красивый жест, непрочитанный текст, не долго, "
        "не разу, ни разу, длинный длиный, раненый жаренный прочитанн, привосходный."
    )
    values = {
        (
            candidate.source.lower(),
            candidate.replacement.lower(),
            candidate.rule_id,
            candidate.group,
            candidate.mode,
            candidate.requires_model,
            candidate.requires,
        )
        for candidate in candidates
        if candidate.edit_type != "keep"
    }

    ne_requires = ("morphology", "syntax", "model")
    n_nn_requires = ("morphology", "syntax", "dictionary", "model")
    assert ("некрасивый", "не красивый", "ne_adjective", "ne", "model_required", True, ne_requires) in values
    assert ("не красивый", "некрасивый", "ne_adjective", "ne", "model_required", True, ne_requires) in values
    assert ("непрочитанный", "не прочитанный", "ne_participle", "ne", "model_required", True, ne_requires) in values
    assert ("не долго", "недолго", "ne_adverb", "ne", "model_required", True, ne_requires) in values
    assert ("не разу", "ни разу", "ni_stable_expression", "ne_ni", "model_required", True, ne_requires) in values
    assert ("ни разу", "не разу", "ni_stable_expression", "ne_ni", "model_required", True, ne_requires) in values
    assert ("длиный", "длинный", "n_nn_adjective", "n_nn", "model_required", True, n_nn_requires) in values
    assert ("раненый", "раненный", "n_nn_participle", "n_nn", "model_required", True, n_nn_requires) in values
    assert ("жаренный", "жареный", "n_nn_deverbal_adjective", "n_nn", "model_required", True, n_nn_requires) in values
    assert ("прочитанн", "прочитан", "n_nn_short_form", "n_nn", "model_required", True, n_nn_requires) in values
    assert ("привосходный", "превосходный", "prefix_pre_pri", "prefix_pre_pri", "model_required", True, ("dictionary", "model")) in values


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

    tsya_candidate = next(item for item in candidates if item.rule_id == "tsya_soft_insert")
    assert tsya_candidate.group == "tsya"
    assert tsya_candidate.requires == ("morphology", "syntax", "model")


def test_candidate_generator_emits_clean_finite_tsya_candidate_as_model_required():
    candidates = CandidateGenerator().generate("Он учится каждый день.")

    candidate = next(
        item
        for item in candidates
        if item.source.lower() == "учится" and item.replacement.lower() == "учиться"
    )

    assert candidate.rule_id == "tsya_soft_insert"
    assert candidate.mode == "model_required"
    assert candidate.requires_model is True
    assert candidate.requires_scoring is True


def test_candidate_generator_emits_injected_dictionary_candidate_metadata():
    generator = CandidateGenerator(dictionary_lexicon=["библиотека"], dictionary_min_score=85)
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


def test_candidate_generator_emits_specific_dictionary_family_rule_ids():
    generator = CandidateGenerator(dictionary_lexicon=["корова", "территория"], dictionary_min_score=85)
    candidates = generator.generate("Клрова корвоа територия.")
    values = {
        (candidate.source.lower(), candidate.replacement.lower(), candidate.rule_id, candidate.mode, candidate.requires_model)
        for candidate in candidates
    }

    assert ("клрова", "корова", "keyboard_typo_candidate", "model_required", True) in values
    assert ("корвоа", "корова", "swapped_letters_candidate", "model_required", True) in values
    assert ("територия", "территория", "double_consonant_candidate", "model_required", True) in values


def test_candidate_generator_does_not_duplicate_frequent_error_exact_fallback():
    generator = CandidateGenerator(dictionary_lexicon=["территория"], dictionary_min_score=85)
    candidates = generator.generate("Територия большая.")
    spelling_candidates = [
        candidate
        for candidate in candidates
        if candidate.source.lower() == "територия" and candidate.replacement.lower() == "территория"
    ]

    assert len(spelling_candidates) == 1
    assert spelling_candidates[0].rule_id == "double_consonant_candidate"


def test_candidate_generator_without_dictionary_lexicon_disables_dictionary_fuzzy():
    candidates = CandidateGenerator().generate("Библеотека открыта.")

    assert not any(candidate.rule_id == "dictionary_fuzzy" for candidate in candidates)


def test_candidate_generator_from_config_uses_dictionary_provider(tmp_path):
    lexicon_path = tmp_path / "lexicon.txt"
    lexicon_path.write_text("\nБИБЛИОТЕКА\nмолоко\nбиблиотека\nтерритория\n", encoding="utf-8")
    generator = CandidateGenerator.from_config(
        {
            "dictionary": {
                "enabled": True,
                "lexicon_path": str(lexicon_path),
                "max_candidates": 2,
                "min_score": 85,
            }
        }
    )

    candidates = generator.generate("Библеотека открыта.")
    values = {(candidate.source, candidate.replacement, candidate.rule_id) for candidate in candidates}

    assert ("Библеотека", "Библиотека", "dictionary_fuzzy") in values


def test_candidate_generator_disables_yo_e_candidates_by_default():
    generator = CandidateGenerator(dictionary_lexicon=["елка", "ёлка"], dictionary_min_score=85)
    candidates = generator.generate("Елка стоит.")

    assert not any(candidate.rule_id == "yo_e_candidate" for candidate in candidates)


def test_candidate_generator_from_config_emits_opt_in_yo_e_candidate(tmp_path):
    lexicon_path = tmp_path / "lexicon.txt"
    lexicon_path.write_text("елка\nёлка\n", encoding="utf-8")
    generator = CandidateGenerator.from_config(
        {
            "dictionary": {
                "enabled": True,
                "lexicon_path": str(lexicon_path),
                "max_candidates": 2,
                "min_score": 85,
                "yo_e": {
                    "enabled": True,
                    "mode": "model_required",
                    "require_lexicon_support": True,
                    "require_model_scoring": True,
                },
            }
        }
    )
    candidates = generator.generate("Елка стоит.")
    values = {
        (
            candidate.source,
            candidate.replacement,
            candidate.rule_id,
            candidate.mode,
            candidate.requires_model,
            candidate.requires_scoring,
        )
        for candidate in candidates
    }

    assert ("Елка", "Ёлка", "yo_e_candidate", "model_required", True, True) in values


def test_candidate_generator_skips_dictionary_candidates_for_known_words_and_protected_spans():
    generator = CandidateGenerator(
        dictionary_lexicon=["корова", "пример", "елка", "ёлка"],
        dictionary_min_score=85,
        dictionary_yo_e_enabled=True,
    )
    candidates = generator.generate("корова test@example.com https://example.com 12345")

    assert not any(candidate.rule_id == "dictionary_fuzzy" for candidate in candidates)
    assert not any(candidate.rule_id == "yo_e_candidate" for candidate in candidates)


def test_candidate_generator_skips_known_word_before_dictionary_choices(monkeypatch):
    generator = CandidateGenerator(dictionary_lexicon=["корова"], dictionary_min_score=85)

    def fail_choices(_token, _lexicon):
        raise AssertionError("known words should skip dictionary choices")

    monkeypatch.setattr(generator, "_dictionary_choices_for_token", fail_choices)

    candidates = generator.generate("корова")

    assert not any(candidate.rule_id == "dictionary_fuzzy" for candidate in candidates)


def test_candidate_generator_reports_dictionary_token_cache_stats():
    generator = CandidateGenerator(
        dictionary_lexicon=["библиотека"],
        dictionary_min_score=85,
        dictionary_token_cache_enabled=True,
        dictionary_token_cache_max_size=10,
    )

    generator.generate("Библеотека открыта.")
    first_stats = generator.dictionary_cache_stats()
    generator.generate("Библеотека открыта.")
    second_stats = generator.dictionary_cache_stats()

    assert first_stats["misses"] > 0
    assert second_stats["hits"] > first_stats["hits"]
    assert 0.0 <= second_stats["hit_rate"] <= 1.0
