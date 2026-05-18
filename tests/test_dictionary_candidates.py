from src.candidates.dictionary_candidates import dictionary_candidate_specs, dictionary_candidates


TEST_LEXICON = ["корова", "молоко", "территория", "апелляция", "группа"]


def test_dictionary_candidates_returns_fuzzy_match_for_vowel_typo():
    candidates = dictionary_candidates("карова", TEST_LEXICON, min_score=85)

    assert "корова" in candidates


def test_dictionary_candidates_returns_fuzzy_match_for_missing_double_consonant():
    candidates = dictionary_candidates("територия", TEST_LEXICON, min_score=85)

    assert "территория" in candidates


def test_dictionary_candidates_return_double_consonant_family_for_common_words():
    cases = [
        ("територия", "территория"),
        ("апеляция", "апелляция"),
        ("група", "группа"),
    ]

    for source, replacement in cases:
        specs = dictionary_candidate_specs(source, TEST_LEXICON, min_score=85)

        assert any(
            spec.replacement == replacement
            and spec.rule_id == "double_consonant_candidate"
            and spec.mode == "model_required"
            and spec.requires_model is True
            for spec in specs
        )


def test_dictionary_candidates_return_typo_family_metadata():
    cases = [
        ("млоко", "молоко", "missing_letter_candidate"),
        ("коорова", "корова", "extra_letter_candidate"),
        ("корвоа", "корова", "swapped_letters_candidate"),
        ("клрова", "корова", "keyboard_typo_candidate"),
    ]

    for source, replacement, rule_id in cases:
        specs = dictionary_candidate_specs(source, TEST_LEXICON, min_score=85)

        assert any(
            spec.replacement == replacement
            and spec.rule_id == rule_id
            and spec.mode == "model_required"
            and spec.requires_model is True
            for spec in specs
        )


def test_dictionary_candidates_respects_total_limit_across_families():
    candidates = dictionary_candidates("група", ["группа", "груша", "груда"], limit=1, min_score=50)

    assert candidates == ["группа"]


def test_dictionary_candidates_generate_yo_e_only_when_enabled_and_lexicon_supported():
    disabled = dictionary_candidate_specs("елка", ["елка", "ёлка"], min_score=85)
    enabled = dictionary_candidate_specs("елка", ["елка", "ёлка"], min_score=85, yo_e_enabled=True)
    missing_source = dictionary_candidate_specs("елка", ["ёлка"], min_score=85, yo_e_enabled=True)

    assert not any(spec.rule_id == "yo_e_candidate" for spec in disabled)
    assert not any(spec.rule_id == "yo_e_candidate" for spec in missing_source)
    assert any(
        spec.replacement == "ёлка"
        and spec.rule_id == "yo_e_candidate"
        and spec.mode == "model_required"
        and spec.requires_model is True
        for spec in enabled
    )


def test_dictionary_candidates_rejects_distant_override_candidate():
    assert dictionary_candidates("стол", ["самолёт"], min_score=85) == []
    assert dictionary_candidates("стлл", ["самолёт"], min_score=85) == []


def test_dictionary_candidates_skips_known_correct_word():
    assert dictionary_candidates("корова", TEST_LEXICON) == []


def test_dictionary_candidates_skips_numbers():
    assert dictionary_candidates("12345", ["корова"]) == []


def test_dictionary_candidates_skips_email_and_url():
    assert dictionary_candidates("test@example.com", ["корова"]) == []
    assert dictionary_candidates("https://example.com", ["корова"]) == []
