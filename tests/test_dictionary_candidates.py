from src.candidates.dictionary_candidates import dictionary_candidates


def test_dictionary_candidates_returns_fuzzy_match_for_vowel_typo():
    candidates = dictionary_candidates("карова", ["корова", "молоко", "территория"])

    assert "корова" in candidates


def test_dictionary_candidates_returns_fuzzy_match_for_missing_double_consonant():
    candidates = dictionary_candidates("територия", ["территория"])

    assert "территория" in candidates


def test_dictionary_candidates_skips_known_correct_word():
    assert dictionary_candidates("корова", ["корова", "молоко"]) == []


def test_dictionary_candidates_skips_numbers():
    assert dictionary_candidates("12345", ["корова"]) == []


def test_dictionary_candidates_skips_email_and_url():
    assert dictionary_candidates("test@example.com", ["корова"]) == []
    assert dictionary_candidates("https://example.com", ["корова"]) == []
