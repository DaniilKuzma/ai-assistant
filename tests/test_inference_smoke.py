from src.inference.corrector import Corrector


def test_corrector_applies_only_allowed_changes():
    corrector = Corrector()

    result = corrector.correct("Я незнаю что делать")

    assert result.corrected_text == "Я не знаю, что делать."
    assert {edit.status for edit in result.edits} == {"accepted"}


def test_corrector_does_not_rewrite_semantics():
    corrector = Corrector()

    result = corrector.correct("Я люблю этот дом")

    assert result.corrected_text == "Я люблю этот дом."


def test_corrector_adds_comma_after_simple_introductory_hyphen_word():
    corrector = Corrector()

    result = corrector.correct("во первых это важно")

    assert result.corrected_text == "Во-первых, это важно."


def test_rule_backed_corrector_does_not_apply_context_dependent_pairs_blindly():
    corrector = Corrector()

    result = corrector.correct("Он также пришел")

    assert result.corrected_text == "Он также пришел."


def test_rule_backed_corrector_removes_obvious_extra_punctuation():
    corrector = Corrector()

    result = corrector.correct("Привет,, мир")

    assert result.corrected_text == "Привет, мир."


def test_rule_backed_corrector_handles_simple_direct_speech_quotes():
    corrector = Corrector()

    result = corrector.correct('Он сказал "Привет"')

    assert result.corrected_text == "Он сказал: «Привет»."


def test_rule_backed_corrector_adds_obvious_subject_predicate_dash():
    corrector = Corrector()

    result = corrector.correct("Москва это столица")

    assert result.corrected_text == "Москва — это столица."
