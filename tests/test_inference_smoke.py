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
