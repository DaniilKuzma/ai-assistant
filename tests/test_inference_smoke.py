from src.inference.corrector import Corrector


def test_plain_corrector_does_not_apply_candidate_only_split_join_without_scorer():
    corrector = Corrector()

    result = corrector.correct("Я незнаю что делать.")

    assert result.corrected_text == "Я незнаю что делать."
    assert not any(edit.source.lower() == "незнаю" and edit.replacement.lower() == "не знаю" for edit in result.edits)


def test_corrector_does_not_rewrite_semantics():
    corrector = Corrector()

    result = corrector.correct("Я люблю этот дом")

    assert result.corrected_text == "Я люблю этот дом."


def test_plain_corrector_does_not_apply_candidate_only_hyphen_or_model_punctuation_without_scorer():
    corrector = Corrector()

    result = corrector.correct("во первых это важно")

    assert result.corrected_text == "Во первых это важно."


def test_rule_backed_corrector_does_not_apply_context_dependent_pairs_blindly():
    corrector = Corrector()

    result = corrector.correct("Он также пришел")
    also_result = corrector.correct("Он тоже пришел")
    chtoby_result = corrector.correct("Он пришел чтобы помочь")

    assert result.corrected_text == "Он также пришел."
    assert also_result.corrected_text == "Он тоже пришел."
    assert chtoby_result.corrected_text == "Он пришел чтобы помочь."


def test_rule_backed_corrector_removes_obvious_extra_punctuation():
    corrector = Corrector()

    result = corrector.correct("Привет,, мир")

    assert result.corrected_text == "Привет, мир."


def test_plain_corrector_does_not_apply_model_required_comma_rule_without_scorer():
    corrector = Corrector()

    result = corrector.correct("Я думаю что это важно")

    assert result.corrected_text == "Я думаю что это важно."
    assert not any(edit.rule_id == "comma_subordinate" and edit.status == "accepted" for edit in result.edits)


def test_plain_corrector_does_not_apply_model_required_direct_speech_without_scorer():
    corrector = Corrector()

    result = corrector.correct('Он сказал "Привет"')

    assert result.corrected_text == 'Он сказал "Привет".'


def test_plain_corrector_does_not_apply_model_required_dash_without_scorer():
    corrector = Corrector()

    result = corrector.correct("Москва это столица")

    assert result.corrected_text == "Москва это столица."


def test_plain_corrector_does_not_apply_model_required_tsya_candidate_without_scorer():
    corrector = Corrector()
    source = "Они могут появится завтра."

    result = corrector.correct(source)

    assert result.corrected_text == source
    assert not any(edit.source.lower() == "появится" and edit.replacement.lower() == "появиться" for edit in result.edits)
