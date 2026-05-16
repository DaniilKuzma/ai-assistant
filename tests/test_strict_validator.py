from src.validation.strict_validator import StrictValidator


def test_validator_accepts_allowed_spelling_and_punctuation_edits():
    validator = StrictValidator()

    result = validator.validate("Я незнаю что делать", "Я не знаю, что делать.")

    assert [edit.status for edit in result.edits] == ["accepted", "accepted", "accepted"]
    assert result.apply_accepted() == "Я не знаю, что делать."


def test_validator_rejects_semantic_rewrite_but_keeps_allowed_edits():
    validator = StrictValidator()

    result = validator.validate("Я люблю этот дом", "Я обожаю этот дом.")

    rejected = [edit for edit in result.edits if edit.status == "rejected"]
    assert rejected
    assert result.apply_accepted() == "Я люблю этот дом."


def test_prevalidator_protects_url_email_and_numbers():
    validator = StrictValidator()

    spans = validator.pre_validate("Напиши на test@example.com и открой https://example.com 12.05.2026")

    protected_texts = {span.text for span in spans}
    assert "test@example.com" in protected_texts
    assert "https://example.com" in protected_texts
    assert "12.05.2026" in protected_texts


def test_validator_rejects_punctuation_insert_inside_protected_number_but_keeps_final_punctuation():
    validator = StrictValidator()

    result = validator.validate("Евро стоил 4016 руб", "Евро стоил 40,16 руб.")

    comma_edits = [edit for edit in result.edits if edit.edit_type == "punctuation_insert"]
    assert comma_edits
    assert comma_edits[0].status == "rejected"
    assert result.apply_accepted() == "Евро стоил 4016 руб."


def test_validator_applies_positioned_punctuation_when_unknown_word_change_is_rejected():
    validator = StrictValidator()

    result = validator.validate("Он сказал привет", "Он сказал: пока.")

    assert any(edit.edit_type == "unknown" and edit.status == "rejected" for edit in result.edits)
    assert any(edit.edit_type == "punctuation_insert" and edit.status == "accepted" for edit in result.edits)
    assert result.apply_accepted() == "Он сказал: привет."
