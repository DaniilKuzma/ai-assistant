from src.validation.diff_analyzer import DiffAnalyzer


def test_diff_analyzer_classifies_split_punctuation_and_final_punctuation():
    analyzer = DiffAnalyzer()

    edits = analyzer.analyze("Я незнаю что делать", "Я не знаю, что делать.")
    edit_types = [edit.edit_type for edit in edits]

    assert "split_word" in edit_types
    assert "punctuation_insert" in edit_types
    assert "final_punctuation" in edit_types


def test_diff_analyzer_classifies_hyphen_and_case():
    analyzer = DiffAnalyzer()

    edits = analyzer.analyze("сегодня что то произошло", "Сегодня что-то произошло.")
    edit_types = [edit.edit_type for edit in edits]
    rule_ids = {edit.rule_id for edit in edits}

    assert "case_change" in edit_types
    assert "hyphen_change" in edit_types
    assert "final_punctuation" in edit_types
    assert "hyphen_particles" in rule_ids


def test_diff_analyzer_rejects_case_change_after_number_or_latin_prefix():
    analyzer = DiffAnalyzer()

    date_edits = analyzer.analyze("14 августа будет встреча", "14 Августа будет встреча.")
    latin_edits = analyzer.analyze("G20 призывает к миру", "G20 Призывает к миру.")
    mid_sentence_edits = analyzer.analyze("Встреча 14 августа", "Встреча 14 Августа.")

    assert not any(edit.edit_type == "case_change" for edit in date_edits)
    assert any(edit.edit_type == "unknown" for edit in date_edits)
    assert not any(edit.edit_type == "case_change" for edit in latin_edits)
    assert any(edit.edit_type == "unknown" for edit in latin_edits)
    assert not any(edit.edit_type == "case_change" for edit in mid_sentence_edits)
    assert any(edit.edit_type == "unknown" for edit in mid_sentence_edits)


def test_diff_analyzer_rejects_unknown_replacement():
    analyzer = DiffAnalyzer()

    edits = analyzer.analyze("Я люблю дом", "Я обожаю дом")

    assert any(edit.edit_type == "unknown" for edit in edits)


def test_diff_analyzer_keeps_position_for_punctuation_insert_inside_number():
    analyzer = DiffAnalyzer()

    edits = analyzer.analyze("Евро стоил 4016 руб", "Евро стоил 40,16 руб.")

    comma_edits = [edit for edit in edits if edit.edit_type == "punctuation_insert" and edit.replacement == ","]
    assert comma_edits
    assert comma_edits[0].start == comma_edits[0].end
    assert 11 < comma_edits[0].start < 16


def test_diff_analyzer_keeps_rule_id_for_generated_orthography_edits():
    analyzer = DiffAnalyzer()

    edits = analyzer.analyze("Я недумаю и вижу чящу.", "Я не думаю и вижу чащу.")
    rule_ids = {(edit.replacement.lower(), edit.rule_id) for edit in edits}

    assert ("не думаю", "ne_verb") in rule_ids
    assert any(replacement == "чащу" and rule.startswith("pattern_") for replacement, rule in rule_ids)
