from __future__ import annotations

from src.nlp.syntax_analyzer import analyze_syntax
from src.nlp.syntax_features import SyntaxFeatureExtractor


def _features(text: str) -> tuple[object, SyntaxFeatureExtractor]:
    analysis = analyze_syntax(text)
    return analysis, SyntaxFeatureExtractor(analysis)


def test_subordinate_and_coordinate_conjunction_features():
    subordinate_text = "Проект готов, потому что команда закончила работу."
    coordinate_text = "Мы пришли, но встреча закончилась."

    subordinate_analysis, subordinate = _features(subordinate_text)
    coordinate_analysis, coordinate = _features(coordinate_text)

    subordinate_markers = subordinate.find_subordinate_conjunctions(subordinate_analysis)
    coordinate_markers = coordinate.find_coordinating_conjunctions(coordinate_analysis)

    assert any(marker["text"] in {"потому что", "что"} for marker in subordinate_markers)
    assert any(marker["text"] == "но" for marker in coordinate_markers)
    assert all(0 <= marker["start"] <= marker["end"] <= len(subordinate_text) for marker in subordinate_markers)
    assert all(0 <= marker["start"] <= marker["end"] <= len(coordinate_text) for marker in coordinate_markers)


def test_clause_features_return_bounded_spans():
    text = "Проект готов, потому что команда закончила работу."
    analysis, extractor = _features(text)

    clauses = extractor.find_clauses(analysis)

    assert clauses
    assert any(clause.kind in {"main", "subordinate", "unknown"} for clause in clauses)
    assert all(0 <= clause.start <= clause.end <= len(text) for clause in clauses)
    assert all(0.0 <= clause.confidence <= 1.0 for clause in clauses)


def test_adverbial_participle_and_participial_phrase_features():
    adverbial_text = "Проверив отчёт, редактор отправил письмо."
    participial_text = "Отчёт, подготовленный командой, отправили утром."

    adverbial_analysis, adverbial = _features(adverbial_text)
    participial_analysis, participial = _features(participial_text)

    adverbial_phrases = adverbial.find_adverbial_participle_phrases(adverbial_analysis)
    participial_phrases = participial.find_participial_phrases(participial_analysis)

    assert any(phrase.kind == "adverbial_participle" for phrase in adverbial_phrases)
    assert any("Проверив" in phrase.evidence for phrase in adverbial_phrases)
    assert any(phrase.kind == "participial" for phrase in participial_phrases)
    assert any("подготовленный" in phrase.evidence for phrase in participial_phrases)


def test_subject_predicate_dash_and_direct_speech_features():
    dash_text = "Москва — столица России."
    speech_text = "«Проект готов», — сказала Мария."

    dash_analysis, dash = _features(dash_text)
    speech_analysis, speech = _features(speech_text)

    dash_candidates = dash.find_subject_predicate_dash_candidates(dash_analysis)
    direct_speech = speech.find_direct_speech_spans(speech_analysis)

    assert dash_candidates
    assert any(candidate.kind == "subject_predicate" for candidate in dash_candidates)
    assert any(candidate.start <= dash_text.index("—") < candidate.end for candidate in dash_candidates)
    assert direct_speech
    assert any(candidate.kind == "direct_speech" for candidate in direct_speech)
    assert any("сказала" in candidate.evidence for candidate in direct_speech)


def test_quote_and_bracket_balance_features_and_safe_gaps():
    text = "Проверь «план» (черновик) завтра."
    analysis, extractor = _features(text)

    quote_spans = extractor.find_quote_spans(analysis)
    bracket_spans = extractor.find_bracket_spans(analysis)

    assert any(span.kind == "quote_span" and text[span.start] == "«" for span in quote_spans)
    assert any(span.kind == "bracket_span" and text[span.start] == "(" for span in bracket_spans)
    assert extractor.is_inside_protected_span(0, 0) is False
    assert extractor.is_safe_gap_for_punctuation(text, len("Проверь"), len("Проверь ")) is True
    assert extractor.is_safe_gap_for_punctuation(text, text.index("«"), text.index("»") + 1) is False
