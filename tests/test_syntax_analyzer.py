from __future__ import annotations

import builtins

from src.nlp.syntax_analyzer import SyntaxAnalyzer, analyze_syntax, syntax_pipeline
from src.nlp.syntax_types import SyntaxAnalysis


SAMPLE_TEXT = "РџСЂРѕРµРєС‚ РіРѕС‚РѕРІ, РїРѕС‚РѕРјСѓ С‡С‚Рѕ РєРѕРјР°РЅРґР° Р·Р°РєРѕРЅС‡РёР»Р° СЂР°Р±РѕС‚Сѓ."


def test_analyze_syntax_returns_canonical_analysis_for_basic_sentence():
    syntax_pipeline.cache_clear()
    analysis = analyze_syntax(SAMPLE_TEXT)

    assert isinstance(analysis, SyntaxAnalysis)
    assert analysis.text == SAMPLE_TEXT
    assert analysis.cache_key
    assert analysis.backend in {"natasha", "fallback"}
    assert all(0 <= start <= end <= len(SAMPLE_TEXT) for start, end in analysis.protected_spans)

    if analysis.backend == "natasha":
        assert analysis.tokens
        assert analysis.sentences
        assert analysis.sentences[0].tokens
        assert analysis.sentences[0].start == 0
        assert analysis.sentences[0].end == len(SAMPLE_TEXT)
        assert any(token.lemma for token in analysis.tokens)
        assert any(token.pos for token in analysis.tokens)
        assert any(token.dep_rel for token in analysis.tokens)
        assert any(token.head_id is not None for token in analysis.tokens)
        assert any(token.text == "РїРѕС‚РѕРјСѓ" for token in analysis.tokens)
        assert any(token.text == "С‡С‚Рѕ" for token in analysis.tokens)


def test_analyze_syntax_reuses_lru_cache_for_repeated_text():
    analyzer = SyntaxAnalyzer({"cache_enabled": True, "cache_max_size": 8})
    analyzer.clear_cache()

    first = analyzer.analyze(SAMPLE_TEXT)
    first_stats = analyzer.cache_info()
    second = analyzer.analyze(SAMPLE_TEXT)
    second_stats = analyzer.cache_info()

    assert second is first
    assert first_stats["misses"] == 1
    assert second_stats["hits"] == first_stats["hits"] + 1
    assert second_stats["size"] == 1


def test_analyze_syntax_skips_cache_for_enormous_text_by_default():
    analyzer = SyntaxAnalyzer({"cache_enabled": True, "cache_max_size": 8, "cache_max_text_length": 10})
    analyzer.clear_cache()

    analyzer.analyze("РўРµРєСЃС‚.")
    analyzer.analyze("РўРµРєСЃС‚.")
    analyzer.analyze("РћС‡РµРЅСЊ РґР»РёРЅРЅС‹Р№ С‚РµРєСЃС‚ РґР»СЏ СЃРёРЅС‚Р°РєСЃРёС‡РµСЃРєРѕРіРѕ Р°РЅР°Р»РёР·Р°.")
    analyzer.analyze("РћС‡РµРЅСЊ РґР»РёРЅРЅС‹Р№ С‚РµРєСЃС‚ РґР»СЏ СЃРёРЅС‚Р°РєСЃРёС‡РµСЃРєРѕРіРѕ Р°РЅР°Р»РёР·Р°.")
    stats = analyzer.cache_info()

    assert stats["hits"] == 1
    assert stats["misses"] == 3
    assert stats["size"] == 1


def test_analyze_syntax_unavailable_backend_fails_open(monkeypatch):
    real_import = builtins.__import__

    def import_without_natasha(name, *args, **kwargs):
        if name == "natasha" or name.startswith("natasha."):
            raise ModuleNotFoundError("No module named 'natasha'")
        return real_import(name, *args, **kwargs)

    syntax_pipeline.cache_clear()
    monkeypatch.setattr(builtins, "__import__", import_without_natasha)

    try:
        analysis = SyntaxAnalyzer({"cache_enabled": False, "fail_open": True}).analyze(SAMPLE_TEXT)
    finally:
        syntax_pipeline.cache_clear()

    assert analysis.backend == "fallback"
    assert analysis.errors
    assert analysis.tokens == []
    assert analysis.sentences
    assert analysis.clauses == []
    assert isinstance(analysis.phrases, list)
    assert analysis.cache_key

