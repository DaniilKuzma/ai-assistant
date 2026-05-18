from __future__ import annotations

import builtins

import pytest

from src.nlp.syntax import parse_syntax, syntax_pipeline


SAMPLE_TEXT = "Проект готов, потому что команда закончила работу."


def test_parse_syntax_returns_stable_tokens_for_russian_sentence():
    syntax_pipeline.cache_clear()
    if syntax_pipeline() is None:
        pytest.skip("Natasha syntax pipeline is unavailable in this environment.")

    tokens = parse_syntax(SAMPLE_TEXT)

    assert tokens
    by_text = {token.text: token for token in tokens}
    assert {"Проект", "готов", "команда"}.issubset(by_text)
    assert any(token.lemma for token in tokens)
    assert any(token.pos for token in tokens)
    assert any(token.rel for token in tokens)
    assert any(token.head_id is not None for token in tokens)

    project = by_text["Проект"]
    assert project.lemma
    assert project.pos
    assert project.start == SAMPLE_TEXT.index("Проект")
    assert project.end == project.start + len("Проект")


def test_parse_syntax_reuses_cached_pipeline():
    syntax_pipeline.cache_clear()

    parse_syntax(SAMPLE_TEXT)
    first = syntax_pipeline.cache_info()
    parse_syntax(SAMPLE_TEXT)
    second = syntax_pipeline.cache_info()

    assert first.misses == 1
    assert second.misses == first.misses
    assert second.hits > first.hits


def test_parse_syntax_empty_text_does_not_load_pipeline():
    syntax_pipeline.cache_clear()

    assert parse_syntax("   ") == []
    assert syntax_pipeline.cache_info().misses == 0


def test_parse_syntax_caches_unavailable_pipeline_when_pkg_resources_is_missing(monkeypatch):
    real_import = builtins.__import__

    def import_without_pkg_resources(name, *args, **kwargs):
        if name == "pkg_resources" or name.startswith("pkg_resources."):
            raise ModuleNotFoundError("No module named 'pkg_resources'")
        return real_import(name, *args, **kwargs)

    syntax_pipeline.cache_clear()
    monkeypatch.setattr(builtins, "__import__", import_without_pkg_resources)

    try:
        assert parse_syntax(SAMPLE_TEXT) == []
        first = syntax_pipeline.cache_info()
        assert parse_syntax(SAMPLE_TEXT) == []
        second = syntax_pipeline.cache_info()

        assert first.misses == 1
        assert second.misses == first.misses
        assert second.hits > first.hits
        assert second.currsize == 1
    finally:
        syntax_pipeline.cache_clear()
