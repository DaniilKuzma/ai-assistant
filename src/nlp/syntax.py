from __future__ import annotations

import logging

from src.nlp.syntax_analyzer import SyntaxAnalyzer, analyze_syntax, clear_syntax_cache, syntax_cache_info, syntax_pipeline
from src.nlp.syntax_types import ClauseSpan, PhraseSpan, SyntaxAnalysis, SyntaxSentence, SyntaxToken


logger = logging.getLogger(__name__)


def parse_syntax(text: str) -> list[SyntaxToken]:
    """Compatibility wrapper returning flat tokens from the canonical analyzer."""

    if not text.strip():
        return []
    try:
        analysis = SyntaxAnalyzer({"cache_enabled": False}).analyze(text)
    except Exception:  # pragma: no cover - fail-open compatibility path.
        logger.debug("Syntax parsing failed", exc_info=True)
        return []
    if analysis.backend != "natasha":
        return []
    return analysis.tokens


__all__ = [
    "ClauseSpan",
    "PhraseSpan",
    "SyntaxAnalysis",
    "SyntaxAnalyzer",
    "SyntaxSentence",
    "SyntaxToken",
    "analyze_syntax",
    "clear_syntax_cache",
    "parse_syntax",
    "syntax_cache_info",
    "syntax_pipeline",
]
