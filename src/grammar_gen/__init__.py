"""AST-first online training example generation support."""

from src.grammar_gen.ast import (
    Clause,
    ComplexSentence,
    DashSubjectPredicateSentence,
    HomogeneousSentence,
    IntroductorySentence,
    NounPhrase,
    SimpleSentence,
    VerbPhrase,
)
from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.lexicon import (
    AdjectiveEntry,
    AdverbEntry,
    IntroductoryEntry,
    Lexeme,
    Lexicon,
    NounEntry,
    PrepositionEntry,
    VerbEntry,
)
from src.grammar_gen.morphology import (
    MorphologyEngine,
    TokenAnalysis,
    is_valid_prepositional_phrase,
    reject_bad_surface,
)
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.semantic_safety import (
    reject_semantic_nonsense,
    validate_clause_semantics,
    validate_frame_fillers,
)
from src.grammar_gen.semantics import PrepSlot, SemanticClass, SemanticFrameLexicon, VerbFrame
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.safety import assert_valid_or_raise, validate_ast_sentence, validate_surface, validate_target_ast_or_raise

__all__ = [
    "AdjectiveEntry",
    "AdverbEntry",
    "Clause",
    "ComplexSentence",
    "DashSubjectPredicateSentence",
    "GrammarBuilder",
    "HomogeneousSentence",
    "IntroductoryEntry",
    "IntroductorySentence",
    "Lexeme",
    "Lexicon",
    "MorphologyEngine",
    "NounPhrase",
    "NounEntry",
    "PrepSlot",
    "PrepositionEntry",
    "RandomSource",
    "Realizer",
    "SemanticClass",
    "SemanticFrameLexicon",
    "SimpleSentence",
    "TokenAnalysis",
    "VerbEntry",
    "VerbPhrase",
    "VerbFrame",
    "assert_valid_or_raise",
    "is_valid_prepositional_phrase",
    "reject_bad_surface",
    "reject_semantic_nonsense",
    "validate_ast_sentence",
    "validate_target_ast_or_raise",
    "validate_clause_semantics",
    "validate_frame_fillers",
    "validate_surface",
]
