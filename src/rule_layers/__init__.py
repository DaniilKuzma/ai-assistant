from __future__ import annotations

from src.rule_layers.base import LayerDirectCase, LayerOperation, LayerRuleSpec, RuleLayer
from src.rule_layers.casing import load_casing_specs
from src.rule_layers.compound_spelling import load_compound_spelling_specs
from src.rule_layers.coverage import LayerCoverageReport, collect_layer_coverage, validate_layer_coverage
from src.rule_layers.dictionary_typo import load_dictionary_typo_corrections, load_dictionary_typo_specs
from src.rule_layers.direct_cases import DirectCasesLayer, LayerRuleProgram
from src.rule_layers.example_builders import (
    build_gap_operations_example,
    build_generated_example_from_case,
    build_token_span_replacement_example,
)
from src.rule_layers.spec_loader import DEFAULT_LAYERS_DIR, load_layer_specs
from src.rule_layers.semantic import load_semantic_specs
from src.rule_layers.syntax_punctuation import load_syntax_punctuation_specs


__all__ = [
    "DEFAULT_LAYERS_DIR",
    "DirectCasesLayer",
    "LayerCoverageReport",
    "LayerDirectCase",
    "LayerOperation",
    "LayerRuleProgram",
    "LayerRuleSpec",
    "RuleLayer",
    "build_gap_operations_example",
    "build_generated_example_from_case",
    "build_token_span_replacement_example",
    "collect_layer_coverage",
    "load_casing_specs",
    "load_dictionary_typo_corrections",
    "load_dictionary_typo_specs",
    "load_compound_spelling_specs",
    "load_layer_specs",
    "load_semantic_specs",
    "load_syntax_punctuation_specs",
    "validate_layer_coverage",
]
