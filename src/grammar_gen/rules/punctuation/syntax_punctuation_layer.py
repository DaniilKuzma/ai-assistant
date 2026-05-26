from __future__ import annotations

from pathlib import Path

from src.rule_layers.direct_cases import LayerRuleProgram, layer_rule_programs
from src.rule_layers.spec_loader import DEFAULT_LAYERS_DIR
from src.rule_layers.syntax_punctuation import load_syntax_punctuation_specs


def syntax_punctuation_rule_programs(
    root: str | Path = DEFAULT_LAYERS_DIR,
) -> tuple[LayerRuleProgram, ...]:
    return layer_rule_programs(load_syntax_punctuation_specs(root))


__all__ = ["syntax_punctuation_rule_programs"]
