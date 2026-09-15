from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING

from src.grammar_gen.rules.base import GenerationMode

if TYPE_CHECKING:
    from src.grammar_gen.builders import GrammarBuilder
    from src.grammar_gen.randomness import RandomSource
    from src.grammar_gen.realizer import Realizer
    from src.schema import GeneratedExample


LAYER_MODES = frozenset(mode.value for mode in GenerationMode)


@dataclass(frozen=True)
class LayerOperation:
    kind: str
    label: str
    source_pattern: str = ""
    target_pattern: str = ""
    token_index: int | None = None
    token_start: int | None = None
    token_end: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind.strip()
        label = self.label.strip()
        if not kind:
            raise ValueError("LayerOperation kind must not be empty.")
        if not label:
            raise ValueError("LayerOperation label must not be empty.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class LayerDirectCase:
    rule_id: str
    family: str
    sub_rule_id: str
    mode: str
    source_text: str
    target_text: str
    token_operations: tuple[LayerOperation, ...] = ()
    gap_operations: tuple[LayerOperation, ...] = ()
    boundary_operations: tuple[LayerOperation, ...] = ()
    expected_token_edit_count: int = 0
    expected_gap_edit_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    direct_token_labels: tuple[str, ...] = ()
    direct_gap_labels: tuple[str, ...] = ()
    direct_boundary_before_labels: tuple[str, ...] = ()
    direct_boundary_after_labels: tuple[str, ...] = ()
    expected_boundary_edit_count: int = 0

    def __post_init__(self) -> None:
        rule_id = self.rule_id.strip()
        family = self.family.strip()
        sub_rule_id = self.sub_rule_id.strip()
        mode = self.mode.strip()
        if not rule_id:
            raise ValueError("LayerDirectCase rule_id must not be empty.")
        if not family:
            raise ValueError("LayerDirectCase family must not be empty.")
        if not sub_rule_id:
            raise ValueError("LayerDirectCase sub_rule_id must not be empty.")
        if mode not in LAYER_MODES:
            raise ValueError(f"Unsupported layer case mode: {mode!r}.")
        if not self.source_text.strip():
            raise ValueError("LayerDirectCase source_text must not be empty.")
        if not self.target_text.strip():
            raise ValueError("LayerDirectCase target_text must not be empty.")
        if self.expected_token_edit_count < 0:
            raise ValueError("expected_token_edit_count must be non-negative.")
        if self.expected_gap_edit_count < 0:
            raise ValueError("expected_gap_edit_count must be non-negative.")
        if self.expected_boundary_edit_count < 0:
            raise ValueError("expected_boundary_edit_count must be non-negative.")
        if self.weight < 0:
            raise ValueError("LayerDirectCase weight must be non-negative.")
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "sub_rule_id", sub_rule_id)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "token_operations", tuple(self.token_operations))
        object.__setattr__(self, "gap_operations", tuple(self.gap_operations))
        object.__setattr__(self, "boundary_operations", tuple(self.boundary_operations))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "direct_token_labels", tuple(self.direct_token_labels))
        object.__setattr__(self, "direct_gap_labels", tuple(self.direct_gap_labels))
        object.__setattr__(self, "direct_boundary_before_labels", tuple(self.direct_boundary_before_labels))
        object.__setattr__(self, "direct_boundary_after_labels", tuple(self.direct_boundary_after_labels))


@dataclass(frozen=True)
class LayerRuleSpec:
    layer: str
    rule_id: str
    family: str
    cases: tuple[LayerDirectCase, ...] = ()
    description: str = ""
    explanation: str = ""
    enabled: bool = True
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        layer = self.layer.strip()
        rule_id = self.rule_id.strip()
        family = self.family.strip()
        if not layer:
            raise ValueError("LayerRuleSpec layer must not be empty.")
        if not rule_id:
            raise ValueError("LayerRuleSpec rule_id must not be empty.")
        if not family:
            raise ValueError("LayerRuleSpec family must not be empty.")
        if self.weight < 0:
            raise ValueError("LayerRuleSpec weight must be non-negative.")
        for case in self.cases:
            if case.rule_id != rule_id:
                raise ValueError(
                    f"Layer case rule_id mismatch for {rule_id!r}: {case.rule_id!r}."
                )
            if case.family != family:
                raise ValueError(
                    f"Layer case family mismatch for {rule_id!r}: {case.family!r}."
                )
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "metadata", dict(self.metadata))


class RuleLayer(Protocol):
    spec: LayerRuleSpec

    @property
    def supported_modes(self) -> tuple[GenerationMode, ...]:
        ...

    def generate(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
        mode: GenerationMode,
    ) -> GeneratedExample:
        ...


__all__ = ["LayerDirectCase", "LayerOperation", "LayerRuleSpec", "RuleLayer"]
