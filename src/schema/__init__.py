"""Shared schema contracts for AST-first generated examples and runtime edits."""

from src.schema.edits import CorrectionResult, RuntimeEdit
from src.schema.edit_types import (
    ALLOWED_EDIT_TYPES,
    PUNCTUATION_TYPES,
    SPELLING_TYPES,
    coarse_error_type,
    is_allowed_edit_type,
    is_context_dependent_pair,
)
from src.schema.examples import GeneratedExample, WordToken
from src.schema.labels import (
    GAP_ID_TO_LABEL,
    GAP_LABEL_TO_ID,
    GAP_PUNCTUATION_LABELS,
    RULE_ID_TO_LABEL,
    RULE_LABELS,
    RULE_LABEL_TO_ID,
    TOKEN_EDIT_LABELS,
    TOKEN_ID_TO_LABEL,
    TOKEN_LABEL_TO_ID,
    GapPunctuationLabel,
    RuleLabel,
    TokenEditLabel,
    gap_id_to_label,
    gap_label_to_id,
    rule_id_to_label,
    rule_tag_to_id,
    token_id_to_label,
    token_label_to_id,
)
from src.schema.serialization import (
    read_jsonl_examples,
    validate_jsonl_examples,
    write_jsonl_examples,
)

__all__ = [
    "ALLOWED_EDIT_TYPES",
    "CorrectionResult",
    "GAP_ID_TO_LABEL",
    "GAP_LABEL_TO_ID",
    "GAP_PUNCTUATION_LABELS",
    "GeneratedExample",
    "GapPunctuationLabel",
    "PUNCTUATION_TYPES",
    "RULE_ID_TO_LABEL",
    "RULE_LABELS",
    "RULE_LABEL_TO_ID",
    "RuleLabel",
    "RuntimeEdit",
    "SPELLING_TYPES",
    "TOKEN_EDIT_LABELS",
    "TOKEN_ID_TO_LABEL",
    "TOKEN_LABEL_TO_ID",
    "TokenEditLabel",
    "WordToken",
    "coarse_error_type",
    "gap_id_to_label",
    "gap_label_to_id",
    "is_allowed_edit_type",
    "is_context_dependent_pair",
    "read_jsonl_examples",
    "rule_id_to_label",
    "rule_tag_to_id",
    "token_id_to_label",
    "token_label_to_id",
    "validate_jsonl_examples",
    "write_jsonl_examples",
]

