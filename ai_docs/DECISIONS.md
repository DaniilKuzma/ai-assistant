# Decisions

- Current architecture: AST-first online generation plus direct edit tagging.
- RuRoBERTa remains encoder-only with LoRA and custom heads.
- Train examples are generated on the fly; large train/val/test CSV datasets are
  not materialized.
- Frozen validation, test, and regression sets may be materialized as JSONL in
  `data/generated_eval`.
- The old offline data-prep stack is removed.
- Runtime safety is handled by conservative direct edit scope guarding.
