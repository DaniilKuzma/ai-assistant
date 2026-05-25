# Decisions

- Current architecture: AST-first online generation plus direct edit tagging.
- RuRoBERTa remains encoder-only with LoRA and custom heads.
- Train examples are generated on the fly; large train/val/test CSV datasets are
  not materialized.
- Frozen validation, test, and regression sets may be materialized as JSONL in
  `data/generated_eval`.
- CandidateGenerator, rule_lab, clean sentence pool scanning, corruption
  operators, and strict synthetic-row validation are removed from data prep.
- CandidateGenerator is not part of the target model-backed runtime.
- The old StrictValidator data-prep role is removed; a future ScopeGuard will
  cover runtime boundaries.
