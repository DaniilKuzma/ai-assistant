# Decisions

- Current architecture is AST-first online generation plus direct edit tagging.
- Training examples are sampled online; `train.csv` is removed.
- Frozen validation, test, and regression sets may be stored as JSONL in
  `data/generated_eval`.
- RuRoBERTa remains encoder-only with LoRA and custom direct edit heads.
- Runtime correction is deterministic rules first, then guarded neural direct
  edits.
- GUI behavior is preserved.
- The old candidate-aware dataset-builder architecture is removed and must not
  be restored.
