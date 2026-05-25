# Open Tasks And Fragile Areas

## Fragile Areas

- `ScopeGuard` controls runtime safety; add negative tests for changes.
- Punctuation gap labels are high-risk for false positives.
- Context-sensitive spelling pairs should remain threshold-gated.
- `-тся/-ться` and similar morphology-dependent edits require conservative
  runtime handling.
- Heavy RuRoBERTa loading must stay lazy; debug smoke must run without it.

## Useful Next Steps

- Expand `RuleProgram` coverage incrementally.
- Keep `configs/rules.yaml` honest: planned or partial entries must not look
  production-ready.
- Add hard negatives for each new rule before enabling it in online training.
- Review frozen eval worst examples after runtime threshold changes.

## Non-Goals

- Do not recreate `train.csv`.
- Do not add seq2seq rewriting.
- Do not restore the old offline data-builder architecture.
