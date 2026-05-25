# Open Tasks And Fragile Areas

## Fragile Areas

- `ScopeGuard`: central runtime protection for unsafe edits; changes require
  negative tests.
- Punctuation predictions: quotes, brackets, dash, final punctuation, and
  delete/replace actions can create false positives.
- Context pairs require high confidence and context-sensitive guardrails.
- `-тся/-ться` remains high-risk and should stay threshold-gated.
- Dictionary-backed lexical edits are useful for recall but increase
  overcorrection risk.
- Syntax support depends on `natasha`; fallback mode may return an intentionally
  sparse syntax layer.
- Heavy model loading should remain lazy so unit tests do not download
  RuRoBERTa unnecessarily.

## Useful Next Steps

- Review clean overcorrection examples after threshold or runtime changes.
- Keep `configs/rules.yaml` as an honest coverage matrix: planned entries must
  not look implemented.
- Add negative tests for each new deterministic or fallback-safe rule.
- Keep generated eval sets small and explicit unless a run is intentionally
  building frozen JSONL evaluation data.
