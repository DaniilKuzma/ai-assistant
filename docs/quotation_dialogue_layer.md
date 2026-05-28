# Quotation Dialogue Layer

`quotation_dialogue` is a bounded YAML-backed `RuleLayer` for Russian quotation
marks and direct speech punctuation. It stays inside the current
`GeneratedExample` / `RuleProgram` / `OnlineGrammarDataset` contract and emits
only direct labels.

## Covered Cases

- `quote_pairing`: insert `«...»` around controlled terms and section/title
  names in curated contexts.
- `quote_normalization`: normalize straight double quotes around controlled
  terms or direct speech to Russian guillemets.
- `quote_extra_marks`: delete clearly extra quotes around ordinary objects such
  as `отчёт` or `файл`.
- `dialogue_author_before`: add colon, quote pair, first-word capitalization,
  and final dot for simple author-before-speech cases.
- `dialogue_speech_before_author`: add quote pair and/or `, —` after the
  closing quote for simple speech-before-author cases.
- `dialogue_author_inside_speech`: limited single-pattern interrupted direct
  speech with two comma-dash gaps.
- `dialogue_bracket_guards`: guard-only identity examples for bracketed text.

## Labels And Counts

Quote and bracket wrappers use boundary labels, not token edits:

- before-token: `INSERT_OPEN_QUOTE`, `DELETE_OPEN_QUOTE`,
  `NORMALIZE_OPEN_QUOTE`;
- after-token: `INSERT_CLOSE_QUOTE`, `DELETE_CLOSE_QUOTE`,
  `NORMALIZE_CLOSE_QUOTE`;
- gaps: `COLON`, `DOT`, `COMMA_DASH`;
- token edits: currently only `CAPITALIZE` for safe direct speech casing.

Counts remain separated:

- `expected_token_edit_count` counts token edits only;
- `expected_gap_edit_count` counts punctuation gap decisions;
- `expected_boundary_edit_count` counts quote/bracket boundary decisions;
- `expected_total_edit_count` and `expected_edit_count` are the sum.

## Runtime Order

Representative runtime application first applies boundary/token labels with
`apply_boundary_and_token_edit_labels`, then retokenizes and applies gap labels
with `apply_gap_labels`. This is required for wrapper-aware `DOT` and
`COMMA_DASH` insertion after a newly inserted closing quote.

## Limitations

The layer does not implement nested quotes, broad citation punctuation, broad
quote style normalization, line-start dialogue replicas, or general bracket
correction. Complex interrupted direct speech remains limited to the safe
single-pattern examples present in `dialogue_author_inside_speech`.
