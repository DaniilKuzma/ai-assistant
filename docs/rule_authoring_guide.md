# Rule Authoring Guide

## Choose The Rule Surface

Prefer the smallest controlled surface that fits the rule:

- Add a YAML `RuleLayer` spec under `lexicon/layers/compound_spelling/`,
  `lexicon/layers/dictionary_typo/`, or `lexicon/layers/syntax_punctuation/`
  when the rule can be expressed as direct token span or punctuation-gap cases.
- Add morpheme coverage through `src/orthography_gen/` specs and lexeme cards
  when the rule depends on a controlled orthographic site in a word form.
- Add a handwritten `RuleProgram` under `src/grammar_gen/rules/orthography/` or
  `src/grammar_gen/rules/punctuation/` only when the example needs generated AST
  context that the layer loaders cannot represent.

Keep each rule responsible for one Russian spelling or punctuation phenomenon.

## Modes

Each rule should define safe behavior for supported modes:

- `POSITIVE` creates a dirty `source_text` and clean `target_text`.
- `HARD_NEGATIVE` creates a tempting but already-correct or unsafe example.
- `CLEAN_IDENTITY` creates source and target text that must remain unchanged.

Do not use external clean-pool scanning to produce these examples.

## Labels

Populate token edit labels and punctuation gap labels directly. Use existing
labels from `src/schema/labels.py`; add a new label only when the edit cannot be
represented by the current direct tag set and the runtime realizer can support
it safely.

Use `SPAN_REPLACE_BY_LEXICON` for bounded token spans whose replacement must be
resolved through `OrthographicCorrectionLexicon`. It is appropriate for
controlled split/merge/hyphen and dictionary typo spans, but it must have an
unambiguous lexicon entry at runtime.

Use `DELETE_PUNCTUATION` only as a gap label when the source text already
contains the punctuation mark to remove. It must not encode broad punctuation
rewriting.

## expected_edit_count

Set `metadata["expected_edit_count"]` when the rule has a known logical edit
count. The audit and safety layers use this to detect accidental extra edits or
missing labels.

## Safety

Generated pairs must preserve meaning and stay within Russian spelling and
punctuation. Add metadata for allowed surface differences where the safety layer
needs rule-specific context. Avoid examples that require semantic rewriting,
style changes, or broad grammar correction.

## Tests

Add tests for:

- every supported mode;
- JSON round-trip through `GeneratedExample`;
- surface safety;
- expected label counts;
- hard negatives that must not produce edits;
- runtime realization if the rule introduces a new direct label behavior.

## Registry

Register the rule in the grammar-generation registry and ensure
`configs/config.yaml:generation.enabled_rule_groups` can include it through an
existing or explicit group. Keep `configs/rules.yaml` honest: mark entries as
implemented only when tests and generator coverage exist.

For layer rules:

- load specs through the existing `src/rule_layers` loader for that family;
- ensure every executable `rule_id` is present in `src/schema/labels.py`;
- include positive plus hard-negative or clean-identity cases;
- add `layer`, `rule_ids`, `sub_rule_ids`, `requires`, and `tests` in
  `configs/rules.yaml`.

Do not mark casing, abbreviations, quotation, dialogue/direct speech, or broad
syntax-only taxonomy entries as implemented unless a bounded generator and tests
exist for them.

## Audit Command

Run:

```bash
python scripts/audit_generator.py configs/config.yaml --count 1000
```

For performance-sensitive rules, also run:

```bash
python scripts/benchmark_generation.py configs/config.yaml --count 1000
```
