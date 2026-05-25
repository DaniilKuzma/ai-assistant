# Rule Authoring Guide

## Create A RuleProgram

Add a focused `RuleProgram` under `src/grammar_gen/rules/orthography/` or
`src/grammar_gen/rules/punctuation/`. Keep the rule responsible for one
orthography or punctuation phenomenon and register it in the appropriate
registry module.

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

## Audit Command

Run:

```bash
python scripts/audit_generator.py configs/config.yaml --count 1000
```

For performance-sensitive rules, also run:

```bash
python scripts/benchmark_generation.py configs/config.yaml --count 1000
```
