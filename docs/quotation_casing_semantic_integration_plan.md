# Quotation, Casing, And Semantic Integration

This document records the implemented integration state for the
`quotation_dialogue`, `casing`, and `semantic` layers. The architecture remains
AST-first online generation plus direct edit tagging. It does not restore
candidate-aware data preparation, seq2seq correction, offline train/val/test
CSV datasets, clean sentence pool scanning, or free-form rewrite correction.

## Config Wiring

`configs/config.yaml` enables the three layers through:

- `generation.enabled_rule_groups`;
- `generation.mix`;
- `generation.rule_layers.groups`;
- `generation.frozen_eval` for config-driven frozen eval builds.

`online_generator_from_config(config, seed=...)` loads the layer specs from
`lexicon/layers/quotation_dialogue`, `lexicon/layers/casing`, and
`lexicon/layers/semantic`. Layer rule ids are grouped ids. The semantic loader
does not register legacy `RuleProgram` ids such as `takzhe_tak_zhe`,
`tozhe_to_zhe`, `zato_za_to`, or `ne_verb`.

## Labels

The layer integration relies on suffix-only label schema extension:

- token labels include `SPAN_REPLACE_BY_LEXICON` and `CAPITALIZE`;
- gap labels include `DELETE_PUNCTUATION` and `COMMA_DASH`;
- boundary-before labels support open quote/bracket insertion, deletion, and
  normalization;
- boundary-after labels support close quote/bracket insertion, deletion, and
  normalization;
- rule labels include grouped quote/dialogue, casing, and semantic rule ids at
  the end of `RULE_LABELS`.

Tests assert prefix compatibility so old token, gap, boundary, and rule ids do
not move when the schema grows.

## Implemented Layers

`quotation_dialogue` is a bounded YAML-backed layer. It covers quote pairing,
quote normalization, extra quote deletion, simple author-before-speech,
speech-before-author, and a limited author-inside-speech pattern. Bracket cases
are guard-only; broad bracket correction is not implemented.

`casing` is a bounded YAML-backed layer. It covers sentence-start
capitalization, controlled person/geographic/organization/document/event names,
ordinary common lowercase normalization, and formal-you guards. It uses
`CAPITALIZE` and bounded `LOWERCASE`; it does not use `UPPERCASE` for proper
names and does not perform broad NER-backed capitalization.

`semantic` is a bounded YAML-backed layer. It covers contextual service-word
spelling, derived prepositions, ne/ni guards, introductory-word homonym guards,
and comparative `как` appositive/role contexts. It emits direct labels or
lexicon-gated `SPAN_REPLACE_BY_LEXICON`; it is not a semantic rewrite layer.

## Runtime Support

Runtime correction remains deterministic-first and conservative:

- quote/bracket wrappers are realized through boundary labels;
- `COMMA_DASH` inserts the comma-dash gap outside a closing quote/bracket;
- `CAPITALIZE` uppercases only the first letter of a token;
- `SPAN_REPLACE_BY_LEXICON` resolves through the trusted orthographic lexicon;
- semantic lexicon entries require a matching model/requested `semantic_*`
  rule id and are not applied as deterministic typo corrections;
- `ScopeGuard` accepts bounded punctuation and casing edits but rejects semantic
  word insertion and unrelated Russian word replacement.

Formal-you correction remains guard-only. Already acceptable lowercase
`вы`/`ваш` examples must remain unchanged unless a future explicit opt-in policy
is added.

## Audit And Frozen Eval

`audit_batch()` and `diversity_report()` expose:

- layer distribution;
- family distribution;
- rule id distribution;
- sub-rule id distribution;
- duplicate rate per layer;
- top duplicate source-target pairs.

`scripts/audit_generation_diversity.py` prints those sections explicitly.
`scripts/build_frozen_eval.py configs/config.yaml` now reads
`generation.frozen_eval` and builds val/test/regression JSONL plus manifests.
The legacy explicit `--split --count --output` mode remains supported.

## Rules Metadata

`configs/rules.yaml` marks only tested executable subrules as executable.
Bounded positive coverage uses `implemented_limited` or
`controlled_implemented`; identity-only coverage uses `guard_only`. New rows use
`implementation.layer` values `quotation_dialogue`, `casing`, or `semantic`
where the listed rule ids belong to those layers, and they list concrete
`rule_ids`, `sub_rule_ids`, and pytest files.

## Adding New Cases

1. Add the case to the appropriate YAML file under `lexicon/layers/<layer>/`.
2. Use an existing direct label. Add a new label only after runtime realization,
   tensorization, model heads, and prefix compatibility tests are updated.
3. For semantic span replacements, add or verify the trusted runtime lexicon
   entry and keep hard negatives for literal/ambiguous contexts.
4. Add loader, generation, runtime, and frozen-eval tests for the new sub-rule.
5. Update `configs/rules.yaml` with the layer, grouped rule id, sub-rule id,
   status, and tests.

## Non-Deterministic By Default

The following remain non-deterministic/model-gated by default:

- semantic service-word and derived-preposition disambiguation;
- semantic punctuation disambiguation around introductory words and `как`;
- broad casing and NER-style proper-name discovery;
- quote/dialogue cases outside the curated YAML templates;
- bracket correction beyond guard-only identity examples.
