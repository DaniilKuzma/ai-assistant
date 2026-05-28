# Quotation, Casing, And Semantic Integration Plan

This note records the current integration state for the next quotation/dialogue,
casing, and semantic-context expansion step. It is documentation only: it does
not add labels, register rules, alter runtime correction, recreate offline CSV
datasets, or restore any candidate-aware data pipeline.

## Current Architecture State

The active architecture remains AST-first online generation plus direct edit
tagging:

- `RuleProgram` and `RuleLayer` implementations emit `GeneratedExample`
  instances online.
- Training examples use direct token edit labels and punctuation gap labels.
- Frozen validation, test, and regression examples may exist as JSONL under
  `data/generated_eval`, but materialized `train.csv`, `val.csv`, or `test.csv`
  datasets are not part of the architecture.
- Runtime correction stays conservative: deterministic rules first, then direct
  neural token/gap edits gated by thresholds and `ScopeGuard`.
- The old candidate-aware dataset pipeline is not restored.

Baseline observed before this document was added:

- `git status --short` showed an already dirty worktree.
- `python -m compileall src tests scripts` passed.
- The initial requested layer/runtime pytest set passed with `84 passed`.
- The final requested pytest subset passed with `32 passed`.

## RuleLayer Contract

`src/rule_layers/base.py` defines the current layer contract:

- `LayerRuleSpec` groups executable cases by `layer`, `rule_id`, and `family`,
  with `enabled`, `weight`, text metadata, and a tuple of `LayerDirectCase`
  values.
- `LayerDirectCase` carries `mode`, `source_text`, `target_text`,
  `token_operations`, `gap_operations`, `direct_token_labels`,
  `direct_gap_labels`, expected token/gap edit counts, `sub_rule_id`,
  metadata, and per-case weight.
- `LayerOperation` describes a bounded token or gap operation with a direct
  label, optional source/target patterns, explicit token indexes or spans, and
  operation metadata.
- `RuleLayer.generate(builder, realizer, rng, mode)` returns a
  `GeneratedExample`.

`src/rule_layers/direct_cases.py` wraps enabled specs in `LayerRuleProgram`.
`src/rule_layers/example_builders.py` resolves operations into source tokens,
token edit labels, gap labels, rule ids, expected edit counts, and metadata.
Positive examples must label the exact direct edit surface; hard negatives and
clean identities must remain identity pairs.

## YAML-Backed Layer Wiring

YAML-backed layers are connected through configuration and the generator
factory:

- `configs/config.yaml:generation.rule_layers` enables layer loading and points
  at `lexicon/layers`.
- `src/grammar_gen/factory.py:online_generator_from_config()` builds the
  generator and calls the configured layer loaders.
- Specialized loaders compile current YAML families:
  `load_compound_spelling_specs()`, `load_dictionary_typo_specs()`, and
  `load_syntax_punctuation_specs()`.
- Generic YAML loading remains available through `load_layer_specs()` for a
  layer shape that already fits `LayerRuleSpec`.
- `register_layered_rules()` validates that each spec `rule_id` exists in
  `src/schema/labels.py` and then registers the layer-backed `RuleProgram`.

The current integrated YAML-backed families are `compound_spelling`,
`dictionary_typo`, and `syntax_punctuation`. They do not implement broad
quotation/dialogue, casing, abbreviations, or free-form syntax rewriting.

## Current Labels

Current token edit labels are:

`KEEP`, `DELETE`, `SKIP_MERGED`, `LOWERCASE`, `UPPERCASE`,
`SPLIT_NE_VERB`, `MERGE_TAK_ZHE_TO_TAKZHE`,
`SPLIT_TAKZHE_TO_TAK_ZHE`, `MERGE_TO_ZHE_TO_TOZHE`,
`SPLIT_TOZHE_TO_TO_ZHE`, `MERGE_ZA_TO_TO_ZATO`,
`SPLIT_ZATO_TO_ZA_TO`, `HYPHENATE_PARTICLE_TO`,
`HYPHENATE_PARTICLE_LIBO`, `HYPHENATE_PARTICLE_NIBUD`,
`HYPHENATE_KOE`, `HYPHENATE_PO_ADVERB`, `FIX_TSYA_TO_TTSYA`,
`FIX_TTSYA_TO_TSYA`, `DICT_REPLACE`, and `SPAN_REPLACE_BY_LEXICON`.

Current gap punctuation labels are:

`NONE`, `COMMA`, `DASH`, `COLON`, `SEMICOLON`, `DOT`, `QUESTION`,
`EXCLAMATION`, `ELLIPSIS`, and `DELETE_PUNCTUATION`.

Current rule label families are:

- legacy contextual orthography: `ne_verb`, `takzhe_tak_zhe`,
  `tozhe_to_zhe`, `zato_za_to`, `hyphen_particles`, `hyphen_koe`,
  `hyphen_po_adverb`, `tsya_ttsya`;
- morphemic orthography: `suffix_its_ets`, `suffix_enn_yan`, `n_nn_basic`,
  and `morpheme_*`;
- handwritten punctuation: `comma_*`, `dash_subject_predicate`,
  `final_punctuation`;
- YAML-backed spelling and typo groups: `compound_*`, `dictionary_*`,
  and `typo_*`;
- YAML-backed syntax punctuation groups: `punct_*`.

Future grouped rule ids should be new ids, for example
`punct_quotation_dialogue`, `casing_orthographic`, and
`semantic_contextual_orthography`. These are planning names only and must not be
added to the schema until a bounded generator and runtime behavior exist.

## Quotation And Dialogue Labels

Quotation and dialogue cannot be represented safely by the current gap-only
punctuation surface. Existing gap labels attach one punctuation action after an
existing word token. That works for comma, dash, colon, semicolon, final marks,
ellipsis, and deletion of punctuation already present after a token, but it does
not model every boundary needed by quotation and dialogue.

The future layer needs token-boundary punctuation labels because it must
represent cases such as:

- an opening quote before the first quoted token;
- a closing quote after the last quoted token;
- a dialogue dash before the first replica token;
- paired quote state across a quoted span;
- normalization between quote characters without broad punctuation rewriting.

A single after-token gap label cannot encode both sides of a quote pair or a
pre-token opening mark. Adding quotation/dialogue should therefore introduce a
bounded token-boundary punctuation surface, plus hard negatives for unmatched
pairs, citations, nested quotes, abbreviations, and ordinary dash punctuation.

## Casing Labels

The current `UPPERCASE` token label is not suitable for proper-name casing. In
`src/runtime/edit_realizer.py`, `UPPERCASE` uppercases the whole source token.
Russian proper-name correction usually requires first-letter capitalization or a
lexicon-exact casing replacement, not all-caps conversion.

Future casing should use direct labels that match safe runtime behavior, for
example:

- `CAPITALIZE_TOKEN` for first-letter capitalization when the lowercase token is
  otherwise unchanged;
- `DECAPITALIZE_TOKEN` for bounded lowercase normalization;
- `CASE_REPLACE_BY_LEXICON` for trusted names, titles, abbreviations, or
  orthographic exceptions whose exact casing must come from a lexicon.

Casing remains in scope only when it is part of Russian orthographic norm. It
must not become named-entity rewriting, title normalization, style correction,
or semantic content insertion.

## Semantic Layer Boundary

The future semantic layer must not become free-form semantic rewriting. Its role
should be limited to disambiguating whether a spelling, punctuation, or
orthographic casing correction is normatively allowed in context. It should
still emit direct labels or lexicon-gated replacements.

Existing contextual rules and compound subrules that are semantic by nature
include:

- legacy `RuleProgram` rules: `takzhe_tak_zhe`, `tozhe_to_zhe`,
  `zato_za_to`, `ne_verb`, `hyphen_particles`, `hyphen_koe`, and
  `hyphen_po_adverb`;
- service-word compound subrules such as `takzhe_tak_zhe`,
  `tozhe_to_zhe`, `chto_by_chto_bi`, `zato_za_to`,
  `potomu_chto_po_tomu_chto`, `ottogo_ot_togo`, `otchego_ot_chego`, and
  `zachem_za_chem`;
- derived-preposition subrules such as `v_techenie_v_techenii`,
  `v_prodolzhenie_v_prodolzhenii`, `vsledstvie_v_sledstvii`,
  `naschet_na_schet`, `vvidu_v_vidu`, `napodobie_na_podobie`,
  `navstrechu_na_vstrechu`, `nesmotrya_na_ne_smotrya_na`, and
  `nevziraya_na_ne_vziraya_na`;
- compound spelling cases whose correctness depends on lexical or semantic
  class, including `ne_*`, appositions, coordinate adjectives, color/direction
  adjectives, and `pol-/polu-` cases.

These existing rules should not be deleted or replaced. New semantic expansion
should use grouped rule ids and direct labels while preserving the legacy
`RuleProgram` ids and the current compound subrule coverage.

## Conflict Avoidance

`RuleRegistry.register_rule()` rejects duplicate `rule_id` values. Therefore a
future `LayerRuleSpec` must not reuse a `rule_id` already present in
`default_rule_registry` unless the registry later gains an explicit safe
override mechanism.

To avoid duplicate rule and pair conflicts:

- keep the old `RuleProgram` ids intact: `ne_verb`, `takzhe_tak_zhe`,
  `tozhe_to_zhe`, `zato_za_to`, `hyphen_particles`, `hyphen_koe`, and
  `hyphen_po_adverb`;
- prefer new grouped rule ids for new layers, such as
  `punct_quotation_dialogue`, `casing_orthographic`, and
  `semantic_contextual_orthography`;
- use unique `sub_rule_id` values inside each grouped layer;
- audit normalized correction entries by `(source, target, rule_id, operation)`;
- audit generated examples by `(source_text, target_text, primary_rule_id,
  mode)`;
- treat ambiguous lexicon mappings as no-op unless safe context guards make a
  single target unambiguous;
- keep hard negatives for semantic traps, quote/bracket lookalikes, proper-name
  false positives, and already-correct punctuation.

No future implementation in these areas should add seq2seq correction, free-form
rewrite correction, `CandidateGenerator`, strict synthetic-row dataset building,
offline train CSVs, or clean sentence pool scanning.
