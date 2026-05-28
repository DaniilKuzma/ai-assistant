# Rule Expansion Plan

This project expands only Russian spelling and punctuation correction. The
active architecture is AST-first online generation plus direct edit tagging;
the old candidate-aware dataset pipeline is not part of the plan.

## Implemented Controlled Layers

- `compound_spelling`: controlled split, merge, hyphen, `пол-/полу-`, service
  word, preposition, pronoun/particle, adverb, noun/adjective, and `не`
  spelling examples. Runtime application is model-assisted and lexicon-gated;
  generic span fixes use `SPAN_REPLACE_BY_LEXICON`.
- `morpheme`: compiler-backed orthographic examples for hissing vowels,
  hard/soft signs, root vowels, prefixes, suffixes, н/нн, consonants, and
  endings. Runtime single-token replacements use `DICT_REPLACE` through the
  orthographic lexicon.
- `dictionary_typo`: trusted dictionary words, borrowed words, domain terms,
  curated common misspellings, bounded character noise, keyboard-neighbor noise,
  and split/glue space noise. It is not a free edit-distance typo corrector.
- `syntax_punctuation`: controlled punctuation gap examples for final marks,
  dash syntax, homogeneous members, detached members, comparative turns,
  introductory/address/interjection cases, complex sentences, BSP, fixed
  expression guards, and extra punctuation deletion through
  `DELETE_PUNCTUATION`.
- `quotation_dialogue`: bounded quote pairing, quote normalization, extra quote
  deletion, direct-speech author/speech templates, bracket guards, boundary
  wrapper labels, and `COMMA_DASH`. Nested quotes, broad citations, and general
  bracket correction are not deterministic.
- `casing`: bounded `CAPITALIZE` and `LOWERCASE` cases for sentence starts,
  proper names, geo names, organizations, document/event titles, ordinary common
  lowercase normalization, and formal-you guards. Broad NER casing is not
  deterministic by default.
- `semantic`: grouped `semantic_*` contextual disambiguation for service words,
  derived prepositions, ne/ni guards, introductory-word homonyms, and
  comparative `как` cases. It is model-assisted and lexicon/direct-label gated,
  not a free-form semantic rewrite layer.

`configs/rules.yaml` is the coverage matrix. A row is `implemented` or
`controlled_implemented` only when the rule is executable by online generation
and has tests. Broader rows that still require a model, dictionary, or syntax
analyzer are marked `model_assisted`, `dictionary_required`,
`syntax_required`, or `planned`.

## Current Non-Goals

- Broad NER casing, title/style normalization, and abbreviation handling are not
  integrated correction layers.
- Nested quotes, broad citations, complex interrupted direct speech, and general
  bracket correction remain outside deterministic runtime correction.
- Semantic rows may disambiguate bounded spelling/punctuation cases only; they
  must not add absent words or change meaning.
- No seq2seq model, free-form rewrite correction, materialized train CSV, or
  candidate-aware training path should be added.

## Next Expansion Order

1. Add more controlled layer cases only when the direct labels and runtime
   realizer already support the operation.
2. Extend the runtime lexicon for `SPAN_REPLACE_BY_LEXICON` or `DICT_REPLACE`
   before enabling new span replacements.
3. Add syntax-backed rules only with explicit hard negatives and tests for
   false-positive boundaries.
4. Keep frozen eval JSONL under `data/generated_eval`; do not create train,
   validation, or test CSV datasets.
5. Keep `configs/rules.yaml` honest: `implemented_limited` means bounded
   positive generation exists, `guard_only` means identity hard negatives only,
   and each executable row must list concrete rule ids, sub-rule ids, and tests.

## Required Test Evidence

- Layer loader tests for rule ids, modes, sub-rule ids, and invalid specs.
- Generator audit tests for expected edit counts and source/target contracts.
- Runtime realizer tests for each direct label used by the rule.
- Coverage matrix tests proving executable rule ids exist in the direct online
  generation registry and that implemented rows list pytest coverage.
- Integration tests proving new layers sample by mix, declared rule ids sample
  in supported modes, mixed examples validate, diversity audit exposes
  layer/family/sub-rule distributions, frozen eval manifests include the new
  layers, and smoke training tensorizes the current label schema.
