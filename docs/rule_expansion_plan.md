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

`configs/rules.yaml` is the coverage matrix. A row is `implemented` or
`controlled_implemented` only when the rule is executable by online generation
and has tests. Broader rows that still require a model, dictionary, or syntax
analyzer are marked `model_assisted`, `dictionary_required`,
`syntax_required`, or `planned`.

## Current Non-Goals

- Casing and capitalization are not an integrated correction layer.
- Abbreviation handling is not an integrated correction layer.
- Quotes, brackets, dialogue, direct speech, citations, and paired
  quote/bracket punctuation are planned outside the current four layers.
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

## Required Test Evidence

- Layer loader tests for rule ids, modes, sub-rule ids, and invalid specs.
- Generator audit tests for expected edit counts and source/target contracts.
- Runtime realizer tests for each direct label used by the rule.
- Coverage matrix tests proving executable rule ids exist in the direct online
  generation registry and that implemented rows list pytest coverage.
