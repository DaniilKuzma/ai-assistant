# dictionary_typo coverage

`dictionary_typo` is a controlled online-generation layer for Russian spelling
and typo training examples. It covers only trusted entries from
`lexicon/layers/dictionary_typo/` and generated typo pairs that compile into the
runtime orthographic correction lexicon.

## Covered rule ids

- `dictionary_normative_words`: frequent normative dictionary spellings.
- `dictionary_borrowed_words`: accepted borrowed-word spellings, including
  explicit correction sources such as `риэлтор -> риелтор`.
- `dictionary_domain_terms`: IT, educational, and business terms that should be
  preserved or corrected only through explicit lexicon pairs.
- `dictionary_common_misspellings`: curated common misspellings.
- `typo_character_noise`: bounded delete, insert, replace, transpose, and
  duplicate character noise.
- `typo_keyboard_neighbor`: bounded Russian keyboard-neighbor replacement noise.
- `typo_space_noise`: curated split-word and glued-word space noise.

## Runtime contract

Runtime does not run deterministic typo autocorrect and does not guess arbitrary
edit-distance corrections. A correction is applicable only when all of these are
true:

- the model predicted a direct edit label with sufficient confidence;
- the source span is present in the trusted correction lexicon;
- the lexicon resolves the source to one unambiguous target.

Single-token dictionary and typo fixes use `DICT_REPLACE`. Split/glue fixes use
`SPAN_REPLACE_BY_LEXICON`. Ambiguous sources stay no-op through the existing
`OrthographicCorrectionLexicon.lookup()` behavior.

## Protected tokens

Generated typo corruption skips protected lexical material: accepted variants,
CSV lexicon forms, orthography card correct forms, domain terms, common
abbreviations and names, URLs, emails, numeric tokens, and code-like tokens.
Generated wrong forms are also dropped when they collide with protected words or
would map to more than one correction target.

This layer is training coverage, not a free-form correction engine.
