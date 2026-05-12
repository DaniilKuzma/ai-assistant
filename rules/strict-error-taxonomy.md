# Strict Error Taxonomy

This document defines the package-1 contract for the strict orthography and
punctuation scope.

Code source of truth: `src/strict_error_taxonomy.py`

## Scopes

- `clean`: unchanged clean examples.
- `orthography`: normative Russian spelling errors.
- `punctuation`: normative Russian punctuation errors.
- `quarantine`: labels that may contain useful cases but are currently too broad
  or too artificial.
- `forbidden`: typo-like, spacing-only, heavy rewrite, unknown, or out-of-scope
  labels.

## Currently Allowed Orthography

- `spelling_tsya`
- `spelling_suffix_pronunciation`
- `spelling_prefix`
- `spelling_n_nn`
- `spelling_soft_hard_sign`
- `spelling_ne_ni`
- `spelling_vowel_after_sibilant`
- `spelling_i_y_after_ts`
- `spelling_hyphen`
- `spelling_dictionary_word`
- `spelling_compound_joining`
- `spelling_capitalization`
- `spelling_abbreviation_case`
- `spelling_borrowed_word`
- `real_spelling`

## Currently Allowed Punctuation

- `punct_remove_comma_before_clause`
- `punct_remove_intro_comma`
- `punct_remove_homogeneous_comma`
- `punct_remove_final`
- `punct_extra_comma_before_single_i`
- `punct_dash_missing`
- `punct_dash_extra`
- `punct_bsp_colon_missing`
- `punct_bsp_dash_missing`
- `punct_bracket_pair_missing`
- `punct_remove_quotes`
- `punct_quote_punct_order`
- `punct_direct_speech_inner_punct`
- `punct_direct_speech_dash_missing`
- `real_punctuation`

## Quarantine

- `spelling_replace`: too broad; split into concrete orthographic rules.
- `spelling_double`: keep only rule-based double-consonant cases later.
- `punct_remove_comma`: too broad; use concrete comma-rule labels.
- `punct_remove_internal_comma`: too broad; use concrete comma-rule labels.
- `punct_remove_all_commas`: too broad; use concrete comma-rule labels.
- `punct_remove_period`, `punct_remove_question`, `punct_remove_exclamation`:
  random non-final deletion; use `punct_remove_final` for final marks.
- `punct_wrong_*`: random sign-to-sign replacement is not a normative task.
- `punct_extra_comma`: replaced by concrete `punct_extra_comma_before_single_i`.
- `punct_bracket_extra`: currently too artificial.
- `punct_bracket_missing_close`: replaced by paired bracket examples.
- `punct_ellipsis_extra`, `punct_ellipsis_missing`: outside the current strict
  punctuation scope.
- `punct_list_missing_colon`, `punct_list_item_punctuation`: outside package-4
  scope; BSP colon/dash labels are used instead.
- `punct_quote_style`: typography/style unless the scope explicitly includes it.

## Forbidden

- `spelling_delete`
- `spelling_swap`
- `spelling_extra`
- `real_other`
- `unknown`
- `keyboard`
- `spacing_only`
- `heavy_rewrite`

Real-pair filtering also rejects `capitalization_only`, `style_rewrite`,
`untrainable_real_pair`, and `morphology_or_grammar` rows before they can enter
train/val. These are filter reasons rather than model labels.

## Runtime Contract

- `DatasetGenerator()` uses strict scope by default and must not emit quarantine
  or forbidden labels.
- `filter_real_pairs(..., strict_scope=True)` must reject rows whose labels are
  not strict.
- Legacy typo-like word operations (`delete`, `swap`, `extra`, random `double`)
  remain unavailable even if old flags are passed through `error_config`.
