# Syntax Punctuation Candidate Coverage Summary

This report covers the fixed syntax-punctuation candidate path only. It does not build a dataset, run training, change thresholds, change checkpoints, or enable auto-apply.

## Candidate Path

- Canonical module: `src/rules/syntax_punctuation.py`
- Registry bridge: `src/rules/punctuation.py`
- Candidate metadata bridge: `src/candidates/candidate_generator.py`
- Validator guard path: `src/validation/strict_validator.py`

## Newly Added Rule IDs

- `detached_participial_comma`
- `apposition_comma`
- `clarification_comma`
- `enumeration_dash`

## Candidate-Backed Rule IDs

- `comma_subordinate`
- `comma_conjunction`
- `introductory_comma`
- `address_comma`
- `homogeneous_comma`
- `enumeration_colon`
- `enumeration_dash`
- `detached_adverbial_comma`
- `detached_participial_comma`
- `apposition_comma`
- `clarification_comma`
- `comparative_turnover_comma`
- `subject_predicate_dash`
- `asyndetic_dash`
- `consequence_dash`
- `explanation_colon`
- `semicolon`
- `direct_speech_colon`
- `direct_speech_dash`
- `direct_speech_quotes`
- `quote_pair_balance`
- `bracket_pair_balance`
- `quote_open`
- `quote_close`
- `punctuation_delete_replace`

## Converted Matrix Entries

Candidate-backed conversion updated 65 syntax-required punctuation entries:

- 64 entries from the pre-conversion syntax-required inventory where `source_section=punctuation`, the syntax family was covered by bounded generators, and `can_be_candidate_backed_now=true`.
- 1 additional bounded asyndetic entry: `punctuation_7_3_1`.

Two existing quote/bracket-balance entries were also refreshed to require syntax/model/validator metadata:

- `punctuation_9_5`
- `punctuation_9_6`

## Family Coverage

- `subordinate_clause_comma`: candidate-backed, `model_required`.
- `coordinating_conjunction_comma`: candidate-backed, `model_required`.
- `introductory_words`: candidate-backed, `model_required`.
- `address_comma`: candidate-backed, `model_required`.
- `homogeneous_members`: candidate-backed, `model_required`.
- `enumeration_colon_dash`: candidate-backed, `model_required`.
- `detached_adverbial_phrases`: candidate-backed, `model_required`.
- `detached_participial_phrases`: candidate-backed, `model_required`.
- `detached_applications`: candidate-backed through `apposition_comma`, `model_required`.
- `clarification_members`: candidate-backed, `model_required`.
- `comparative_turnovers`: candidate-backed for `slovno`, `budto`, and `kak_budto`; unsafe bare `kak` remains blocked.
- `subject_predicate_dash`: candidate-backed, `model_required`.
- `asyndetic_complex_sentence`: bounded candidate-backed only when two finite-like clause spans are available.
- `direct_speech_syntax`: candidate-backed, `model_required`.
- `quote_bracket_balance`: candidate-backed obvious one-sided imbalance only.
- `punctuation_combinations`: candidate-backed obvious noise only, `candidate_only`.

## Candidate Recall

Fixed audit file: `reports/syntax_module/syntax_punctuation_candidate_recall.csv`.

- Covered audit rows: 63
- Candidate recall: 1.00
- Gap coverage: 1.00
- Hard negative pass rate: 1.00
- Unsafe candidate count on clean hard negatives: 0

## Remaining Blockers

- Bare `kak` comparative/role patterns, including role and identity uses: `punctuation_5_8_7`, `punctuation_5_8_10`.
- Stable or non-decomposable punctuation combinations that need lexical or semantic context: `punctuation_5_7_1` through `punctuation_5_7_10`.
- Footnote and layout formatting: `punctuation_9_7`.
- Semantic-only asyndetic punctuation outside the bounded two-clause patterns remains blocked by validator guard `unsafe_asyndetic`.

## Test Status

Targeted commands:

- `python -m pytest -q tests/test_syntax_punctuation_candidates.py`
- `python -m pytest -q tests/test_punctuation_rules.py`
- `python -m pytest -q tests/test_strict_validator.py`
- `python -m pytest -q tests/test_negative_rule_suites.py`

Verdict: `SYNTAX_PUNCTUATION_READY` if the targeted commands pass.
