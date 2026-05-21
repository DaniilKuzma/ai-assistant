# Syntax Rules Dataset Readiness

This report is eval/audit-only. It does not build the training dataset, run training, change checkpoints, or change thresholds.

## Summary

- positive eval rows: 2150
- hard negative rows: 1050
- rules with synthetic support: 43
- minimum eval examples per rule: 50
- minimum candidate recall: 1.00
- unsafe accepted hard-negative edits: 0

## Syntax Rules With Synthetic Support

- address_comma
- apposition_comma
- asyndetic_dash
- bracket_pair_balance
- clarification_comma
- comma_conjunction
- comma_subordinate
- comparative_turnover_comma
- consequence_dash
- context_chto_by
- context_nesmotrya
- context_tak_zhe
- context_to_zhe
- context_vsledstvie
- context_za_to
- detached_adverbial_comma
- detached_participial_comma
- direct_speech_colon
- direct_speech_dash
- direct_speech_quotes
- enumeration_colon
- enumeration_dash
- explanation_colon
- homogeneous_comma
- introductory_comma
- n_nn_adjective
- n_nn_deverbal_adjective
- n_nn_participle
- n_nn_short_form
- ne_adjective
- ne_adverb
- ne_participle
- ne_predicative
- ne_short_form
- ne_verb
- ni_particle_context
- ni_stable_expression
- punctuation_delete_replace
- quote_pair_balance
- semicolon
- subject_predicate_dash
- tsya_soft_delete
- tsya_soft_insert

## Remaining Blockers

- grammatical_endings_context: no current dictionary-backed lexical/morphology candidate path
- bare как comparative patterns: blocked by unsafe_comparative_as guard
- semantic-only asyndetic punctuation: blocked by unsafe_asyndetic guard
- stable punctuation combinations and footnote/layout formatting: no safe candidate path

Verdict: SYNTAX_DATASET_SUPPORT_READY
