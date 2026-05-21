# Syntax Orthography Coverage Summary

This audit uses fixed probe examples only. It does not build datasets, train models, change thresholds, change checkpoints, or enable auto-apply.

## Candidate-backed rule_ids

- ne_verb
- ne_adjective
- ne_adverb
- ne_participle
- ne_short_form
- ne_predicative
- ni_stable_expression
- ni_particle_context
- n_nn_adjective
- n_nn_participle
- n_nn_deverbal_adjective
- n_nn_short_form
- tsya_soft_insert
- tsya_soft_delete
- context_tak_zhe
- context_to_zhe
- context_chto_by
- context_za_to
- context_vsledstvie
- context_nesmotrya

## Converted syntax_required orthography entries

- orthography_3_4_1
- orthography_3_4_3_2
- orthography_3_5_4
- orthography_3_6_1_2
- orthography_3_6_4_1
- orthography_3_6_4_2
- orthography_3_7_1_4_4
- orthography_3_7_1_4_5
- orthography_3_7_1_5
- orthography_3_7_2_8
- orthography_3_7_2_9
- orthography_3_7_2_10

## Recall

- Candidate probes present: 20/20
- Candidate recall: 1.00
- Hard-negative plain Corrector pass: 1.00

## Remaining blockers

- grammatical_endings_context remains blocked: safe broad ending correction still requires dictionary-backed lexical and morphology support.

Verdict: SYNTAX_ORTHOGRAPHY_READY
