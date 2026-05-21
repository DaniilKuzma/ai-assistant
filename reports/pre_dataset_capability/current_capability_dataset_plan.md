# Current Capability Dataset Plan

This plan is for a future dataset build only. It does not build data, train, change checkpoints, or change production thresholds.

## Include Rule IDs

### INCLUDE_NOW

- address_comma: risk=high; quota=120; needs=none
- comma_conjunction: risk=high; quota=120; needs=none
- comma_subordinate: risk=high; quota=120; needs=none
- comparative_turnover_comma: risk=high; quota=120; needs=none
- detached_adverbial_comma: risk=high; quota=120; needs=none
- direct_speech_dash: risk=medium; quota=80; needs=none
- homogeneous_comma: risk=high; quota=120; needs=none
- hyphen_koe_koy: risk=low; quota=50; needs=none
- hyphen_particles: risk=low; quota=50; needs=none
- hyphen_whitelist: risk=medium; quota=80; needs=none
- introductory_comma: risk=high; quota=120; needs=none
- subject_predicate_dash: risk=medium; quota=80; needs=none

### INCLUDE_AFTER_VALIDATOR

- abbreviation_case_protection: risk=high; quota=120; needs=validator, hard_negatives
- bracket_pair_balance: risk=high; quota=120; needs=validator, hard_negatives
- capitalization_sentence_start: risk=high; quota=120; needs=validator, hard_negatives
- context_chto_by: risk=high; quota=120; needs=validator, hard_negatives
- context_nesmotrya: risk=high; quota=120; needs=validator, hard_negatives
- context_tak_zhe: risk=high; quota=120; needs=validator, hard_negatives
- context_to_zhe: risk=high; quota=120; needs=validator, hard_negatives
- context_vsledstvie: risk=high; quota=120; needs=validator, hard_negatives
- context_za_to: risk=high; quota=120; needs=validator, hard_negatives
- dictionary_fuzzy: risk=high; quota=120; needs=validator, hard_negatives
- double_consonant_candidate: risk=high; quota=120; needs=validator, hard_negatives
- extra_letter_candidate: risk=high; quota=120; needs=validator, hard_negatives
- final_punctuation_default: risk=high; quota=120; needs=validator, hard_negatives
- hyphen_po_adverbs: risk=high; quota=120; needs=validator, hard_negatives
- keyboard_typo_candidate: risk=high; quota=120; needs=validator, hard_negatives
- missing_letter_candidate: risk=high; quota=120; needs=validator, hard_negatives
- punctuation_delete_replace: risk=high; quota=120; needs=validator, hard_negatives
- quote_pair_balance: risk=high; quota=120; needs=validator, hard_negatives
- swapped_letters_candidate: risk=high; quota=120; needs=validator, hard_negatives
- tsya_soft_delete: risk=high; quota=120; needs=validator, hard_negatives
- tsya_soft_insert: risk=high; quota=120; needs=validator, hard_negatives

### INCLUDE_AFTER_THRESHOLD_CALIBRATION

- direct_speech_colon: risk=medium; quota=80; needs=threshold_calibration
- direct_speech_quotes: risk=medium; quota=80; needs=threshold_calibration
- enumeration_colon: risk=medium; quota=80; needs=threshold_calibration
- explanation_colon: risk=medium; quota=80; needs=threshold_calibration
- frequent_error_exact: risk=medium; quota=80; needs=threshold_calibration
- missing_hard_sign: risk=medium; quota=80; needs=threshold_calibration
- n_nn_adjective: risk=medium; quota=80; needs=threshold_calibration
- n_nn_deverbal_adjective: risk=medium; quota=80; needs=threshold_calibration
- n_nn_participle: risk=medium; quota=80; needs=threshold_calibration
- n_nn_short_form: risk=medium; quota=80; needs=threshold_calibration
- ne_adjective: risk=medium; quota=80; needs=threshold_calibration
- ne_adverb: risk=medium; quota=80; needs=threshold_calibration
- ne_participle: risk=medium; quota=80; needs=threshold_calibration
- ne_verb: risk=medium; quota=80; needs=threshold_calibration
- ni_stable_expression: risk=high; quota=120; needs=threshold_calibration
- pattern_жо_же: risk=medium; quota=80; needs=threshold_calibration
- pattern_жы_жи: risk=medium; quota=80; needs=threshold_calibration
- pattern_цы_ци: risk=medium; quota=80; needs=threshold_calibration
- pattern_чо_че: risk=medium; quota=80; needs=threshold_calibration
- pattern_чю_чу: risk=medium; quota=80; needs=threshold_calibration
- pattern_чя_ча: risk=medium; quota=80; needs=threshold_calibration
- pattern_шо_ше: risk=medium; quota=80; needs=threshold_calibration
- pattern_шы_ши: risk=medium; quota=80; needs=threshold_calibration
- pattern_що_ще: risk=medium; quota=80; needs=threshold_calibration
- pattern_щю_щу: risk=medium; quota=80; needs=threshold_calibration
- pattern_щя_ща: risk=medium; quota=80; needs=threshold_calibration
- pol_polu_compounds: risk=medium; quota=80; needs=threshold_calibration
- prefix_pre_pri: risk=medium; quota=80; needs=threshold_calibration
- prefix_s_to_z: risk=medium; quota=80; needs=threshold_calibration
- prefix_z_to_s: risk=medium; quota=80; needs=threshold_calibration
- sdelat_prefix: risk=medium; quota=80; needs=threshold_calibration
- soft_to_hard_sign: risk=medium; quota=80; needs=threshold_calibration

### INCLUDE_AFTER_TRAINING

- asyndetic_dash: risk=medium; quota=80; needs=more_training
- consequence_dash: risk=medium; quota=80; needs=more_training
- cy_exception: risk=medium; quota=80; needs=more_training
- semicolon: risk=medium; quota=80; needs=more_training

## Hard Negatives Needed

- dictionary/typo candidates: clean known words, proper names, morphology-compatible distractors
- capitalization and abbreviations: sentence-initial traps, acronym/proper-name casing traps
- punctuation: clean punctuation contexts, quote/bracket balance traps, direct-speech traps
- context pairs: clean minimal pairs for each split/join alternative

## Do Not Include

- metadata-only taxonomy groups
- planned or disabled rows
- rows blocked by syntax, dictionary, NER, or no candidate path
