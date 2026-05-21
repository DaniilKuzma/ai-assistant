# Rule Expansion Backlog V2

Blocked phase2 rules by required next work.

## NEEDS_VALIDATOR

- abbreviation_case_protection: Add bounded StrictValidator guard for abbreviation_case_protection and preserve known positives.
- capitalization_sentence_start: Add bounded StrictValidator guard for capitalization_sentence_start and preserve known positives.
- dictionary_fuzzy: Add bounded StrictValidator guard for dictionary_fuzzy and preserve known positives.
- final_punctuation_default: Add bounded StrictValidator guard for final_punctuation_default and preserve known positives.
- keyboard_typo_candidate: Add bounded StrictValidator guard for keyboard_typo_candidate and preserve known positives.

## NEEDS_THRESHOLD_CALIBRATION

- bracket_pair_balance: Calibrate bracket_pair_balance on validation data without tuning on test.
- context_chto_by: Calibrate context_chto_by on validation data without tuning on test.
- context_nesmotrya: Calibrate context_nesmotrya on validation data without tuning on test.
- context_tak_zhe: Calibrate context_tak_zhe on validation data without tuning on test.
- context_to_zhe: Calibrate context_to_zhe on validation data without tuning on test.
- context_vsledstvie: Calibrate context_vsledstvie on validation data without tuning on test.
- context_za_to: Calibrate context_za_to on validation data without tuning on test.
- direct_speech_colon: Calibrate direct_speech_colon on validation data without tuning on test.
- direct_speech_quotes: Calibrate direct_speech_quotes on validation data without tuning on test.
- double_consonant_candidate: Calibrate double_consonant_candidate on validation data without tuning on test.
- enumeration_colon: Calibrate enumeration_colon on validation data without tuning on test.
- explanation_colon: Calibrate explanation_colon on validation data without tuning on test.
- extra_letter_candidate: Calibrate extra_letter_candidate on validation data without tuning on test.
- frequent_error_exact: Calibrate frequent_error_exact on validation data without tuning on test.
- hyphen_po_adverbs: Calibrate hyphen_po_adverbs on validation data without tuning on test.
- missing_hard_sign: Calibrate missing_hard_sign on validation data without tuning on test.
- missing_letter_candidate: Calibrate missing_letter_candidate on validation data without tuning on test.
- n_nn_adjective: Calibrate n_nn_adjective on validation data without tuning on test.
- n_nn_deverbal_adjective: Calibrate n_nn_deverbal_adjective on validation data without tuning on test.
- n_nn_participle: Calibrate n_nn_participle on validation data without tuning on test.
- n_nn_short_form: Calibrate n_nn_short_form on validation data without tuning on test.
- ne_adjective: Calibrate ne_adjective on validation data without tuning on test.
- ne_adverb: Calibrate ne_adverb on validation data without tuning on test.
- ne_participle: Calibrate ne_participle on validation data without tuning on test.
- ne_verb: Calibrate ne_verb on validation data without tuning on test.
- ni_stable_expression: Calibrate ni_stable_expression on validation data without tuning on test.
- pattern_жо_же: Calibrate pattern_жо_же on validation data without tuning on test.
- pattern_жы_жи: Calibrate pattern_жы_жи on validation data without tuning on test.
- pattern_цы_ци: Calibrate pattern_цы_ци on validation data without tuning on test.
- pattern_чо_че: Calibrate pattern_чо_че on validation data without tuning on test.
- pattern_чю_чу: Calibrate pattern_чю_чу on validation data without tuning on test.
- pattern_чя_ча: Calibrate pattern_чя_ча on validation data without tuning on test.
- pattern_шо_ше: Calibrate pattern_шо_ше on validation data without tuning on test.
- pattern_шы_ши: Calibrate pattern_шы_ши on validation data without tuning on test.
- pattern_що_ще: Calibrate pattern_що_ще on validation data without tuning on test.
- pattern_щю_щу: Calibrate pattern_щю_щу on validation data without tuning on test.
- pattern_щя_ща: Calibrate pattern_щя_ща on validation data without tuning on test.
- pol_polu_compounds: Calibrate pol_polu_compounds on validation data without tuning on test.
- prefix_pre_pri: Calibrate prefix_pre_pri on validation data without tuning on test.
- prefix_s_to_z: Calibrate prefix_s_to_z on validation data without tuning on test.
- prefix_z_to_s: Calibrate prefix_z_to_s on validation data without tuning on test.
- punctuation_delete_replace: Calibrate punctuation_delete_replace on validation data without tuning on test.
- quote_pair_balance: Calibrate quote_pair_balance on validation data without tuning on test.
- sdelat_prefix: Calibrate sdelat_prefix on validation data without tuning on test.
- soft_to_hard_sign: Calibrate soft_to_hard_sign on validation data without tuning on test.
- swapped_letters_candidate: Calibrate swapped_letters_candidate on validation data without tuning on test.
- tsya_soft_delete: Calibrate tsya_soft_delete on validation data without tuning on test.
- tsya_soft_insert: Calibrate tsya_soft_insert on validation data without tuning on test.

## NEEDS_MORE_TRAINING

- cy_exception: Include safe cy_exception positives in a future training cycle.

## NEEDS_CANDIDATE_GENERATOR

- : Implement bounded candidate generator support for neural_punctuation.
- quote_close: Implement bounded candidate generator support for quote_close.
- quote_open: Implement bounded candidate generator support for quote_open.
- yo_e_candidate: Implement bounded candidate generator support for yo_e_candidate.

## NEEDS_DICTIONARY_SUPPORT

- : Add dictionary support needed for letter_y candidate generation.
- : Add dictionary support needed for alternating_roots candidate generation.
- : Add dictionary support needed for difficult_root_vowels candidate generation.
- : Add dictionary support needed for suffix_vowels candidate generation.
- : Add dictionary support needed for fleeting_connecting_vowels candidate generation.
- : Add dictionary support needed for verb_endings candidate generation.
- : Add dictionary support needed for consonants_voiced_devoiced candidate generation.
- : Add dictionary support needed for consonants_unchecked candidate generation.
- : Add dictionary support needed for consonants_sk_st_zg_zd candidate generation.
- : Add dictionary support needed for consonants_unpronounced candidate generation.
- : Add dictionary support needed for consonant_clusters candidate generation.
- : Add dictionary support needed for consonants_ch_sh_before_n_t candidate generation.
- : Add dictionary support needed for consonants_g_ogo_ego candidate generation.
- : Add dictionary support needed for foreign_words candidate generation.
- : Add dictionary support needed for typos_space_operations candidate generation.
- : Add dictionary support needed for typical_unsettled_neologisms candidate generation.

## NEEDS_SYNTAX_FEATURES

- : Add syntax-backed features and validators before activating case_endings.
- : Add syntax-backed features and validators before activating nonletter_signs.
- : Add syntax-backed features and validators before activating solid_hyphen_separate_nouns.
- : Add syntax-backed features and validators before activating solid_hyphen_separate_adjectives.
- : Add syntax-backed features and validators before activating capitalization_stylistic_pronouns.
- : Add syntax-backed features and validators before activating homophones.
- : Add syntax-backed features and validators before activating sentence_boundary_and_middle_punctuation.
- : Add syntax-backed features and validators before activating sentence_inner_end_marks.
- : Add syntax-backed features and validators before activating comma_subject_predicate.
- : Add syntax-backed features and validators before activating extra_comma_government.
- : Add syntax-backed features and validators before activating dash_incomplete_sentence.
- : Add syntax-backed features and validators before activating dash_connection.
- : Add syntax-backed features and validators before activating comma_conjunction.
- : Add syntax-backed features and validators before activating homogeneous_definitions.
- : Add syntax-backed features and validators before activating detached_definitions.
- : Add syntax-backed features and validators before activating detached_limiting_specifying_joining.
- : Add syntax-backed features and validators before activating indivisible_combinations.
- : Add syntax-backed features and validators before activating complex_mark_interaction.
- asyndetic_dash: Add syntax-backed features and validators before activating asyndetic_dash.
- consequence_dash: Add syntax-backed features and validators before activating consequence_dash.
- semicolon: Add syntax-backed features and validators before activating semicolon.

## NEEDS_NER_SUPPORT

- : Add NER-backed spans and negative tests for typos_initials_period.
- : Add NER-backed spans and negative tests for detached_applications.
- capitalization_ner: Add NER-backed spans and negative tests for capitalization_ner.
