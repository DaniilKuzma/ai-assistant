# Short Dataset V2 Generation Report

- verdict: BLOCKED
- total: 60000
- split_sizes: {"test": 5000, "train": 50000, "val": 5000}
- composition: {"clean_identity_from_open_clean": 7800, "hard_negative_from_open_clean": 7800, "real_error_pair": 0, "synthetic_augmented_from_open_clean": 44400}
- clean_source_counts: {"lenta_news": 44370, "opencorpora": 15630}
- real_source_counts: {}
- real_pair_acceptance_rate: 0.000000
- candidate_recall_min_mean: 1.000000 / 1.000000
- gap_coverage_min_mean: 1.000000 / 1.000000
- template_leakage: {"test_overlap_with_train_rate": 0.0, "val_overlap_with_train_rate": 0.0}
- synthetic_normalized_duplicate_rate: 0.000000
- top_normalized_pair_count: 3
- meta_language_counts: {"context-pairs": 0, "n-nn": 0, "ne-pos": 0, "technical_rule_names": 0, "готовит важный примере": 0, "готовит итоговый примере": 0, "готовит рабочий примере": 0, "готовит точный примере": 0, "правило": 0, "пример 123": 0, "проверяет семейство": 0, "семейство": 0, "серии": 0, "серия": 0}
- suspicious_template_counts: {"готовит важный примере": 0, "готовит итоговый примере": 0, "готовит рабочий примере": 0, "готовит точный примере": 0, "проверяет семейство": 0}
- low_count_active_rule_ids: comma_conjunction, comma_subordinate, context_chto_by, hyphen_whitelist, introductory_comma, n_nn_deverbal_adjective

## Top 20 Rule Counts

- n_nn_adjective: 14000
- clean_identity: 7800
- clean_identity_hard_negative: 7800
- n_nn_short_form: 5428
- final_punctuation_default: 4734
- pattern_чо_че: 4562
- prefix_pre_pri: 4175
- pattern_жо_же: 3278
- pattern_цы_ци: 3144
- prefix_s_to_z: 2549
- n_nn_participle: 2413
- pattern_чя_ча: 2248
- pattern_шо_ше: 2137
- prefix_z_to_s: 1955
- tsya_soft_delete: 1783
- pattern_що_ще: 1728
- pattern_жы_жи: 1378
- pattern_шы_ши: 1277
- ne_verb: 1052
- pattern_щя_ща: 796

## Audit Errors

- active_rule_count_below_min:address_comma,comma_conjunction,comma_subordinate,comparative_turnover_comma,context_chto_by,context_nesmotrya,context_tak_zhe,context_to_zhe,context_vsledstvie,context_za_to,cy_exception,detached_adverbial_comma,dictionary_fuzzy,direct_speech_colon,direct_speech_dash,direct_speech_quotes,double_consonant_candidate,enumeration_colon,extra_letter_candidate,homogeneous_comma,hyphen_koe_koy,hyphen_particles,hyphen_po_adverbs,hyphen_whitelist,introductory_comma,keyboard_typo_candidate,missing_letter_candidate,n_nn_deverbal_adjective,ne_adjective,ne_adverb,ne_participle,pol_polu_compounds,subject_predicate_dash,swapped_letters_candidate
- missing_source_type:real_error_pair
