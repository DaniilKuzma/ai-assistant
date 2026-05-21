# Short Dataset V3 Generation Report

- verdict: BLOCKED
- total: 48687
- split_sizes: {"test": 4058, "train": 40572, "val": 4057}
- composition: {"clean_identity_from_open_clean": 7200, "hard_negative_from_open_clean": 7200, "real_error_pair": 3600, "synthetic_augmented_from_open_clean": 30687}
- clean_source_counts: {"lenta_news": 42341, "nerus_news": 1805, "opencorpora": 941}
- real_source_counts: {"sage_multidomain_gold": 1295, "sage_ruspellru": 1144, "spellcheck_punctuation_benchmark": 1161}
- real_pair_acceptance_rate: 1.000000
- candidate_recall_min_mean: 0.453425 / 0.952415
- gap_coverage_min_mean: 1.000000 / 1.000000
- template_leakage: {"test_overlap_with_train_rate": 0.0, "val_overlap_with_train_rate": 0.0}
- synthetic_normalized_duplicate_rate: 0.088572
- top_normalized_pair_count: 8
- meta_language_counts: {"context-pairs": 0, "n-nn": 0, "ne-pos": 0, "technical_rule_names": 0, "готовит важный примере": 0, "готовит итоговый примере": 0, "готовит рабочий примере": 0, "готовит точный примере": 0, "правило": 0, "пример 123": 0, "проверяет семейство": 0, "семейство": 0, "серии": 0, "серия": 0}
- suspicious_template_counts: {"готовит важный примере": 0, "готовит итоговый примере": 0, "готовит рабочий примере": 0, "готовит точный примере": 0, "проверяет семейство": 0}
- active_rule_quota_summary: {"active_rule_count": 56, "excluded_count": 7, "min_total_per_active_rule": 80, "preferred_total_per_active_rule": 200, "underfilled_count": 8}
- low_count_active_rule_ids: n_nn_adjective, n_nn_participle, n_nn_short_form, pattern_жо_же, pattern_цы_ци, pattern_чо_че, pattern_чя_ча, prefix_s_to_z
- excluded_active_rule_ids: asyndetic_dash, consequence_dash, final_punctuation_default, hyphen_whitelist, ne_verb, prefix_pre_pri, tsya_soft_insert
- rule_caps_applied: final_punctuation_default, n_nn_adjective, n_nn_short_form, pattern_жо_же, pattern_цы_ци, pattern_чо_че, prefix_pre_pri
- error_type_caps_applied: 

## Top 30 Rule Counts

- clean_identity: 7200
- clean_identity_hard_negative: 7200
- n_nn_adjective: 2000
- missing_letter_candidate: 939
- n_nn_short_form: 932
- pattern_чо_че: 860
- dictionary_fuzzy: 748
- final_punctuation_default: 664
- pattern_жо_же: 651
- address_comma: 593
- comma_conjunction: 593
- comma_subordinate: 593
- comparative_turnover_comma: 593
- context_chto_by: 593
- context_nesmotrya: 593
- context_tak_zhe: 593
- context_to_zhe: 593
- context_vsledstvie: 593
- context_za_to: 593
- cy_exception: 593
- detached_adverbial_comma: 593
- direct_speech_colon: 593
- direct_speech_dash: 593
- direct_speech_quotes: 593
- double_consonant_candidate: 593
- enumeration_colon: 593
- explanation_colon: 593
- extra_letter_candidate: 593
- homogeneous_comma: 593
- hyphen_koe_koy: 593

## Audit Errors

- synthetic_augmented_from_open_clean_shortage:30687<42000
- dataset_size_below_requested:48687!=60000
- candidate_recall_active_min_below_threshold
- active_rule_quota_underfilled:n_nn_adjective,n_nn_participle,n_nn_short_form,pattern_жо_же,pattern_цы_ци,pattern_чо_че,pattern_чя_ча,prefix_s_to_z
