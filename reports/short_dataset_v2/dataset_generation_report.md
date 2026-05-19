# Short Dataset V2 Generation Report

- verdict: BLOCKED
- total: 0
- split_sizes: {"test": 0, "train": 0, "val": 0}
- composition: {"clean_identity": 0, "hard_negative": 0, "real_error_pair": 0, "synthetic_augmented": 0}
- clean_source_counts: {}
- real_source_counts: {}
- real_pair_acceptance_rate: 0.000000
- candidate_recall_min_mean: 1.000000 / 1.000000
- gap_coverage_min_mean: 1.000000 / 1.000000
- template_leakage: {"test_overlap_with_train_rate": 0.0, "val_overlap_with_train_rate": 0.0}
- synthetic_normalized_duplicate_rate: 0.000000
- top_normalized_pair_count: 0
- meta_language_counts: {"context-pairs": 0, "n-nn": 0, "ne-pos": 0, "technical_rule_names": 0, "готовит важный примере": 0, "готовит итоговый примере": 0, "готовит рабочий примере": 0, "готовит точный примере": 0, "правило": 0, "пример 123": 0, "проверяет семейство": 0, "семейство": 0, "серии": 0, "серия": 0}
- suspicious_template_counts: {"готовит важный примере": 0, "готовит итоговый примере": 0, "готовит рабочий примере": 0, "готовит точный примере": 0, "проверяет семейство": 0}
- low_count_active_rule_ids: 

## Top 20 Rule Counts


## Audit Errors

- synthetic_augmented_shortage:0<44400
- clean_identity_shortage:0<7800
- hard_negative_shortage:0<7800
- dataset_size_below_requested:0!=60000
- clean_pool_below_min:0
- empty_clean_pool
- missing_source_type:synthetic_augmented
- missing_source_type:real_error_pair
- missing_source_type:clean_identity
- missing_source_type:hard_negative
