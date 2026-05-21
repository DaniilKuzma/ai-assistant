# Threshold Calibration GUI Spelling v2

- Rows: 292
- Baseline config: `configs/config.yaml` mode `calibrated_val_guarded_v3`
- Tuned config: `configs/config_threshold_calibrated_gui_spelling_v2.yaml` mode `calibrated_gui_spelling_v2`

## Summary

| metric | baseline | tuned v2 |
|---|---:|---:|
| exact_match_rate | 0.3562 | 0.5240 |
| dirty_exact_rate | 0.1931 | 0.4034 |
| dirty_improved_rate | 0.1931 | 0.4249 |
| dirty_worse_rate | 0.0043 | 0.0000 |
| clean_overcorrection_rate | 0.0000 | 0.0000 |
| mean_distance_delta | 0.1507 | 0.3562 |

## Tuned Thresholds

| key | current | tuned | floor | expected | tp | fp | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| double_consonant_candidate_threshold | 0.8400 | 0.4500 | 0.4500 | 31 | 23 | 0 | 1.000 | 0.742 |
| frequent_error_exact_threshold | 0.9500 | 0.1800 | 0.1800 | 29 | 28 | 0 | 1.000 | 0.966 |
| hyphen_po_adverbs_threshold | 0.8800 | 0.4000 | 0.4000 | 8 | 8 | 1 | 0.889 | 1.000 |
| hyphen_threshold | 0.9500 | 0.8200 | 0.8000 | 2 | 2 | 0 | 1.000 | 1.000 |
| n_nn_deverbal_adjective_threshold | 0.9200 | 0.6200 | 0.6200 | 5 | 5 | 0 | 1.000 | 1.000 |

## Category Comparison

| category | rows | baseline exact | tuned exact | baseline improved | tuned improved |
|---|---:|---:|---:|---:|---:|
| clean_hard_negative | 24 | 1.000 | 1.000 | 0.000 | 0.000 |
| clean_identity | 35 | 1.000 | 1.000 | 0.000 | 0.000 |
| context_pair | 14 | 0.000 | 0.000 | 0.000 | 0.000 |
| dataset_comma_conjunction | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| dataset_comma_subordinate | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| dataset_context_chto_by | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| dataset_context_nesmotrya | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| dataset_context_tak_zhe | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| dataset_context_to_zhe | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| dataset_dictionary_fuzzy | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| dataset_double_consonant_candidate | 8 | 0.000 | 0.875 | 0.000 | 0.875 |
| dataset_extra_letter_candidate | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| dataset_homogeneous_comma | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| dataset_hyphen_koe_koy | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| dataset_hyphen_particles | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| dataset_hyphen_po_adverbs | 7 | 0.000 | 1.000 | 0.000 | 1.000 |
| dataset_introductory_comma | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| dataset_keyboard_typo_candidate | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| dataset_missing_letter_candidate | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| dataset_n_nn_adjective | 6 | 0.000 | 0.167 | 0.000 | 0.333 |
| dataset_n_nn_deverbal_adjective | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| dataset_n_nn_participle | 6 | 0.000 | 0.000 | 0.000 | 0.667 |
| dataset_swapped_letters_candidate | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| frequent_spelling_exact | 66 | 0.015 | 0.424 | 0.015 | 0.424 |
| hyphen_whitelist | 6 | 0.500 | 1.000 | 0.500 | 1.000 |
| split_join_exact | 5 | 0.000 | 0.800 | 0.000 | 0.800 |
