# Final Dataset Readiness Report

Generated: 2026-05-24T21:23:26.613922+00:00

## Verdict

- final verdict: DATASET_BLOCKED
- ready for training: no
- dataset_contract: candidate_opportunity
- dataset_hash: c117cfdbd66b626c6457d09523c743b654877c7d2c77b174decbcb6edac29b72
- config_hash: da01fb81ceab5900dcc43ee15f1b4398c8974ee6efc7ded0881c4c15415afea1
- total rows: 320000
- split sizes: {"test": 32000, "train": 256000, "val": 32000}
- report_freshness before final report refresh: {"errors": [], "report_count": 67, "status": "fresh"}
- report_freshness after final report refresh: {"errors": [], "report_count": 68, "status": "fresh"}

## Layer Counts

- layer_counts: {"atomic_hard_negative": 118378, "atomic_positive": 0, "clean_identity": 200914, "real_atomic": 353, "stress_multi_error": 355}
- requested/effective/selected layer targets:

| layer | requested | effective | available | selected | deficit | adjustment |
|---|---:|---:|---:|---:|---:|---|
| atomic_positive | 190000 | 190000 | 92411 | 0 | 190000 |  |
| atomic_hard_negative | 76000 | 76000 | 86759 | 118378 | 0 |  |
| clean_identity | 32000 | 32000 | 547607 | 200914 | 0 |  |
| real_atomic | 6000 | 6000 | 353 | 353 | 5647 |  |
| stress_multi_error | 16000 | 16000 | 16000 | 355 | 15645 |  |

- layer_target_adjustments: {}

## Rule Activation

- production_ready_rule_count: 12
- training_candidate_rule_count: 76
- final_active_rule_count: 0
- final_active_rule_ids: []

No rule remained final-active after gates. The per-rule table below is from `under_quota_active_rules_report.csv`, i.e. the 76 attempted active rules before final pruning.

| rule_id | atomic_positive_count | hard_negative_count | candidate_recall | status | reason |
|---|---:|---:|---:|---|---|
| abbreviation_case_protection | 0 | 58 |  | under_quota | below_min_atomic_positive_quota |
| address_comma | 0 | 32 |  | under_quota | below_min_atomic_positive_quota |
| apposition_comma | 0 | 1667 | 1.0 | under_quota | below_min_atomic_positive_quota |
| asyndetic_dash | 0 | 32 |  | under_quota | below_min_atomic_positive_quota |
| bracket_pair_balance | 0 | 12 |  | under_quota | below_min_atomic_positive_quota |
| capitalization_sentence_start | 0 | 1845 | 1.0 | under_quota | below_min_atomic_positive_quota |
| clarification_comma | 0 | 198 | 1.0 | under_quota | below_min_atomic_positive_quota |
| comma_conjunction | 0 | 389 | 1.0 | under_quota | below_min_atomic_positive_quota |
| comma_subordinate | 0 | 1842 | 1.0 | under_quota | below_min_atomic_positive_quota |
| comparative_turnover_comma | 0 | 194 |  | under_quota | below_min_atomic_positive_quota |
| consequence_dash | 0 | 6 |  | under_quota | below_min_atomic_positive_quota |
| context_chto_by | 0 | 1684 |  | under_quota | below_min_atomic_positive_quota |
| context_nesmotrya | 0 | 947 | 1.0 | under_quota | below_min_atomic_positive_quota |
| context_tak_zhe | 0 | 1749 |  | under_quota | below_min_atomic_positive_quota |
| context_to_zhe | 0 | 2057 |  | under_quota | below_min_atomic_positive_quota |
| context_vsledstvie | 0 | 178 |  | under_quota | below_min_atomic_positive_quota |
| context_za_to | 0 | 704 |  | under_quota | below_min_atomic_positive_quota |
| cy_exception | 0 | 0 |  | under_quota | below_min_atomic_positive_quota |
| detached_adverbial_comma | 0 | 1429 |  | under_quota | below_min_atomic_positive_quota |
| detached_participial_comma | 0 | 1282 |  | under_quota | below_min_atomic_positive_quota |
| dictionary_fuzzy | 0 | 905 | 1.0 | under_quota | below_min_atomic_positive_quota |
| direct_speech_colon | 0 | 1287 |  | under_quota | below_min_atomic_positive_quota |
| direct_speech_dash | 0 | 318 |  | under_quota | below_min_atomic_positive_quota |
| direct_speech_quotes | 0 | 661 |  | under_quota | below_min_atomic_positive_quota |
| double_consonant_candidate | 0 | 1879 | 1.0 | under_quota | below_min_atomic_positive_quota |
| enumeration_colon | 0 | 268 |  | under_quota | below_min_atomic_positive_quota |
| enumeration_dash | 0 | 1789 |  | under_quota | below_min_atomic_positive_quota |
| explanation_colon | 0 | 841 |  | under_quota | below_min_atomic_positive_quota |
| extra_letter_candidate | 0 | 1162 | 1.0 | under_quota | below_min_atomic_positive_quota |
| final_punctuation_default | 0 | 0 | 1.0 | under_quota | below_min_atomic_positive_quota |
| frequent_error_exact | 0 | 2 | 1.0 | under_quota | below_min_atomic_positive_quota |
| homogeneous_comma | 0 | 1907 | 1.0 | under_quota | below_min_atomic_positive_quota |
| hyphen_koe_koy | 0 | 1 |  | under_quota | below_min_atomic_positive_quota |
| hyphen_particles | 0 | 44 | 1.0 | under_quota | below_min_atomic_positive_quota |
| hyphen_po_adverbs | 0 | 1901 |  | under_quota | below_min_atomic_positive_quota |
| hyphen_whitelist | 0 | 2 | 1.0 | under_quota | below_min_atomic_positive_quota |
| introductory_comma | 0 | 454 | 1.0 | under_quota | below_min_atomic_positive_quota |
| keyboard_typo_candidate | 0 | 877 | 1.0 | under_quota | below_min_atomic_positive_quota |
| missing_hard_sign | 0 | 2078 |  | under_quota | below_min_atomic_positive_quota |
| missing_letter_candidate | 0 | 1193 | 1.0 | under_quota | below_min_atomic_positive_quota |
| n_nn_adjective | 0 | 1805 |  | under_quota | below_min_atomic_positive_quota |
| n_nn_deverbal_adjective | 0 | 299 |  | under_quota | below_min_atomic_positive_quota |
| n_nn_participle | 0 | 2091 | 1.0 | under_quota | below_min_atomic_positive_quota |
| n_nn_short_form | 0 | 1574 |  | under_quota | below_min_atomic_positive_quota |
| ne_adjective | 0 | 1690 | 1.0 | under_quota | below_min_atomic_positive_quota |
| ne_adverb | 0 | 1822 |  | under_quota | below_min_atomic_positive_quota |
| ne_participle | 0 | 1920 |  | under_quota | below_min_atomic_positive_quota |
| ne_predicative | 0 | 1870 |  | under_quota | below_min_atomic_positive_quota |
| ne_short_form | 0 | 2063 |  | under_quota | below_min_atomic_positive_quota |
| ne_verb | 0 | 2097 | 1.0 | under_quota | below_min_atomic_positive_quota |
| ni_particle_context | 0 | 78 |  | under_quota | below_min_atomic_positive_quota |
| ni_stable_expression | 0 | 247 |  | under_quota | below_min_atomic_positive_quota |
| pattern_жо_же | 0 | 874 |  | under_quota | below_min_atomic_positive_quota |
| pattern_жы_жи | 0 | 2 |  | under_quota | below_min_atomic_positive_quota |
| pattern_цы_ци | 0 | 74 |  | under_quota | below_min_atomic_positive_quota |
| pattern_чо_че | 0 | 192 |  | under_quota | below_min_atomic_positive_quota |
| pattern_чю_чу | 0 | 0 |  | under_quota | below_min_atomic_positive_quota |
| pattern_чя_ча | 0 | 0 |  | under_quota | below_min_atomic_positive_quota |
| pattern_шо_ше | 0 | 2039 |  | under_quota | below_min_atomic_positive_quota |
| pattern_шы_ши | 0 | 2 |  | under_quota | below_min_atomic_positive_quota |
| pattern_що_ще | 0 | 4 |  | under_quota | below_min_atomic_positive_quota |
| pattern_щю_щу | 0 | 1 |  | under_quota | below_min_atomic_positive_quota |
| pattern_щя_ща | 0 | 0 |  | under_quota | below_min_atomic_positive_quota |
| pol_polu_compounds | 0 | 92 |  | under_quota | below_min_atomic_positive_quota |
| prefix_pre_pri | 0 | 1795 |  | under_quota | below_min_atomic_positive_quota |
| prefix_s_to_z | 0 | 271 |  | under_quota | below_min_atomic_positive_quota |
| prefix_z_to_s | 0 | 2 |  | under_quota | below_min_atomic_positive_quota |
| punctuation_delete_replace | 0 | 54 | 1.0 | under_quota | below_min_atomic_positive_quota |
| quote_pair_balance | 0 | 24 |  | under_quota | below_min_atomic_positive_quota |
| sdelat_prefix | 0 | 0 |  | under_quota | below_min_atomic_positive_quota |
| semicolon | 0 | 6 |  | under_quota | below_min_atomic_positive_quota |
| soft_to_hard_sign | 0 | 0 | 1.0 | under_quota | below_min_atomic_positive_quota |
| subject_predicate_dash | 0 | 1977 | 1.0 | under_quota | below_min_atomic_positive_quota |
| swapped_letters_candidate | 0 | 1881 | 1.0 | under_quota | below_min_atomic_positive_quota |
| tsya_soft_delete | 0 | 1800 |  | under_quota | below_min_atomic_positive_quota |
| tsya_soft_insert | 0 | 1836 |  | under_quota | below_min_atomic_positive_quota |

## Gate Metrics

- atomic_positive_count per active rule: no final active rules; all 76 attempted active rules have 0 selected atomic positives
- rules under atomic quota: 76
- rules under hard-negative quota: 38
- min atomic_positive_count across attempted active rules: 0
- max atomic_positive_count across attempted active rules: 0
- min hard_negative_count across attempted active rules: 0
- max hard_negative_count across attempted active rules: 2097
- candidate recall min: 1.0
- candidate recall mean: 1.0
- candidate recall failures: 0
- mixed_script_clean_count: 0
- unknown_rule_train_count: 0
- multi_edit_atomic_positive_count: 0
- real_atomic_count: 353
- stress_count: 355

## Blocking Quality Findings

| check_name | count | example |
|---|---:|---|
| artificial_marker_metka | 18 |  |
| artificial_marker_metka | 18 |  |
| clean_hard_balance_unbalanced_guillemets | 13 | Редактор сказал: «открытая заявка готов. |
| clean_hard_balance_unbalanced_ascii_quotes | 1 | И я готов этого добиватся в своей дальнейшей политической деятельности". |
| clean_hard_balance_unbalanced_parentheses | 12 | Проверь документ свежая сводка). |

## Warnings And Audit Errors

- warnings: ["layer_target_deficit:atomic_positive", "layer_target_deficit:real_atomic", "layer_target_deficit:stress_multi_error", "final_active_rule_count_below_target:0<76"]
- audit_errors: ["stress_ratio_outside_3_5_percent", "known_quality_bugs_present:artificial_marker_metka", "clean_hard_balance_bugs_present:unbalanced_guillemets", "clean_hard_balance_bugs_present:unbalanced_ascii_quotes", "clean_hard_balance_bugs_present:unbalanced_parentheses", "corpus_opportunity_share_below_threshold", "extended_quality_audit_blocking_issues", "final_active_rule_count_below_min:0<76"]

## Files

- dataset: data\processed\correction_dataset.csv.gz
- manifest: data\processed\dataset_manifest.json
- missing required dataset files: []
- missing required report files: []
- missing required dataset columns: []

## Commands And Exit Codes

| command | exit_code |
|---|---:|
| `git status` | 0 |
| `local source file size check` | 0 |
| `python -m pytest tests/test_config_profile.py -q` | 0 |
| `python scripts/preflight_candidate_dataset.py --config configs/config.yaml --clean-stale-artifacts` | 0 |
| `python scripts/preflight_candidate_dataset.py --config configs/config.yaml --json` | 0 |
| `python scripts/setup_data_sources.py --config configs/config.yaml --clean --real --validate --force --no-download --report-dir reports/source_setup --processed-dir data/processed` | 0 |
| `python scripts/preflight_candidate_dataset.py --config configs/config.yaml --json` | 0 |
| `python scripts/build_dataset.py --config configs/config.yaml --force --fail-on-blocked` | 2 |
| `python -m compileall src scripts tests` | 0 |
| `python -m pytest tests/test_dataset_contract.py tests/test_clean_sentence_pool.py tests/test_atomic_verifier.py tests/test_real_error_sources.py tests/test_hard_negative_generation.py tests/test_stress_generation.py -q` | 0 |
| `python -m pytest tests/test_rule_capabilities.py tests/test_rule_activation_policy.py tests/test_candidate_dataset_preflight.py tests/test_candidate_opportunity_pre_generation_readiness.py -q` | 0 |
| `python -m pytest tests/test_training_dataset_build.py tests/test_training_dataset_candidate_recall.py tests/test_training_dataset_rule_quotas.py tests/test_training_dataset_quality.py -q` | 0 |
| `python -m pytest tests/test_training_features.py tests/test_model_training.py tests/test_training_trainer.py tests/test_loss_weights.py -q` | 0 |

## Reports To Inspect

- reports\dataset_build\dataset_generation_report.md
- reports\dataset_build\layer_target_report.csv
- reports\dataset_build\under_quota_active_rules_report.csv
- reports\dataset_build\active_rule_quota_report.csv
- reports\dataset_build\candidate_recall_gate_report.csv
- reports\dataset_build\extended_quality_audit.csv
- reports\dataset_build\clean_hard_balance_audit.csv
- reports\dataset_build\known_quality_bugs_report.md
- reports\dataset_build\final_validation_summary.json
