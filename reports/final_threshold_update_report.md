# Final Threshold Update Report

## A. Input

- selected recommended thresholds: `reports/short_dataset_v2/threshold_calibration/latest_recommended_thresholds.yaml`
- selected calibration manifest: `reports/short_dataset_v2/threshold_calibration/latest_manifest.json`
- calibration timestamp: `20260520_173850`
- calibration verdict: `READY_FOR_THRESHOLD_UPDATE`
- config path: `configs/config.yaml`
- dataset path: `data/processed/short_dataset_v2/correction_dataset.csv.gz`
- dataset manifest: `reports/short_dataset_v2/dataset_manifest.json`
- adapter path: `models/adapters/latest`
- heads path: `models/heads/latest`
- threshold profile: `calibrated_val_guarded`
- val config: `/tmp/config.eval_val_calibrated.yaml`
- test config: `/tmp/config.eval_test_calibrated.yaml`

## B. Val Results

| metric | value |
|---|---:|
| `exact_match` | 0.538400 |
| `edit_precision` | 0.995355 |
| `edit_recall` | 0.321612 |
| `edit_f1` | 0.486145 |
| `spelling_precision` | 1.000000 |
| `spelling_recall` | 0.173982 |
| `spelling_f1` | 0.296396 |
| `punctuation_precision` | 0.998938 |
| `punctuation_recall` | 0.619487 |
| `punctuation_f1` | 0.764730 |
| `clean_exact_match` | 0.995385 |
| `clean_overcorrection_rate` | 0.004615 |
| `dirty_worse_rate` | 0.000270 |
| `real_edit_f1` | 0.155844 |
| `real_spelling_f1` | 0.233010 |
| `real_punctuation_f1` | 0.057554 |
| `real_dirty_worse_rate` | 0.011765 |

Metadata: evaluation_backend=`existing_checkpoint`, threshold_profile_used=`calibrated_val_guarded`, fallback_used=`False`.

## C. Test Results

| metric | value |
|---|---:|
| `exact_match` | 0.517800 |
| `edit_precision` | 0.984911 |
| `edit_recall` | 0.301364 |
| `edit_f1` | 0.461514 |
| `spelling_precision` | 0.998035 |
| `spelling_recall` | 0.196594 |
| `spelling_f1` | 0.328484 |
| `punctuation_precision` | 0.994771 |
| `punctuation_recall` | 0.516984 |
| `punctuation_f1` | 0.680376 |
| `clean_exact_match` | 0.986154 |
| `clean_overcorrection_rate` | 0.013846 |
| `dirty_worse_rate` | 0.001622 |
| `real_edit_f1` | 0.196970 |
| `real_spelling_f1` | 0.289474 |
| `real_punctuation_f1` | 0.211382 |
| `real_dirty_worse_rate` | 0.070588 |

Metadata: evaluation_backend=`existing_checkpoint`, threshold_profile_used=`calibrated_val_guarded`, fallback_used=`False`.

## D. Rule-Level Highlights

- top improved rule on val vs previous short_dataset_v2 report: `subject_predicate_dash` with true_positive delta=104.000000, f1 delta=1.000000.

Rules still at predicted_count=0 on test:
- `context_nesmotrya`: gold=403, predicted=0, group=solid_hyphen_separate_service_words
- `context_za_to`: gold=397, predicted=0, group=solid_hyphen_separate_service_words
- `context_vsledstvie`: gold=287, predicted=0, group=solid_hyphen_separate_service_words
- `context_chto_by`: gold=237, predicted=0, group=solid_hyphen_separate_service_words
- `ne_adverb`: gold=232, predicted=0, group=solid_hyphen_separate_ne
- `ne_participle`: gold=225, predicted=0, group=solid_hyphen_separate_ne
- `ne_adjective`: gold=174, predicted=0, group=solid_hyphen_separate_ne
- `context_to_zhe`: gold=148, predicted=0, group=solid_hyphen_separate_service_words
- `direct_speech_colon`: gold=135, predicted=0, group=direct_speech_quotes
- `direct_speech_quotes`: gold=135, predicted=0, group=direct_speech_quotes
- `explanation_colon`: gold=124, predicted=0, group=complex_sentence_asyndetic
- `ni_stable_expression`: gold=120, predicted=0, group=ne_ni_particles

Risky rules with false positives on test:
- `dictionary_fuzzy`: false_positive=45, predicted=45, precision=0.000000
- `final_punctuation_default`: false_positive=13, predicted=13, precision=0.000000
- `comma_subordinate`: false_positive=3, predicted=122, precision=0.975410
- `double_consonant_candidate`: false_positive=3, predicted=56, precision=0.946429
- `swapped_letters_candidate`: false_positive=2, predicted=114, precision=0.982456
- `hyphen_koe_koy`: false_positive=1, predicted=152, precision=0.993421
- `homogeneous_comma`: false_positive=1, predicted=83, precision=0.987952
- `capitalization_sentence_start`: false_positive=1, predicted=1, precision=0.000000

Rules with high rejected_by_validator on test:
- `dictionary_fuzzy`: rejected_by_validator=130, threshold_rejects=735, reasons={"conflict": 19, "morphology_agreement_guard": 240, "threshold": 735, "unsafe_fuzzy_spelling_candidate": 20}
- `explanation_colon`: rejected_by_validator=124, threshold_rejects=2, reasons={"threshold": 2, "unsafe_colon_candidate": 248}
- `enumeration_colon`: rejected_by_validator=82, threshold_rejects=0, reasons={"unsafe_colon_candidate": 164}
- `prefix_pre_pri`: rejected_by_validator=11, threshold_rejects=60, reasons={"threshold": 60, "unsafe_fuzzy_spelling_candidate": 22}
- `hyphen_po_adverbs`: rejected_by_validator=10, threshold_rejects=66, reasons={"threshold": 66, "unsafe_hyphen_po_adverb": 20}
- `pattern_шо_ше`: rejected_by_validator=6, threshold_rejects=4, reasons={"known_source_lexical_guard": 12, "threshold": 4}
- `missing_hard_sign`: rejected_by_validator=3, threshold_rejects=10, reasons={"known_source_lexical_guard": 6, "threshold": 10}
- `double_consonant_candidate`: rejected_by_validator=3, threshold_rejects=88, reasons={"conflict": 1, "morphology_agreement_guard": 6, "threshold": 88}
- `pattern_цы_ци`: rejected_by_validator=2, threshold_rejects=0, reasons={"known_source_lexical_guard": 4}
- `pattern_жо_же`: rejected_by_validator=2, threshold_rejects=0, reasons={"known_source_lexical_guard": 4}
- `tsya_soft_insert`: rejected_by_validator=2, threshold_rejects=30, reasons={"dangerous_tsya": 4, "threshold": 30}
- `capitalization_sentence_start`: rejected_by_validator=1, threshold_rejects=67, reasons={"abbreviation_sentence_start_capitalization": 2, "threshold": 67}

Real punctuation status:
- val real_punctuation_f1=0.057554, real_dirty_worse_rate=0.011765
- test real_punctuation_f1=0.211382, real_dirty_worse_rate=0.070588

## E. Thresholds

Accepted threshold changes from recommendation:
- `subject_predicate_dash_threshold`: 0.99
- `hyphen_particles_threshold`: 0.5
- `hyphen_koe_koy_threshold`: 0.45
- `dictionary_fuzzy_threshold`: 0.75
- `cy_exception_threshold`: 0.58
- `swapped_letters_candidate_threshold`: 0.76
- `double_consonant_candidate_threshold`: 0.553223
- `hyphen_po_adverbs_threshold`: 0.41333

Thresholds kept strict by calibration manifest:
- `direct_speech_colon`
- `context_to_zhe`
- `ne_adverb`
- `ne_participle`
- `ne_adjective`
- `context_za_to`
- `direct_speech_quotes`
- `enumeration_colon`
- `explanation_colon`
- `n_nn_deverbal_adjective`

Thresholds not recommended and why:
- `direct_speech_colon` kept at 0.995000: no safety-safe threshold improves validation objective
- `context_to_zhe` kept at 0.950000: no safety-safe threshold improves validation objective
- `ne_adverb` kept at 0.999950: no safety-safe threshold improves validation objective
- `ne_participle` kept at 0.999950: no safety-safe threshold improves validation objective
- `ne_adjective` kept at 0.999950: no safety-safe threshold improves validation objective
- `context_za_to` kept at 0.950000: no safety-safe threshold improves validation objective
- `direct_speech_quotes` kept at 0.995000: no safety-safe threshold improves validation objective
- `enumeration_colon` kept at 0.920000: no safety-safe threshold improves validation objective
- `explanation_colon` kept at 0.920000: no safety-safe threshold improves validation objective
- `n_nn_deverbal_adjective` kept at 0.920000: no safety-safe threshold improves validation objective

Config/deployment update status:
- `configs/config.yaml` updated: no
- config backup path: n/a
- `models/adapters/latest/thresholds.json` updated: no
- reason: test safety gates failed; no test tuning performed

## F. Safety Gates

Val gates: PASSED

| gate | status |
|---|---|
| clean_overcorrection_rate <= 0.005 | PASS |
| dirty_worse_rate <= 0.003 | PASS |
| edit_precision >= 0.90 | PASS |
| punctuation_f1 >= 0.70 | PASS |
| spelling_f1 >= 0.20 | PASS |
| real_dirty_worse_rate <= 0.05 | PASS |
| clean_exact_match >= 0.99 | PASS |
| punctuation_f1 calibration regression <= 0.02 | PASS |
| clean_overcorrection <= calibration final | PASS |
| dirty_worse <= calibration final | PASS |

Test gates: FAILED

| gate | status |
|---|---|
| clean_overcorrection_rate <= 0.007 | FAIL |
| dirty_worse_rate <= 0.005 | PASS |
| edit_precision >= 0.90 | PASS |
| punctuation_f1 >= 0.65 | PASS |
| spelling_f1 > 0.05 | PASS |
| real_dirty_worse_rate <= 0.07 | FAIL |

Failing test gates:
- clean_overcorrection_rate <= 0.007
- real_dirty_worse_rate <= 0.07

Sanity tests before final answer:
- `.venv/bin/python -m pytest -q tests/test_strict_validator.py`: 69 passed
- `.venv/bin/python -m pytest -q tests/test_negative_rule_suites.py`: 133 passed
- `.venv/bin/python -m pytest -q tests/test_threshold_calibration.py`: 4 passed
- `.venv/bin/python -m pytest -q tests/test_model_corrector.py`: 32 passed

## G. Final Verdict

`TEST_REGRESSION`

Reason: val passed, but test failed `clean_overcorrection_rate <= 0.007` and `real_dirty_worse_rate <= 0.07`. Thresholds were not changed after test, and no retraining, dataset rebuild, LoRA change, max_candidates change, max_sequence_length change, epochs/lr change, or test tuning was performed.
