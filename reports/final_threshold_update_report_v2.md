# Final Threshold Update Report v2

## Verdict

- final_verdict: `TEST_REGRESSION_V2`
- config_updated: `no`
- current_config_threshold_mode: `conservative`
- reason: val v2 passed, but the single permitted test check failed `clean_overcorrection_rate <= 0.007`.
- no training, dataset rebuild, LoRA, max_candidates, max_sequence_length, or balanced-mode changes were made.

## Val v1 vs v2

| metric | calibrated_val_guarded | calibrated_val_guarded_v2 | delta |
|---|---:|---:|---:|
| exact_match | 0.538400 | 0.496000 | -0.042400 |
| edit_precision | 0.995355 | 1.000000 | +0.004645 |
| spelling_f1 | 0.296396 | 0.226508 | -0.069888 |
| punctuation_f1 | 0.764730 | 0.763533 | -0.001197 |
| clean_overcorrection_rate | 0.004615 | 0.000000 | -0.004615 |
| dirty_worse_rate | 0.000270 | 0.000000 | -0.000270 |
| real_dirty_worse_rate | 0.011765 | 0.000000 | -0.011765 |

## Test v1 vs v2

| metric | calibrated_val_guarded | calibrated_val_guarded_v2 | delta |
|---|---:|---:|---:|
| exact_match | 0.517800 | 0.471600 | -0.046200 |
| edit_precision | 0.984911 | 0.989203 | +0.004292 |
| spelling_f1 | 0.328484 | 0.286945 | -0.041539 |
| punctuation_f1 | 0.680376 | 0.677723 | -0.002653 |
| clean_overcorrection_rate | 0.013846 | 0.007692 | -0.006154 |
| dirty_worse_rate | 0.001622 | 0.001351 | -0.000270 |
| real_dirty_worse_rate | 0.070588 | 0.058824 | -0.011765 |

## Val v2 Gates

| gate | value | limit | status |
|---|---:|---:|---|
| clean_overcorrection_rate | 0.000000 | <= 0.005 | PASS |
| dirty_worse_rate | 0.000000 | <= 0.003 | PASS |
| edit_precision | 1.000000 | >= 0.9 | PASS |
| punctuation_f1 | 0.763533 | >= 0.7 | PASS |
| spelling_f1 | 0.226508 | >= 0.15 | PASS |
| real_dirty_worse_rate | 0.000000 | <= 0.05 | PASS |

## Test v2 Gates

| gate | value | limit | status |
|---|---:|---:|---|
| clean_overcorrection_rate | 0.007692 | <= 0.007 | FAIL |
| dirty_worse_rate | 0.001351 | <= 0.005 | PASS |
| edit_precision | 0.989203 | >= 0.9 | PASS |
| punctuation_f1 | 0.677723 | >= 0.65 | PASS |
| spelling_f1 | 0.286945 | > 0.05 | PASS |
| real_dirty_worse_rate | 0.058824 | <= 0.07 | PASS |

## Original Test Regression by Rule

| rule_id | clean_overcorrection | dirty_worse | real_dirty_worse | accepted | recommendation |
|---|---:|---:|---:|---:|---|
| dictionary_fuzzy | 12 | 0 | 0 | 45 | raise threshold candidate during v2 selection |
| double_consonant_candidate | 2 | 1 | 1 | 56 | raise threshold candidate during v2 selection |
| swapped_letters_candidate | 2 | 0 | 0 | 114 | raise threshold candidate during v2 selection |
| final_punctuation_default | 0 | 4 | 4 | 13 | safety sentinel: inspect before changing |
| hyphen_koe_koy | 0 | 1 | 1 | 152 | safety sentinel: inspect before changing |
| homogeneous_comma | 0 | 1 | 1 | 83 | safety sentinel: inspect before changing |
| unattributed_dirty_worse | 0 | 1 | 1 | 0 | inspect partial-correction metric artifact or rejected gold edit |
| comma_subordinate | 0 | 0 | 0 | 122 | monitor |

## v2 Test Regression by Rule

| rule_id | clean_overcorrection | dirty_worse | real_dirty_worse | accepted | recommendation |
|---|---:|---:|---:|---:|---|
| dictionary_fuzzy | 6 | 0 | 0 | 9 | raise threshold candidate during v2 selection |
| swapped_letters_candidate | 2 | 0 | 0 | 2 | raise threshold candidate during v2 selection |
| final_punctuation_default | 0 | 3 | 3 | 7 | safety sentinel: inspect before changing |
| hyphen_koe_koy | 0 | 1 | 1 | 152 | safety sentinel: inspect before changing |
| homogeneous_comma | 0 | 1 | 1 | 83 | safety sentinel: inspect before changing |
| unattributed_dirty_worse | 0 | 1 | 1 | 0 | inspect partial-correction metric artifact or rejected gold edit |
| comma_subordinate | 0 | 0 | 0 | 122 | monitor |
| hyphen_particles | 0 | 0 | 0 | 112 | monitor |

## Rules Raised or Rolled Back

| threshold_key | v1 | v2 | note |
|---|---:|---:|---|
| dictionary_fuzzy_threshold | 0.750000 | 0.950000 | rolled back to conservative; still the largest clean FP source on test v2 |
| double_consonant_candidate_threshold | 0.553223 | 0.840000 | raised from val-calibrated value; v2 removed its observed test clean/real dirty-worse impact |
| swapped_letters_candidate_threshold | 0.760000 | 0.840000 | raised, but still leaves clean FP on `приемлет -> примелет` |
| hyphen_po_adverbs_threshold | 0.413330 | 0.880000 | raised; no direct test safety impact in v1/v2 reports |
| final_punctuation_threshold | 0.880000 | 0.950000 | sentinel raised; real dirty-worse reduced but not eliminated |

## Lowered Rules Kept

| threshold_key | v2 | note |
|---|---:|---|
| subject_predicate_dash_threshold | 0.990000 | kept from guarded profile pending separate rule-level safety work |
| hyphen_particles_threshold | 0.500000 | kept from guarded profile pending separate rule-level safety work |
| hyphen_koe_koy_threshold | 0.450000 | kept from guarded profile pending separate rule-level safety work |
| cy_exception_threshold | 0.580000 | kept from guarded profile pending separate rule-level safety work |

## Strict Rules Kept

- context_pair_threshold: 0.995000
- context_za_to_threshold: not present in the current profile; no relaxed v2 override added
- direct_speech_quotes_threshold: 0.995000
- enumeration_colon_threshold: not present in the current profile; no relaxed v2 override added
- explanation_colon_threshold: not present in the current profile; no relaxed v2 override added
- n_nn_deverbal_adjective_threshold: not present in the current profile; no relaxed v2 override added

## Remaining Weak Areas

- Clean overcorrection remains above the test gate by one clean row rate unit: `0.007692` vs `0.007`.
- v2 clean FP causes are `dictionary_fuzzy` and `swapped_letters_candidate`; examples include named/entity-like or valid lexical forms such as `заминировании -> ламинировании`, `Щеголев -> Щеголяв`, `отлаживанию -> отваживанию`, `приемлет -> примелет`.
- Real dirty-worse is below the configured test gate after v2, but remaining cases are mostly partial-correction interactions: `final_punctuation_default`, `homogeneous_comma`, `hyphen_koe_koy`, and one row-level unattributed case.
- `dictionary_fuzzy_threshold=0.95` is not enough by itself because accepted clean FPs score above 0.95; this needs an additional val-side safety discriminator or stricter candidate-family handling, not test tuning.

## Artifacts

- v1 test regression: `reports/eval_test_calibrated_short_v2/test_regression_by_rule.csv`
- v1 test regression examples: `reports/eval_test_calibrated_short_v2/test_regression_examples_by_rule.csv`
- v2 test regression: `reports/eval_test_calibrated_v2_short_v2/test_regression_by_rule.csv`
- v2 test regression examples: `reports/eval_test_calibrated_v2_short_v2/test_regression_examples_by_rule.csv`
- safety set report: `reports/threshold_safety_set_report.md`
- v2 decision report: `reports/threshold_safety_v2/threshold_safety_decision_v2.csv`
- v2 thresholds: `reports/threshold_safety_v2/recommended_thresholds_v2.yaml`

## Test Commands

- `.venv/bin/python -m pytest -q tests/test_strict_validator.py` -> 69 passed
- `.venv/bin/python -m pytest -q tests/test_negative_rule_suites.py` -> 133 passed
- `.venv/bin/python -m pytest -q tests/test_threshold_calibration.py` -> 4 passed
- `.venv/bin/python -m pytest -q tests/test_model_corrector.py` -> 32 passed
- `.venv/bin/python -m pytest -q tests/test_eval_threshold_profile.py` -> 4 passed
- `.venv/bin/python -m pytest -q tests/test_threshold_safety_v2.py` -> 2 passed
