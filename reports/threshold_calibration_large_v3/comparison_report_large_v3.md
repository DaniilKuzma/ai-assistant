# Large Threshold Calibration v3

- Dataset: `data/processed/short_dataset_v2/correction_dataset.csv.gz`
- Calibration split: `val`
- Holdout split: `test`
- Output config: `configs/config_threshold_calibrated_gui_spelling_v3.yaml`

## Profile Summary

| profile:split | rows | dirty exact | dirty improved | dirty worse | clean overcorr | mean distance delta |
|---|---:|---:|---:|---:|---:|---:|
| baseline:val | 5000 | 0.3189 | 0.3216 | 0.0003 | 0.0000 | 0.2780 |
| baseline:test | 5000 | 0.2886 | 0.2932 | 0.0008 | 0.0015 | 0.2476 |
| baseline:all | 10000 | 0.3038 | 0.3074 | 0.0005 | 0.0008 | 0.2628 |
| large_v3:val | 5000 | 0.3946 | 0.4024 | 0.0003 | 0.0031 | 0.3556 |
| large_v3:test | 5000 | 0.3857 | 0.3908 | 0.0005 | 0.0062 | 0.3414 |
| large_v3:all | 10000 | 0.3901 | 0.3966 | 0.0004 | 0.0046 | 0.3485 |
| seed_v2:val | 5000 | 0.3603 | 0.3678 | 0.0003 | 0.0031 | 0.3122 |
| seed_v2:test | 5000 | 0.3478 | 0.3527 | 0.0005 | 0.0046 | 0.2912 |
| seed_v2:all | 10000 | 0.3541 | 0.3603 | 0.0004 | 0.0038 | 0.3017 |

## Changed Thresholds

| threshold | seed_v2 | large_v3 |
|---|---:|---:|
| colon_threshold | 0.920000 | 0.995000 |
| comma_conjunction_threshold | 0.995000 | 0.980000 |
| comma_subordinate_threshold | 0.920000 | 0.970000 |
| comma_threshold | 0.900000 | 0.940000 |
| cy_exception_threshold | 0.580000 | 0.570000 |
| dash_threshold | 0.940000 | 0.999000 |
| default_threshold | 0.900000 | 0.640000 |
| double_consonant_candidate_threshold | 0.450000 | 0.330000 |
| final_punctuation_threshold | 0.950000 | 0.930000 |
| frequent_error_exact_threshold | 0.180000 | 0.150000 |
| hyphen_particles_threshold | 0.500000 | 0.520000 |
| introductory_comma_threshold | 0.970000 | 0.950000 |
| n_nn_deverbal_adjective_threshold | 0.620000 | 0.590000 |
| swapped_letters_candidate_threshold | 0.840000 | 0.760000 |

## Search Evidence

| threshold | current | tuned | val decisions | positives | fp | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| colon_threshold | 0.920000 | 0.995000 | 160 | 160 | 0 | 1.0000 | 1.0000 |
| comma_conjunction_threshold | 0.995000 | 0.980000 | 208 | 200 | 0 | 1.0000 | 0.9900 |
| comma_subordinate_threshold | 0.920000 | 0.970000 | 239 | 168 | 0 | 1.0000 | 0.9405 |
| comma_threshold | 0.900000 | 0.940000 | 426 | 426 | 0 | 1.0000 | 1.0000 |
| cy_exception_threshold | 0.580000 | 0.570000 | 117 | 117 | 0 | 1.0000 | 1.0000 |
| dash_threshold | 0.940000 | 0.999000 | 95 | 95 | 0 | 1.0000 | 1.0000 |
| default_threshold | 0.900000 | 0.640000 | 756 | 689 | 0 | 1.0000 | 0.9260 |
| double_consonant_candidate_threshold | 0.450000 | 0.330000 | 88 | 80 | 6 | 0.9268 | 0.9500 |
| final_punctuation_threshold | 0.950000 | 0.930000 | 11 | 5 | 0 | 1.0000 | 0.2000 |
| frequent_error_exact_threshold | 0.180000 | 0.150000 | 9 | 9 | 0 | 1.0000 | 1.0000 |
| hyphen_particles_threshold | 0.500000 | 0.520000 | 80 | 80 | 0 | 1.0000 | 1.0000 |
| introductory_comma_threshold | 0.970000 | 0.950000 | 278 | 276 | 0 | 1.0000 | 0.9710 |
| n_nn_deverbal_adjective_threshold | 0.620000 | 0.590000 | 223 | 221 | 2 | 0.9910 | 1.0000 |
| swapped_letters_candidate_threshold | 0.840000 | 0.760000 | 101 | 89 | 0 | 1.0000 | 0.9775 |
