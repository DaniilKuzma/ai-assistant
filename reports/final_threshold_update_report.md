# Final Threshold Update Report canonical

## Verdict

READY_FOR_WORKING_V1

## Scope

- No retraining was run.
- No dataset rebuild was run.
- LoRA, max_candidates, max_sequence_length, dataset paths, and training parameters were not changed.
- canonical keeps the core threshold values and relies on stricter lexical validation guards.

## Guard Changes

- `dictionary_fuzzy` now rejects protected/acronym/all-caps/mixed-script/mixed-case technical/proper-like source tokens.
- `dictionary_fuzzy` now rejects known or probable-clean source rewrites into known replacements when the edit is lexically risky.
- `dictionary_fuzzy` now rejects morphology/POS-incompatible replacements and long same-length one-substitution rewrites with plausible clean morphology.
- `swapped_letters_candidate` now requires an actual adjacent swap, unknown/suspicious source, known replacement, and no protected/proper/acronym source.
- `swapped_letters_candidate` now rejects known/probable-clean source rewrites and morphology/POS-incompatible swaps.

## Clean FP Examples Blocked

Audit: `reports/eval_test_calibrated_core_short_core/clean_fp_dictionary_swapped_audit.csv`

- `dictionary_fuzzy`: `заминировании -> ламинировании` rejected by `known_source_lexical_guard`.
- `dictionary_fuzzy`: `отлаживанию -> отваживанию` rejected by `known_source_lexical_guard`.
- `dictionary_fuzzy`: `Щеголев -> Щеголяв` rejected by `protected_lexical_guard`.
- `swapped_letters_candidate`: `приемлет -> примелет` rejected by `known_source_lexical_guard`.

## Threshold Profile

Profile: `calibrated_guarded`

- `dictionary_fuzzy_threshold`: 0.95
- `double_consonant_candidate_threshold`: 0.84
- `swapped_letters_candidate_threshold`: 0.84
- `hyphen_po_adverbs_threshold`: 0.88
- `final_punctuation_threshold`: 0.95
- `hyphen_particles_threshold`: 0.50
- `hyphen_koe_koy_threshold`: 0.45
- `cy_exception_threshold`: 0.58
- `subject_predicate_dash_threshold`: 0.99

## Validation Metrics

Report dir: `reports/eval_val_calibrated_canonical_short_core`

- `exact_match`: 0.4960
- `edit_precision`: 1.0000
- `spelling_f1`: 0.2290
- `punctuation_f1`: 0.7635
- `clean_overcorrection_rate`: 0.0000
- `dirty_worse_rate`: 0.0000
- `real_dirty_worse_rate`: 0.0000

## Test Metrics

Report dir: `reports/eval_test_calibrated_canonical_short_core`

- `exact_match`: 0.4732
- `edit_precision`: 0.9958
- `spelling_f1`: 0.2869
- `punctuation_f1`: 0.6777
- `clean_overcorrection_rate`: 0.001538
- `dirty_worse_rate`: 0.001351
- `real_dirty_worse_rate`: 0.058824

## Config Update

- Config updated: yes
- Backup: `configs/config.yaml.backup_before_threshold_update_canonical_20260520_214323`
- Active mode: `calibrated_guarded`
