# Pre-Generation Readiness Report

Date: 2026-05-24

## Status

- final_status: READY_TO_GENERATE_FINAL_DATASET
- preflight_status: pass
- stale_artifact_status: cleaned_then_fresh
- non_blocking_warnings: clean_pool_contamination_detected in sampled clean pool
- remaining_blockers: none

## Activation Counts

- production_ready_rule_count: 12
- training_candidate_rule_count: 76
- target_training_candidate_rule_count: 69
- final_active_rule_count: 76
- exact_active_rule_count_12_required: no

## Tiny Candidate Opportunity Build

- verdict: READY_FOR_TRAINING_DATASET
- dataset_contract: candidate_opportunity
- tiny_final_active_rule_count: 2
- tiny_production_ready_rule_count: 1
- tiny_training_candidate_rule_count: 2
- tiny_target_training_candidate_rule_count: 2
- tiny_layer_counts: atomic_positive=4, atomic_hard_negative=3, clean_identity=3, real_atomic=1, stress_multi_error=1
- syntax_synthetic_atomic_positive_count: 2
- syntax_synthetic_atomic_positive_count_by_rule: {"comma_subordinate": 2}
- syntax_synthetic_hard_negative_count_by_rule: {"comma_subordinate": 1}
- quota_gate_status: pass; under_quota_rule_ids=[], rules_under_hard_negative_min=[]
- report_freshness: fresh; report_manifest_errors=[]
- stale_report_handling: pass; mutated `candidate_recall_by_rule.csv` blocks `force=False` rebuild with `stale_reports_hash_mismatch:candidate_recall_by_rule.csv`

## Command Results

- `python scripts/preflight_candidate_dataset.py --config configs/config.yaml --clean-stale-artifacts`: exit 0
- `python scripts/preflight_candidate_dataset.py --config configs/config.yaml --json`: exit 0
- `python scripts/setup_data_sources.py --config configs/config.yaml --dry-run --processed-dir <temp-dir> --report-dir <temp-dir> --allow-partial`: exit 0
- `python scripts/setup_data_sources.py --help`: exit 0
- `python scripts/build_dataset.py --help`: exit 0
- `python scripts/preflight_candidate_dataset.py --help`: exit 0
- `python -m compileall src scripts tests`: exit 0
- `python -m pytest tests/test_dataset_contract.py tests/test_clean_sentence_pool.py tests/test_atomic_verifier.py tests/test_real_error_sources.py tests/test_hard_negative_generation.py tests/test_stress_generation.py -q`: exit 0, 47 passed
- `python -m pytest tests/test_rule_capabilities.py tests/test_rule_activation_policy.py tests/test_candidate_dataset_preflight.py tests/test_candidate_opportunity_pre_generation_readiness.py -q`: exit 0, 29 passed
- `python -m pytest tests/test_training_dataset_build.py tests/test_training_dataset_candidate_recall.py tests/test_training_dataset_rule_quotas.py tests/test_training_dataset_quality.py -q`: exit 0, 37 passed
- `python -m pytest tests/test_training_features.py tests/test_model_training.py tests/test_training_trainer.py tests/test_loss_weights.py -q`: exit 0, 30 passed

## Notes

- No 200k dataset build was run.
- No model training command was run.
- `legacy_builder: false` remains explicit in the tiny readiness config.
- Top-level `data.composition` was verified to drive canonical layer targets despite conflicting legacy `source_type_targets`.
