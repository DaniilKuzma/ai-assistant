# Current Pre-Dataset Capability Summary

- total taxonomy entries: 409
- metadata_only count: 87
- true leaf/actionable entries count: 322
- production_ready_now count: 12
- training_eligible_now count: 47
- training_eligible active rule_id count: 69
- INCLUDE_NOW count: 12
- INCLUDE_AFTER_VALIDATOR count: 15
- INCLUDE_AFTER_THRESHOLD_CALIBRATION count: 18
- INCLUDE_AFTER_TRAINING count: 2
- BLOCK_NO_CANDIDATE count: 5
- BLOCK_NEEDS_SYNTAX count: 122
- BLOCK_NEEDS_DICTIONARY count: 77
- BLOCK_NEEDS_NER count: 12
- BLOCK_METADATA_ONLY count: 87
- BLOCK_PLANNED count: 58
- BLOCK_DISABLED count: 1
- recommended dataset active rule_id count: 69
- recommended dataset size estimate: 6620

## Count Interpretation

The taxonomy-entry count is below 55 because several Orfogrammka entries group multiple executable rule_ids.
This is an explicit audited justification: dataset planning should use the active rule_id count, not only the taxonomy-entry count.
