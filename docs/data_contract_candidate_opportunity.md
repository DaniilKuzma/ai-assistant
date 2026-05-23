# Candidate Opportunity Data Contract

`candidate_opportunity` is the canonical training dataset contract. It is layered, atomic-first, and candidate-aware; it is not a generic bag of sentence pairs.

## Layers

- `atomic_positive`: one verified edit from a clean corpus opportunity; counts toward active rule quota.
- `atomic_hard_negative`: clean identity row targeted at an active rule candidate opportunity; trains against overcorrection and does not count toward positive quota.
- `clean_identity`: clean sentence with no intended edit.
- `real_atomic`: real source/target pair admitted to train only when it has one known-rule edit, candidate coverage, and strict validator acceptance.
- `stress_multi_error`: multi-edit synthetic or real stress row; never counts toward rule quota and uses `loss_weight` from `data.stress.loss_weight`.
- `real_holdout` and `real_mining`: real-pair side outputs for evaluation/mining, not canonical train layers.

## Columns

Required training columns still include `source`, `target`, `split`, `source_type`, `rule_ids`, `edits`, `metadata`, `rule_id`, and `error_types`.

Contract columns are added by `src.data.dataset_contract.ensure_contract_columns`:

- `dataset_contract`: always `candidate_opportunity`.
- `dataset_layer`: one of the layers above.
- `is_atomic`, `is_stress`, `count_toward_rule_quota`.
- `loss_weight`: per-row sample weight; stress defaults to `0.4` from config.
- `gold_edit_count`, `target_rule_id`, candidate span/source/replacement fields.
- `verification_status` and `rejection_reason` for audit/debugging.

## Gates

Atomic positives must have exactly one gold edit, candidate coverage, known rule id, and strict validator acceptance. Hard negatives and clean identity rows must pass clean-text filters, including mixed-script/confusable rejection. Real pairs use `data.real_pairs`: single known-rule edits can enter `real_atomic`; unknown rules go to mining; multi-edit known-rule pairs go to stress/eval.

Canonical manifests include `dataset_hash`, `config_hash`, `verdict`, `audit_errors`, `layer_counts`, active rule quota reports, candidate recall reports, and quality audit reports. A canonical build is ready only when these gates pass; scripts print the contract summary without launching model training.
