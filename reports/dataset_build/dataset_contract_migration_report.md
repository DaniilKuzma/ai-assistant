# Candidate Opportunity Dataset Contract Migration Report

## Summary

`candidate_opportunity` is the canonical training dataset contract. It is layered, candidate-aware, and atomic-first: training rows are not a generic mix of source/target pairs, and only verified one-edit positives count toward active rule quotas.

## Row Semantics

- `atomic_positive`: one verified edit generated from a clean-corpus opportunity. The row must have candidate coverage, exactly one gold edit, a known rule id, and `StrictValidator` acceptance. Only these rows count toward rule quota.
- `atomic_hard_negative`: a clean identity row that exposes a candidate opportunity for an active rule and teaches the model not to overcorrect. It has zero gold edits and does not count toward positive quota.
- `real_atomic`: a real source/target pair admitted to train only when it has one known-rule edit, candidate coverage, and strict-validator acceptance.
- `stress_multi_error`: a multi-edit synthetic or real row for robustness/evaluation pressure. It uses the configured lowered `loss_weight` and never counts toward rule quota.
- `real_mining` and `real_holdout`: side outputs for unknown-rule mining and evaluation/holdout routing; they are not canonical train layers.

## Real Pair Routing

Real pairs are atomized before training use. Single-edit known-rule pairs can enter `real_atomic`; unknown-rule rows go to mining; multi-edit known-rule rows go to stress or evaluation only. Multi-edit real or synthetic rows must not be used to satisfy active rule quotas.

## Hard Gates

Dataset readiness is blocked by atomic positives with extra or non-one edits, unknown train rules, mixed-script or confusable clean/hard rows, failed real-pair atomization, low active-rule candidate recall, and stale or hash-mismatched reports. The manifest records `verdict`, `audit_errors`, `dataset_hash`, `config_hash`, and `layer_counts`.

## Source Of Truth Reports

- `data/processed/dataset_manifest.json`: canonical dataset verdict, hashes, layer counts, audit errors, and summaries.
- `reports/dataset_build/report_manifest.json`: freshness manifest for generated reports, keyed by `dataset_hash` and `config_hash`.
- `reports/dataset_build/active_rule_quota_report.csv`: active rule quota status based only on atomic positives plus hard-negative support.
- `reports/dataset_build/candidate_recall_by_rule.csv` and `candidate_recall_gate_report.csv`: candidate recall computed with the real generator, not trusted metadata.
- `reports/dataset_build/atomic_purity_report.csv`, `extra_edit_report.csv`, `unknown_rule_train_report.csv`, `mixed_script_clean_report.csv`, and `real_pair_atomization_report.csv`: hard-gate quality reports.
