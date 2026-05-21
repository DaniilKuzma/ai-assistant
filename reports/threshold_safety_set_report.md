# Threshold Safety Set Report

## Inputs

- source dataset: `data/processed/short_dataset_v2/correction_dataset.csv.gz`
- excluded split: `test`
- purpose: threshold selection safety only; no model training or dataset rebuild

## Safety Rows

- calibration_safety_train_clean: 1000
- calibration_safety_train_hard_negative: 1000
- calibration_safety_train_real: 851
- val_clean_identity: 650
- val_hard_negative: 650
- val_real: 85
- total_safety_rows: 4236

## Source Types

- calibration_safety_train_real: 851
- clean_identity_from_open_clean: 1650
- hard_negative_from_open_clean: 1650
- real_error_pair: 85

## Selection Rows

- selection rows are full val plus non-test train safety rows
- split_train: 2851
- split_val: 5000
- total_selection_rows: 7851
