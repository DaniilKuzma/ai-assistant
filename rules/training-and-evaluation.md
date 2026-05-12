# Training And Evaluation

This file owns the executable training/evaluation commands, report locations,
notebook scenario and V10.5 acceptance targets. Metric definitions live in
`rules/evaluation-metrics.md`.

## Training

```bash
PYTHONPATH=src .venv/bin/python src/train.py \
  --dataset data/processed/dataset.csv \
  --output-dir models \
  --model-version 10 \
  --max-length 128 \
  --max-vocab-size 80000 \
  --candidate-min-freq 1 \
  --candidate-max-distance 1 \
  --candidate-top-k 16 \
  --long-oov-max-distance 2 \
  --long-oov-min-length 8 \
  --min-dictionary-score 0.25 \
  --clean-action-keep-weight 2.0 \
  --dirty-action-keep-weight 0.8 \
  --action-change-weight 4.0 \
  --punct-change-weight 8.0 \
  --final-punct-weight 8.0 \
  --clean-punct-keep-weight 4.0 \
  --dirty-punct-keep-weight 1.0 \
  --context-model-name DeepPavlov/rubert-base-cased \
  --context-device auto \
  --d-model 128 \
  --num-layers 2 \
  --ff-dim 256 \
  --dropout 0.30 \
  --learning-rate 1e-4 \
  --batch-size 64 \
  --epochs 30
```

Training writes:

```text
models/hybrid_corrector.keras
models/hybrid_preprocessor.pkl
models/candidate_generator.pkl
models/hybrid_config.json
models/hybrid_training_log.csv
```

## Smoke Evaluation

```bash
PYTHONPATH=src .venv/bin/python src/evaluate.py \
  --dataset data/processed/test.csv \
  --sample-size 1000 \
  --output-dir report/synthetic_v11 \
  --strictness strict \
  --punctuation-mode conservative \
  --candidate-top-k 16 \
  --context-device auto \
  --min-dictionary-score 0.25 \
  --no-diagnostics \
  --inference-batch-size 512
```

Smoke evaluation writes:

```text
report/synthetic_v11/summary.json
report/synthetic_v11/error_analysis.csv
report/synthetic_v11/error_metrics_by_type.csv
report/synthetic_v11/worst_cases.csv
```

Run without `--no-diagnostics` to write larger diagnostic CSV files:

```text
report/synthetic_v11/word_edit_diagnostics.csv
report/synthetic_v11/word_decision_diagnostics.csv
report/synthetic_v11/punct_edit_diagnostics.csv
```

Training-time clean audits are written under `report/`:

```text
report/clean_candidate_audit.csv
report/clean_noise_filter.csv
```

## External Evaluation

External real-pair reports are separate from the synthetic-compatible test:

```text
report/external_ai_forever_v11/summary.json
report/external_ruspellgold_v11/summary.json
```

Run full external evaluation only after the synthetic smoke report preserves
clean-safety and does not increase harmful edits.

## Notebook Scenario

The main experimental scenario is `notebook.ipynb`.

Current V10/V10.5 defaults:

```text
MODEL_VERSION: 10
TRAIN_CLEAN_RATIO: 0.35
EVAL_CLEAN_RATIO: 0.35
SAMPLES_PER_TEXT: 4
REAL_TRAIN_REPEAT: 6
ACTION_CHANGE_WEIGHT: 4.0
PUNCT_CHANGE_WEIGHT: 8.0
FINAL_PUNCT_WEIGHT: 8.0
CLEAN_PUNCT_KEEP_WEIGHT: 4.0
DIRTY_PUNCT_KEEP_WEIGHT: 1.0
EVAL_SAMPLE_SIZE: 1000
RUN_EXTERNAL_EVAL: False
WRITE_DIAGNOSTICS: False
INFERENCE_BATCH_SIZE: 512
EVAL_OUTPUT_DIR: report/synthetic_v11
CONTEXT_DEVICE: auto
MIN_DICTIONARY_SCORE: 0.25
USE_ENTITY_GUARD: True
```

The notebook can rebuild synthetic+real labels, train the hybrid model, plot
loss/accuracy, run synthetic/external evaluation and inspect worst cases.

## Acceptance Targets

After a full V10.5 evaluation:

```text
clean_overcorrection_rate <= 0.002
worse_rate <= 0.01
overall exact_match above 0.5699
dirty exact_match above 0.3398
punctuation-only unchanged_wrong_rate below the previous level
punct_applied_count above the current 856 level
punct_target_applied_count grows
punct_target_predicted_but_blocked_rate falls
safe_comma_delete_recovery_count grows
safe_service_comma_delete_recovery_count > 0
safe_date_comma_delete_recovery_count > 0
safe_final_period_recovery_count >= 74
target_present_but_not_applied_rate falls
low_action_confidence_or_margin noticeably falls
source_kind_slices and source_dataset_slices are present in summary.json
```
