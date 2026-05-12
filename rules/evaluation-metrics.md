# Evaluation Metrics

Evaluation runs through:

```bash
PYTHONPATH=src .venv/bin/python src/evaluate.py
```

The evaluator uses `HybridCorrector` in inference mode and compares
`predicted_text` with `correct_text`.

Report paths and acceptance targets live in `rules/training-and-evaluation.md`.

## Core Metrics

`exact_match` - full match between corrected text and reference text. This is a
strict metric: if a string is partially fixed but still has one error,
`exact_match` remains `False`.

`input_cer` - CER between erroneous input and reference text.

`pred_cer` - CER between prediction and reference text.

`cer_delta = input_cer - pred_cer`.

```text
cer_delta > 0  ассистент улучшил текст
cer_delta = 0  качество не изменилось
cer_delta < 0  ассистент сделал хуже
```

`improved_rate` - share of examples where `pred_cer < input_cer`.

`worse_rate` - share of examples where `pred_cer > input_cer`. This is one of
the most important safety metrics: the assistant must rarely make text worse.

`unchanged_wrong_rate` - share of erroneous examples left unchanged.

`clean_overcorrection_rate` - share of clean examples changed unnecessarily.

`word_overcorrection_rate`, `punct_overcorrection_rate` - split
clean-overcorrection by edit type.

`punct_input_similarity` and `punct_pred_similarity` - punctuation sequence
similarity before and after correction.

`punct_count_error` / `mean_punct_count_error` - punctuation count error.

`confidence` - average runtime confidence.

`accepted` and `accepted_rate` - share of results accepted by `QualityGuard`.

`guard_reason` - reason for accepting or rejecting the result.

## Error-Type Metrics

`error_metrics_by_type.csv` expands `error_types`. Both `|` and `,` separators
are supported so old and new datasets remain readable.

For each type, the evaluator writes summary metrics such as:

```text
examples
exact_match
mean_input_cer
mean_pred_cer
mean_cer_delta
improved_rate
worse_rate
clean_overcorrection_rate
punct_pred_similarity
accepted_rate
```

## V10.5 Diagnostics

`summary.json` should include:

```text
runtime_version: 10.5
write_diagnostics
source_kind_slices
source_dataset_slices
low_action_dictionary_recovery_count
safe_comma_delete_recovery_count
safe_service_comma_delete_recovery_count
safe_date_comma_delete_recovery_count
safe_final_period_recovery_count
punct_target_change_count
punct_target_predicted_count
punct_target_applied_count
punct_target_predicted_but_blocked_count
punct_target_predicted_but_blocked_rate
```

`punct_target_change_count`, `punct_target_predicted_count`,
`punct_target_applied_count` and
`punct_target_predicted_but_blocked_rate` diagnose punctuation targets: how many
marks needed to change, how many the model predicted correctly, how many were
actually applied and what share of correct target predictions was blocked by
runtime guards.

`low_action_dictionary_recovery_count` counts safe top-1 dictionary candidates
that passed narrow recovery instead of staying under
`low_action_confidence_or_margin`.

`safe_comma_delete_recovery_count` counts extra comma deletions after short
service words without relaxing the global `comma_delete` threshold.

`safe_service_comma_delete_recovery_count` and
`safe_date_comma_delete_recovery_count` split lowercase service-word recovery
and day-month date cleanup, for example `11, января -> 11 января`.

`safe_final_period_recovery_count` counts final periods applied through narrow
recovery without relaxing internal punctuation thresholds.

## Reading Results

Good assistant behavior usually means:

```text
mean_pred_cer < mean_input_cer
mean_cer_delta > 0
worse_rate is low
clean_overcorrection_rate is close to 0
accepted_rate is high without worse_rate growth
punct_pred_similarity > punct_input_similarity
```

Useful diagnostic signals:

```text
target_present_but_not_applied_rate
low_action_confidence_or_margin
low_action_dictionary_recovery_count
safe_comma_delete_recovery_count
safe_service_comma_delete_recovery_count
safe_date_comma_delete_recovery_count
safe_final_period_recovery_count
punct_applied_count
punct_change_candidate_count
punct_target_applied_count
punct_target_predicted_but_blocked_rate
source_kind_slices
```

The most useful rows for manual analysis are in each report directory's
`worst_cases.csv`, for example `report/synthetic_v11/worst_cases.csv`. Often
the correct candidate is already present in the candidate list, but action
confidence or runtime gates prevent applying it. V10.5 improves safe word
recall, comma-delete recall and final-period recall; real candidate coverage and
bracket-pair recovery remain separate directions that must not weaken
clean/entity/protected guards.
