# Метрики HybridCorrector

Оценка выполняется через:

```bash
PYTHONPATH=src .venv/bin/python src/evaluate.py
```

Основной evaluator использует `HybridCorrector` в inference-режиме и сравнивает
`predicted_text` с `correct_text`.

## Файлы Отчетов

```text
report/synthetic_v11/summary.json
report/synthetic_v11/error_analysis.csv
report/synthetic_v11/error_metrics_by_type.csv
report/synthetic_v11/worst_cases.csv
```

## Основные Метрики

`exact_match` - полное совпадение исправленного текста с эталоном.
Это строгая метрика: если строка исправлена частично, но осталась хотя бы одна
ошибка, `exact_match` все равно равен `False`.

`input_cer` - CER между ошибочным входом и правильным текстом.

`pred_cer` - CER между предсказанием и правильным текстом.

`cer_delta = input_cer - pred_cer`.

```text
cer_delta > 0  ассистент улучшил текст
cer_delta = 0  качество не изменилось
cer_delta < 0  ассистент сделал хуже
```

`improved_rate` - доля примеров, где `pred_cer < input_cer`.

`worse_rate` - доля примеров, где `pred_cer > input_cer`. Для ассистента это
одна из главных метрик: правильная система не должна часто портить текст.

`unchanged_wrong_rate` - доля ошибочных примеров, которые остались без
исправления.

`clean_overcorrection_rate` - доля чистых примеров, которые были изменены зря.

`word_overcorrection_rate`, `punct_overcorrection_rate` - разбиение
clean-overcorrection по типу правки: слово или пунктуация.

`punct_input_similarity` и `punct_pred_similarity` - сходство
последовательностей пунктуации до и после модели.

`punct_count_error` - ошибка по количеству знаков препинания.

`punct_target_change_count`, `punct_target_predicted_count`,
`punct_target_applied_count`, `punct_target_predicted_but_blocked_rate` -
диагностика пунктуационных целей: сколько знаков надо было изменить, сколько
модель предсказала правильно, сколько реально применилось и какая доля
правильно предсказанных целей была заблокирована runtime.

`confidence` - средняя уверенность runtime-системы.

`accepted` и `accepted_rate` - доля результатов, которые прошли
`QualityGuard`.

`guard_reason` - причина принятия или отклонения результата.

## Метрики По Типам Ошибок

`error_metrics_by_type.csv` разворачивает `error_types`. Разделителями
поддерживаются `|` и `,`, чтобы старые и новые датасеты читались одинаково.

Для каждого типа считаются те же summary-метрики:

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

## Как Читать Результаты

Хороший результат для ассистента:

```text
mean_pred_cer < mean_input_cer
mean_cer_delta > 0
worse_rate низкий
clean_overcorrection_rate близок к 0
accepted_rate высокий, но без роста worse_rate
punct_pred_similarity > punct_input_similarity
```

Самые полезные строки для ручного анализа лежат в:

```text
report/worst_cases.csv
```

## Текущий V10.5 Evaluator

Evaluator пишет:

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

Для быстрых итераций можно отключить большие diagnostic CSV:

```bash
PYTHONPATH=src .venv/bin/python src/evaluate.py \
  --dataset data/processed/test.csv \
  --sample-size 1000 \
  --output-dir report/synthetic_v11 \
  --strictness strict \
  --context-device auto \
  --no-diagnostics \
  --inference-batch-size 512
```

## V10.5 Reports

Основной synthetic-compatible отчет:

```text
report/synthetic_v11/summary.json
report/synthetic_v11/error_analysis.csv
report/synthetic_v11/error_metrics_by_type.csv
report/synthetic_v11/worst_cases.csv
```

Внешние real-pair проверки для отдельного полного прогона:

```text
report/external_ai_forever_v11/summary.json
report/external_ruspellgold_v11/summary.json
```

V10.5 использует V10-обучение с реальными парами только в train/val, отбрасывает spacing-only пары,
а основной `test.csv` остается сопоставимым synthetic split. Поэтому после
полного прогона нужно смотреть два типа результата:

```text
1. report/synthetic_v11/summary.json - контролирует clean-safety и сопоставимый synthetic test.
2. external_*_v9/summary.json - показывает перенос на реальные spellcheck/punctuation пары.
```

Ключевые V10.5 acceptance targets:

```text
clean_overcorrection_rate <= 0.002
worse_rate <= 0.01
overall exact_match выше 0.5699
dirty exact_match выше 0.3398
punct_applied_count растет без роста punct_overcorrection_rate
punct_target_applied_count растет
punct_target_predicted_but_blocked_rate падает
safe_comma_delete_recovery_count растет
safe_service_comma_delete_recovery_count > 0
safe_date_comma_delete_recovery_count > 0
safe_final_period_recovery_count >= 74
target_present_but_not_applied_rate снижается
low_action_confidence_or_margin заметно падает
```

V10.5 сохраняет clean-safety после расширения structural punctuation recall:
ложные comma-insert и разрушение сбалансированных скобок должны исчезнуть, а
полный external evaluation запускается только после успешного synthetic-прогона.
Новый `low_action_dictionary_recovery_count` показывает, сколько безопасных
top-1 dictionary-кандидатов прошло узкий recovery вместо раннего
`low_action_confidence_or_margin`.
Новый `safe_comma_delete_recovery_count` показывает, сколько лишних запятых
после коротких служебных слов было удалено без ослабления глобального
`comma_delete` threshold.
Новые `safe_service_comma_delete_recovery_count` и
`safe_date_comma_delete_recovery_count` отделяют расширенный lowercase-сценарий
для служебных слов и day-month даты вроде `11, января -> 11 января`.
Новый `safe_final_period_recovery_count` показывает, сколько финальных точек
runtime применил через узкий recovery без ослабления внутренних punctuation
thresholds.

Главные diagnostic-сигналы:

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

Правильный кандидат часто уже есть в candidate list, но action confidence или
runtime-гейты не дают его применить. V10.5 исправляет безопасную часть
word-recall, comma-delete recall и final-period recall; real candidate coverage и bracket-pair
recovery остаются отдельными направлениями без ослабления clean/entity/protected
guards.
