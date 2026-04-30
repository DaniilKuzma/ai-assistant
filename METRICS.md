# Метрики HybridCorrector

Оценка выполняется через:

```bash
PYTHONPATH=src .venv/bin/python src/evaluate.py
```

Основной evaluator использует `HybridCorrector` в inference-режиме и сравнивает
`predicted_text` с `correct_text`.

## Файлы Отчетов

```text
report/summary.json
report/error_analysis.csv
report/error_metrics_by_type.csv
report/worst_cases.csv
```

## Основные Метрики

`exact_match` - полное совпадение исправленного текста с эталоном.

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

`punct_input_similarity` и `punct_pred_similarity` - сходство
последовательностей пунктуации до и после модели.

`punct_count_error` - ошибка по количеству знаков препинания.

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
