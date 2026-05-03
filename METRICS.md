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

`word_overcorrection_rate`, `punct_overcorrection_rate`,
`space_overcorrection_rate` - разбиение clean-overcorrection по типу правки:
слово, пунктуация или deterministic spacing.

`space_edit_count` - сколько SPACE-правок появилось в строке.

`deterministic_spacing_applied_count` - сколько раз runtime применил
детерминированный слой пробелов.

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

## Текущий V8 Evaluator

Новый evaluator пишет:

```text
runtime_version: 8.1
deterministic_spacing_enabled
space_edit_count
deterministic_spacing_applied_count
space_overcorrection_rate
```

Для A/B сравнения deterministic spacing можно отключить:

```bash
PYTHONPATH=src .venv/bin/python src/evaluate.py \
  --dataset data/processed/test.csv \
  --all \
  --output-dir report/no_spacing_ab \
  --strictness strict \
  --disable-deterministic-spacing
```

## Текущий Полный Отчет V8

Файл:

```text
report/summary.json
```

Ключевые значения:

```text
runtime_version: 8.1
examples: 5913
exact_match: 0.475731
clean exact_match: 0.998555
dirty exact_match: 0.192859
mean_cer_delta: 0.003985
dirty_slice.mean_cer_delta: 0.006150
worse_rate: 0.003044
clean_overcorrection_rate: 0.001445
word_overcorrection_rate: 0.0
punct_overcorrection_rate: 0.001445
space_overcorrection_rate: 0.0
target_present_but_not_applied_rate: 0.491918
candidate_coverage_rate: 0.802004
deterministic_spacing_applied_count: 518
space_edit_count: 508
reranker_non_finite_count: 0
entity_guard_blocked_count: 1831
context_source_veto_relaxed_count: 5
punct_input_similarity: 0.887929
punct_pred_similarity: 0.892349
punct_wrong_period_to_comma_improved_rate: 0.386905
```

Как читать этот результат:

```text
clean-тексты почти не портятся;
word-overcorrection на clean отсутствует;
deterministic spacing не портит clean;
основная слабость - dirty recall, пунктуация и space_merge_words.
```

Почему `exact_match` остается низким:

```text
dirty unchanged_wrong_rate: 0.592390
punct_applied_count: 85
punct_change_candidate_count: 3835
space_merge_words exact_match: 0.057971
target_present_but_not_applied_rate: 0.491918
```

Модель часто уменьшает CER, но не исправляет все ошибки в строке. Для роста
`exact_match` нужно повышать recall, а не только снижать overcorrection.

Главный diagnostic-сигнал:

```text
target_present_but_not_applied_rate: 0.491918
```

Правильный кандидат часто есть в candidate list, но runtime-гейты или action
confidence не дают его применить. V8.1.1 первым делом закрывает вредные
glued-service-token split случаи и clean-overcorrection по reporting commas.
После smoke/full evaluation следующий отдельный этап должен снижать
`low_action_confidence_or_margin` и улучшать punctuation recall, не ослабляя
clean/entity/protected guards.
