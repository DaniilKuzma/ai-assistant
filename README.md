# Интеллектуальный ассистент для исправления орфографии и пунктуации

Проект реализует одну основную систему коррекции русского текста:

```text
HybridCorrector = deterministic spacing + top-k кандидаты + edit-based Transformer + source-aware punctuation head + RuBERT context reranker + quality guard
```

Модель не переписывает предложение целиком. Она предсказывает точечные действия
над токенами (`KEEP`, `DELETE`, `REPLACE_0 ... REPLACE_7`) и знак препинания после каждого токена.
За счет этого ассистент ведет себя осторожнее: действие по умолчанию - оставить
текст как есть.

Runtime работает в консервативном режиме. Отдельный deterministic spacing слой
чинит только безопасные пробельные ошибки: двойные пробелы, пробел перед
простыми знаками, пропущенный пробел после простого знака и пробелы вокруг
`—` между кириллическими словами. URL, email, decimal/date-like фрагменты,
латиница, кавычки, скобки, слэши, `№`, числа и структурная пунктуация
сохраняются. Пунктуационная голова может менять только простые знаки
`, . ? ! : ;` при высокой уверенности.
Runtime обрабатывает строку целиком или overlapping windows, поэтому ошибочная
точка внутри предложения больше не превращается в искусственный конец сегмента.
Пунктуационная голова получает на вход исходный знак после токена
(`source_punct_ids`). Текущий runtime - v8.1: он использует rank-consistent
top-k actions, строгие safe split-candidates, RuBERT/RuRoBERTa context reranking
на CPU как управляющий слой, полную диагностику word decisions,
candidate-aware KEEP training, clean-noise filter, entity/noise guard,
усиленные веса punctuation changes и отдельные action weights для clean KEEP,
dirty KEEP и CHANGE.

Словарные исправления дополнительно защищены: `min_dictionary_score=0.25`,
`candidate_min_freq=1` и морфологический guard на `pymorphy2` блокируют
опасные замены правильных словоформ вроде `летнем -> летним`.

Keyboard-опечатки намеренно не используются.

## Структура

```text
src/data_preparation.py      подготовка корпуса и synthetic augmentation
src/candidate_generator.py   словарные и правиловые кандидаты
src/context_reranker.py      RuBERT/RuRoBERTa context reranking
src/entity_guard.py          lexical/entity guard для clean/raw слов
src/morphology_guard.py      морфологическая защита словарных замен
src/training_augmentation.py top-k candidates, KEEP augmentation и clean/noise audit
src/edit_labels.py           разметка error_text -> edit labels
src/hybrid_preprocessor.py   словари и векторизация
src/hybrid_model.py          Transformer с двумя головами
src/train_hybrid.py          обучение основной модели
src/evaluate_hybrid.py       оценка качества
src/hybrid_corrector.py      runtime API корректора
src/quality_guard.py         защита от вредных исправлений
src/text_utils.py            токенизация и сборка текста
app/main.py                  GUI
```

Старые `src/train.py` и `src/evaluate.py` оставлены как основные entry point,
но теперь они запускают hybrid trainer/evaluator.

## Данные

Основные файлы:

```text
data/raw/texts.txt
data/raw/train_clean.txt
data/raw/val_clean.txt
data/raw/test_clean.txt
data/processed/dataset.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Split делается по чистым предложениям до augmentation, чтобы одинаковые
исходные предложения не попадали одновременно в train и test.

## Типы Ошибок

Генератор покрывает:

```text
clean
light spelling typos
orthography rules: тся/ться, ого/ово, его/ево, пре/при и похожие пары
punctuation errors
spacing errors
mixed
hard
```

Keyboard-neighbor typos не добавляются.

## Обучение

```bash
PYTHONPATH=src .venv/bin/python src/train.py \
  --dataset data/processed/dataset.csv \
  --output-dir models \
  --max-length 128 \
  --max-vocab-size 80000 \
  --candidate-min-freq 1 \
  --candidate-max-distance 1 \
  --candidate-top-k 8 \
  --long-oov-max-distance 2 \
  --long-oov-min-length 8 \
  --min-dictionary-score 0.25 \
  --context-model-name DeepPavlov/rubert-base-cased \
  --context-device cpu \
  --d-model 128 \
  --num-layers 2 \
  --ff-dim 256 \
  --dropout 0.30 \
  --learning-rate 1e-4 \
  --batch-size 64 \
  --epochs 30
```

После обучения появляются:

```text
models/hybrid_corrector.keras
models/hybrid_preprocessor.pkl
models/candidate_generator.pkl
models/hybrid_config.json
models/hybrid_training_log.csv
```

## Оценка

```bash
PYTHONPATH=src .venv/bin/python src/evaluate.py \
  --dataset data/processed/test.csv \
  --all \
  --output-dir report \
  --strictness strict \
  --punctuation-mode conservative \
  --candidate-top-k 8 \
  --context-device cpu \
  --min-dictionary-score 0.25
```

Отчеты:

```text
report/summary.json
report/error_analysis.csv
report/error_metrics_by_type.csv
report/word_edit_diagnostics.csv
report/word_decision_diagnostics.csv
report/punct_edit_diagnostics.csv
report/clean_candidate_audit.csv
report/clean_noise_filter.csv
report/worst_cases.csv
```

Ключевые метрики:

```text
exact_match
mean_input_cer
mean_pred_cer
mean_cer_delta
improved_rate
worse_rate
unchanged_wrong_rate
clean_overcorrection_rate
candidate_coverage_rate
target_rank_distribution
reranker_called_count
reranker_changed_top1_count
reranker_error_reason_counts
split_candidate_accept_rate
comma_delete_blocked_count
internal_period_overcorrection_rate
spacing_only_overcorrection_rate
layout_changed_without_edits_count
punct_pred_similarity
mean_punct_count_error
accepted_rate
```

## Текущий Статус V8

Последний полный отчет notebook записан в:

```text
report/summary.json
report/error_analysis.csv
report/word_decision_diagnostics.csv
report/punct_edit_diagnostics.csv
```

Состояние на полном `test.csv` после текущего V8-прогона:

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
candidate_coverage_rate: 0.802004
target_present_but_not_applied_rate: 0.491918
deterministic_spacing_applied_count: 518
space_edit_count: 508
reranker_non_finite_count: 0
entity_guard_blocked_count: 1831
context_source_veto_relaxed_count: 5
punct_input_similarity: 0.887929
punct_pred_similarity: 0.892349
punct_wrong_period_to_comma_improved_rate: 0.386905
```

Главный вывод: clean-safety остается хорошей, словарных overcorrection на clean
нет (`word_overcorrection_rate=0.0`), deterministic spacing не портит clean
(`space_overcorrection_rate=0.0`). Основная слабость по-прежнему находится в
dirty recall: много строк остаются частично исправленными или неизмененными.

Самые важные сигналы для следующего шага:

```text
dirty unchanged_wrong_rate: 0.592390
target_present_but_not_applied_rate: 0.491918
punct_change_candidate_count: 3835
punct_applied_count: 85
space_merge_words exact_match: 0.057971
space_add_before_punct improved_rate: 0.941860
space_duplicate improved_rate: 0.936620
```

## Текущий Runtime-Fix V8.1.1

V8.1.1 закрывает самые явные runtime-ошибки без переобучения:

```text
1. space_merge_words:
   сообщаетсяна -> сообщается на
   незаставляет -> не заставляет
   ихпоражения -> их поражения

2. dictionary-drop guard:
   блокировать сообщаетсяна -> сообщается,
   незаставляет -> заставляет,
   ихпоражения -> поражения.

3. comma-insert guard:
   не вставлять запятую после сообщалось/сообщается перед о/об/обо/ранее.
```

Следующий эксперимент после V8.1.1:

```text
1. Запустить smoke/full evaluation.
2. Проверить, что clean_overcorrection_rate <= 0.002 и worse_rate <= 0.01.
3. Если safety сохранилась, отдельно планировать punctuation recall.
```

## Notebook

Основной экспериментальный сценарий находится в:

```text
notebook.ipynb
```

В нем можно менять гиперпараметры обучения, пересобирать dataset labels,
обучать hybrid-модель, смотреть графики loss/accuracy, запускать evaluation и
просматривать худшие примеры. По умолчанию notebook пересобирает датасет с
`TRAIN_CLEAN_RATIO=0.40`, `EVAL_CLEAN_RATIO=0.35` и `SAMPLES_PER_TEXT=4`.
Train curriculum усилен multi-error и space-профилями, а evaluation curriculum
оставлен сопоставимым.

Чтобы заново посчитать метрики в notebook:

```text
1. Restart Kernel.
2. Выполнить ячейки с импортами и конфигом, где заданы:
   CONTEXT_DEVICE = "cpu"
   MIN_DICTIONARY_SCORE = 0.25
   USE_ENTITY_GUARD = True
3. Запустить evaluation-ячейку.
```

## Использование

```python
from hybrid_corrector import HybridCorrector

corrector = HybridCorrector()
print(corrector.correct("Машиное обучение это интерестно."))
```

Если `models/hybrid_corrector.keras` еще не обучена, корректор запускается в
консервативном словарном режиме. Финальная рабочая модель проекта - это
`models/hybrid_corrector.keras`.

## GUI

```bash
PYTHONPATH=src .venv/bin/python app/main.py
```

GUI использует `HybridCorrector` и сохраняет подсветку различий между исходным
и исправленным текстом.

## Быстрые проверки

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```
