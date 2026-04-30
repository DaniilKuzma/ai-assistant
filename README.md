# Интеллектуальный ассистент для исправления орфографии и пунктуации

Проект реализует одну основную систему коррекции русского текста:

```text
HybridCorrector = top-k кандидаты + edit-based Transformer + source-aware punctuation head + RuBERT context reranker + quality guard
```

Модель не переписывает предложение целиком. Она предсказывает точечные действия
над токенами (`KEEP`, `DELETE`, `REPLACE_0 ... REPLACE_4`) и знак препинания после каждого токена.
За счет этого ассистент ведет себя осторожнее: действие по умолчанию - оставить
текст как есть.

Runtime работает в консервативном режиме: исходные кавычки, тире, скобки,
слэши, `№`, числа и латиница сохраняются, а пунктуационная голова может менять
только простые знаки `, . ? ! : ;` при высокой уверенности.
Runtime обрабатывает строку целиком или overlapping windows, поэтому ошибочная
точка внутри предложения больше не превращается в искусственный конец сегмента.
Пунктуационная голова получает на вход исходный знак после токена
(`source_punct_ids`). Текущий формат модели - v6: он также использует
rank-consistent top-k actions, safe split-candidates, RuBERT/RuRoBERTa context
reranking, candidate-aware KEEP training, clean-noise filter и усиленные веса
punctuation changes, поэтому старые модели нужно переобучить с начала.

Словарные исправления дополнительно защищены: `min_dictionary_score=1.0`,
`candidate_min_freq=1` и морфологический guard на `pymorphy2` блокируют
опасные замены правильных словоформ вроде `летнем -> летним`.

Keyboard-опечатки намеренно не используются.

## Структура

```text
src/data_preparation.py      подготовка корпуса и synthetic augmentation
src/candidate_generator.py   словарные и правиловые кандидаты
src/context_reranker.py      RuBERT/RuRoBERTa context reranking
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
  --candidate-top-k 5 \
  --min-dictionary-score 1.0 \
  --context-model-name DeepPavlov/rubert-base-cased \
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
  --sample-size 500 \
  --output-dir report \
  --strictness strict \
  --punctuation-mode conservative \
  --candidate-top-k 5
```

Отчеты:

```text
report/summary.json
report/error_analysis.csv
report/error_metrics_by_type.csv
report/word_edit_diagnostics.csv
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
split_candidate_accept_rate
internal_period_overcorrection_rate
spacing_only_overcorrection_rate
layout_changed_without_edits_count
punct_pred_similarity
mean_punct_count_error
accepted_rate
```

## Notebook

Основной экспериментальный сценарий находится в:

```text
notebook.ipynb
```

В нем можно менять гиперпараметры обучения, пересобирать dataset labels,
обучать hybrid-модель, смотреть графики loss/accuracy, запускать evaluation и
просматривать худшие примеры. По умолчанию notebook пересобирает датасет с
`TRAIN_CLEAN_RATIO=0.45` и `EVAL_CLEAN_RATIO=0.35`, чтобы была измерима
метрика `clean_overcorrection_rate`. `SAMPLES_PER_TEXT=3` оставлен без
раздувания; пунктуационные профили слегка усилены через curriculum.

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
