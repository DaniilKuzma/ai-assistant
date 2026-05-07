# Интеллектуальный ассистент для исправления орфографии и пунктуации

Проект реализует одну основную систему коррекции русского текста:

```text
HybridCorrector = top-k кандидаты + edit-based Transformer + source-aware punctuation head + RuBERT context reranker + quality guard
```

Модель не переписывает предложение целиком. Она предсказывает точечные действия
над токенами (`KEEP`, `DELETE`, `REPLACE_0 ... REPLACE_7`) и знак препинания после каждого токена.
За счет этого ассистент ведет себя осторожнее: действие по умолчанию - оставить
текст как есть.

Runtime работает в консервативном режиме без отдельного класса пробельных
исправлений. URL, email, decimal/date-like фрагменты, латиница, слэши, `№`,
числа и неизвестная структурная пунктуация сохраняются. Пунктуационная голова
может менять простые знаки и whitelist структурных gap-labels для кавычек,
тире, скобок, многоточия, прямой речи и списков при высокой уверенности.
Runtime обрабатывает строку целиком или overlapping windows, поэтому ошибочная
точка внутри предложения больше не превращается в искусственный конец сегмента.
Пунктуационная голова получает на вход исходный знак после токена
(`source_punct_ids`). Текущий сценарий - V10.5: он сохраняет осторожный runtime,
добавляет реальные пары spellcheck/punctuation в обучение, пишет разрезы
метрик по `source_kind/source_dataset`, использует rank-consistent top-k
actions, строгие safe split-candidates, RuBERT/RuRoBERTa context reranking через
`context-device auto` с CUDA и безопасным CPU fallback, candidate-aware KEEP training, clean-noise filter, entity/noise guard и
дополнительные clean-safety guards для ложных comma-insert и разрушения
сбалансированной структурной пунктуации. V10.5 добавляет узкие runtime recovery:
безопасные top-1 dictionary-опечатки и удаление лишней запятой после коротких
служебных слов, отдельное удаление запятой в day-month датах, а также
безопасное восстановление финальной точки, если модель уже выбрала ее на
последнем gap.

Словарные исправления дополнительно защищены: `min_dictionary_score=0.25`,
`candidate_min_freq=1` и морфологический guard на `pymorphy2` блокируют
опасные замены правильных словоформ вроде `летнем -> летним`.

Keyboard-опечатки намеренно не используются.

## Структура

```text
src/data_preparation.py      подготовка корпуса и synthetic augmentation
src/external_datasets.py     загрузка и фильтрация внешних real-pairs
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
data/external/ai_forever_spellcheck_punctuation_test.csv
data/external/ruspellgold_test.csv
```

Split делается по чистым предложениям до augmentation, чтобы одинаковые
исходные предложения не попадали одновременно в train и test.
В V10 train/val дополнительно получают реальные пары из
`ai-forever/spellcheck_punctuation_benchmark`; официальные test-сплиты AI
Forever и RuSpellGold сохраняются отдельно в `data/external/` и не смешиваются
с основным `test.csv`.

## Типы Ошибок

Генератор покрывает:

```text
clean
light spelling typos
orthography rules: тся/ться, ого/ово, его/ево, пре/при и похожие пары
compound/case/abbreviation/borrowed-word spelling rules
punctuation errors
structural punctuation errors
mixed
hard
real spellcheck/punctuation pairs without spacing-only rows
```

Keyboard-neighbor typos не добавляются.

## Обучение

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

Отчеты:

```text
report/synthetic_v11/summary.json
report/synthetic_v11/error_analysis.csv
report/synthetic_v11/error_metrics_by_type.csv
report/synthetic_v11/worst_cases.csv
```

Большие diagnostic CSV включаются без `--no-diagnostics`:

```text
report/word_edit_diagnostics.csv
report/word_decision_diagnostics.csv
report/punct_edit_diagnostics.csv
report/clean_candidate_audit.csv
report/clean_noise_filter.csv
report/external_ai_forever_v11/summary.json
report/external_ruspellgold_v11/summary.json
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
safe_comma_delete_recovery_count
safe_service_comma_delete_recovery_count
safe_date_comma_delete_recovery_count
safe_final_period_recovery_count
punct_target_change_count
punct_target_predicted_count
punct_target_applied_count
punct_target_predicted_but_blocked_rate
internal_period_overcorrection_rate
layout_changed_without_edits_count
punct_pred_similarity
mean_punct_count_error
accepted_rate
source_kind_slices
source_dataset_slices
```

## Текущий Статус V10.5

Notebook теперь готовит V10-переобучение и V10.5 runtime evaluation:

```text
MODEL_VERSION: 10
TRAIN_CLEAN_RATIO: 0.35
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
```

После полного прогона основной synthetic-compatible отчет записывается в:

```text
report/synthetic_v11/summary.json
report/synthetic_v11/error_analysis.csv
report/synthetic_v11/error_metrics_by_type.csv
report/synthetic_v11/worst_cases.csv
```

Внешние проверки записываются отдельно:

```text
report/external_ai_forever_v11/
report/external_ruspellgold_v11/
```

Acceptance targets для V10.5 после полного прогона:

```text
clean_overcorrection_rate <= 0.002
worse_rate <= 0.01
overall exact_match выше 0.5699
dirty exact_match выше 0.3398
punctuation-only unchanged_wrong_rate ниже предыдущего уровня
punct_applied_count выше текущих 856
punct_target_applied_count растет
punct_target_predicted_but_blocked_rate падает
safe_comma_delete_recovery_count растет
safe_final_period_recovery_count >= 74
safe_service_comma_delete_recovery_count > 0
safe_date_comma_delete_recovery_count > 0
target_present_but_not_applied_rate ниже текущего уровня
low_action_confidence_or_margin заметно падает
```

## Notebook

Основной экспериментальный сценарий находится в:

```text
notebook.ipynb
```

В нем можно менять гиперпараметры обучения, пересобирать dataset labels,
обучать hybrid-модель, смотреть графики loss/accuracy, запускать основной и
external evaluation, а также просматривать худшие примеры. По умолчанию
notebook пересобирает synthetic+real train с `TRAIN_CLEAN_RATIO=0.35`,
`EVAL_CLEAN_RATIO=0.35`, `SAMPLES_PER_TEXT=4` и `REAL_TRAIN_REPEAT=6`.
Evaluation curriculum остается сопоставимым, external test не смешивается с
основным `test.csv`, а полный external evaluation включается вручную после
успешного smoke-прогона.

Чтобы заново посчитать метрики в notebook:

```text
1. Restart Kernel.
2. Выполнить ячейки с импортами и конфигом, где заданы:
   CONTEXT_DEVICE = "auto"
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
