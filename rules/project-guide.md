# Project Guide

Проект реализует ассистента для осторожного исправления русской орфографии и
пунктуации. Основной runtime - `HybridCorrector`: он не переписывает
предложение целиком, а выбирает локальные token-edit действия и знаки
препинания после токенов.

Подробности разнесены по отдельным rules:

- `rules/runtime-architecture.md` - устройство `HybridCorrector`, guards,
  recovery-сценарии и контракт артефактов.
- `rules/training-and-evaluation.md` - команды обучения, оценки, notebook
  scenario, отчеты и acceptance targets.
- `rules/evaluation-metrics.md` - определения метрик и diagnostic-сигналов.
- `rules/strict-error-taxonomy.md` - строгий scope ошибок и runtime contract.
- `rules/strict-baseline-package0.md` - baseline package 0.

## Source Map

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
src/hybrid_model.py          Keras Transformer с action/punctuation головами
src/train_hybrid.py          обучение основной модели
src/evaluate_hybrid.py       оценка качества
src/hybrid_corrector.py      runtime API корректора
src/quality_guard.py         защита от вредных исправлений
src/text_utils.py            токенизация и сборка текста
app/main.py                  tkinter GUI
```

Старые `src/train.py` и `src/evaluate.py` оставлены как стабильные entry
points: они делегируют в hybrid trainer/evaluator.

## Data

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
исходные предложения не попадали одновременно в train и test. В текущем
V10/V10.5 сценарии реальные пары AI Forever добавляются только в train/val, а
held-out AI Forever и RuSpellGold остаются отдельными external checks.

## Error Scope

Текущий scope - строгая орфография и нормативная пунктуация:

```text
clean
orthography: тся/ться, ого/ово, пре/при, н/нн, ъ/ь, не/ни,
  и/ы после ц, hyphen/compound spelling, capitalization,
  abbreviation case, borrowed words, dictionary word pairs
punctuation: subordinate clauses, introductory words, homogeneous members,
  extra comma before single и, final .?!, subject-predicate dash,
  BSP colon/dash, quotes/direct speech, paired brackets
real_spelling
real_punctuation
```

Spacing-only, capitalization-only, style-only, heavy rewrite, untrainable
punctuation and morphology/grammar-only real rows are rejected before training.
Keyboard-neighbor typos and typo-like random operations (`delete`, `swap`,
`extra`, random `double`) are outside the current scope.

The authoritative label contract is `rules/strict-error-taxonomy.md`.

## Runtime Usage

```python
from hybrid_corrector import HybridCorrector

corrector = HybridCorrector()
print(corrector.correct("Машиное обучение это интерестно."))
```

If `models/hybrid_corrector.keras` is not trained yet, the runtime can still
start in a conservative dictionary mode so the GUI remains usable.

## GUI

```bash
PYTHONPATH=src .venv/bin/python app/main.py
```

The GUI uses `HybridCorrector` and highlights differences between source and
corrected text.

## Quick Checks

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```
