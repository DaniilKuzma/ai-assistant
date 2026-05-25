# Deployment And Operations

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Не используй `pip install src`: это сторонний пакет с PyPI, не локальный код проекта.

В текущем shell bare `python` может отсутствовать. Надежная команда:

```bash
.venv/bin/python -m pytest -q
```

## Тесты

```bash
.venv/bin/python -m pytest -q
```

Для быстрых проверок полезны targeted suites:

```bash
.venv/bin/python -m pytest -q tests/test_streamlit_app.py
.venv/bin/python -m pytest -q tests/test_rules_coverage.py tests/test_no_legacy_candidate_architecture.py
```

## Training

```bash
.venv/bin/python -m src.training.train configs/config.yaml --smoke --debug-model
```

Главные настройки:

- `training.run_model_training`
- `training.max_train_examples`, `max_val_examples`, `max_test_examples`
- `model.primary_encoder`, `model.fallback_encoder`
- `model.lora`
- `thresholds.mode`
- `paths.adapter_output_dir`, `paths.heads_output_dir`, `paths.reports_dir`

Отключить обучение и оставить pipeline в no-training режиме:

```bash
RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING=1 .venv/bin/python -m src.training.train configs/config.yaml
```

## Evaluation And Reports

Reports пишутся в `reports/`.

Важные файлы:

- `dataset_report.md`
- `training_report.md`
- `evaluation_summary.csv`
- `error_by_type.csv`
- `rule_precision_recall.csv`
- `gap_label_coverage_by_rule.csv`
- `clean_overcorrection_examples.csv`
- `dirty_worse_examples.csv`
- `accepted_edits.csv`
- `rejected_edits.csv`
- `threshold_precision_recall.png`

## Streamlit

```bash
streamlit run src/app/streamlit_app.py
```

UI поддерживает:

- plain text correction;
- highlighted diff;
- accepted/rejected edits table;
- `.docx` upload/download.

## Notebook

```bash
jupyter notebook notebooks/main_pipeline.ipynb
```

Notebook ведет через config, dataset, labels/features, training, evaluation, reports и ручные примеры.

