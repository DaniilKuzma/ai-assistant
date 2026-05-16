# Russian Edit Corrector

Интеллектуальный ассистент для исправления только двух типов ошибок в русском тексте:

- орфография;
- пунктуация.

Система не является литературным редактором и не должна менять стиль, смысл, порядок слов, падежи, времена, синонимы или добавлять новые смысловые слова.

## Архитектура

Проект использует candidate-aware edit-based correction:

1. preprocessing и protected spans;
2. tokenizer/aligner;
3. encoder-only модель `ai-forever/ruRoberta-large`;
4. fallback encoder `ai-forever/ruBert-base`;
5. candidate scoring head;
6. punctuation gap head;
7. confidence и error type heads;
8. strict validator;
9. edit realizer.

Seq2seq-модели и готовые correction-модели не используются.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Не запускайте `pip install src`: это сторонний пакет с PyPI, а не локальный код проекта. Для локального проекта нужна команда `pip install -e .` из корня репозитория.

## Тесты

```bash
pytest -q
```

## Notebook Pipeline

```bash
jupyter notebook notebooks/main_pipeline.ipynb
```

Notebook загружает `configs/config.yaml`, проверяет окружение, собирает/загружает датасет, показывает его состав, строит labels/features, запускает обучение, evaluation, отчёты и ручные примеры.

## Датасет

Главный датасет собирается в notebook и/или через data builder из `configs/config.yaml`:

- путь: `data/processed/correction_dataset.csv.gz`;
- размер: `450000` строк;
- поля: `source`, `target`, `error_types`, `source_dataset`, `is_clean`, `is_synthetic`, `split`, `domain`, `edit_operations`;
- состав текущей сборки: `1597` real, `403403` synthetic, `45000` clean identity;
- splits: `405000` train, `22500` val, `22500` test.

Реальные пары берутся из открытых Hugging Face JSONL-файлов `RussianNLP/RuSpellGold` и `ai-forever/spellcheck_punctuation_benchmark`, затем проходят strict-filtering.

Остальной объем добирается synthetic corpus, разрешенным в ТЗ. Чистая основа для synthetic-части берется из внешних корпусов `Leipzig news`, `Leipzig Wikipedia`, `Taiga rest`, строго отфильтрованного `Taiga proza`, `UD Russian Taiga` и `OpenCorpora`; затем в эти предложения вносятся только разрешенные орфографические и пунктуационные ошибки. Художественные диалоги, жаргон, URL/код и слишком шумные предложения фильтруются до генерации ошибок.

## Обучение

```bash
python -m src.training.train
```

Параметры задаются в `configs/config.yaml`: режимы quick-debug/small/full, LoRA, batch size, gradient accumulation, mixed precision, thresholds и loss weights.

Главный `configs/config.yaml` настроен на full-train fine-tune по датасету `450000` строк: загружается локально кешированная `ai-forever/ruRoberta-large` через `AutoModel`, при нехватке памяти используется fallback `ai-forever/ruBert-base`. Seq2seq-классы не используются.

```bash
python -m src.training.train configs/config.yaml
```

Проект принудительно отключает TensorFlow/Flax backend для Transformers (`USE_TF=0`, `USE_FLAX=0`), потому что здесь нужен только PyTorch encoder.

## Evaluation

Метрики:

- exact_match;
- dirty_improved_rate;
- dirty_worse_rate;
- clean_overcorrection_rate;
- spelling precision/recall/F1;
- punctuation precision/recall/F1;
- edit precision/recall/F1;
- combined_score.

Reports пишутся в `reports/`: dataset, training, evaluation summary, accepted/rejected edits и threshold sweep.

Обязательные файлы:

- `dataset_report.md`;
- `training_report.md`;
- `evaluation_summary.csv`;
- `error_by_type.csv`;
- `clean_overcorrection_examples.csv`;
- `dirty_worse_examples.csv`;
- `accepted_edits.csv`;
- `rejected_edits.csv`.

## Streamlit

```bash
streamlit run src/app/streamlit_app.py
```

Интерфейс по умолчанию загружает обученную edit-based модель из `models/adapters/latest` и `models/heads/latest`. Если артефакты отсутствуют или не загружаются, приложение явно показывает предупреждение и использует rule fallback только как аварийный режим.

Интерфейс показывает исходный текст, исправленный текст и список accepted/rejected edits. Также есть загрузка `.docx`, исправление по параграфам и скачивание нового документа.

## DOCX

DOCX pipeline:

1. извлекает текст параграфов;
2. исправляет каждый параграф;
3. собирает новый `.docx`;
4. сохраняет количество параграфов и базовую структуру.

## Ограничения

Текущая rule-backed inference-часть нужна для безопасного smoke/integration контура. Качество полноценного корректора должно достигаться fine-tuning encoder-модели и подбором thresholds. Validator остается обязательной защитой и после обучения.
