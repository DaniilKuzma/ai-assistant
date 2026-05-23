# Project Overview

## Назначение

`russian-edit-corrector` — ассистент для контролируемого исправления русского текста. Проект исправляет только:

- орфографические ошибки;
- пунктуационные ошибки.

Система не является литературным редактором. Она не должна менять смысл, стиль, порядок слов, падежи, времена, синонимы или добавлять новые смысловые слова.

## Текущий Подход

Проект использует candidate-aware edit-based correction:

- генерируются ограниченные кандидаты исправлений;
- encoder-only модель оценивает кандидатов и пунктуационные gaps;
- `StrictValidator` фильтрует все изменения перед применением;
- итоговый текст собирается только из accepted edits.

Seq2seq и готовые generative correction-модели намеренно не используются.

## Уникальная особенность: контекстная память решений и инкрементальная проверка

Проект поддерживает контекстную память корректорских решений. Это не пользовательский словарь:

- пользовательский словарь хранит слова или допустимые формы;
- `CorrectionMemory` хранит решение по конкретному исправлению в конкретном документе и контексте;
- решение может быть `accepted`, `rejected`, `ignored` или `manual`;
- ключ памяти учитывает правило, тип правки, исходный фрагмент, замену и левый/правый контекст.

Память меняет inference/pipeline: уже известное решение может повторно принять candidate или подавить ранее отклоненный candidate в таком же контексте. Она не обучает веса ruRoberta, LoRA adapters или custom heads.

Память не расширяет strict scope проекта. Даже memory-selected исправления остаются candidate-aware edits и проходят через `StrictValidator`; система по-прежнему исправляет только орфографию и пунктуацию.

Инкрементальная проверка дополняет этот сценарий: для новой версии текста или DOCX можно переиспользовать кеш неизмененных сегментов/абзацев и заново проверять только измененные части. Это оптимизация pipeline, а не изменение качества модели или правил.

## Основные Зависимости

- Python `>=3.10`
- PyTorch
- Transformers `4.57.3`
- PEFT/LoRA
- pandas, numpy, scikit-learn, matplotlib
- pymorphy3, natasha, razdel, rapidfuzz
- Streamlit
- python-docx
- pytest, jupyter

Полный список — `requirements.txt`.

## Главные Артефакты

- dataset: `data/processed/correction_dataset.csv.gz`
- model adapters: `models/current/adapters`
- custom heads: `models/current/heads/heads.pt`
- reports: `reports/`
- notebook pipeline: `notebooks/main_pipeline.ipynb`

## Текущий Риск

По `reports/pretraining_readiness_report.md` проект функционально проходит тесты и dry-run, но статус pretraining readiness остается `BLOCKED` из-за clean overcorrection в no-training evaluation с существующим checkpoint.

