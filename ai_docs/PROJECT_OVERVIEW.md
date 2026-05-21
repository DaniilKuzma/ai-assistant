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

