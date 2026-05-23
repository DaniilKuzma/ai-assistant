# AI Index

Это входная точка AI-memory для проекта `russian-edit-corrector`.

Будущий AI должен читать этот файл первым, а не сканировать весь репозиторий. Здесь собрана карта документов, которые помогают быстро понять проект и выбрать точечные файлы для чтения.

## Быстрый Маршрут

1. `ai_docs/PROJECT_OVERVIEW.md` — зачем существует проект и что он не должен делать.
2. `ai_docs/ARCHITECTURE.md` — как устроен pipeline исправлений.
3. `ai_docs/FILE_MAP.md` — где лежат важные модули, конфиги, данные, модели и отчеты.
4. `ai_docs/FUNCTIONS_MAP.md` — ключевые классы и функции по подсистемам.
5. `ai_docs/OPEN_TASKS.md` — известные блокеры и хрупкие места.
6. `docs/correction_memory.md` — краткое описание контекстной памяти решений и инкрементальной проверки.

## Документы

- `ai_docs/PROJECT_OVERVIEW.md` — цель, scope, зависимости, текущий статус.
- `ai_docs/ARCHITECTURE.md` — data flow, inference, training, validation.
- `ai_docs/FILE_MAP.md` — папки, entry points, артефакты.
- `ai_docs/FUNCTIONS_MAP.md` — публичные классы/функции и роли модулей.
- `ai_docs/API.md` — Python API, Streamlit, DOCX, отсутствие REST API.
- `ai_docs/DEPLOYMENT.md` — установка, запуск, тесты, обучение, evaluation.
- `ai_docs/CODING_RULES.md` — локальные правила разработки.
- `ai_docs/DECISIONS.md` — принятые архитектурные решения.
- `ai_docs/CHANGELOG_AI.md` — изменения AI-memory.
- `ai_docs/OPEN_TASKS.md` — открытые задачи и риски.
- `docs/correction_memory.md` — пользовательский словарь, контекстная память, feedback и incremental correction.

`ai_docs/DATABASE.md` намеренно отсутствует: в проекте не найдено базы данных, ORM, миграций или DB-схемы.

## Самые Важные Факты

- Проект исправляет только русскую орфографию и пунктуацию.
- Система не должна менять стиль, смысл, порядок слов или добавлять новые смысловые слова.
- Архитектура: candidate-aware edit-based correction, encoder-only ruRoberta, LoRA, custom heads.
- Plain `Corrector` применяет только безопасные deterministic fallback-исправления.
- Model-backed исправления проходят через `TrainedModelCorrector`, thresholds и `StrictValidator`.
- Контекстная память решений хранит `accepted/rejected/ignored/manual` для конкретной правки в конкретном контексте; она меняет inference/pipeline, но не обучает веса модели.
- Память не применяется в обход `StrictValidator`: memory-selected candidates остаются в strict scope и проходят validation.
- Инкрементальная проверка переиспользует кеш исправленных сегментов текста или абзацев DOCX и заново проверяет только измененные части.
- Главный конфиг: `configs/config.yaml`.
- Coverage matrix правил: `configs/rules.yaml`.
- Streamlit UI: `src/app/streamlit_app.py`.
- Correction memory modules: `src/memory/`.
- Incremental text correction: `src/inference/incremental_corrector.py`.
- Incremental DOCX correction: `src/docx/docx_corrector.py`.
- Training entrypoint: `python -m src.training.train configs/config.yaml`.
- Старый README удален; package readme указывает на этот `AI_INDEX.md`.

