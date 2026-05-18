# Coding Rules

## Scope Rules

- Исправлять только русскую орфографию и пунктуацию.
- Не менять стиль, смысл, порядок слов, грамматическую форму вне strict spelling/punctuation scope.
- Не добавлять смысловые слова.
- Не превращать проект в generative rewrite system.

## Architecture Rules

- Сохранять edit-based/candidate-aware подход.
- Не добавлять seq2seq correction-модель без явного архитектурного решения.
- Все model-backed изменения должны проходить через thresholds и `StrictValidator`.
- Plain `Corrector` должен оставаться conservative fallback.
- Candidate-only/model-required edits не должны применяться без scorer/trusted path.

## Rules And Candidates

- Новый rule должен иметь понятный `rule_id`, mode, requires/group metadata.
- Если rule отображается в `configs/rules.yaml`, проверь coverage tests.
- Для risky rules добавляй negative tests против overcorrection.
- Dictionary/syntax/model-required зоны не делай deterministic без доказанной безопасности.

## Data And Training

- Не пересобирай большой dataset и не запускай full training случайно.
- Для dry-run используй лимиты окружения или small config.
- Generated reports/model artifacts могут быть тяжелыми и уже измененными; не откатывай их без явной просьбы.
- `USE_TF=0`, `TRANSFORMERS_NO_TF=1`, `USE_FLAX=0` выставляются кодом для PyTorch-only Transformers path.

## Tests

- Основной запуск: `.venv/bin/python -m pytest -q`.
- Для inference/rules изменений нужны targeted tests по candidates, validator, negative suites, metrics/reports.
- Для Streamlit/DOCX изменений нужны `tests/test_streamlit_app.py` и `tests/test_docx_io.py`.

## Git Safety

- Всегда смотри `git status --short` перед правками.
- Не откатывай чужие изменения.
- Не делай destructive git-команды без явной просьбы.

