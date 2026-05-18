# File Map

## Корень

- `AI_INDEX.md` — первый файл для AI-навигации.
- `AGENTS.md` — инструкции будущим AI-агентам.
- `pyproject.toml` — package metadata; `readme` указывает на `AI_INDEX.md`.
- `requirements.txt` — runtime/dev зависимости.
- `pytest.ini` — pytest запускает тесты из `tests/` с `-s`.

## Конфиги

- `configs/config.yaml` — главный конфиг модели, данных, thresholds, labels, metrics, paths.
- `configs/rules.yaml` — coverage matrix правил Орфограммки: implemented, partial, planned, model/syntax/dictionary required.

## Source Layout

- `src/preprocessing/` — tokenizer, sentence splitter, protected spans, punctuation gaps.
- `src/rules/` — rule specs, orthography/punctuation/synthetic rules, registry, coverage validation.
- `src/candidates/` — candidate generation, dictionary candidates, morphology, ranking, matching.
- `src/validation/` — diff analyzer, edit classifier, strict validator.
- `src/alignment/` — source-target alignment and label builders.
- `src/model/` — encoder loading, multitask edit model, heads, losses.
- `src/training/` — tensorization, trainer, train entrypoint, save/load artifacts.
- `src/inference/` — plain corrector, trained corrector, edit realizer, postprocess.
- `src/evaluation/` — metrics, reports, threshold sweep, rule/candidate recall reports.
- `src/data/` — dataset builders, external sources, clean corpus sources, splits, stats.
- `src/docx/` — DOCX read/correct/write flow.
- `src/app/` — Streamlit UI.
- `src/nlp/` — Natasha syntax wrapper.

## Entry Points

- Training: `python -m src.training.train configs/config.yaml`
- Evaluation helper: `src.training.train.evaluate_trained_model`
- Streamlit: `streamlit run src/app/streamlit_app.py`
- Notebook: `notebooks/main_pipeline.ipynb`
- Tests: `.venv/bin/python -m pytest -q`

## Data And Generated Artifacts

- `data/raw/` — raw/external corpora.
- `data/processed/correction_dataset.csv.gz` — processed correction dataset.
- `models/adapters/latest/` — LoRA adapter/config/labels/thresholds.
- `models/heads/latest/heads.pt` — custom heads checkpoint.
- `reports/` — generated evaluation/training/dataset reports.

Некоторые файлы в `reports/`, `models/` и `data/processed/` уже изменены в рабочем дереве; не откатывай их без явной просьбы.

