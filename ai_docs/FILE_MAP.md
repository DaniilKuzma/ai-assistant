# File Map

## Корень

- `AI_INDEX.md` — первый файл для AI-навигации.
- `AGENTS.md` — инструкции будущим AI-агентам.
- `pyproject.toml` — package metadata; `readme` указывает на `AI_INDEX.md`.
- `requirements.txt` — runtime/dev зависимости.
- `pytest.ini` — pytest запускает тесты из `tests/` с `-s`.

## Конфиги

- `configs/config.yaml` — главный конфиг модели, thresholds, labels, metrics, paths и единственный canonical dataset block.
- `configs/config.yaml:data.candidate_opportunity` — единственный source of truth для candidate dataset settings; `data.training_dataset`, `data.training_dataset_core` и sibling dataset-дубли удалены.
- `configs/rules.yaml` — coverage matrix правил Орфограммки: implemented, partial, planned, model/syntax/dictionary required.

## Source Layout

- `src/preprocessing/` — tokenizer, sentence splitter, protected spans, punctuation gaps.
- `src/rules/` — rule specs, orthography/punctuation/synthetic rules, registry, coverage validation.
- `src/candidates/` — candidate generation, dictionary candidates, morphology, ranking, matching.
- `src/memory/` — контекстная память решений, feedback service, document segment cache для incremental correction.
- `src/validation/` — diff analyzer, edit classifier, strict validator.
- `src/alignment/` — source-target alignment and label builders.
- `src/model/` — encoder loading, multitask edit model, heads, losses.
- `src/training/` — tensorization, trainer, train entrypoint, save/load artifacts.
- `src/inference/` — plain corrector, trained corrector, edit realizer, postprocess.
- `src/inference/incremental_corrector.py` — text incremental correction wrapper; переиспользует кеш неизмененных сегментов.
- `src/evaluation/` — metrics, reports, threshold sweep, rule/candidate recall reports.
- `src/data/` — dataset builders, external sources, clean corpus sources, splits, stats.
- `src/config/candidate_dataset_config.py` — lightweight resolver/validator для `data.candidate_opportunity`; production code должен читать dataset settings только через него.
- `src/data/dataset_contract.py` — contract constants, layer inference, contract columns, stable dataset hash.
- `src/data/operator_dataset_builder.py` — canonical `candidate_opportunity` builder: clean-pool opportunities, atomic positives, hard negatives, real atomic rows, stress rows, layer files, manifest.
- `src/data/training_quality_audit.py` — quality/audit gates for atomic purity, unknown rules, mixed script clean rows, real-pair atomization and report writers.
- `src/docx/` — DOCX read/correct/write flow.
- `src/app/` — Streamlit UI.
- `src/nlp/` — Natasha syntax wrapper.

## Entry Points

- Training: `python -m src.training.train configs/config.yaml`
- Canonical layered dataset build: `python scripts/build_dataset.py --force`
- Rebuild helper: `python scripts/rebuild_training_dataset.py`
- Source setup: `python scripts/setup_data_sources.py --clean --real`
- Evaluation helper: `src.training.train.evaluate_trained_model`
- Streamlit: `streamlit run src/app/streamlit_app.py`
- Notebook: `notebooks/main_pipeline.ipynb`
- Tests: `.venv/bin/python -m pytest -q`

## Data And Generated Artifacts

- `data/raw/` — raw/external corpora.
- `data/processed/correction_dataset.csv.gz` — processed correction dataset.
- `data/processed/train_<layer>.csv.gz` — split/layer files for `candidate_opportunity` train rows.
- `data/processed/dataset_manifest.json` — canonical manifest with `dataset_contract`, `dataset_hash`, `verdict`, `audit_errors`, `layer_counts`.
- `models/current/adapters/` — LoRA adapter/config/labels/thresholds.
- `models/current/heads/heads.pt` — custom heads checkpoint.
- `reports/` — generated evaluation/training/dataset reports.
- `docs/correction_memory.md` — краткое руководство по контекстной памяти решений и инкрементальной проверке.

Некоторые файлы в `reports/`, `models/` и `data/processed/` уже изменены в рабочем дереве; не откатывай их без явной просьбы.

