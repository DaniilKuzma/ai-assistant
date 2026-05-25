# File Map

## Root

- `AI_INDEX.md` - first file for AI navigation.
- `AGENTS.md` - instructions for future AI agents.
- `pyproject.toml` - package metadata.
- `requirements.txt` - runtime and development dependencies.
- `pytest.ini` - pytest configuration.

## Config

- `configs/config.yaml` - current AST-first online generation, model, training,
  runtime, and paths config.
- `configs/rules.yaml` - rule coverage matrix and taxonomy metadata.

## Source Layout

- `src/grammar_gen/` - future AST-first online generation.
- `src/runtime/` - future deterministic-first runtime orchestration.
- `src/schema/` - future shared schemas for generated examples, labels, and
  runtime edits.
- `src/data/` - intentionally empty legacy namespace; old dataset builders were
  removed.
- `src/preprocessing/` - tokenizer, sentence splitter, protected spans,
  punctuation gaps.
- `src/rules/` - deterministic rules and rule metadata.
- `src/model/` - encoder loading, multitask model, heads, losses.
- `src/training/` - tensorization, trainer, train entry point, artifact
  save/load.
- `src/inference/` - current inference code retained until the runtime rewrite.
- `src/evaluation/` - evaluation and report helpers retained where they do not
  require the removed data-prep pipeline.
- `src/memory/` - correction memory and incremental segment cache.
- `src/docx/` - DOCX read/correct/write flow.
- `src/app/` - Streamlit UI.

## Entry Points

- Training smoke entry: `python -m src.training.train configs/config.yaml`
- Streamlit: `streamlit run src/app/streamlit_app.py`
- Tests: `python -m pytest -q`

## Data And Generated Artifacts

- `data/processed/` - no materialized training datasets; keep `.gitkeep` only.
- `data/generated_eval/` - future frozen val/test/regression JSONL outputs.
- `reports/` - generated reports; keep `.gitkeep` only in source control.
- `models/adapters/latest` - configured LoRA adapter output.
- `models/heads/latest` - configured custom head output.
- `lexicon/` - future lexicon resources.

Old CSV/GZIP dataset artifacts and dataset reports were removed with the
candidate-aware offline data-prep pipeline.
