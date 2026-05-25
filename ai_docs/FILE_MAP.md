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

- `src/grammar_gen/` - AST-first online generation.
- `src/runtime/` - deterministic-first runtime orchestration, scope guarding,
  and morphology helpers.
- `src/schema/` - shared schemas for generated examples, labels, edit types,
  lexical resources, and runtime edits.
- `src/data/` - intentionally empty legacy namespace.
- `src/preprocessing/` - tokenizer, sentence splitter, protected spans,
  punctuation gaps.
- `src/rules/` - deterministic rules and rule metadata.
- `src/model/` - encoder loading, multitask model, heads, losses.
- `src/training/` - tensorization, trainer, train entry point, artifact
  save/load.
- `src/inference/` - direct model-backed inference adapter.
- `src/evaluation/` - direct evaluation and report helpers.
- `src/memory/` - correction memory and incremental segment cache.
- `src/docx/` - DOCX read/correct/write flow.
- `src/app/` - Streamlit UI.

## Entry Points

- Generator audit: `python scripts/audit_generator.py configs/config.yaml`
- Frozen eval build: `python scripts/build_frozen_eval.py configs/config.yaml`
- Generation benchmark: `python scripts/benchmark_generation.py configs/config.yaml`
- Training smoke entry: `python -m src.training.train configs/config.yaml --smoke --debug-model`
- Model evaluation: `python scripts/evaluate_model.py configs/config.yaml`
- Streamlit: `streamlit run src/app/streamlit_app.py`
- Tests: `python -m pytest -q`

## Data And Generated Artifacts

- `data/processed/` - no materialized training datasets; keep `.gitkeep` only.
- `data/generated_eval/` - frozen val/test/regression JSONL outputs.
- `reports/` - generated reports; keep `.gitkeep` only in source control.
- `models/adapters/latest` - configured LoRA adapter output.
- `models/heads/latest` - configured custom head output.
- `lexicon/` - lexicon resources.

Old CSV/GZIP training artifacts and offline data-prep reports are not part of
the current architecture.
