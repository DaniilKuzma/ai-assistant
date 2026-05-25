# File Map

## Root

- `AI_INDEX.md` - first-read project map for AI agents.
- `AGENTS.md` - persistent agent instructions.
- `configs/config.yaml` - active generation, model, training, runtime, and path
  configuration.
- `configs/rules.yaml` - rule taxonomy and coverage matrix.

## Source Layout

- `src/grammar_gen/` - AST, realizer, morphology, lexicon, safety checks,
  `OnlineExampleGenerator`, and `RuleProgram` implementations.
- `src/schema/` - `GeneratedExample`, runtime edits, labels, and serialization.
- `src/model/` - RuRoBERTa encoder loading, direct edit model, heads, and losses.
- `src/training/` - online dataset, direct tensorization, direct trainer, and
  train entry point.
- `src/runtime/` - deterministic-first correction, neural backend, edit
  realization, tokenization, thresholds, and `ScopeGuard`.
- `src/evaluation/` - direct runtime/model evaluation over frozen JSONL.
- `src/app/` - Streamlit GUI.
- `src/docx/`, `src/memory/`, `src/preprocessing/`, `src/nlp/` - document,
  memory, text-processing, and syntax support.

## Entry Points

- `python scripts/audit_generator.py configs/config.yaml`
- `python scripts/benchmark_generation.py configs/config.yaml`
- `python scripts/build_frozen_eval.py configs/config.yaml --split val --count 100 --output data/generated_eval/val.jsonl`
- `python -m src.training.train configs/config.yaml --smoke --debug-model --steps 2`
- `python scripts/evaluate_model.py configs/config.yaml --dataset data/generated_eval/val.jsonl --output reports/eval_val`
- `streamlit run src/app/streamlit_app.py`

## Artifact Policy

- `data/processed/` keeps `.gitkeep` only; no generated CSV datasets.
- `data/generated_eval/` stores explicit frozen eval JSONL when intentionally
  built.
- `reports/` and `models/` are generated output locations and should not carry
  stale smoke or dataset-builder artifacts.
