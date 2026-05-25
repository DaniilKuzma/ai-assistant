# AI Index

This is the first file to read before project analysis in
`russian-edit-corrector`.

## Quick Route

1. `ai_docs/PROJECT_OVERVIEW.md` - purpose, scope, and artifact policy.
2. `ai_docs/ARCHITECTURE.md` - AST-first online generation and direct edit
   tagging architecture.
3. `docs/new_training_pipeline.md` - training pipeline details.
4. `docs/rule_authoring_guide.md` - how to add a new `RuleProgram`.
5. `ai_docs/FILE_MAP.md` - important folders and generated output policy.

## Current Facts

- The project corrects only Russian spelling and punctuation.
- The system must not change meaning, style, word order, tense, case, or add
  semantic words.
- Training examples are generated online by AST-first `RuleProgram` objects.
- Materialized `train.csv` / train-val-test CSV datasets are not part of the
  architecture.
- Frozen validation, test, and regression examples may be materialized as JSONL
  under `data/generated_eval`.
- The model is an encoder-only RuRoBERTa direct edit tagger with LoRA and custom
  token, punctuation-gap, and rule heads.
- Runtime correction is deterministic rules first, then conservative neural
  direct edits guarded by `ScopeGuard`.
- GUI behavior is preserved: user enters text, clicks the correction button, and
  receives corrected text plus edit details.
- The old candidate-aware data-builder architecture is removed and must not be
  restored.

## Current Entry Points

- Generator audit: `python scripts/audit_generator.py configs/config.yaml`
- Generation benchmark: `python scripts/benchmark_generation.py configs/config.yaml`
- Frozen eval build: `python scripts/build_frozen_eval.py configs/config.yaml`
- Training smoke/debug: `python -m src.training.train configs/config.yaml --smoke --debug-model`
- Model/runtime evaluation: `python scripts/evaluate_model.py configs/config.yaml`
