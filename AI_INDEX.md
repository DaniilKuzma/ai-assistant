# AI Index

This is the entry point for AI-memory in `russian-edit-corrector`.

## Quick Route

1. `ai_docs/PROJECT_OVERVIEW.md` - project purpose and scope.
2. `ai_docs/ARCHITECTURE.md` - current AST-first generation and runtime shape.
3. `ai_docs/FILE_MAP.md` - important folders, entry points, and generated
   artifact policy.
4. `docs/architecture_decision_ast_first.md` - decision record for removing the
   old data-prep pipeline.
5. `docs/correction_memory.md` - correction memory behavior.

## Current Facts

- The project corrects only Russian spelling and punctuation.
- The system must not change meaning, style, word order, tense, case, or add new
  semantic words.
- Current target architecture: AST-first online generation plus direct edit
  tagging.
- RuRoBERTa remains encoder-only with LoRA and custom heads.
- The model predicts direct token edit tags and punctuation gap labels.
- Train examples are generated online. Large `train.csv`, `val.csv`,
  `test.csv`, and `correction_dataset.csv.gz` artifacts were removed.
- Frozen validation, test, and regression sets may be materialized as JSONL in
  `data/generated_eval`.
- CandidateGenerator is no longer used in data prep or the target model-backed
  runtime.
- The old `StrictValidator` data-prep role is removed. It will be replaced later
  by a simpler runtime `ScopeGuard`.
- Main config: `configs/config.yaml`.
- Future generation/runtime folders:
  - `src/grammar_gen/`
  - `src/runtime/`
  - `src/schema/`
  - `lexicon/`
  - `data/generated_eval/`
