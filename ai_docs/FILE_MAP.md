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
- `src/rule_layers/` - YAML-backed `RuleLayer` specs, loaders, coverage helpers,
  and direct-case builders for controlled generated examples.
- `src/orthography_gen/` - compiler-backed morpheme layer: orthographic rule
  specs, lexeme cards, context wrapping, and safe error injection.
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
- `python scripts/build_frozen_eval.py configs/config.yaml --split val --count 500 --output data/generated_eval/val.jsonl`
- `python scripts/build_frozen_eval.py configs/config.yaml --split test --count 500 --output data/generated_eval/test.jsonl`
- `python scripts/build_frozen_eval.py configs/config.yaml --split regression --count 200 --output data/generated_eval/regression.jsonl`
- `python -m src.training.train configs/config.yaml --smoke --debug-model --steps 2`
- `python scripts/evaluate_model.py configs/config.yaml --dataset data/generated_eval/val.jsonl --output reports/eval_val`
- `streamlit run src/app/streamlit_app.py`

## Integrated Layers

- `lexicon/layers/compound_spelling/` - controlled split, merge, hyphen, and
  `не` spelling examples. Generic span operations use
  `SPAN_REPLACE_BY_LEXICON`.
- `src/orthography_gen/` - `morpheme` examples for roots, prefixes, suffixes,
  endings, consonants, signs, and н/нн.
- `lexicon/layers/dictionary_typo/` - trusted dictionary, borrowed-word,
  domain-term, common misspelling, character-noise, keyboard-neighbor, and
  space-noise examples.
- `lexicon/layers/syntax_punctuation/` - controlled punctuation gap examples,
  including `DELETE_PUNCTUATION`.

## Artifact Policy

- `data/processed/` keeps `.gitkeep` only; no generated CSV datasets.
- `data/generated_eval/` stores explicit frozen eval JSONL when intentionally
  built.
- `reports/` and `models/` are generated output locations and should not carry
  stale smoke or dataset-builder artifacts.
