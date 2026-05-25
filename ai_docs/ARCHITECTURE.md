# Architecture

## High-Level Flow

```text
RuleProgram
  -> Grammar AST
  -> Morphological realizer
  -> Error renderer
  -> token edit labels + punctuation gap labels
  -> online training batches
```

The project now targets AST-first online generation plus direct edit tagging.
The old candidate-aware offline dataset build is not part of the architecture.

## Training Flow

- `configs/config.yaml:generation` defines online AST generation, rule-group mix,
  grammar limits, seed, and frozen eval sizes.
- Train examples are sampled on the fly.
- Frozen val/test/regression examples may be materialized as JSONL in
  `data/generated_eval`.
- Large `data/processed/train.csv`, `val.csv`, `test.csv`,
  `correction_dataset.csv.gz`, layer CSV/GZIP files, clean pools, and dataset
  manifests are obsolete and removed.

## Model Shape

- Encoder: `ai-forever/ruRoberta-large`.
- Adaptation: LoRA.
- Heads: direct token edit tagging and punctuation gap classification.
- The model is encoder-only and must not become a free-form seq2seq rewriter.

## Runtime Shape

- Deterministic rule engine runs first for ironclad spelling and punctuation
  fixes.
- Neural token edits and neural punctuation run after deterministic rules when
  confidence passes configured thresholds.
- CandidateGenerator is not used by the target model-backed runtime.
- The old StrictValidator data-prep boundary is removed. A future ScopeGuard
  will enforce runtime scope.
- GUI behavior remains stable: user enters text, presses `Исправить`, and gets
  corrected text plus edits.

## Future Packages

- `src/grammar_gen/` - online AST generation.
- `src/runtime/` - deterministic-first runtime orchestration.
- `src/schema/` - generated example, label, and edit schemas.
- `lexicon/` - lexicon resources.
- `data/generated_eval/` - frozen JSONL evaluation sets.
