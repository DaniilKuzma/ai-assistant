# Architecture

## High-Level Flow

```text
RuleProgram
  -> Grammar AST
  -> Morphology and lexicon realization
  -> Error rendering
  -> GeneratedExample
  -> online training batches
  -> RuRoBERTa direct edit tagger
```

The project is AST-first and direct-tagging-first. The old offline dataset
builder and candidate-aware training path are not part of the architecture.

## Generation And Training

- `configs/config.yaml:generation` controls rule groups, mix, seed, grammar
  limits, and frozen eval sizes.
- `OnlineExampleGenerator` samples `GeneratedExample` instances on demand.
- `OnlineGrammarDataset` tensorizes generated examples during training.
- Frozen eval sets are explicit JSONL files under `data/generated_eval`.
- Large `train.csv`, validation CSV, test CSV, and CSV.GZ datasets are obsolete.

## Model

- Encoder: RuRoBERTa.
- Adaptation: LoRA where configured.
- Heads: token edit labels, punctuation gap labels, rule tags, and confidence
  logits.
- The model outputs direct edits; it must not become seq2seq or a free-form
  rewrite model.

## Runtime

- Deterministic rules run first for conservative spelling and punctuation fixes.
- Neural direct token edits and punctuation are applied only above configured
  confidence thresholds.
- `ScopeGuard` keeps runtime edits within Russian spelling and punctuation
  boundaries.
- GUI behavior remains stable and user-facing workflow is unchanged.
