# Architecture Decision: AST-First Online Generation

## Decision

The project no longer builds a large candidate-aware CSV training dataset. The old
clean sentence pool scan, rule_lab recipes, rule cards, corruption operators,
CandidateGenerator-backed data prep, strict synthetic row validation, and
multi-hour offline train/val/test CSV build are removed.

Training examples are generated online from controlled grammar programs. Frozen
validation, test, and regression sets may be materialized as compact JSONL under
`data/generated_eval`, but train data is not materialized as `train.csv`.

## Rationale

Clean pool scanning made the training pipeline slow, stateful, and hard to
reason about. It coupled data quality to whichever external corpus happened to
be cached locally and produced large generated artifacts that obscured the
actual rule coverage.

Materialized train CSV files also forced every architecture change through an
expensive rebuild. The new design keeps the training distribution in code and
configuration, so experiments can change rule mixes without checking in or
moving huge generated files.

`rule_lab` and rule cards are no longer sources of train rows because they
encoded sentence templates outside the grammar model. The replacement is a
single controlled generation path that owns syntax, morphology, rendering, and
labels together.

CandidateGenerator is not part of data prep anymore. It also is not the target
model-backed runtime path. Candidate coverage is no longer a training gate.

## New Flow

```text
RuleProgram
  -> Grammar AST
  -> Morphological realizer
  -> Error renderer
  -> token edit labels + punctuation gap labels
  -> online training batches
```

The model remains encoder-only RuRoBERTa with LoRA and custom heads. It becomes
a direct edit tagger plus punctuation gap classifier instead of a scorer for
pre-generated candidates.

Deterministic rules still run first for ironclad corrections. A later runtime
ScopeGuard will replace the old StrictValidator boundary with a simpler guard
focused on the final runtime scope.

## Performance Contract

Generation must not perform a full clean pool scan, per-rule corpus mining, or
any materialized train CSV build. Train examples are generated online from the
AST-first rule programs.

Frozen evaluation may be materialized only as compact JSONL under
`data/generated_eval`; it is not part of the train generation path.

The generation benchmark is expected to run in seconds or minutes for normal
counts, not hours. Configs that reference removed clean-pool, rule_lab,
candidate-opportunity, correction-dataset, or `data/processed/train.csv` paths
are invalid for generation.

## User Experience

The GUI behavior stays the same. A user enters Russian text, presses
`Исправить`, and receives `corrected_text` plus edits. This decision changes
how training examples and model predictions are represented, not the basic UI
workflow.
