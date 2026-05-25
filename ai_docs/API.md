# API

## User-Facing API

The GUI workflow is unchanged:

1. User enters Russian text.
2. User clicks the correction button.
3. The app returns corrected text and edit details.

## Internal Training API

- Training data is generated online from `src.grammar_gen`.
- `GeneratedExample` is the serialized example contract.
- Frozen eval datasets are JSONL files, not train CSV files.
- `src.training.train.train` delegates to the direct online trainer.

## Model API

- `DirectEditTaggerModel` predicts direct token edit labels, punctuation gap
  labels, and rule tags.
- `DirectNeuralBackend` loads saved direct heads and optional LoRA adapter
  artifacts for runtime predictions.
- Seq2seq and free-form rewrite APIs are intentionally absent.

## Runtime API

- `Corrector.correct(text)` preserves the existing correction workflow.
- Deterministic rules run before neural direct edits.
- `ScopeGuard` rejects edits outside Russian spelling and punctuation scope.
