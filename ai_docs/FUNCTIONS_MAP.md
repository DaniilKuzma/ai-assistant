# Functions Map

## Generation

- `src.grammar_gen.rules.base.RuleProgram` - base class for rule programs.
- `src.grammar_gen.generator.OnlineExampleGenerator` - samples examples online.
- `src.grammar_gen.audit.audit_batch` - validates generated batches.
- `src.grammar_gen.safety.validate_generated_pair` - conservative generation
  safety checks.
- `src.schema.examples.GeneratedExample` - source/target and direct-label
  contract.

## Training

- `src.training.online_dataset.OnlineGrammarDataset` - iterable online training
  dataset.
- `src.training.online_dataset.FrozenJsonlDataset` - direct eval dataset from
  frozen JSONL.
- `src.training.tensorization.build_direct_training_feature` - maps
  `GeneratedExample` to direct model tensors.
- `src.training.trainer.train_model` - full direct training loop.
- `src.training.train.train` - public training entry point.

## Model And Runtime

- `src.model.edit_model.DirectEditTaggerModel` - RuRoBERTa direct edit tagger.
- `src.runtime.corrector.Corrector` - deterministic-first correction
  orchestrator.
- `src.runtime.neural_backend.DirectNeuralBackend` - loads direct model
  artifacts and predicts edit labels.
- `src.runtime.scope_guard.ScopeGuard` - conservative runtime safety boundary.
- `src.runtime.edit_realizer` - turns direct labels into runtime edits.

## Evaluation

- `src.evaluation.evaluate.evaluate_corrector` - evaluates a corrector on frozen
  `GeneratedExample` JSONL.
- `scripts/audit_generator.py` - generation quality audit.
- `scripts/benchmark_generation.py` - generation throughput benchmark.
- `scripts/build_frozen_eval.py` - frozen eval JSONL builder.
- `scripts/evaluate_model.py` - runtime/model evaluation CLI.
