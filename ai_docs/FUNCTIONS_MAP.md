# Functions Map

This map reflects the active AST-first online generation and direct edit tagging
architecture.

## Active Stable Areas

- `src.config.load_config.load_config` - load YAML configuration.
- `src.grammar_gen.generator.OnlineExampleGenerator` - sample AST-generated
  train/eval examples online.
- `src.grammar_gen.audit.audit_batch` - validate generated examples and report
  generation failures.
- `src.schema.examples.GeneratedExample` - shared generated-example contract.
- `src.schema.edits.RuntimeEdit` - runtime edit contract used by correction,
  memory, and feedback.
- `src.schema.edit_types.coarse_error_type` - shared edit taxonomy grouping for
  evaluation reports.
- `src.model.edit_model.DirectEditTaggerModel` - encoder-only direct edit
  tagger with token, gap, and rule heads.
- `src.training.tensorization.build_direct_training_feature` - tensorize
  generated examples for direct edit tagging.
- `src.training.train.train` - online-generation training entry point.
- `src.runtime.corrector.Corrector` - deterministic-first runtime orchestration.
- `src.runtime.scope_guard.ScopeGuard` - conservative runtime safety boundary.
- `src.inference.model_corrector.TrainedModelCorrector` - adapter for trained
  direct models.
- `src.evaluation.evaluate.evaluate_corrector` - direct runtime/model
  evaluation entry.

## Script Entrypoints

- `scripts/audit_generator.py` - generator audit CLI.
- `scripts/build_frozen_eval.py` - frozen JSONL eval set builder.
- `scripts/benchmark_generation.py` - generation throughput benchmark.
- `scripts/evaluate_model.py` - direct model/runtime evaluation.
- `scripts/tune_thresholds.py` - direct runtime threshold tuning.
