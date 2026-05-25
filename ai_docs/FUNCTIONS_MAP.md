# Functions Map

This map reflects the AST-first cleanup state. The old data-prep builders,
candidate dataset config resolver, rule_lab helpers, and matrix dataset helpers
were removed.

## Active Stable Areas

- `src.config.load_config.load_config` - load YAML configuration.
- `src.model.encoder.load_tokenizer` - load the encoder tokenizer when model
  training or model-tokenized evaluation is enabled.
- `src.model.edit_model.CandidateAwareEditModel` - current encoder-only model
  wrapper retained until the direct tagger rename/refactor lands.
- `src.training.train.train` - training entry point; currently uses smoke
  online-generation placeholders until `src.grammar_gen` is implemented.
- `src.training.tensorization.build_features_from_rows` - current feature
  builder retained for compatibility during the architecture transition.
- `src.inference.corrector.Corrector` - deterministic fallback corrector.
- `src.inference.model_corrector.TrainedModelCorrector` - current model-backed
  corrector retained until the runtime rewrite.
- `src.evaluation.evaluate.evaluate_rows_detailed` - row-level evaluation.

## Planned Areas

- `src.grammar_gen` - RuleProgram, Grammar AST, morphology realization, error
  rendering, and online example sampling.
- `src.schema` - generated example, token edit, punctuation gap, and runtime edit
  schemas.
- `src.runtime` - deterministic-first runtime orchestration and future
  ScopeGuard.
