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

## RuleLayer Architecture

The current generator has two controlled rule surfaces:

- handwritten `RuleProgram` classes for legacy direct orthography and
  punctuation rules;
- `RuleLayer` specs loaded from `lexicon/layers/` and compiled into direct
  `RuleProgram` instances by `src/rule_layers/direct_cases.py`.

`compound_spelling`, `dictionary_typo`, `syntax_punctuation`,
`quotation_dialogue`, and `casing` are YAML-backed RuleLayer families.
`morpheme` is a controlled compiler-backed layer under
`src/orthography_gen/`; it uses lexeme cards and orthographic scenario specs,
but presents the same `GeneratedExample` contract to training.

Layer examples must emit direct token or gap labels. `SPAN_REPLACE_BY_LEXICON`
marks a token span whose replacement is resolved only through the trusted
runtime orthographic lexicon. `DELETE_PUNCTUATION` marks an existing punctuation
character after a token as removable; it is a gap label, not a token rewrite.

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
- Layer-driven span replacements remain lexicon-gated at runtime. The model can
  request `SPAN_REPLACE_BY_LEXICON`, but ambiguous or missing lexicon entries
  are no-ops.
- Extra punctuation deletion uses `DELETE_PUNCTUATION` and only removes a
  punctuation character that is already present in the source text.
- GUI behavior remains stable and user-facing workflow is unchanged.

## Explicitly Out Of Scope

The current integrated layers do not implement broad NER-backed casing,
abbreviation expansion as a correction layer, nested quotation marks, broad
dialogue/direct speech, general paired bracket/quote punctuation, or free-form
rewrite correction. The bounded `quotation_dialogue` layer covers only curated
quote and direct-speech cases with direct boundary, token, and gap labels. The
bounded `casing` layer covers only controlled Russian orthographic casing cases
with `CAPITALIZE` and `LOWERCASE`; formal `Вы`/`Ваш` correction is guard-only
unless a future explicit opt-in policy enables it.
