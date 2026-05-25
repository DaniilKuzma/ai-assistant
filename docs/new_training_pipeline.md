# New Training Pipeline

## RuleProgram

`RuleProgram` is the unit of controlled example generation. Each rule owns its
positive, hard-negative, and clean-identity examples and returns a
`GeneratedExample` with direct labels. A rule must stay inside Russian spelling
and punctuation scope.

## Grammar AST

Rules build examples from grammar AST nodes rather than scanning a clean
sentence pool. The AST describes sentence structure, slots, agreement, and
punctuation positions before text is rendered.

## Morphology And Lexicon

The lexicon supplies controlled Russian lemmas, frames, exceptions, and safe
word lists. The morphology layer realizes forms used by the AST and keeps
agreement explicit enough for generator audits.

## OnlineExampleGenerator

`OnlineExampleGenerator` samples enabled `RuleProgram` objects according to the
configured mix. It validates `GeneratedExample` contracts, applies safety
checks, retries failed samples, and supports deterministic sampling by index for
repeatable training and frozen eval builds.

## GeneratedExample Contract

Each example contains:

- `source_text` and `target_text`;
- `source_tokens` with character offsets;
- `token_edit_labels`;
- `gap_labels`;
- per-token `rule_ids`;
- `primary_rule_id`;
- `mode`;
- `explanation_ids`;
- metadata such as seed, expected edit counts, and safety notes.

Training labels are direct labels, not pre-generated candidate rows.

## DirectEditTaggerModel

`DirectEditTaggerModel` wraps RuRoBERTa with optional LoRA and direct heads for
token edits, punctuation gaps, rule tags, and confidence logits. It is
encoder-only and must not be converted into a seq2seq rewrite model.

## Training

`OnlineGrammarDataset` samples generated examples online and
`build_direct_training_feature` tensorizes them for the direct heads. Training
writes head artifacts and optional adapters under configured `models/` paths.
No `train.csv` is built or consumed.

## Runtime Correction

Runtime correction is deterministic first. After deterministic edits, the neural
backend may predict direct token and punctuation labels. `ScopeGuard` and
thresholds keep accepted edits conservative.

## Evaluation

Frozen eval datasets are compact JSONL files under `data/generated_eval`.
`scripts/evaluate_model.py` evaluates the runtime corrector over those
`GeneratedExample` records and writes reports under `reports/`.

## Performance And Quality Guards

- `scripts/audit_generator.py` checks generation quality and safety failures.
- `scripts/benchmark_generation.py` checks generation throughput.
- Unit tests cover schema contracts, tensorization, direct model heads, rule
  programs, runtime realization, and no legacy data-builder restoration.
- Generated CSV datasets are forbidden; only intentional frozen eval JSONL is
  allowed.
