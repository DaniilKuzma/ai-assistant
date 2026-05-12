# Runtime Architecture

`HybridCorrector` is the single correction strategy for the project:

```text
HybridCorrector = top-k candidate generator + edit-based Transformer +
source-aware punctuation head + RuBERT context reranker + quality guard
```

The model does not rewrite the whole sentence. It predicts local token actions
and punctuation labels after tokens, then the runtime applies only edits that
pass conservative guards.

## Action Space

Token actions:

```text
KEEP
DELETE
REPLACE_0 ... REPLACE_7
```

Each `REPLACE_N` selects one of the generated candidates for the token. The
default safe behavior is `KEEP`.

The punctuation head receives the original punctuation after each token through
`source_punct_ids` and predicts the output gap label. In
`punctuation_mode="conservative"` it can change simple trailing marks and
whitelisted structural gap labels for quotes, dashes, brackets, direct speech,
lists and selected ellipsis cases.

## Protected Text

Runtime protects input that should not be rewritten casually:

```text
URL
email
decimal/date-like fragments
Latin text
slashes
№
numbers
unknown structural punctuation
```

The runtime works on full lines and falls back to overlapping windows for long
inputs. This prevents an accidental internal period from forcing an artificial
sentence boundary.

## Candidate And Context Layers

`src/candidate_generator.py` creates dictionary and rule candidates. Dictionary
corrections use:

```text
candidate_min_freq=1
min_dictionary_score=0.25
```

Ambiguous dictionary and split edits are rescored by
`DeepPavlov/rubert-base-cased` through `src/context_reranker.py`. With
`context-device auto`, CUDA is used when available and safe; runtime falls back
to CPU on scoring errors. Split edits fail closed when context scoring is not
reliable.

## Guards

The runtime is intentionally conservative:

- `src/morphology_guard.py` blocks dangerous dictionary replacements of valid
  word forms, for example `летнем -> летним`.
- `src/entity_guard.py` protects clean/raw words, names, toponyms and rare terms
  loaded from `data/raw/texts.txt` and `data/processed/*.csv` `correct_text`
  values.
- `src/quality_guard.py` rejects results that look harmful after reconstruction.
- Structural punctuation edits are rejected when they worsen quote or bracket
  balance.

## Current V10.5 Behavior

V10.5 keeps cautious dirty-aware recall gates and adds narrow recovery paths:

```text
safe top-1 dictionary typo recovery
extra comma deletion after short service words
lowercase service-word comma deletion
day-month date comma cleanup
safe final-period recovery on the last gap
```

These recovery paths must not relax punctuation, entity, protected-word,
morphology or clean-safety guards.

## Artifact Contract

Current model artifacts:

```text
models/hybrid_corrector.keras
models/hybrid_preprocessor.pkl
models/candidate_generator.pkl
models/hybrid_config.json
models/hybrid_training_log.csv
```

The active format expects `model_version=10`, `runtime_version="10.5"`,
`source_punct_ids` and top-k `candidate_ids`. Stale model artifacts that do not
match the current format are obsolete and should be retrained.
