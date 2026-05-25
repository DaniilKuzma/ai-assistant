# Project Overview

## Purpose

`russian-edit-corrector` is a controlled assistant for Russian spelling and
punctuation correction. It is not a literary editor and must not rewrite
meaning, style, word order, tense, case, or semantic content.

## Current Approach

The active architecture is AST-first online generation plus direct edit tagging:

- `RuleProgram` objects build controlled grammar AST examples.
- Morphology and lexicon resources realize valid Russian forms.
- Error rendering produces bounded spelling or punctuation mistakes.
- `GeneratedExample` stores source/target text, source tokens, direct token edit
  labels, punctuation gap labels, rule ids, mode, and metadata.
- Training samples examples online instead of reading a materialized train CSV.

The neural model is an encoder-only RuRoBERTa direct edit tagger with LoRA and
custom heads. It predicts token edits, punctuation gap labels, and rule tags; it
is not seq2seq and not a free-form rewriter.

## Data Policy

`train.csv` no longer exists. Large materialized train/val/test CSV or CSV.GZ
datasets must not be recreated. Frozen evaluation sets may be generated as JSONL
under `data/generated_eval` for validation, test, and regression runs.

## User Experience

GUI behavior is preserved. A user enters Russian text, clicks the correction
button, and receives corrected text plus accepted/rejected edit details.

## Main Artifacts

- Config: `configs/config.yaml`
- Online generation: `src/grammar_gen/`
- Direct schemas: `src/schema/`
- Training: `src/training/`
- Runtime: `src/runtime/`
- Frozen eval JSONL: `data/generated_eval/`
