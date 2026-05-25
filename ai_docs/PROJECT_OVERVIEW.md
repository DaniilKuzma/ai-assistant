# Project Overview

## Purpose

`russian-edit-corrector` is a controlled assistant for Russian spelling and
punctuation correction.

The project does not do literary editing. It must not change meaning, style,
word order, tense, case, synonyms, or add semantic content.

## Current Approach

The target architecture is AST-first online generation plus direct edit tagging:

- controlled `RuleProgram` objects render grammar ASTs;
- morphology realizes grammatical Russian forms;
- an error renderer creates spelling or punctuation mistakes;
- labels are produced directly as token edit tags and punctuation gap labels;
- training consumes generated examples online instead of reading a huge
  materialized train CSV.

RuRoBERTa remains an encoder-only model with LoRA and custom heads. The model is
not a seq2seq rewriter and no longer scores pre-generated correction candidates
as the main model-backed path.

## Data Policy

The old offline data-prep stack and large CSV/GZIP train/val/test artifacts
were removed.

Frozen validation, test, and regression sets may be written as JSONL under
`data/generated_eval`. Training data is generated online.

## Main Artifacts

- Config: `configs/config.yaml`
- Architecture decision: `docs/architecture_decision_ast_first.md`
- Online generation package: `src/grammar_gen/`
- Runtime package: `src/runtime/`
- Shared schemas: `src/schema/`
- Lexicon assets: `lexicon/`
- Frozen eval JSONL target: `data/generated_eval/`
