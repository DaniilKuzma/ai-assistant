# Instructions For AI Agents

Read `AI_INDEX.md` before project analysis. Use it as the project map instead
of scanning the whole repository first.

Working rules:

1. Check `git status` before edits and do not revert changes you did not make.
2. Keep the project scope strict: Russian spelling and punctuation only.
3. The current architecture is AST-first online generation plus direct edit
   tagging. Do not restore the old candidate-aware dataset pipeline.
4. Do not add seq2seq or free-form rewrite correction.
5. Do not recreate materialized train/val/test CSV datasets.
6. Do not use `CandidateGenerator`, rule_lab, clean sentence pool scanning, or
   strict synthetic-row validation for data prep.
7. Runtime safety remains conservative. The old `StrictValidator` data-prep
   role is gone; a future `ScopeGuard` will cover runtime boundaries.
