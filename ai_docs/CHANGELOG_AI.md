# AI Changelog

## 2026-05-25

- Stabilized AI memory around the completed AST-first online generation and
  direct edit tagging architecture.
- Documented that training data is generated online and `train.csv` is gone.
- Documented frozen eval JSONL under `data/generated_eval`.
- Documented RuRoBERTa direct edit tagging, deterministic-first runtime, and
  preserved GUI behavior.
- Removed AI-memory references that implied the old offline data-builder path
  was still active.

## 2026-05-18

- Created Markdown-based AI memory for the project.
- Added `AI_INDEX.md` as first-read entry point.
- Added compact documents under `ai_docs/`.
