# Coding Rules

- Keep scope strict: Russian spelling and punctuation only.
- Do not add seq2seq or free-form rewrite correction.
- Do not restore materialized CSV training datasets.
- Do not reintroduce the old offline data-prep stack or materialized training
  dataset builders.
- Prefer the new packages for future work:
  - `src/grammar_gen/`
  - `src/runtime/`
  - `src/schema/`
- Do not create large generated files in source control. Reports stay generated
  output and `reports/` keeps only `.gitkeep`.
