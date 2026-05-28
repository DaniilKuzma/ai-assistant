# Casing Layer

`casing` is a bounded YAML-backed `RuleLayer` for Russian orthographic casing.
It emits only direct token edit labels and stays inside the existing
`GeneratedExample` / `RuleProgram` / online generation contract.

## Covered Cases

- Sentence start capitalization in controlled examples, including a simple
  post-period sentence-start case.
- Controlled person names: `Иван Петров`, `Мария Иванова`, `Анна Кузнецова`.
- Controlled geographic names: `Москва`, `Рязань`, `Россия`,
  `Российская Федерация`.
- Limited organization and document/title names from the project dictionary,
  including `Рязанский государственный радиотехнический университет`,
  `Министерство науки и высшего образования`, `Федеральный закон`.
- Limited historical event and holiday titles:
  `Великая Отечественная война`, `День Победы`.
- Extra capital letters on ordinary common words in mid-sentence contexts,
  such as `Директора` -> `директора`, `Президента` -> `президента`,
  `Понедельник` -> `понедельник`, `Январь` -> `январь`.

## Labels

The layer allows only token operations with:

- `CAPITALIZE` for first-letter capitalization;
- `LOWERCASE` for bounded lowercase normalization.

`UPPERCASE` is intentionally forbidden in the casing layer. It uppercases the
whole token at runtime, which is wrong for person names, geographic names,
organization names, and most Russian orthographic casing corrections.

The layer does not use gap operations, boundary operations, seq2seq correction,
free-form rewrite correction, or candidate-aware dataset generation.

## Guards And Limitations

The layer is dictionary-limited. It is not a broad NER system, title normalizer,
style rewriter, or semantic rewrite layer. Hard negatives protect already
correct starts, abbreviations, numeric starts, ordinary nouns such as `город`
and `страна`, ordinary `министерство`/`университет`/`закон`, proper names that
must stay capitalized, and abbreviations such as `РФ`, `США`, `ИИ`, `ООО`.

Formal `Вы`/`Ваш` correction is opt-in and is not enabled by default.
`casing_formal_you_guard` is guard-only: it generates hard-negative and
clean-identity examples, but no positive examples and no automatic
`вы/ваш` -> `Вы/Ваш` correction.
