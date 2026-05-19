# Open Tasks And Fragile Areas
НЕАКТАУЛЬНО!!!
## Current Blocker

По `reports/pretraining_readiness_report.md` текущий readiness status — `BLOCKED`.

Главная причина: clean overcorrection в no-training evaluation с existing checkpoint:

- `clean_overcorrection_rate = 0.28`;
- примеры лежат в `reports/clean_overcorrection_examples.csv`;
- встречаются unwanted quote normalization, comma insertion, capitalization и lexical changes.

До устранения этого риска проект не стоит считать готовым к большому dataset build или short training.

## Fragile Areas

- `StrictValidator`: центральная защита от unsafe edits; изменения требуют negative tests.
- Punctuation predictions: кавычки, скобки, тире, финальная пунктуация и delete/replace actions легко дают false positives.
- Context pairs: `также/так же`, `тоже/то же`, `чтобы/что бы`, `зато/за то` требуют высокого confidence и контекстных guardrails.
- `-тся/-ться`: опасная зона, требует trusted/model path и thresholds.
- Dictionary fuzzy candidates: полезны для recall, но повышают риск lexical overcorrection.
- `yo_e_candidate`: выключен по умолчанию; включать только при lexicon support и model scoring.
- Syntax layer зависит от `natasha`; при отсутствии зависимости fallback может вернуть пустой синтаксический слой.
- Heavy model loading должен оставаться lazy, чтобы unit tests не скачивали ruRoberta без необходимости.

## Useful Next Steps

- Разобрать `reports/clean_overcorrection_examples.csv` и ужесточить validator/thresholds для ложных accepted edits.
- Проверить threshold profiles по rule families после каждого изменения candidates/rules.
- Поддерживать `configs/rules.yaml` как честную coverage matrix: planned entries не должны выглядеть implemented.
- Добавлять negative tests для каждого нового deterministic или fallback-safe rule.
- Сохранять small/dry-run путь для dataset builder и training pipeline.

