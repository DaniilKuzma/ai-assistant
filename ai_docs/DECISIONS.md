# Decisions

## Accepted Decisions

- Проект — controlled Russian spelling/punctuation assistant, не литературный редактор.
- Архитектура — candidate-aware edit ranking + punctuation gap tagging.
- Model backbone — encoder-only `ai-forever/ruRoberta-large`.
- Fine-tuning — PEFT/LoRA adapters плюс custom heads.
- Seq2seq/generative correction intentionally absent.
- Strict validator остается обязательной защитой после model scoring.
- Contextual correction memory is pipeline-level adaptation, not model fine-tuning: память решений может менять selection/reuse/suppression в inference, но не обучает ruRoberta, LoRA adapters или custom heads.
- Plain fallback corrector применяет только deterministic safe edits.
- Trained inference грузит adapters из `models/current/adapters` и heads из `models/current/heads/heads.pt`.
- `configs/rules.yaml` используется как coverage matrix, а не только список executable rules.
- `AI_INDEX.md` заменяет старый README как package readme и AI entrypoint.

## Important Defaults

- `thresholds.mode`: `balanced`
- `decoder.max_passes`: `3`
- `model.max_sequence_length`: `128`
- `model.max_candidates`: `16`
- `dictionary.yo_e.enabled`: `false`
- `training.mode`: `full-train`
- `training.evaluation_split`: `test`

## Deferred / Not Implemented

- Нет REST API.
- Нет базы данных, ORM или миграций.
- Нет NER-backed capitalization/entity correction.
- `ё/е` выключено по умолчанию и требует lexicon+model scoring.
- Многие Орфограммка-группы остаются planned/model_required/syntax_required/dictionary_model_required.

