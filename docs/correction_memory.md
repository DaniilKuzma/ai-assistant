# Контекстная память решений

## Зачем нужна

Контекстная память помогает повторять уже принятое корректорское решение в похожей рабочей ситуации: принять исправление, отклонить его, проигнорировать или сохранить ручной выбор. Это полезно при повторной проверке одного документа или его новых версий, где одна и та же спорная правка встречается в том же контексте.

Проект по-прежнему исправляет только орфографию и пунктуацию русского текста. Память не делает систему литературным редактором и не должна расширять strict scope.

## Отличие от пользовательского словаря

Пользовательский словарь хранит слова или допустимые формы.

`CorrectionMemory` хранит не слово, а решение по конкретной правке:

- документ (`doc_id`);
- правило (`rule_id`);
- тип правки (`edit_type`);
- исходный фрагмент и замену;
- левый и правый контекст;
- решение `accepted`, `rejected`, `ignored` или `manual`;
- служебные metadata.

Поэтому запись памяти для `так же -> также` не означает глобально заменять все такие вхождения. Она относится к конкретному candidate в конкретном контексте.

## Как работает

1. Candidate generator создает bounded candidates для орфографии и пунктуации.
2. Pipeline ищет candidate в `CorrectionMemory` по contextual key.
3. Если найдено `accepted`, candidate может быть переиспользован согласно `correction_memory.accepted_reuse`.
4. Если найдено `rejected` или `ignored`, candidate может быть подавлен согласно `correction_memory.rejected_suppress`.
5. Model-backed path по-прежнему использует scoring, thresholds и validation.
6. `StrictValidator` остается обязательной границей перед применением edits.
7. UI или `CorrectionFeedbackService` явно записывает feedback в память и сохраняет JSONL storage.

Конфиг находится в `configs/config.yaml`:

```yaml
correction_memory:
  enabled: false
  storage_path: data/user/correction_memory.jsonl
  context_window_chars: 48
  accepted_reuse: true
  rejected_suppress: true
```

`enabled: false` означает, что память не загружается по умолчанию.

## Инкрементальная проверка

`IncrementalCorrector` делит текст на сегменты, строит cache по normalized hash и на следующем запуске переиспользует исправления для неизмененных сегментов. Новые или измененные сегменты проходят через переданный corrector и обычный validation path.

Для DOCX `correct_docx_incremental` работает на уровне непустых абзацев: неизмененные абзацы берутся из `paragraph_cache`, измененные проверяются заново, затем записывается новый `.docx`.

## Ограничения

- Память не обучает веса ruRoberta, LoRA adapters или custom heads.
- Память не применяется в обход `StrictValidator`.
- Память не является глобальным правилом замены и не заменяет словарь.
- Память должна использоваться только для исправлений в strict scope: орфография и пунктуация.
- Exact contextual lookup чувствителен к документу, rule id, типу правки, source/replacement и контексту.
- Кеш incremental correction ускоряет повторную проверку версий, но не меняет правила, thresholds или качество модели.

## Как тестировать

Запуск focused tests:

```bash
python -m pytest tests/test_correction_memory.py tests/test_feedback_service.py tests/test_document_index.py tests/test_incremental_corrector.py -q
```

Проверка Streamlit/DOCX сценариев:

```bash
python -m pytest tests/test_streamlit_app.py tests/test_docx_io.py -q
```

Что покрывают тесты:

- сохранение и lookup `accepted/rejected/ignored/manual`;
- JSONL save/load и corrupted-line tolerance;
- отличие контекстов, документов, rule id и replacements;
- feedback service для `Edit` и `Candidate`;
- сегментацию текста, reuse cache и пересчет offsets;
- incremental DOCX reuse/check counters и запись output `.docx`.
