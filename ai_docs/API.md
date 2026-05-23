# API

## Python API

Проект не предоставляет REST/HTTP API. Основные интерфейсы используются как Python classes/functions.

### Plain Corrector

`src.inference.corrector.Corrector`

- `Corrector.from_config(config)` — создать fallback corrector из dict config.
- `Corrector.correct(text)` — вернуть `CorrectionResult`.
- Применяет только conservative deterministic edits.

### Trained Corrector

`src.inference.model_corrector.TrainedModelCorrector`

- `TrainedModelCorrector.from_config(config)` — загрузить tokenizer, encoder, LoRA adapters и heads.
- `correct(text)` — score candidates, apply thresholds, validate, return `CorrectionResult`.
- Требует `models/current/adapters` и `models/current/heads/heads.pt`.

### Correction Memory

`src.memory.CorrectionMemory`

- `CorrectionMemory(storage_path=None, enabled=True, context_window_chars=48)` — создать in-memory или JSONL-backed память решений.
- `remember_candidate(text, candidate, decision, doc_id="default", metadata=None)` — сохранить решение по `Candidate`.
- `remember_edit(text, edit, decision, doc_id="default", metadata=None)` — сохранить решение по `Edit`.
- `lookup_candidate(text, candidate, doc_id="default")` и `lookup_edit(text, edit, doc_id="default")` — найти точное совпадение по документу, rule id, edit type, source/replacement и контексту.
- `load()` / `save()` — загрузить или записать JSONL storage.
- `entries()` / `clear()` — получить entries или очистить память.

`CorrectionMemoryEntry` хранит `decision`, `doc_id`, `rule_id`, `edit_type`, `source`, `replacement`, левый/правый контекст, normalized context, timestamp и metadata. Допустимые решения: `accepted`, `rejected`, `ignored`, `manual`.

`CorrectionMemoryMatch` возвращает найденную entry, `reason` и признак exact match. Текущий lookup использует exact contextual key.

`src.memory.build_memory_from_config(config)` возвращает `CorrectionMemory` только при `correction_memory.enabled: true`; иначе возвращает `None`.

Память влияет на inference/pipeline selection: accepted candidates могут переиспользоваться, rejected/ignored candidates могут подавляться. Она не обучает ruRoberta и не отменяет `StrictValidator`.

### Correction Feedback Service

`src.memory.CorrectionFeedbackService`

- `CorrectionFeedbackService(memory, doc_id="default")` — связать feedback с конкретным документом.
- `accept_edit(source_text, edit, metadata=None)` — записать `accepted`.
- `reject_edit(source_text, edit, metadata=None)` — записать `rejected`.
- `ignore_edit(source_text, edit, metadata=None)` — записать `ignored`.
- `remember_edit_decision(...)` и `remember_candidate_decision(...)` — записать любое допустимое решение для `Edit` или `Candidate`.

Сервис является pipeline-level feedback API. Он не запускает training и не изменяет веса модели.

### Incremental Corrector

`src.inference.incremental_corrector.IncrementalCorrector`

- `IncrementalCorrector(corrector)` — обернуть существующий corrector с методом `correct(text)`.
- `correct_incremental(text, previous_cache=None)` — исправить текст по сегментам, переиспользуя `SegmentCorrectionCache` для неизмененных сегментов.
- Возвращает `IncrementalCorrectionResult` с `corrected_text`, edits, счетчиками checked/reused/changed segments и новым `segment_cache`.
- Не заменяет старый `correct(text)` и не меняет validation path: новые сегменты проходят через переданный corrector.

### Candidate Generator

`src.candidates.candidate_generator.CandidateGenerator`

- `CandidateGenerator.from_config(config)` — подключает dictionary provider и лимиты.
- `generate(text)` — возвращает bounded candidates с metadata: `rule_id`, `mode`, `requires`, `group`.

### Validation

`src.validation.strict_validator.StrictValidator`

- `validate(source, target, trusted_edits=...)` — принимает или отклоняет edits.
- `pre_validate(source, trusted_edits=...)` — проверка trusted edits перед применением.

`src.validation.diff_analyzer.DiffAnalyzer`

- `analyze(source, target)` — классифицирует diff в edit objects.

### Training

`src.training.train.train(config_path="configs/config.yaml")`

- готовит data/features;
- опционально обучает модель;
- сохраняет artifacts;
- пишет reports.

`src.training.train.evaluate_trained_model(config_path, split="test")`

- оценивает сохраненный/fine-tuned corrector.

## Streamlit UI

Entry point:

```bash
streamlit run src/app/streamlit_app.py
```

Функция `build_streamlit_corrector` сначала пытается загрузить trained model. Если artifacts отсутствуют или загрузка падает, используется `Corrector` fallback.

## DOCX Interface

`src.docx.docx_corrector.correct_docx(input_path, output_path, corrector=None)`

- читает paragraphs;
- исправляет каждый paragraph;
- пишет новый `.docx`;
- возвращает список edits.

Если `corrector` не передан, используется plain fallback `Corrector`.

`src.docx.docx_corrector.correct_docx_incremental(input_path, output_path, corrector=None, previous_cache=None)`

- читает paragraphs;
- переиспользует `SegmentCorrectionCache` для неизмененных непустых paragraphs;
- проверяет только новые или измененные paragraphs через переданный corrector;
- пишет новый `.docx`;
- возвращает `DocxIncrementalResult` с `edits`, `checked_paragraphs`, `reused_paragraphs` и `paragraph_cache`.

Если `corrector` не передан, используется plain fallback `Corrector`. Новые/измененные paragraphs проходят обычный validation path внутри выбранного corrector.

