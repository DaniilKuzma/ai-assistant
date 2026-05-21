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

