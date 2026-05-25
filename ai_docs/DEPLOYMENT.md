# Deployment And Operations

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
```

Use `pip install -e .`, not `pip install src`.

## Verification

```bash
python -m compileall src scripts
pytest -q
python scripts/audit_generator.py configs/config.yaml --count 1000
python scripts/benchmark_generation.py configs/config.yaml --count 1000
```

## Frozen Eval

```bash
python scripts/build_frozen_eval.py configs/config.yaml --split val --count 100 --output data/generated_eval/val.jsonl
```

Frozen eval JSONL is allowed under `data/generated_eval`. Do not create
`data/processed/train.csv` or other materialized train/val/test CSV datasets.

## Training

```bash
python -m src.training.train configs/config.yaml --smoke --debug-model --steps 2
```

`--debug-model` must run without downloading or loading real RuRoBERTa. Full
training uses online examples from the generator and writes model heads/adapters
to configured `models/` paths.

## Evaluation

```bash
python scripts/evaluate_model.py configs/config.yaml --dataset data/generated_eval/val.jsonl --output reports/eval_val
```

Reports under `reports/` are generated outputs. Remove stale smoke or
dataset-builder outputs before committing cleanup work.

## GUI

```bash
streamlit run src/app/streamlit_app.py
```

The GUI remains deterministic-first and direct-tagging-backed while preserving
the existing user flow.
