# Russian Edit Corrector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reproducible candidate-aware Russian spelling and punctuation corrector project.

**Architecture:** The implementation is a modular Python package. Tests exercise a lightweight rule-backed inference path, while the model/training modules provide the real encoder-only LoRA fine-tuning path without importing heavyweight dependencies at package import time.

**Tech Stack:** Python 3.10, PyTorch, Transformers, PEFT, datasets, pandas, scikit-learn, razdel, rapidfuzz, Streamlit, python-docx, pytest.

---

### Task 1: Tests First

**Files:**
- Create: `tests/test_candidate_generator.py`
- Create: `tests/test_strict_validator.py`
- Create: `tests/test_diff_analyzer.py`
- Create: `tests/test_alignment.py`
- Create: `tests/test_synthetic_generator.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_docx_io.py`
- Create: `tests/test_inference_smoke.py`

- [ ] Write tests that describe the strict scope and expected public APIs.
- [ ] Run `pytest -q` and verify the tests fail because modules are missing.

### Task 2: Core Package

**Files:**
- Create package `src/*`
- Implement config loading, preprocessing, candidates, diff analyzer, validator, aligner, synthetic generator, metrics, DOCX I/O, and inference.

- [ ] Implement minimal code to satisfy the tests.
- [ ] Keep all edits whitelist/candidate based.
- [ ] Run `pytest -q`.

### Task 3: Model And Training Scaffold

**Files:**
- Create: `src/model/encoder.py`
- Create: `src/model/heads.py`
- Create: `src/model/edit_model.py`
- Create: `src/model/losses.py`
- Create: `src/training/*.py`

- [ ] Implement encoder-only Hugging Face loading.
- [ ] Reject seq2seq usage by design.
- [ ] Add LoRA-compatible training and save/load helpers.

### Task 4: Evaluation, Reports, Notebook, App

**Files:**
- Create: `src/evaluation/*.py`
- Create: `notebooks/main_pipeline.ipynb`
- Create: `src/app/streamlit_app.py`
- Create: `README.md`
- Create: `requirements.txt`
- Create: `configs/config.yaml`

- [ ] Implement evaluation reports and threshold sweep.
- [ ] Add a top-to-bottom notebook.
- [ ] Add Streamlit text and DOCX UI.
- [ ] Document install, train, eval, app, DOCX, reports, and limitations.

### Task 5: Verification

- [ ] Run `pytest -q`.
- [ ] Fix any failures.
- [ ] Report the verified status and any remaining limitations.
