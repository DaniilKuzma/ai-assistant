# Test Audit Report

Date: 2026-05-18

## Initial Status

- Command requested by task: `python -m pytest -q`
  - Result: failed before pytest collection in the current shell.
  - Cause: `python` was not on PATH (`/bin/bash: line 1: python: command not found`).
- Command used with the project virtual environment: `.venv/bin/python -m pytest -q`
  - Result: `5 failed, 305 passed in 32.51s`.

## Failing Tests And Root Causes

| Test | Category | Root cause | Fix |
| --- | --- | --- | --- |
| `tests/test_docx_io.py::test_docx_correction_preserves_paragraph_count` | Outdated model-centric expectation | The test expected default `Corrector()` inside `correct_docx()` to apply candidate-only/model-scored split/join and comma edits. Plain `Corrector()` is intentionally conservative and only applies deterministic fallback edits. | Updated the test to inject a `TrainedModelCorrector` with a fake backend that scores the split/join candidate above threshold and predicts the comma. |
| `tests/test_docx_io.py::test_docx_correction_preserves_first_run_bold_formatting` | Outdated model-centric expectation | Same as above; the formatting assertion was valid, but the correction source had to be model-backed, not plain fallback. | Updated the test to inject the fake model-backed corrector and assert accepted model/validator edits. |
| `tests/test_docx_io.py::test_docx_correction_preserves_inline_run_formatting_boundaries` | Outdated model-centric expectation | Same as above; inline run boundary preservation should be tested after model-approved edits. | Updated the test to inject the fake model-backed corrector and keep the run-formatting assertions. |
| `tests/test_syntax_layer.py::test_parse_syntax_returns_stable_tokens_for_russian_sentence` | Broken dependency/environment | `natasha` was declared in `requirements.txt` but missing from `.venv`, so `parse_syntax()` returned `[]` through its optional-stack fallback path. | Synced `.venv` from `requirements.txt`; `natasha==1.6.0` is now installed. |
| `tests/test_syntax_layer.py::test_parse_syntax_reuses_cached_pipeline` | Broken dependency/environment | Because `syntax_pipeline()` raised `ModuleNotFoundError`, the LRU cache never retained a pipeline and misses increased on every call. | Synced `.venv` from `requirements.txt`; syntax pipeline caching now works with the installed dependency. |

## Changes Made

- Updated `tests/test_docx_io.py` only:
  - added a local fake model backend;
  - used existing `correct_docx(..., corrector=...)` injection;
  - kept paragraph count and run formatting checks;
  - added assertions that validator-accepted model-backed edits are present.
- Installed declared dependency packages into `.venv` via `python -m pip install -r requirements.txt`.
- Added this audit report.

## Verification

- Targeted check after dependency sync and DOCX test rewrite:
  - Command: `python -m pytest tests/test_docx_io.py tests/test_syntax_layer.py -q`
  - Result: `6 passed in 3.38s`.
- Full check before writing this report:
  - Command: `python -m pytest -q`
  - Result: `310 passed in 37.14s`.
- Final full-suite status after this report was added:
  - Command: `python -m pytest -q`
  - Result: `310 passed in 33.09s`.

## Remaining Issues

- No code-level blockers identified.
- Environment note: running `python -m pytest -q` requires activating `.venv` in this shell because bare `python` was not available before activation.
