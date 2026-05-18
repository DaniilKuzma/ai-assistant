# Russian Edit Corrector Design

## Goal

Build a controlled Russian spelling and punctuation assistant from scratch. The system corrects only spelling and punctuation, never style, meaning, grammar outside spelling, word order, or semantic content.

## Chosen Architecture

Two edit-based variants were considered:

1. Sequence tagging edit model: simple token labels such as KEEP, insert comma, split, or case fix. It is compact, but weaker at controlling exact word replacements.
2. Candidate-aware edit ranking model with punctuation gap tagging: each word/span gets a bounded candidate list and the model can choose only KEEP or an allowed candidate. Punctuation is handled by a separate gap head.

The project uses the second option because it gives strict control over possible edits and prevents generative rewrites.

## Components

- `preprocessing`: Russian tokenization, sentence boundaries, and protected spans for URLs, email, dates, numbers, usernames, code-like fragments, and English words.
- `candidates`: whitelist-backed spelling, split/join, hyphen, frequent-error, and sentence-start case candidates.
- `validation`: diff analysis and strict pre/post validation. Only spelling, punctuation, split/join whitelist, hyphen whitelist, sentence-start capitalization, and final punctuation are accepted.
- `alignment`: source-target alignment for filtering datasets and building edit/punctuation labels.
- `model`: encoder-only multitask model using `ai-forever/ruRoberta-large`, LoRA-compatible, with candidate scoring, punctuation gap, confidence, and error-type heads.
- `training`: configurable training pipeline with PEFT/LoRA, mixed precision, gradient accumulation, and checkpoint saving for adapters, heads, tokenizer reference, label mappings, and thresholds.
- `inference`: iterative correction with `max_passes`, thresholds, candidate ranking, validator, and edit realization.
- `evaluation`: exact match, edit F1, spelling/punctuation F1, dirty improved/worse, clean overcorrection, grouped reports, and threshold sweeps.
- `docx`: paragraph-preserving Word document correction.
- `app`: Streamlit UI for text and DOCX correction.
- `notebooks/main_pipeline.ipynb`: single top-to-bottom pipeline entrypoint.

## Data Flow

Input text is tokenized and protected spans are marked. Candidate generation produces bounded word/span candidates. The encoder scores candidates and punctuation gaps. The strict validator filters all proposed edits. The edit realizer applies accepted edits only, and iterative decoding stops on stabilization or `max_passes`.

For training, source-target pairs are diffed and aligned. Examples that cannot be decomposed into allowed edits are removed. Synthetic examples are generated only from allowed spelling, punctuation, split/join, hyphen, and final-punctuation classes.

## Verification

The MVP must run without downloading large models during tests. Unit and smoke tests cover candidate generation, diff classification, strict validation, alignment, synthetic generation, metrics, DOCX I/O, and inference. Heavy Hugging Face loading is lazy and only happens when training or model-backed inference is explicitly invoked.
