"""FRED-T5 baseline evaluation on the project datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import silent_tqdm
from tqdm.auto import tqdm as tqdm_auto
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from quality_guard import levenshtein_distance


SAGE_FRED_T5_MODEL_NAME = "ai-forever/sage-fredt5-large"
DEFAULT_MODEL_NAME = SAGE_FRED_T5_MODEL_NAME
DEFAULT_PROMPT_TEMPLATE = (
    "<LM>Исправь орфографические и пунктуационные ошибки в русском тексте. "
    "Сохрани смысл и верни только исправленный текст:\n{text}"
)
MODEL_FILE_MANIFESTS = {
    "ai-forever/FRED-T5-1.7B": (
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "pytorch_model.bin",
    ),
    "ai-forever/sage-fredt5-large": (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    ),
    "ai-forever/FRED-T5-large-spell": (
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    ),
}
MODEL_FILE_SIZES = {
    "ai-forever/FRED-T5-1.7B": {
        "config.json": "653 B",
        "tokenizer_config.json": "19.74 KiB",
        "special_tokens_map.json": "574 B",
        "added_tokens.json": "2.44 KiB",
        "vocab.json": "1.63 MiB",
        "merges.txt": "1.21 MiB",
        "pytorch_model.bin": "6.48 GiB",
    },
    "ai-forever/sage-fredt5-large": {
        "model.safetensors": "3.06 GiB",
    },
    "ai-forever/FRED-T5-large-spell": {
        "model.safetensors": "3.06 GiB",
    },
}
PUNCT_CHARS = set('.,!?;:-"()[]{}—–«»')
SOURCE_META_COLUMNS = ("source_kind", "source_dataset", "source_domain")
SPECIAL_TOKEN_RE = re.compile(r"</s>|<pad>|<s>|<LM>|<SC\d+>|<extra_id_\d+>")


@dataclass(frozen=True)
class GeneratedPrediction:
    predicted_text: str
    raw_generated_text: str
    was_empty: bool
    used_input_fallback: bool


def char_error_rate(predicted: str, target: str) -> float:
    target = str(target)
    if not target:
        return 0.0 if not predicted else 1.0
    return levenshtein_distance(str(predicted), target) / len(target)


def punctuation_only(text: str) -> str:
    return "".join(ch for ch in str(text) if ch in PUNCT_CHARS)


def punctuation_similarity(predicted: str, target: str) -> float:
    return SequenceMatcher(None, punctuation_only(predicted), punctuation_only(target)).ratio()


def punctuation_count_error(predicted: str, target: str) -> int:
    pred_counts = Counter(punctuation_only(predicted))
    target_counts = Counter(punctuation_only(target))
    return sum(abs(pred_counts[ch] - target_counts[ch]) for ch in set(pred_counts) | set(target_counts))


def explode_error_types(value: str) -> list[str]:
    value = str(value or "unknown")
    labels: list[str] = []
    for chunk in value.replace("|", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            labels.append(chunk)
    return labels or ["unknown"]


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.mean()
    if pd.isna(value):
        return None
    return float(value)


def summarize_generation(df: pd.DataFrame, include_slices: bool = False) -> dict[str, Any]:
    clean_mask = df["input_cer"] == 0
    wrong_mask = df["input_cer"] > 0
    result: dict[str, Any] = {
        "examples": int(len(df)),
        "exact_match": _safe_mean(df["exact_match"]),
        "mean_input_cer": _safe_mean(df["input_cer"]),
        "mean_pred_cer": _safe_mean(df["pred_cer"]),
        "mean_cer_delta": _safe_mean(df["cer_delta"]),
        "improved_rate": _safe_mean(df["improved"]),
        "worse_rate": _safe_mean(df["worse"]),
        "unchanged_rate": _safe_mean(df["unchanged"]),
        "unchanged_wrong_rate": _safe_mean(df.loc[wrong_mask, "unchanged"]) if wrong_mask.any() else None,
        "clean_overcorrection_rate": (
            _safe_mean(df.loc[clean_mask, "clean_overcorrection"]) if clean_mask.any() else None
        ),
        "punct_input_similarity": _safe_mean(df["punct_input_similarity"]),
        "punct_pred_similarity": _safe_mean(df["punct_pred_similarity"]),
        "punct_similarity_delta": (
            float(df["punct_pred_similarity"].mean() - df["punct_input_similarity"].mean())
            if len(df)
            else None
        ),
        "mean_punct_count_error": _safe_mean(df["punct_count_error"]),
        "generation_empty_rate": _safe_mean(df["generation_empty"]),
        "input_fallback_rate": _safe_mean(df["used_input_fallback"]),
    }
    if include_slices:
        result["clean_slice"] = summarize_generation(df.loc[clean_mask].copy()) if clean_mask.any() else None
        result["dirty_slice"] = summarize_generation(df.loc[wrong_mask].copy()) if wrong_mask.any() else None
    return result


def summarize_by_error_type(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, value in enumerate(df["error_types"].fillna("unknown")):
        for label in explode_error_types(value):
            groups[label].append(idx)

    rows: list[dict[str, Any]] = []
    for label, indices in groups.items():
        if len(indices) < min_count:
            continue
        item = summarize_generation(df.iloc[indices].copy())
        item["error_type"] = label
        rows.append(item)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(["mean_cer_delta", "worse_rate"], ascending=[True, False])


def summarize_by_metadata(df: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    if column not in df.columns:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for value, part in df.groupby(column, dropna=False):
        key = str(value or "unknown")
        result[key] = summarize_generation(part.copy(), include_slices=False)
    return result


def format_prompt(text: str, prompt_template: str) -> str:
    if "{text}" in prompt_template:
        return prompt_template.format(text=text)
    return f"{prompt_template}{text}"


def clean_generated_text(
    raw_generated_text: str,
    *,
    source_text: str,
    fallback_to_input_on_empty: bool = True,
) -> GeneratedPrediction:
    raw_text = str(raw_generated_text or "")
    text = SPECIAL_TOKEN_RE.sub("", raw_text)
    for marker in (
        "Исправленный текст:",
        "Исправленный вариант:",
        "Ответ:",
        "Correction:",
        "Corrected text:",
    ):
        if marker in text:
            text = text.split(marker, 1)[1]
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    was_empty = not bool(text)
    used_input_fallback = bool(was_empty and fallback_to_input_on_empty)
    if used_input_fallback:
        text = str(source_text)
    return GeneratedPrediction(
        predicted_text=text,
        raw_generated_text=raw_text,
        was_empty=was_empty,
        used_input_fallback=used_input_fallback,
    )


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str = "auto", device: torch.device | None = None) -> torch.dtype | None:
    dtype = str(dtype or "auto").lower()
    if dtype in {"none", "float32", "fp32"}:
        return torch.float32 if dtype != "none" else None
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype == "auto":
        if device is not None and device.type == "cuda":
            return torch.float16
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def load_seq2seq_model(model_name: str, dtype: torch.dtype | None, *, local_files_only: bool = False):
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if local_files_only:
        kwargs["local_files_only"] = True
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    except TypeError:
        if "dtype" not in kwargs:
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)


def format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown size"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GiB"


def prepare_model_source(model_name: str, show_download_progress: bool = True) -> tuple[str, bool]:
    """Ensure a HF model is cached and return a source plus local-only flag."""
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return str(local_path), False

    print(f"Preparing Hugging Face model cache: {model_name}", flush=True)
    tqdm_class = tqdm_auto if show_download_progress else silent_tqdm
    manifest = MODEL_FILE_MANIFESTS.get(model_name)
    if manifest is None:
        model_source = snapshot_download(
            repo_id=model_name,
            max_workers=1,
            tqdm_class=tqdm_class,
            etag_timeout=30,
        )
        return model_source, False

    sizes = MODEL_FILE_SIZES.get(model_name, {})
    for filename in manifest:
        print(f"Downloading/checking {filename} ({sizes.get(filename, 'unknown size')})", flush=True)
        hf_hub_download(
            repo_id=model_name,
            filename=filename,
            etag_timeout=30,
            tqdm_class=tqdm_class,
        )
    return model_name, True


def clear_hf_progress_environment() -> None:
    if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1":
        print("HF_HUB_DISABLE_PROGRESS_BARS=1 detected; enabling Hugging Face progress bars for this run.")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"


def generate_predictions(
    texts: list[str],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    device: str = "auto",
    torch_dtype: str = "auto",
    batch_size: int = 4,
    max_input_tokens: int = 256,
    max_new_tokens: int = 192,
    use_dynamic_max_length: bool = True,
    max_length_multiplier: float = 1.5,
    num_beams: int = 1,
    do_sample: bool = False,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    length_penalty: float = 1.0,
    fallback_to_input_on_empty: bool = True,
    show_download_progress: bool = True,
) -> tuple[list[GeneratedPrediction], dict[str, Any]]:
    if show_download_progress:
        clear_hf_progress_environment()
    runtime_device = resolve_device(device)
    dtype = resolve_dtype(torch_dtype, runtime_device)
    model_source, local_files_only = prepare_model_source(
        model_name,
        show_download_progress=show_download_progress,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        eos_token="</s>",
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_seq2seq_model(model_source, dtype, local_files_only=local_files_only)
    model.to(runtime_device)
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "num_beams": int(num_beams),
        "do_sample": bool(do_sample),
        "repetition_penalty": float(repetition_penalty),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "length_penalty": float(length_penalty),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if num_beams and int(num_beams) > 1:
        generation_kwargs["early_stopping"] = True

    predictions: list[GeneratedPrediction] = []
    batch_size = max(1, int(batch_size))
    batch_starts = range(0, len(texts), batch_size)
    progress = tqdm_auto(
        batch_starts,
        total=(len(texts) + batch_size - 1) // batch_size,
        desc="FRED-T5 evaluation",
        disable=not show_download_progress,
    )
    for batch_start in progress:
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        prompts = [format_prompt(text, prompt_template) for text in batch_texts]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=max_input_tokens is not None,
            max_length=int(max_input_tokens) if max_input_tokens is not None else None,
        )
        encoded = {
            key: value.to(runtime_device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask"}
        }
        batch_generation_kwargs = dict(generation_kwargs)
        if use_dynamic_max_length:
            batch_generation_kwargs.pop("max_new_tokens", None)
            input_length = int(encoded["input_ids"].shape[1])
            batch_generation_kwargs["max_length"] = max(8, int(round(input_length * float(max_length_multiplier))))
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **batch_generation_kwargs)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(
            clean_generated_text(
                raw_text,
                source_text=source_text,
                fallback_to_input_on_empty=fallback_to_input_on_empty,
            )
            for raw_text, source_text in zip(decoded, batch_texts)
        )
        progress.set_postfix_str(f"{batch_end}/{len(texts)} examples")
        print(f"FRED-T5 evaluated {batch_end}/{len(texts)} examples", flush=True)

    runtime_info = {
        "model_name": model_name,
        "device": str(runtime_device),
        "torch_dtype": str(dtype).replace("torch.", "") if dtype is not None else None,
        "prompt_template": prompt_template,
        "batch_size": int(batch_size),
        "max_input_tokens": int(max_input_tokens) if max_input_tokens is not None else None,
        "max_new_tokens": int(max_new_tokens),
        "use_dynamic_max_length": bool(use_dynamic_max_length),
        "max_length_multiplier": float(max_length_multiplier),
        "num_beams": int(num_beams),
        "do_sample": bool(do_sample),
        "repetition_penalty": float(repetition_penalty),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "length_penalty": float(length_penalty),
        "fallback_to_input_on_empty": bool(fallback_to_input_on_empty),
        "download_progress": bool(show_download_progress),
        "model_source": model_source,
        "local_files_only": bool(local_files_only),
    }
    return predictions, runtime_info


def evaluate_fred_t5(
    dataset_path: str = "data/processed/test.csv",
    sample_size: int | None = 200,
    output_dir: str = "report/fred_t5_test_sample",
    split: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    device: str = "auto",
    torch_dtype: str = "auto",
    batch_size: int = 4,
    max_input_tokens: int = 256,
    max_new_tokens: int = 192,
    use_dynamic_max_length: bool = True,
    max_length_multiplier: float = 1.5,
    num_beams: int = 1,
    do_sample: bool = False,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    length_penalty: float = 1.0,
    fallback_to_input_on_empty: bool = True,
    show_download_progress: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_state)
    np.random.seed(random_state)

    df = pd.read_csv(dataset_path, low_memory=False)
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split].copy()
    required_columns = {"error_text", "correct_text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {dataset_path}: {sorted(missing)}")
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=int(sample_size), random_state=random_state).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    start = time.time()
    error_texts = df["error_text"].astype(str).tolist()
    predictions, runtime_info = generate_predictions(
        error_texts,
        model_name=model_name,
        prompt_template=prompt_template,
        device=device,
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        use_dynamic_max_length=use_dynamic_max_length,
        max_length_multiplier=max_length_multiplier,
        num_beams=num_beams,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        fallback_to_input_on_empty=fallback_to_input_on_empty,
        show_download_progress=show_download_progress,
    )
    if len(predictions) != len(df):
        raise RuntimeError(f"FRED-T5 returned {len(predictions)} predictions for {len(df)} inputs.")

    rows: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        error_text = str(row["error_text"])
        correct_text = str(row["correct_text"])
        prediction = predictions[i]
        predicted_text = prediction.predicted_text
        input_cer = char_error_rate(error_text, correct_text)
        pred_cer = char_error_rate(predicted_text, correct_text)
        is_clean = input_cer == 0
        unchanged = predicted_text.strip() == error_text.strip()
        row_data: dict[str, Any] = {
            "error_text": error_text,
            "correct_text": correct_text,
            "predicted_text": predicted_text,
            "raw_generated_text": prediction.raw_generated_text,
            "difficulty": row.get("difficulty", "unknown"),
            "error_types": row.get("error_types", "unknown"),
            "input_cer": input_cer,
            "pred_cer": pred_cer,
            "cer_delta": input_cer - pred_cer,
            "exact_match": predicted_text.strip() == correct_text.strip(),
            "improved": pred_cer < input_cer,
            "unchanged": unchanged,
            "worse": pred_cer > input_cer,
            "clean_overcorrection": bool(is_clean and not unchanged),
            "punct_input_similarity": punctuation_similarity(error_text, correct_text),
            "punct_pred_similarity": punctuation_similarity(predicted_text, correct_text),
            "punct_count_error": punctuation_count_error(predicted_text, correct_text),
            "generation_empty": bool(prediction.was_empty),
            "used_input_fallback": bool(prediction.used_input_fallback),
        }
        for column in SOURCE_META_COLUMNS:
            if column in df.columns:
                row_data[column] = row.get(column, "")
        rows.append(row_data)

    analysis = pd.DataFrame(rows)
    output_path = Path(output_dir)
    analysis.to_csv(output_path / "error_analysis.csv", index=False, encoding="utf-8")
    by_type = summarize_by_error_type(analysis)
    by_type.to_csv(output_path / "error_metrics_by_type.csv", index=False, encoding="utf-8")
    analysis.sort_values("cer_delta").head(50).to_csv(
        output_path / "worst_cases.csv",
        index=False,
        encoding="utf-8",
    )

    summary = summarize_generation(analysis, include_slices=True)
    summary.update(runtime_info)
    summary["runtime_version"] = "fred_t5_baseline_v1"
    summary["dataset_path"] = str(dataset_path)
    summary["split"] = split
    summary["sample_size"] = sample_size
    summary["elapsed_minutes"] = (time.time() - start) / 60
    for column in ("source_kind", "source_dataset"):
        slices = summarize_by_metadata(analysis, column)
        if slices:
            summary[f"{column}_slices"] = slices

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FRED-T5 as a seq2seq correction baseline.")
    parser.add_argument("--dataset", default="data/processed/test.csv")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--all", action="store_true", help="Evaluate the full dataset.")
    parser.add_argument("--output-dir", default="report/fred_t5_test_sample")
    parser.add_argument("--split", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16", "none"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--disable-dynamic-max-length", action="store_true")
    parser.add_argument("--max-length-multiplier", type=float, default=1.5)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-input-fallback", action="store_true")
    parser.add_argument("--no-download-progress", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_fred_t5(
        dataset_path=args.dataset,
        sample_size=None if args.all else args.sample_size,
        output_dir=args.output_dir,
        split=args.split,
        model_name=args.model_name,
        prompt_template=args.prompt_template,
        device=args.device,
        torch_dtype=args.torch_dtype,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        use_dynamic_max_length=not args.disable_dynamic_max_length,
        max_length_multiplier=args.max_length_multiplier,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
        fallback_to_input_on_empty=not args.no_input_fallback,
        show_download_progress=not args.no_download_progress,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
