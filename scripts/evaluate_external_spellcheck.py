from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_pair_dataset import _summary, _write_report
from src.config.load_config import load_config
from src.evaluation.direct_metrics import normalize_text_for_eval
from src.runtime.corrector import Corrector


def evaluate_external_spellcheck(
    config_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    corrector = Corrector.from_config(config, strict_neural=True, allow_fallback=allow_fallback)
    rows = _load_rows(Path(dataset_path), limit=limit)
    evaluated = [_evaluate_external_row(corrector, row, index) for index, row in enumerate(rows)]
    backend = _backend_metadata(corrector, evaluated)
    subsets = {
        "all": evaluated,
        "in_scope_by_simple_diff_filter": [row for row in evaluated if bool(row["in_scope_by_simple_diff_filter"])],
        "clean_no_change": [row for row in evaluated if not bool(row["changed_needed"])],
    }
    summary = {
        name: _summary(subset_rows, backend, allow_fallback=allow_fallback)
        for name, subset_rows in subsets.items()
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output / "all_examples", summary["all"], evaluated)
    return summary


def _evaluate_external_row(corrector: Any, row: Mapping[str, Any], index: int) -> dict[str, Any]:
    source = str(row.get("source") or "")
    target = str(row.get("correction") or row.get("target") or "")
    result = corrector.correct(source)
    normalized_source = normalize_text_for_eval(source)
    normalized_target = normalize_text_for_eval(target)
    normalized_prediction = normalize_text_for_eval(result.corrected_text)
    return {
        "row_id": index,
        "source": source,
        "target": target,
        "prediction": result.corrected_text,
        "tags": ["external_spellcheck"],
        "scope": "diagnostic",
        "exact_match": normalized_prediction == normalized_target,
        "changed_needed": normalized_source != normalized_target,
        "prediction_changed": normalized_source != normalized_prediction,
        "in_scope_by_simple_diff_filter": _is_simple_in_scope_diff(source, target),
        "runtime_metadata": dict(result.metadata),
        "metadata": {key: value for key, value in row.items() if key not in {"source", "correction", "target"}},
    }


def _is_simple_in_scope_diff(source: str, target: str) -> bool:
    normalized_source = normalize_text_for_eval(source)
    normalized_target = normalize_text_for_eval(target)
    if normalized_source == normalized_target:
        return False
    if not normalized_source or not normalized_target:
        return False
    if "\n" in normalized_source or "\n" in normalized_target:
        return False
    if abs(len(normalized_source) - len(normalized_target)) > 24:
        return False
    if SequenceMatcher(None, normalized_source, normalized_target).ratio() < 0.70:
        return False
    return _changed_token_count(normalized_source, normalized_target) <= 4


def _changed_token_count(source: str, target: str) -> int:
    source_tokens = source.split()
    target_tokens = target.split()
    matcher = SequenceMatcher(None, source_tokens, target_tokens)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")


def _load_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        raw_rows = json.loads(text)
        if not isinstance(raw_rows, list):
            raise ValueError(f"Expected a JSON array in {path}")
        rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
    else:
        rows = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
        rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def _backend_metadata(corrector: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    backend = getattr(corrector, "neural_backend", None)
    metadata = {
        "backend_kind": "direct_neural" if backend is not None else "deterministic_fallback",
        "model_loaded": backend is not None,
        "adapter_path": str(getattr(backend, "adapter_path", "") or ""),
        "heads_path": str(getattr(backend, "heads_path", "") or ""),
        "selected_epoch": getattr(backend, "selected_epoch", None),
    }
    if rows:
        runtime_metadata = rows[0].get("runtime_metadata")
        if isinstance(runtime_metadata, Mapping):
            for key in ("backend_kind", "model_loaded", "adapter_path", "heads_path", "selected_epoch"):
                if runtime_metadata.get(key) not in (None, ""):
                    metadata[key] = runtime_metadata.get(key)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate strict neural corrector on external spellcheck source/correction JSON.")
    parser.add_argument("config")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args(argv)
    summary = evaluate_external_spellcheck(
        args.config,
        args.dataset,
        args.output,
        limit=args.limit,
        allow_fallback=args.allow_fallback,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
