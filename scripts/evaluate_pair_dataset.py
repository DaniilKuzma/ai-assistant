from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.evaluation.direct_metrics import exact_match, normalize_text_for_eval
from src.runtime.corrector import Corrector


def evaluate_pair_dataset(
    config_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    corrector = Corrector.from_config(config, strict_neural=True, allow_fallback=allow_fallback)
    rows = _read_jsonl(Path(dataset_path))
    evaluated = [_evaluate_row(corrector, row, index) for index, row in enumerate(rows)]
    summary = _summary(evaluated, _backend_metadata(corrector, evaluated), allow_fallback=allow_fallback)
    _write_report(Path(output_dir), summary, evaluated)
    return summary


def _evaluate_row(corrector: Any, row: Mapping[str, Any], index: int) -> dict[str, Any]:
    source = str(row.get("source") or "")
    target = str(row.get("target") or row.get("correction") or "")
    result = corrector.correct(source)
    tags = _string_list(row.get("tags")) or ["untagged"]
    prediction = result.corrected_text
    return {
        "row_id": index,
        "source": source,
        "target": target,
        "prediction": prediction,
        "tags": tags,
        "scope": str(row.get("scope") or ""),
        "exact_match": exact_match(target, prediction),
        "changed_needed": normalize_text_for_eval(source) != normalize_text_for_eval(target),
        "prediction_changed": normalize_text_for_eval(source) != normalize_text_for_eval(prediction),
        "runtime_metadata": dict(result.metadata),
    }


def _summary(rows: Sequence[Mapping[str, Any]], backend: Mapping[str, Any], *, allow_fallback: bool) -> dict[str, Any]:
    exact = sum(int(bool(row["exact_match"])) for row in rows)
    changed = [row for row in rows if bool(row["changed_needed"])]
    clean = [row for row in rows if not bool(row["changed_needed"])]
    per_tag: dict[str, dict[str, Any]] = {}
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for tag in _string_list(row.get("tags")):
            buckets[tag].append(row)
    for tag, tag_rows in sorted(buckets.items()):
        per_tag[tag] = {
            "example_count": len(tag_rows),
            "exact_match": _rate(sum(int(bool(row["exact_match"])) for row in tag_rows), len(tag_rows)),
        }
    clean_overcorrected = sum(int(bool(row["prediction_changed"])) for row in clean)
    return {
        "example_count": len(rows),
        "exact_match": _rate(exact, len(rows)),
        "changed_when_needed": _rate(sum(int(bool(row["prediction_changed"])) for row in changed), len(changed)),
        "unchanged_when_clean": _rate(
            sum(int(not bool(row["prediction_changed"])) for row in clean),
            len(clean),
        ),
        "overcorrection_rate": _rate(clean_overcorrected, len(clean)),
        "per_tag": per_tag,
        "allow_fallback": bool(allow_fallback),
        **dict(backend),
    }


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


def _write_report(output_dir: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "worst_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            if bool(row.get("exact_match")):
                continue
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate strict neural corrector on a source/target JSONL pair dataset.")
    parser.add_argument("config")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args(argv)
    summary = evaluate_pair_dataset(args.config, args.dataset, args.output, allow_fallback=args.allow_fallback)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
