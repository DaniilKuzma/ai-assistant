from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_frozen_eval import build_frozen_eval
from src.app.streamlit_app import build_streamlit_corrector
from src.config.load_config import load_config
from src.evaluation.evaluate import evaluate_corrector
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.factory import online_generator_from_config
from src.runtime.corrector import Corrector
from src.runtime.tokenization import tokenize_runtime_words
from src.schema.edits import CorrectionResult
from src.schema.serialization import write_jsonl_examples
from src.training.trainer import train_model


FIXED_RUNTIME_EXAMPLES = (
    "\u041e\u043d \u043d\u0435\u0437\u043d\u0430\u043b \u0447\u0442\u043e \u0434\u0435\u043b\u0430\u0442\u044c",
    "\u041a\u043e\u043c\u0438\u0441\u0441\u0438\u044f \u0442\u0430\u043a \u0436\u0435 \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b\u0430 \u043e\u0442\u0447\u0451\u0442",
    "\u0421\u0442\u0443\u0434\u0435\u043d\u0442 \u0445\u043e\u0442\u0435\u043b \u0443\u0447\u0438\u0442\u0441\u044f",
    "\u0420\u0435\u0434\u0430\u043a\u0442\u043e\u0440 \u0441\u043a\u0430\u0437\u0430\u043b \u0447\u0442\u043e \u043e\u0442\u0447\u0451\u0442 \u0433\u043e\u0442\u043e\u0432",
)


@dataclass(frozen=True)
class _FakePrediction:
    token_labels: list[str]
    token_confidences: list[float]
    gap_labels: list[str]
    gap_confidences: list[float]
    rule_ids: list[str]
    token_margins: list[float]
    gap_margins: list[float]


class _FakeDirectNeuralBackend:
    def predict(self, text: str) -> _FakePrediction:
        words = tokenize_runtime_words(text)
        lowered = [word.text.lower() for word in words]
        token_labels = ["KEEP"] * len(words)
        gap_labels = ["NONE"] * len(words)
        rule_ids = ["none"] * len(words)

        for index, word in enumerate(lowered):
            if word == "\u043d\u0435\u0437\u043d\u0430\u043b":
                token_labels[index] = "SPLIT_NE_VERB"
                rule_ids[index] = "ne_verb"
            elif word == "\u0443\u0447\u0438\u0442\u0441\u044f":
                token_labels[index] = "FIX_TSYA_TO_TTSYA"
                rule_ids[index] = "tsya_ttsya"

        for index in range(len(lowered) - 1):
            pair = (lowered[index], lowered[index + 1])
            if pair == ("\u0442\u0430\u043a", "\u0436\u0435"):
                token_labels[index] = "MERGE_TAK_ZHE_TO_TAKZHE"
                rule_ids[index] = "takzhe_tak_zhe"
            elif pair in {
                ("\u0437\u043d\u0430\u043b", "\u0447\u0442\u043e"),
                ("\u0441\u043a\u0430\u0437\u0430\u043b", "\u0447\u0442\u043e"),
            }:
                gap_labels[index] = "COMMA"
                rule_ids[index] = "comma_subordinate"

        return _FakePrediction(
            token_labels=token_labels,
            token_confidences=[1.0] * len(words),
            gap_labels=gap_labels,
            gap_confidences=[1.0] * len(words),
            rule_ids=rule_ids,
            token_margins=[1.0] * len(words),
            gap_margins=[1.0] * len(words),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a small end-to-end smoke pipeline.")
    parser.add_argument("config", help="Path to the project YAML config.")
    parser.add_argument("--output", required=True, help="Directory for smoke artifacts.")
    parser.add_argument("--count", type=int, default=100, help="Generated audit and frozen val example count.")
    parser.add_argument("--steps", type=int, default=2, help="Debug-model training steps.")
    args = parser.parse_args(argv)

    if args.count <= 0:
        parser.error("--count must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    summary_path = output_dir / "e2e_summary.json"
    state = _SmokeState(
        config_path=config_path,
        output_dir=output_dir,
        count=args.count,
        steps=args.steps,
    )
    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "count": args.count,
        "steps": args.steps,
    }

    stages: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("generator_audit", state.run_generator_audit),
        ("frozen_eval", state.run_frozen_eval),
        ("training_smoke", state.run_training_smoke),
        ("evaluation", state.run_evaluation),
        ("runtime_corrector", state.run_runtime_corrector),
        ("streamlit_builder", state.run_streamlit_builder),
    ]

    for stage_name, stage in stages:
        try:
            summary[stage_name] = {"status": "ok", **stage()}
        except Exception as exc:
            summary[stage_name] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(summary_path, summary)
            print(json.dumps(summary[stage_name], ensure_ascii=False, sort_keys=True), file=sys.stderr)
            return 1

    _assert_no_legacy_train_csv()
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


@dataclass
class _SmokeState:
    config_path: Path
    output_dir: Path
    count: int
    steps: int

    @property
    def val_path(self) -> Path:
        return self.output_dir / "val.jsonl"

    @property
    def frozen_eval_config_path(self) -> Path:
        return self.output_dir / "frozen_eval_smoke_config.yaml"

    @property
    def training_config_path(self) -> Path:
        return self.output_dir / "training_smoke_config.yaml"

    @property
    def runtime_config_path(self) -> Path:
        return self.output_dir / "deterministic_runtime_config.yaml"

    def run_generator_audit(self) -> dict[str, Any]:
        config = load_config(self.config_path)
        generator = online_generator_from_config(config, seed=int(config.get("generation", {}).get("seed", 0)))
        examples = [generator.sample_by_index(index) for index in range(self.count)]
        audit = audit_batch(examples)
        if audit["failed_examples_count"] > 0:
            raise RuntimeError(f"generator audit failed: {audit['failure_reasons']}")
        return {
            "example_count": len(examples),
            "rule_distribution": audit["rule_distribution"],
            "mode_distribution": audit["mode_distribution"],
            "audit_failures_count": audit["failed_examples_count"],
        }

    def run_frozen_eval(self) -> dict[str, Any]:
        config = _smoke_frozen_eval_config(load_config(self.config_path), self.count)
        _write_yaml(self.frozen_eval_config_path, config)
        manifest, examples = build_frozen_eval(
            config_path=self.frozen_eval_config_path,
            split="val",
            count=self.count,
            output_path=self.val_path,
        )
        if manifest["audit_failures_count"] > 0:
            raise RuntimeError(f"frozen eval audit failed: {manifest['audit_failure_reasons']}")
        write_jsonl_examples(self.val_path, examples)
        manifest_path = self.val_path.with_suffix(".manifest.json")
        _write_json(manifest_path, manifest)
        return {
            "example_count": len(examples),
            "config_path": str(self.frozen_eval_config_path),
            "dataset_path": str(self.val_path),
            "manifest_path": str(manifest_path),
            "audit_failures_count": manifest["audit_failures_count"],
        }

    def run_training_smoke(self) -> dict[str, Any]:
        config = _smoke_training_config(load_config(self.config_path), self.output_dir)
        _write_yaml(self.training_config_path, config)
        result = train_model(
            self.training_config_path,
            overrides={"smoke": True, "steps": self.steps},
            debug_model=True,
        )
        return {
            "config_path": str(self.training_config_path),
            "summary_path": result["summary_path"],
            "heads_path": result["heads_path"],
            "actual_train_steps": result["actual_train_steps"],
            "debug_model": result["debug_model"],
            "smoke": result["smoke"],
        }

    def run_evaluation(self) -> dict[str, Any]:
        config = _deterministic_runtime_config(load_config(self.config_path), self.output_dir)
        _write_yaml(self.runtime_config_path, config)
        eval_dir = self.output_dir / "eval_val"
        summary = evaluate_corrector(self.runtime_config_path, self.val_path, eval_dir)
        summary_path = eval_dir / "evaluation_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"evaluation summary was not created: {summary_path}")
        return {
            "config_path": str(self.runtime_config_path),
            "output_dir": str(eval_dir),
            "summary_path": str(summary_path),
            "example_count": summary["example_count"],
        }

    def run_runtime_corrector(self) -> dict[str, Any]:
        corrector = Corrector(
            neural_backend=_FakeDirectNeuralBackend(),
            config={
                "runtime": {
                    "deterministic_first": False,
                    "neural_token_edits": True,
                    "neural_punctuation": True,
                    "confidence_thresholds": {
                        "token_edit": 0.0,
                        "punctuation": 0.0,
                    },
                }
            },
        )
        rows = []
        for source_text in FIXED_RUNTIME_EXAMPLES:
            result = corrector.correct(source_text)
            if not isinstance(result, CorrectionResult):
                raise TypeError(f"corrector returned {type(result).__name__}, expected CorrectionResult")
            if not isinstance(result.corrected_text, str):
                raise TypeError("CorrectionResult.corrected_text must be a string")
            serialized_edits = [_serialize_edit(edit) for edit in result.edits]
            json.dumps(serialized_edits, ensure_ascii=False)
            rows.append(
                {
                    "source_text": result.source_text,
                    "corrected_text": result.corrected_text,
                    "edits": serialized_edits,
                    "metadata": dict(result.metadata),
                }
            )

        runtime_path = self.output_dir / "runtime_examples.json"
        _write_json(runtime_path, rows)
        return {
            "example_count": len(rows),
            "result_path": str(runtime_path),
            "changed_count": sum(1 for row in rows if row["source_text"] != row["corrected_text"]),
        }

    def run_streamlit_builder(self) -> dict[str, Any]:
        if not self.runtime_config_path.exists():
            config = _deterministic_runtime_config(load_config(self.config_path), self.output_dir)
            _write_yaml(self.runtime_config_path, config)
        result = build_streamlit_corrector(self.runtime_config_path)
        if result.kind != "deterministic_fallback":
            raise RuntimeError(f"expected deterministic_fallback, got {result.kind}")
        return {
            "config_path": str(self.runtime_config_path),
            "kind": result.kind,
            "corrector_class": result.corrector.__class__.__name__,
        }


def _smoke_training_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config_copy = copy.deepcopy(config)
    model = config_copy.setdefault("model", {})
    model["local_files_only"] = True
    model["max_sequence_length"] = min(64, int(model.get("max_sequence_length", 64)))
    lora = model.setdefault("lora", {})
    lora["enabled"] = False

    training = config_copy.setdefault("training", {})
    training.update(
        {
            "batch_size": 2,
            "eval_batch_size": 2,
            "epochs": 1,
            "mixed_precision": False,
            "num_workers": 0,
            "save_each_epoch": False,
            "validate_each_epoch": True,
            "gradient_accumulation_steps": 1,
        }
    )

    paths = config_copy.setdefault("paths", {})
    paths["generated_eval_dir"] = str(output_dir)
    paths["adapter_output_dir"] = str(output_dir / "models" / "adapters")
    paths["heads_output_dir"] = str(output_dir / "models" / "heads")
    return config_copy


def _smoke_frozen_eval_config(config: dict[str, Any], count: int) -> dict[str, Any]:
    config_copy = copy.deepcopy(config)
    generation = config_copy.setdefault("generation", {})
    training = config_copy.setdefault("training", {})
    generation["samples_per_epoch"] = max(count, int(training.get("batch_size", 1) or 1))
    training["epochs"] = 1
    return config_copy


def _deterministic_runtime_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config_copy = copy.deepcopy(config)
    runtime = config_copy.setdefault("runtime", {})
    runtime["neural_token_edits"] = False
    runtime["neural_punctuation"] = False

    paths = config_copy.setdefault("paths", {})
    paths["generated_eval_dir"] = str(output_dir)
    paths["adapter_output_dir"] = str(output_dir / "models" / "adapters")
    paths["heads_output_dir"] = str(output_dir / "models" / "heads")
    return config_copy


def _serialize_edit(edit: Any) -> dict[str, Any]:
    return {
        "start": int(getattr(edit, "start", 0)),
        "end": int(getattr(edit, "end", 0)),
        "source": str(getattr(edit, "source", "")),
        "replacement": str(getattr(edit, "replacement", "")),
        "edit_type": str(getattr(edit, "edit_type", "")),
        "rule_id": str(getattr(edit, "rule_id", "")),
        "confidence": float(getattr(edit, "confidence", 0.0)),
        "explanation": str(getattr(edit, "explanation", "")),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _assert_no_legacy_train_csv() -> None:
    train_csv = PROJECT_ROOT / "data" / "processed" / "train.csv"
    if train_csv.exists():
        raise RuntimeError(f"legacy train.csv exists after smoke run: {train_csv}")


if __name__ == "__main__":
    raise SystemExit(main())
