from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.load_config import load_config


COLAB_COMMANDS = [
    "pip install -r requirements.txt",
    "python scripts/build_short_dataset_v3.py --config configs/config.short_dataset_v3.yaml --force",
    "RUN_WAVE1_TRAINING=1 python scripts/run_wave1_v3_training.py --config configs/config.train_short_v3_e1.yaml",
    "RUN_WAVE1_TRAINING=1 python scripts/run_wave1_v3_training.py --config configs/config.train_short_v3_e2.yaml --only-if-e1-passed",
    "python scripts/calibrate_thresholds.py --config configs/config.train_short_v3_e1.yaml --split val --baseline-mode calibrated_val_guarded_v3 --batch-size 256",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate and run Wave 1 v3 training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-config", default="configs/config.yaml")
    parser.add_argument("--only-if-e1-passed", action="store_true")
    args = parser.parse_args()

    config = _merged_config(load_config(args.base_config), load_config(args.config))
    if args.only_if_e1_passed and not _e1_passed():
        print(json.dumps(_ready_for_colab("E1 gate report is absent or failed; E2 not started"), ensure_ascii=False, indent=2))
        return
    if os.environ.get("RUN_WAVE1_TRAINING") != "1":
        print(json.dumps(_ready_for_colab("RUN_WAVE1_TRAINING is not set to 1"), ensure_ascii=False, indent=2))
        return
    cuda_status = _cuda_status()
    if not cuda_status["available"]:
        print(json.dumps(_ready_for_colab(cuda_status["reason"]), ensure_ascii=False, indent=2))
        return

    from src.training.train import train

    result = train(config)
    gate_result = _write_gate_report(config, result)
    print(json.dumps({"verdict": gate_result["verdict"], "training": result, "gate": gate_result}, ensure_ascii=False, indent=2))


def _cuda_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return {"available": False, "reason": f"torch import failed: {exc}"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA is not available"}
    return {"available": True, "reason": ""}


def _ready_for_colab(reason: str) -> dict[str, Any]:
    return {
        "verdict": "READY_FOR_COLAB_TRAINING",
        "training_ran": False,
        "reason": reason,
        "commands": COLAB_COMMANDS,
    }


def _e1_passed() -> bool:
    gate_path = Path("reports/train_short_v3_e1/e1_gate_report.json")
    if not gate_path.exists():
        return False
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(gate.get("passed"))


def _write_gate_report(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports/train_short_v3_e1"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(result.get("metrics") or result.get("evaluation_metrics") or {})
    gates = {
        "clean_overcorrection_rate": _float(metrics.get("clean_overcorrection_rate")) <= 0.007,
        "dirty_worse_rate": _float(metrics.get("dirty_worse_rate")) <= 0.005,
        "edit_precision": _float(metrics.get("edit_precision")) >= 0.90,
        "punctuation_f1": _float(metrics.get("punctuation_f1")) >= 0.65,
        "spelling_f1": _float(metrics.get("spelling_f1")) >= 0.10,
    }
    passed = all(gates.values()) and bool(result.get("model_training_ran"))
    report = {"passed": passed, "verdict": "TRAINING_GATE_PASSED" if passed else "NEEDS_TARGETED_FIX", "gates": gates, "metrics": metrics}
    filename = "e2_gate_report.json" if str(config.get("training", {}).get("mode", "")).endswith("e2") else "e1_gate_report.json"
    (reports_dir / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _merged_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merged_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
