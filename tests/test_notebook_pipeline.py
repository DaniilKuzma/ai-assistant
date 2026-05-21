from pathlib import Path
import json

import yaml


def test_main_pipeline_notebook_executes_top_to_bottom(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING", "1")
    monkeypatch.setenv("RUSSIAN_CORRECTOR_DATASET_LIMIT", "30")
    notebook_path = Path("notebooks/main_pipeline.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook_test__"}

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
            if "config = load_config(config_path)" in "".join(cell["source"]):
                config = namespace["config"]
                config["data"]["processed_train_path"] = str(tmp_path / "data" / "processed" / "correction_dataset.csv.gz")
                config["data"]["manifest_path"] = str(tmp_path / "reports" / "dataset_manifest.json")
                config["data"]["use_external_sources"] = False
                config["data"].setdefault("clean_corpus", {})["enabled"] = False
                config["data"].setdefault("training_dataset", {})["enabled"] = False
                config["data"].setdefault("training_dataset_core", {})["enabled"] = False
                config["data"].pop("exact_split_sizes", None)
                config["data"].pop("train_examples", None)
                config["data"].pop("val_examples", None)
                config["data"].pop("test_examples", None)
                config["data"]["target_total_examples"] = 30
                config["data"]["debug_clean_texts"] = [
                    f"В тестовой записи редактор не знает, что делать, и пишет по-русски в разделе {index}."
                    for index in range(12)
                ]
                config["paths"]["adapter_output_dir"] = str(tmp_path / "models" / "adapters" / "current")
                config["paths"]["heads_output_dir"] = str(tmp_path / "models" / "heads" / "current")
                config["paths"]["reports_dir"] = str(tmp_path / "reports")
                config_path = tmp_path / "config.yaml"
                config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
                namespace["config_path"] = config_path

    assert len(notebook["cells"]) > 3
    assert (tmp_path / "reports" / "evaluation_summary.csv").exists()
