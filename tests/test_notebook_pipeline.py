from pathlib import Path
import json

import yaml

from tests.candidate_contract_fixtures import candidate_contract_config, patch_unit_operator_pipeline, write_unit_clean_pool


def test_main_pipeline_notebook_executes_top_to_bottom(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING", "1")
    monkeypatch.setenv("RUSSIAN_CORRECTOR_DATASET_LIMIT", "10")
    patch_unit_operator_pipeline(monkeypatch)
    notebook_path = Path("notebooks/main_pipeline.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook_test__"}

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
            if "config = load_config(config_path)" in "".join(cell["source"]):
                clean_pool_path = write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
                config = candidate_contract_config(tmp_path, clean_pool_path)
                config["paths"]["adapter_output_dir"] = str(tmp_path / "models" / "adapters" / "current")
                config["paths"]["heads_output_dir"] = str(tmp_path / "models" / "heads" / "current")
                config["paths"]["reports_dir"] = str(tmp_path / "reports")
                config["training"]["feature_build"]["enable_syntax"] = False
                config["training"]["show_progress"] = False
                config["data"]["use_external_sources"] = False
                config["data"].setdefault("clean_corpus", {})["enabled"] = False
                config_path = tmp_path / "config.yaml"
                config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
                namespace["config"] = config
                namespace["config_path"] = config_path

    assert len(notebook["cells"]) > 3
    assert (tmp_path / "reports" / "evaluation_summary.csv").exists()
