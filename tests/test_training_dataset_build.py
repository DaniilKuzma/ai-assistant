from pathlib import Path

import json
import pandas as pd
import yaml

from src.candidates.candidate_generator import Candidate
from src.config.load_config import load_config
from src.data.corruption_operators import CorruptionResult, Opportunity, RuleOperatorRegistry, _verification
from src.data.full_dataset_builder import build_dataset_from_config
from src.rules.capabilities import RuleCapability


def test_canonical_config_uses_broad_dataset_targets_and_no_versioned_paths():
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))

    assert config["data"]["processed_train_path"] == "data/processed/correction_dataset.csv.gz"
    assert config["data"]["manifest_path"] == "data/processed/dataset_manifest.json"
    assert config["data"]["total_examples"] >= 200000
    assert config["data"]["train_examples"] == int(config["data"]["total_examples"] * 0.8)
    assert config["data"]["val_examples"] == int(config["data"]["total_examples"] * 0.1)
    assert config["data"]["test_examples"] == (
        config["data"]["total_examples"] - config["data"]["train_examples"] - config["data"]["val_examples"]
    )
    assert config["training"]["max_train_examples"] == config["data"]["train_examples"]
    assert config["training"]["max_val_examples"] == config["data"]["val_examples"]
    assert config["training"]["max_test_examples"] == config["data"]["test_examples"]

    serialized = yaml.safe_dump(config["data"], allow_unicode=True)
    forbidden = ("short_dataset_v2", "short_dataset_v3", "current_capability_v", "wave", "phase", "latest")
    assert not any(marker in serialized for marker in forbidden)


class UnitCandidateGenerator:
    def generate(self, text: str):
        start = text.find("млоко")
        if start >= 0:
            return [
                Candidate(
                    source="млоко",
                    replacement="молоко",
                    edit_type="spelling",
                    start=start,
                    end=start + len("млоко"),
                    rule_id="unit_atomic",
                )
            ]
        return []


class UnitCandidateGeneratorFactory:
    @classmethod
    def from_config(cls, config: dict):
        del config
        return UnitCandidateGenerator()


class UnitAtomicOperator:
    rule_id = "unit_atomic"
    error_type = "spelling"
    requires = ()

    def find_opportunities(self, clean_sentence: str, syntax_analysis=None):
        del syntax_analysis
        if "молоко" not in clean_sentence:
            return []
        start = clean_sentence.index("молоко")
        return [Opportunity(start, start + len("молоко"), "молоко", self.rule_id)]

    def corrupt(self, clean_sentence: str, opportunity: Opportunity):
        source = clean_sentence[: opportunity.start] + "млоко" + clean_sentence[opportunity.end :]
        return CorruptionResult(
            source=source,
            target=clean_sentence,
            rule_id=self.rule_id,
            edits=[
                {
                    "source": "млоко",
                    "replacement": "молоко",
                    "edit_type": "spelling_replace",
                    "start": opportunity.start,
                    "end": opportunity.start + len("млоко"),
                }
            ],
            error_bearing_span=(opportunity.start, opportunity.start + len("млоко")),
            error_form="млоко",
            target_form="молоко",
            generation_strategy="unit_atomic",
        )

    def verify(self, source: str, target: str, opportunity: Opportunity):
        return _verification(self.rule_id, source, target, opportunity, candidate_generator=UnitCandidateGenerator())


def test_candidate_contract_pipeline_builds_atomic_from_clean_pool_without_output_seed(tmp_path: Path, monkeypatch):
    import src.data.operator_dataset_builder as operator_builder

    clean_pool_path = _write_unit_clean_pool(tmp_path / "clean_sentence_pool.csv.gz")
    config = _candidate_contract_config(tmp_path, clean_pool_path)
    output_path = Path(config["data"]["processed_train_path"])
    _patch_unit_operator_pipeline(monkeypatch)

    def fail_if_output_is_read(path: Path):
        raise AssertionError(f"output dataset must not be read as seed: {path}")

    monkeypatch.setattr(operator_builder, "_read_seed_dataset", fail_if_output_is_read, raising=False)

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(Path(config["data"]["manifest_path"]).read_text(encoding="utf-8"))
    frame = pd.read_csv(output_path)
    atomic = frame[frame["rule_id"].astype(str).eq("unit_atomic")]

    assert result["total"] == 4
    assert output_path.exists()
    assert manifest["rule_id_counts"]["unit_atomic"] == 1
    assert len(atomic) == 1
    assert atomic.iloc[0]["source"] != atomic.iloc[0]["target"]
    assert atomic.iloc[0]["dataset_contract"] == "candidate_opportunity"
    assert atomic.iloc[0]["dataset_layer"] == "atomic_positive"
    assert bool(atomic.iloc[0]["count_toward_rule_quota"]) is True
    assert int(atomic.iloc[0]["gold_edit_count"]) == 1


def test_candidate_contract_pipeline_blocks_when_clean_pool_missing(tmp_path: Path, monkeypatch):
    _patch_unit_operator_pipeline(monkeypatch)
    config = _candidate_contract_config(tmp_path, tmp_path / "missing_clean_sentence_pool.csv.gz")
    output_path = Path(config["data"]["processed_train_path"])

    result = build_dataset_from_config(config, force=True)
    manifest = json.loads(Path(config["data"]["manifest_path"]).read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert result["verdict"] == "DATASET_BLOCKED"
    assert manifest["audit_errors"] == ["missing_clean_sentence_pool"]
    assert not output_path.exists()


def _patch_unit_operator_pipeline(monkeypatch) -> None:
    import src.data.operator_dataset_builder as operator_builder

    registry = RuleOperatorRegistry()
    registry.register(UnitAtomicOperator())
    monkeypatch.setattr(operator_builder, "build_default_operator_registry", lambda: registry)
    monkeypatch.setattr(operator_builder, "load_rule_capabilities", lambda _path: [_unit_capability()])
    monkeypatch.setattr(operator_builder, "CandidateGenerator", UnitCandidateGeneratorFactory, raising=False)


def _candidate_contract_config(tmp_path: Path, clean_pool_path: Path) -> dict:
    config = load_config("configs/config.yaml")
    config["paths"]["reports_dir"] = str(tmp_path / "reports")
    config["data"]["processed_train_path"] = str(tmp_path / "data" / "operator_dataset.csv.gz")
    config["data"]["manifest_path"] = str(tmp_path / "reports" / "dataset_manifest.json")
    config["data"]["dataset_contract"] = "candidate_opportunity"
    config["data"]["clean_pool_path"] = str(clean_pool_path)
    config["data"]["clean_pool_chunksize"] = 2
    config["data"]["total_examples"] = 4
    config["data"]["target_total_examples"] = 4
    config["data"]["train_examples"] = 4
    config["data"]["val_examples"] = 0
    config["data"]["test_examples"] = 0
    config["data"]["rule_quota"] = {
        "rule_ids": ["unit_atomic"],
        "min_atomic_positives_per_active_rule": 1,
        "preferred_atomic_positives_per_active_rule": 1,
        "max_total_per_rule_id": 1,
    }
    core = config["data"]["training_dataset_core"]
    core["dataset_contract"] = "candidate_opportunity"
    core["legacy_builder"] = False
    core["active_rule_quota"] = {
        "rule_ids": ["unit_atomic"],
        "min_total_per_active_rule": 1,
        "preferred_total_per_active_rule": 1,
        "split_minimums": {},
    }
    core["rule_caps"]["max_total_per_rule_id"] = 1
    core["audit"]["require_all_source_types"] = False
    return config


def _write_unit_clean_pool(path: Path) -> Path:
    rows = [
        "В отчете комиссии встретилось слово молоко сегодня.",
        "Редакция отметила, что библиотека открылась после долгой реконструкции.",
        "Аналитики считают, что цифровой отчет помогает оценить работу региона.",
        "Исследователи подчеркнули, что длинный период наблюдений повысил точность.",
    ]
    pd.DataFrame(
        [
            {
                "text": text,
                "source_name": "unit",
                "source_subcorpus": "unit",
                "domain": "unit",
                "sentence_id": f"s{index}",
                "hash": f"h{index}",
            }
            for index, text in enumerate(rows)
        ]
    ).to_csv(path, index=False)
    return path


def _unit_capability() -> RuleCapability:
    return RuleCapability(
        taxonomy_key="unit_atomic",
        domain="unit",
        entry_type="rule",
        title="unit atomic",
        orfogrammka_id="",
        project_rule_ids=["unit_atomic"],
        implementation_status="current",
        requires=[],
        executable=True,
        training_eligible=True,
        training_decision="INCLUDE_NOW",
        training_reason="unit",
        has_candidate_path=True,
        has_synthetic_support=True,
        has_hard_negative_support=True,
        has_validator_support=True,
        has_dictionary_support=False,
        has_syntax_support=False,
        has_morphology_support=False,
        has_ner_support=False,
        risk_level="low",
    )
