from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.candidates.candidate_generator import Candidate
from src.config.load_config import load_config
from src.data.corruption_operators import CorruptionResult, Opportunity, RuleOperatorRegistry, _verification
from src.rules.capabilities import RuleCapability


class UnitCandidateGenerator:
    def generate(self, text: str):
        candidates = []
        typo_start = text.find("млоко")
        while typo_start >= 0:
            candidates.append(
                Candidate(
                    source="млоко",
                    replacement="молоко",
                    edit_type="spelling",
                    start=typo_start,
                    end=typo_start + len("млоко"),
                    rule_id="unit_atomic",
                )
            )
            typo_start = text.find("млоко", typo_start + len("млоко"))
        clean_start = text.find("молоко")
        while clean_start >= 0:
            candidates.append(
                Candidate(
                    source="молоко",
                    replacement="млоко",
                    edit_type="spelling",
                    start=clean_start,
                    end=clean_start + len("молоко"),
                    rule_id="unit_atomic",
                )
            )
            clean_start = text.find("молоко", clean_start + len("молоко"))
        return candidates


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


def patch_unit_operator_pipeline(monkeypatch) -> None:
    import src.data.operator_dataset_builder as operator_builder

    registry = RuleOperatorRegistry()
    registry.register(UnitAtomicOperator())
    monkeypatch.setattr(operator_builder, "build_default_operator_registry", lambda: registry)
    monkeypatch.setattr(operator_builder, "load_rule_capabilities", lambda _path: [unit_capability()])
    monkeypatch.setattr(operator_builder, "CandidateGenerator", UnitCandidateGeneratorFactory, raising=False)


def candidate_contract_config(tmp_path: Path, clean_pool_path: Path) -> dict:
    config = load_config("configs/config.yaml")
    config["paths"]["reports_dir"] = str(tmp_path / "reports")
    config["data"]["processed_train_path"] = str(tmp_path / "data" / "operator_dataset.csv.gz")
    config["data"]["manifest_path"] = str(tmp_path / "reports" / "dataset_manifest.json")
    config["data"]["dataset_contract"] = "candidate_opportunity"
    config["data"]["clean_pool_path"] = str(clean_pool_path)
    config["data"]["clean_pool_chunksize"] = 2
    config["data"]["total_examples"] = 10
    config["data"]["target_total_examples"] = 10
    config["data"]["train_examples"] = 8
    config["data"]["val_examples"] = 1
    config["data"]["test_examples"] = 1
    config["data"]["composition"] = {
        "atomic_positive_target": 4,
        "atomic_hard_negative_target": 3,
        "clean_identity_target": 3,
        "stress_multi_error_target": 0,
        "real_atomic_train_target": 0,
    }
    config["data"]["rule_quota"] = {
        "rule_ids": ["unit_atomic"],
        "min_atomic_positives_per_active_rule": 1,
        "preferred_atomic_positives_per_active_rule": 2,
        "max_total_per_rule_id": 2,
        "min_hard_negatives_per_active_rule": 1,
        "disable_rule_if_quota_not_met": True,
    }
    config["data"]["rule_activation"]["expected_min_final_active_rule_count"] = 0
    config["data"]["rule_activation"]["target_final_active_rule_count"] = 0
    config["data"]["rule_activation"]["fail_below_final_active_rule_count"] = False
    config["data"]["rule_activation"]["warn_below_target_final_active_rule_count"] = False
    config["data"]["audit"] = {
        "min_candidate_recall_for_active_rule": 0.95,
    }
    config["data"]["stress"] = {"enabled": True, "loss_weight": 0.4, "count_toward_rule_quota": False}
    core = config["data"]["training_dataset_core"]
    core["enabled"] = True
    core["dataset_contract"] = "candidate_opportunity"
    core["legacy_builder"] = False
    core["active_rule_quota"] = {
        "rule_ids": ["unit_atomic"],
        "min_total_per_active_rule": 1,
        "preferred_total_per_active_rule": 2,
        "split_minimums": {},
    }
    core["rule_caps"]["max_total_per_rule_id"] = 2
    core["audit"]["require_all_source_types"] = False
    return config


def write_unit_clean_pool(path: Path) -> Path:
    rows = [
        "В отчете комиссии встретилось слово молоко сегодня.",
        "Редакция отметила, что свежее молоко поступило после долгой проверки.",
        "Аналитики считают, что молоко помогает оценить работу нового рынка.",
        "Исследователи подчеркнули, что молоко повысило точность эксперимента.",
        "Комиссия решила, что молоко нужно проверить перед публикацией отчета.",
        "Авторы сообщили, что молоко осталось в холодильнике после проверки.",
        "Когда молоко будет готово, команда отправит образцы в архив.",
        "Эксперты сообщили, что молоко в городе стало заметно дешевле.",
        "Компания заявила, что молоко нового поставщика будет доступно осенью.",
        "Докладчики отметили, что молоко прошло лабораторную проверку вчера.",
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


def write_unit_real_jsonl(path: Path) -> Path:
    rows = [
        {
            "source": "В городе жызнь стала заметно спокойнее после проверки.",
            "target": "В городе жизнь стала заметно спокойнее после проверки.",
        },
        {
            "source": "В городе жызнь стала заметно спокойнее и жызнь продолжалась.",
            "target": "В городе жизнь стала заметно спокойнее и жизнь продолжалась.",
        },
        {
            "source": "В городе жызнь стала заметно спокойнее после ошипки.",
            "target": "В городе жизнь стала заметно спокойнее после ошибки.",
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def write_unit_real_outputs(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    atomic_path = data_dir / "real_error_pairs_atomic.csv.gz"
    stress_path = data_dir / "real_error_pairs_stress.csv.gz"
    mining_path = data_dir / "real_error_pairs_mining.csv.gz"
    validated_path = data_dir / "real_error_pairs_validated.csv.gz"

    atomic_rows = [
        _unit_real_row(
            "В городе млоко стало дешевле после проверки.",
            "В городе молоко стало дешевле после проверки.",
            edit_sources=["млоко"],
            edit_replacements=["молоко"],
            layer="real_atomic",
        )
    ]
    stress_rows = [
        _unit_real_row(
            "В городе млоко и млоко стали дешевле после проверки.",
            "В городе молоко и молоко стали дешевле после проверки.",
            edit_sources=["млоко", "млоко"],
            edit_replacements=["молоко", "молоко"],
            layer="stress_multi_error",
            loss_weight=0.4,
        )
    ]
    mining_rows = [
        _unit_real_row(
            "В городе ошипка стала заметнее после проверки.",
            "В городе ошибка стала заметнее после проверки.",
            edit_sources=["ошипка"],
            edit_replacements=["ошибка"],
            rule_id="unknown",
            layer="real_mining",
            routing_category="mining",
            routing_reason="unknown_rule_mining_only",
            candidate_present=False,
            strict_validator_passed=False,
        )
    ]

    pd.DataFrame(atomic_rows).to_csv(atomic_path, index=False)
    pd.DataFrame(stress_rows).to_csv(stress_path, index=False)
    pd.DataFrame(mining_rows).to_csv(mining_path, index=False)
    pd.DataFrame(atomic_rows).to_csv(validated_path, index=False)
    return {
        "atomic": atomic_path,
        "stress": stress_path,
        "mining": mining_path,
        "validated": validated_path,
    }


def _unit_real_row(
    source: str,
    target: str,
    *,
    edit_sources: list[str],
    edit_replacements: list[str],
    rule_id: str = "unit_atomic",
    layer: str,
    loss_weight: float = 1.0,
    routing_category: str | None = None,
    routing_reason: str = "unit",
    candidate_present: bool = True,
    strict_validator_passed: bool = True,
) -> dict:
    edits = []
    search_start = 0
    for edit_source, replacement in zip(edit_sources, edit_replacements, strict=True):
        start = source.index(edit_source, search_start)
        edits.append(
            {
                "source": edit_source,
                "replacement": replacement,
                "edit_type": "spelling_replace",
                "start": start,
                "end": start + len(edit_source),
                "rule_id": rule_id,
            }
        )
        search_start = start + len(edit_source)
    edit_count = len(edits)
    is_stress = layer == "stress_multi_error"
    category = routing_category or ("stress" if is_stress else "atomic_train")
    metadata = {
        "candidate_present": candidate_present,
        "strict_validator_passed": strict_validator_passed,
        "routing_category": category,
        "routing_reason": routing_reason,
        "gold_edit_count": edit_count,
        "dataset_layer": layer,
        "count_toward_rule_quota": False,
        "loss_weight": loss_weight,
    }
    return {
        "source": source,
        "target": target,
        "source_dataset": "unit_real",
        "source_subdataset": "",
        "domain": "real_error_pair",
        "detected_error_types": json.dumps(["spelling"], ensure_ascii=False),
        "candidate_present": candidate_present,
        "candidate_rule_ids": json.dumps([rule_id], ensure_ascii=False),
        "edit_count": edit_count,
        "gold_edit_count": edit_count,
        "char_edit_ratio": 0.05,
        "token_edit_ratio": 0.05,
        "is_real_pair": True,
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "raw_id": "",
        "raw_source_path": "",
        "error_types": json.dumps(["spelling"], ensure_ascii=False),
        "error_type": "spelling",
        "source_type": "real_error_pair",
        "is_clean": False,
        "is_hard_negative": False,
        "is_synthetic": False,
        "split": "stress" if is_stress else ("mining" if layer == "real_mining" else "train"),
        "rule_id": rule_id,
        "rule_ids": json.dumps([rule_id], ensure_ascii=False),
        "edit_operations": json.dumps(edits, ensure_ascii=False),
        "edits": json.dumps(edits, ensure_ascii=False),
        "dataset_layer": layer,
        "is_stress": is_stress,
        "count_toward_rule_quota": False,
        "loss_weight": loss_weight,
        "routing_category": category,
        "routing_reason": routing_reason,
        "strict_validator_passed": strict_validator_passed,
    }


def unit_capability() -> RuleCapability:
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
