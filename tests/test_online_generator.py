from __future__ import annotations

import importlib.util
import time
from pathlib import Path

from src.config.load_config import load_config
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.generator import OnlineExampleGenerator


ROOT = Path(__file__).resolve().parents[1]


def test_sample_batch_generates_requested_size() -> None:
    generator = _generator()

    examples = generator.sample_batch(200)

    assert len(examples) == 200


def test_audit_batch_has_no_failures_for_online_samples() -> None:
    from src.grammar_gen.audit import audit_batch

    examples = _generator().sample_batch(200)

    result = audit_batch(examples)

    assert result["failed_examples_count"] == 0
    assert result["failure_reasons"] == {}


def test_sample_by_index_is_deterministic() -> None:
    generator = _generator()

    first = generator.sample_by_index(17)
    second = generator.sample_by_index(17)

    assert second == first
    assert second.stable_id() == first.stable_id()


def test_online_generator_can_sample_orthography_morphemic_family() -> None:
    generator = _generator()

    example = generator.sample(rule_id="suffix_its_ets", mode="positive")

    assert example.primary_rule_id == "suffix_its_ets"
    assert example.token_edit_labels.count("DICT_REPLACE") == 1
    assert example.metadata["replacement"]["source"] in example.source_text
    assert example.metadata["replacement"]["target"] in example.target_text


def test_different_indexes_usually_have_different_stable_ids() -> None:
    generator = _generator()

    stable_ids = {generator.sample_by_index(index).stable_id() for index in range(20)}

    assert len(stable_ids) > 1


def test_generation_does_not_create_train_csv() -> None:
    train_path = ROOT / "data" / "processed" / "train.csv"
    assert not train_path.exists()

    _generator().sample_batch(20)

    assert not train_path.exists()


def test_online_generator_factory_uses_production_morphology() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["generation"]["grammar"]["use_pymorphy"] = True

    generator = online_generator_from_config(config, seed=config["generation"]["seed"])

    if importlib.util.find_spec("pymorphy3") is not None:
        assert generator.morphology.uses_pymorphy is True
    else:
        assert generator.morphology.uses_pymorphy is False
        assert generator.sample_by_index(0).target_text


def test_generation_speed_sanity() -> None:
    generator = _generator()
    started = time.perf_counter()

    examples = [generator.sample_by_index(index) for index in range(500)]
    elapsed = time.perf_counter() - started

    assert len(examples) == 500
    assert elapsed < 20


def _generator() -> OnlineExampleGenerator:
    config = load_config(ROOT / "configs" / "config.yaml")
    return online_generator_from_config(config, seed=config["generation"]["seed"])
