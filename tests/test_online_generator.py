from __future__ import annotations

import time
from pathlib import Path

from src.config.load_config import load_config
from src.grammar_gen import Lexicon, MorphologyEngine
from src.grammar_gen.generator import OnlineExampleGenerator
from src.grammar_gen.rules.registry import default_rule_registry


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


def test_different_indexes_usually_have_different_stable_ids() -> None:
    generator = _generator()

    stable_ids = {generator.sample_by_index(index).stable_id() for index in range(20)}

    assert len(stable_ids) > 1


def test_generation_does_not_create_train_csv() -> None:
    train_path = ROOT / "data" / "processed" / "train.csv"
    assert not train_path.exists()

    _generator().sample_batch(20)

    assert not train_path.exists()


def test_generation_does_not_access_clean_sentence_pool(monkeypatch) -> None:
    original_open = Path.open
    forbidden = Path("data/processed/clean_sentence_pool.csv.gz")

    def guarded_open(self: Path, *args, **kwargs):
        try:
            relative = self.resolve().relative_to(ROOT)
        except ValueError:
            relative = self
        if relative.as_posix() == forbidden.as_posix():
            raise AssertionError(f"unexpected access to {forbidden}")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    _generator().sample_batch(20)


def test_generation_speed_sanity() -> None:
    generator = _generator()
    started = time.perf_counter()

    examples = [generator.sample_by_index(index) for index in range(500)]
    elapsed = time.perf_counter() - started

    assert len(examples) == 500
    assert elapsed < 20


def _generator() -> OnlineExampleGenerator:
    config = load_config(ROOT / "configs" / "config.yaml")
    return OnlineExampleGenerator(
        default_rule_registry(),
        Lexicon.default(),
        MorphologyEngine(use_pymorphy=False),
        config,
        seed=config["generation"]["seed"],
    )
