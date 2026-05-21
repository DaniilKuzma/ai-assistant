from pathlib import Path

import pandas as pd
import yaml

from src.candidates.candidate_generator import CandidateGenerator
from src.training.feature_profile import FeatureBuildProfiler
from src.training.tensorization import DebugTokenizer, build_features_from_rows
from src.training.train import train


def test_feature_build_profile_writes_csv_and_summary(tmp_path: Path):
    profile_csv = tmp_path / "feature_build_profile.csv"
    summary_md = tmp_path / "feature_build_profile_summary.md"
    profiler = FeatureBuildProfiler(
        enabled=True,
        output_path=profile_csv,
        summary_path=summary_md,
        sample_size=10,
        split="train",
        syntax_enabled=False,
        dictionary_lexicon_size=0,
    )

    features = build_features_from_rows(
        [{"source": "Я незнаю что делать", "target": "Я не знаю, что делать.", "split": "train"}],
        tokenizer=DebugTokenizer(),
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        punctuation_action_label_map={"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=16,
        max_candidates=6,
        candidate_generator=CandidateGenerator(syntax_provider=lambda _text: ()),
        profiler=profiler,
    )
    profiler.write_reports(
        dictionary_cache_stats={"hits": 2, "misses": 1, "hit_rate": 2 / 3},
        feature_cache_stats={"enabled": True, "hit": False, "miss": True},
    )

    assert len(features) == 1
    frame = pd.read_csv(profile_csv)
    assert list(frame["split"]) == ["train"]
    assert frame.loc[0, "candidate_generation_ms"] >= 0.0
    assert frame.loc[0, "candidate_count"] > 0
    summary = summary_md.read_text(encoding="utf-8")
    assert "total rows profiled: 1" in summary
    assert "dictionary cache hit/miss stats" in summary
    assert "clear bottleneck conclusion" in summary


def test_feature_build_profile_disabled_does_not_write_reports(tmp_path: Path):
    profiler = FeatureBuildProfiler(
        enabled=False,
        output_path=tmp_path / "feature_build_profile.csv",
        summary_path=tmp_path / "feature_build_profile_summary.md",
        sample_size=10,
        split="train",
        syntax_enabled=False,
        dictionary_lexicon_size=0,
    )

    build_features_from_rows(
        [{"source": "Чистый текст.", "target": "Чистый текст.", "split": "train"}],
        tokenizer=DebugTokenizer(),
        punctuation_label_map={"NONE": 0, "DOT": 1},
        error_type_label_map={"keep": 0},
        max_length=8,
        max_candidates=4,
        candidate_generator=CandidateGenerator(syntax_provider=lambda _text: ()),
        profiler=profiler,
    )
    profiler.write_reports(dictionary_cache_stats={}, feature_cache_stats={})

    assert not profiler.output_path.exists()
    assert not profiler.summary_path.exists()


def test_training_feature_build_syntax_disabled_does_not_call_parse_syntax(monkeypatch):
    def fail_parse_syntax(_text: str):
        raise AssertionError("parse_syntax should not be called for training feature build")

    monkeypatch.setattr("src.nlp.syntax.parse_syntax", fail_parse_syntax)

    generator = CandidateGenerator.from_config(
        {
            "training": {"feature_build": {"enable_syntax": False}},
            "dictionary": {"enabled": False},
        },
        purpose="training_features",
    )

    candidates = generator.generate("Даниил проверь текст.")

    assert candidates


def test_training_report_includes_feature_cache_and_dataset_manifest_metadata(tmp_path: Path):
    processed_path = tmp_path / "dataset.csv.gz"
    manifest_path = tmp_path / "dataset_manifest.json"
    pd.DataFrame(
        [
            {
                "source": "Я незнаю что делать",
                "target": "Я не знаю, что делать.",
                "split": "train",
                "source_type": "synthetic_augmented_from_open_clean",
                "is_clean": False,
                "is_synthetic": True,
                "error_types": '["split_join"]',
            }
        ]
    ).to_csv(processed_path, index=False)
    manifest_path.write_text(
        '{"verdict":"READY_FOR_TRAINING_DATASET","total":1,"split_sizes":{"train":1}}',
        encoding="utf-8",
    )
    config = {
        "model": {"max_sequence_length": 16, "max_candidates": 4},
        "training": {
            "run_model_training": False,
            "max_train_examples": 1,
            "show_progress": False,
            "feature_build": {"enable_syntax": False, "profile": False},
            "feature_cache": {"enabled": True, "cache_dir": str(tmp_path / "features_cache")},
        },
        "data": {"processed_train_path": str(processed_path), "manifest_path": str(manifest_path)},
        "labels": {
            "punctuation": {"NONE": 0, "COMMA": 1, "DOT": 2},
            "punctuation_actions": {"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2},
            "error_types": {"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        },
        "dictionary": {"enabled": False},
        "paths": {
            "adapter_output_dir": str(tmp_path / "models" / "adapters"),
            "heads_output_dir": str(tmp_path / "models" / "heads"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    result = train(config_path)

    report = (tmp_path / "reports" / "training_report.md").read_text(encoding="utf-8")
    assert result["feature_cache_enabled"] is True
    assert "- feature_cache_enabled: True" in report
    assert "- feature_cache_hit: False" in report
    assert f"- dataset_path: {processed_path}" in report
    assert f"- manifest_path: {manifest_path}" in report
    assert "- manifest_verdict: READY_FOR_TRAINING_DATASET" in report
    assert "- dataset_rows: 1" in report
    assert "- split_counts: {'train': 1}" in report
