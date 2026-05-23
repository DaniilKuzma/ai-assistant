import json
from pathlib import Path

import pandas as pd

import scripts.setup_data_sources as setup_data_sources
from scripts.setup_data_sources import main
from src.data.clean_sentence_pool import CleanSentencePoolResult
from src.data.real_error_sources import RealErrorLoadResult


def test_setup_script_dry_run_writes_manifest_without_outputs(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n"
        "  training_dataset_core:\n"
        "    open_corpora_sources_path: missing-clean.yaml\n"
        "    real_error_sources_path: missing-real.yaml\n",
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"
    processed_dir = tmp_path / "processed"
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--clean",
            "--real",
            "--dry-run",
            "--report-dir",
            str(report_dir),
            "--processed-dir",
            str(processed_dir),
            "--allow-partial",
        ]
    )

    manifest_path = processed_dir / "source_ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["verdict"] == "DATA_SOURCES_BLOCKED"
    assert manifest["dry_run"] is True
    assert manifest["downloads_enabled"] is False
    assert not (processed_dir / "clean_sentence_pool.csv.gz").exists()


def test_setup_script_writes_manifest_for_small_local_sources(tmp_path: Path, monkeypatch):
    clean_path = tmp_path / "clean.txt"
    clean_path.write_text(
        "Эксперты сообщили, что новый индекс вырос после публикации отчета.\n",
        encoding="utf-8",
    )
    real_path = tmp_path / "real.jsonl"
    real_path.write_text(
        '{"source": "Жызнь в городе стала заметно спокойнее.", "target": "Жизнь в городе стала заметно спокойнее."}\n',
        encoding="utf-8",
    )
    clean_config = tmp_path / "clean.yaml"
    clean_config.write_text(
        f"clean_sources:\n"
        f"  unit_news:\n"
        f"    enabled: true\n"
        f"    type: local_text\n"
        f"    local_path: {clean_path.as_posix()}\n"
        f"    source_subcorpus: news\n"
        f"    domain: news\n"
        f"    style: neutral\n"
        f"    license_status: unit\n"
        f"pool:\n"
        f"  min_clean_sentences: 1\n"
        f"  max_source_share: 1.0\n"
        f"  max_subcorpus_share: 1.0\n",
        encoding="utf-8",
    )
    real_config = tmp_path / "real.yaml"
    real_config.write_text(
        f"real_sources:\n"
        f"  unit_pairs:\n"
        f"    enabled: true\n"
        f"    type: local_jsonl\n"
        f"    local_path: {real_path.as_posix()}\n"
        f"    max_pairs: 10\n"
        f"validation:\n"
        f"  min_tokens: 5\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"data:\n"
        f"  training_dataset_core:\n"
        f"    open_corpora_sources_path: {clean_config.as_posix()}\n"
        f"    real_error_sources_path: {real_config.as_posix()}\n",
        encoding="utf-8",
    )
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--clean",
            "--real",
            "--validate",
            "--report-dir",
            str(reports_dir),
            "--processed-dir",
            str(processed_dir),
            "--allow-partial",
        ]
    )

    manifest = json.loads((processed_dir / "source_ingestion_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["clean_pool_size"] == 1
    assert manifest["real_pair_accepted_count"] == 1
    assert manifest["verdict"] == "DATA_SOURCES_BLOCKED"
    assert pd.read_csv(processed_dir / "clean_sentence_pool.csv.gz").shape[0] == 1
    assert pd.read_csv(processed_dir / "real_error_pairs_validated.csv.gz").shape[0] == 1


def test_prepare_materialized_real_sources_rewrites_punctuation_hf_to_local_jsonl(tmp_path: Path, monkeypatch):
    output_path = tmp_path / "spellcheck_punctuation_benchmark.jsonl"

    def fake_prepare_punctuation_jsonl_file(**kwargs):
        Path(kwargs["output_path"]).write_text(
            '{"source": "Я думаю что пора идти.", "target": "Я думаю, что пора идти."}\n',
            encoding="utf-8",
        )
        return 1, "unit"

    monkeypatch.setattr(setup_data_sources, "prepare_punctuation_jsonl_file", fake_prepare_punctuation_jsonl_file)
    config = {
        "real_sources": {
            "spellcheck_punctuation_benchmark": {
                "enabled": True,
                "type": "huggingface_dataset",
                "hf_id": "ai-forever/spellcheck_punctuation_benchmark",
                "local_path": str(output_path),
            }
        }
    }

    setup_data_sources._prepare_materialized_real_sources(config)

    spec = config["real_sources"]["spellcheck_punctuation_benchmark"]
    assert spec["type"] == "local_jsonl"
    assert spec["local_path"] == str(output_path)


def test_setup_script_applies_top_level_contract_overlays_and_prints_summary(tmp_path: Path, monkeypatch, capsys):
    clean_config = tmp_path / "clean.yaml"
    clean_config.write_text(
        "pool:\n"
        "  reject_mixed_script_tokens: false\n"
        "  reject_latin_confusable_inside_cyrillic_word: false\n"
        "  reject_if_candidate_generator_finds_high_confidence_fix: false\n"
        "clean_sources:\n"
        "  unit:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    real_config = tmp_path / "real.yaml"
    real_config.write_text(
        "validation:\n"
        "  train_policy: old\n"
        "  unknown_rule_policy: old\n"
        "  multi_edit_policy: old\n"
        "  stress_loss_weight: 1.0\n"
        "real_sources:\n"
        "  unit:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"data:\n"
        f"  dataset_contract: candidate_opportunity\n"
        f"  clean_pool:\n"
        f"    reject_mixed_script_tokens: true\n"
        f"    reject_latin_confusable_inside_cyrillic_word: true\n"
        f"    reject_if_candidate_generator_finds_high_confidence_fix: true\n"
        f"  real_pairs:\n"
        f"    train_policy: atomize_single_edit_known_rule_only\n"
        f"    unknown_rule_policy: mining_only\n"
        f"    multi_edit_policy: stress_or_eval_only\n"
        f"  stress:\n"
        f"    loss_weight: 0.4\n"
        f"  training_dataset_core:\n"
        f"    open_corpora_sources_path: {clean_config.as_posix()}\n"
        f"    real_error_sources_path: {real_config.as_posix()}\n",
        encoding="utf-8",
    )
    captured: dict[str, dict] = {}

    def fake_build_clean_sentence_pool(config, *, output_path, reports_dir):
        del output_path, reports_dir
        captured["clean"] = config
        return CleanSentencePoolResult(
            accepted_count=1,
            total_seen=1,
            output_path=str(tmp_path / "clean.csv.gz"),
            source_counts={"unit": 1},
            subcorpus_counts={"unit": 1},
            rejection_reason_counts={},
            source_reports=[{"source_name": "unit", "accepted": 1, "used": True}],
            dominance_violations=[],
        )

    def fake_load_real_error_pairs(config, *, candidate_generator, output_path, reports_dir):
        del candidate_generator, output_path, reports_dir
        captured["real"] = config
        return RealErrorLoadResult(
            rows=[{"source_dataset": "unit", "candidate_present": True}],
            accepted_count=1,
            rejected_count=0,
            source_reports=[{"source_dataset": "unit", "accepted": 1, "used": True}],
            rejection_reason_counts={},
            output_path=str(tmp_path / "real.csv.gz"),
        )

    monkeypatch.setattr(setup_data_sources, "build_clean_sentence_pool", fake_build_clean_sentence_pool)
    monkeypatch.setattr(setup_data_sources, "load_real_error_pairs", fake_load_real_error_pairs)
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--clean",
            "--real",
            "--report-dir",
            str(tmp_path / "reports"),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--allow-partial",
        ]
    )

    manifest = json.loads((tmp_path / "processed" / "source_ingestion_manifest.json").read_text(encoding="utf-8"))
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert captured["clean"]["pool"]["reject_mixed_script_tokens"] is True
    assert captured["clean"]["pool"]["reject_latin_confusable_inside_cyrillic_word"] is True
    assert captured["clean"]["pool"]["reject_if_candidate_generator_finds_high_confidence_fix"] is True
    assert captured["real"]["validation"]["train_policy"] == "atomize_single_edit_known_rule_only"
    assert captured["real"]["validation"]["unknown_rule_policy"] == "mining_only"
    assert captured["real"]["validation"]["multi_edit_policy"] == "stress_or_eval_only"
    assert captured["real"]["validation"]["stress_loss_weight"] == 0.4
    assert manifest["dataset_contract"] == "candidate_opportunity"
    assert manifest["dataset_hash"] == ""
    assert manifest["audit_errors"] == []
    assert manifest["layer_counts"] == {}
    assert "dataset contract: candidate_opportunity" in stdout
    assert "dataset hash:" in stdout
    assert "audit errors: []" in stdout
    assert "layer counts: {}" in stdout
