from pathlib import Path
import json

import pandas as pd
import yaml

from src.candidates.candidate_generator import CandidateGenerator
from src.data.real_error_sources import load_real_error_pairs, validate_real_error_pair
from src.data.sage_sources import materialize_punctuation_jsonl_from_directory, materialize_sage_jsonl_from_directory


def test_real_pair_validation_requires_minimal_candidate_backed_edit():
    accepted = validate_real_error_pair(
        "Жызнь в городе стала заметно спокойнее.",
        "Жизнь в городе стала заметно спокойнее.",
        source_dataset="unit",
        candidate_generator=CandidateGenerator(),
    )
    rejected = validate_real_error_pair(
        "Я люблю этот старый дом возле широкой реки.",
        "Мы полностью изменили смысл предложения без сохранения исходной мысли.",
        source_dataset="unit",
        candidate_generator=CandidateGenerator(),
    )

    assert accepted.accepted is True
    assert accepted.row is not None
    assert accepted.row["source_type"] == "real_error_pair"
    assert accepted.row["is_real_pair"] is True
    assert rejected.accepted is False
    assert rejected.reason in {"char_edit_distance_too_high", "token_edit_distance_too_high", "unsupported_edit_type"}


def test_real_pair_validation_requires_sentence_length_and_rejects_social_noise():
    too_short = validate_real_error_pair(
        "Жызнь прекрасна.",
        "Жизнь прекрасна.",
        source_dataset="unit",
        candidate_generator=CandidateGenerator(),
    )
    social = validate_real_error_pair(
        "Я думал, что жызнь прекрасна, чувак, и написал это вечером.",
        "Я думал, что жизнь прекрасна, чувак, и написал это вечером.",
        source_dataset="unit",
        candidate_generator=CandidateGenerator(),
    )

    assert too_short.accepted is False
    assert too_short.reason == "too_few_tokens"
    assert social.accepted is False
    assert social.reason == "forbidden_domain_noise"


def test_real_pair_validation_rejects_markup_code_and_github_fragments():
    generator = CandidateGenerator()

    markdown = validate_real_error_pair(
        "## Запросы и ответа содержат важные заголовки раздела.",
        "## Запросы и ответы содержат важные заголовки раздела.",
        source_dataset="GitHubTypoCorpusRu",
        candidate_generator=generator,
    )
    code_prefix = validate_real_error_pair(
        "text: Пожалуйста выберите чат чтобы начать общение сейчас.",
        "text: Пожалуйста, выберите чат чтобы начать общение сейчас.",
        source_dataset="GitHubTypoCorpusRu",
        candidate_generator=generator,
    )

    assert markdown.accepted is False
    assert markdown.reason == "markup_or_code_fragment"
    assert code_prefix.accepted is False
    assert code_prefix.reason == "markup_or_code_fragment"


def test_real_error_config_uses_controlled_download_schema():
    config = yaml.safe_load(Path("configs/real_error_sources.yaml").read_text(encoding="utf-8"))

    policy = config["sources"]["download_policy"]
    validation = config["validation"]
    assert policy["mode"] == "local_first_with_controlled_downloads"
    assert policy["allow_downloads_env"] == "RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS"
    assert policy["fail_if_insufficient_sources"] is False
    assert policy["write_reports"] is True
    assert validation["train_policy"] == "atomize_single_edit_known_rule_only"
    assert validation["multi_edit_policy"] == "stress_or_eval_only"
    assert validation["unknown_rule_policy"] == "mining_only"
    assert validation["require_all_edits_candidate_covered"] is True
    assert validation["require_strict_validator"] is True
    assert validation["stress_loss_weight"] == 0.4
    assert config["real_sources"]["spellcheck_benchmark"]["type"] == "huggingface_dataset"
    assert config["real_sources"]["spellcheck_benchmark"]["hf_id"] == "ai-forever/spellcheck_benchmark"
    assert config["real_sources"]["spellcheck_punctuation_benchmark"]["cap_share"] == 0.25
    assert config["real_sources"]["sage_ruspellru"]["cap_share"] == 0.20
    assert config["real_sources"]["sage_multidomain_gold"]["cap_share"] == 0.30
    assert config["real_sources"]["rulec_gec"]["enabled"] is False
    assert config["real_sources"]["rulec_gec"]["type"] == "m2_local"
    assert "url" not in config["real_sources"]["rulec_gec"]


def test_real_pair_loader_reports_rejections(tmp_path: Path):
    source_path = tmp_path / "pairs.jsonl"
    source_path.write_text(
        '{"source": "Жызнь прекрасна.", "correction": "Жизнь прекрасна.", "domain": "unit"}\n'
        '{"source": "Я люблю дом.", "correction": "Мы полностью изменили смысл предложения.", "domain": "unit"}\n',
        encoding="utf-8",
    )
    reports_dir = tmp_path / "reports"
    output_path = tmp_path / "real_error_pairs_validated.csv.gz"

    result = load_real_error_pairs(
        {
            "real_sources": {
                "unit_pairs": {
                    "enabled": True,
                    "type": "local_jsonl",
                    "local_path": str(source_path),
                    "max_examples": 10,
                }
            },
            "validation": {"min_tokens": 2},
        },
        candidate_generator=CandidateGenerator(),
        output_path=output_path,
        reports_dir=reports_dir,
    )

    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert output_path.exists()
    assert (reports_dir / "real_pair_filter_report.csv").exists()
    assert (reports_dir / "rejected_real_pairs.csv").exists()
    assert (reports_dir / "rejected_real_pair_reasons.csv").exists()
    assert result.source_reports[0]["status"] == "loaded"


def test_real_pair_loader_routes_only_single_edit_known_rule_to_atomic_train(tmp_path: Path):
    source_path = tmp_path / "pairs.jsonl"
    source_path.write_text(
        '{"source": "В городе жызнь стала заметно спокойнее после проверки.", "target": "В городе жизнь стала заметно спокойнее после проверки."}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "real_error_pairs_validated.csv.gz"

    result = load_real_error_pairs(
        {
            "real_sources": {
                "unit_pairs": {
                    "enabled": True,
                    "type": "local_jsonl",
                    "local_path": str(source_path),
                    "max_pairs": 10,
                }
            },
            "validation": {"min_tokens": 5},
        },
        candidate_generator=CandidateGenerator(),
        output_path=output_path,
        reports_dir=tmp_path / "reports",
    )

    assert len(result.rows) == 1
    assert result.rows[0]["rule_id"] != "unknown"
    assert result.rows[0]["edit_count"] == 1
    assert result.stress_rows == []
    assert result.mining_rows == []
    assert result.holdout_rows == []
    assert output_path.exists()
    assert Path(result.atomic_output_path).exists()
    assert Path(result.atomic_output_path).name == "real_error_pairs_atomic.csv.gz"
    assert pd.read_csv(output_path).shape[0] == 1
    assert pd.read_csv(result.atomic_output_path).shape[0] == 1


def test_real_pair_loader_routes_explicit_eval_split_to_holdout(tmp_path: Path):
    source_path = tmp_path / "pairs.jsonl"
    source_path.write_text(
        '{"source": "В городе жызнь стала заметно спокойнее после проверки.", "target": "В городе жизнь стала заметно спокойнее после проверки.", "metadata": {"split": "test"}}\n',
        encoding="utf-8",
    )

    result = load_real_error_pairs(
        {
            "real_sources": {
                "unit_pairs": {
                    "enabled": True,
                    "type": "local_jsonl",
                    "local_path": str(source_path),
                    "max_pairs": 10,
                }
            },
            "validation": {"min_tokens": 5},
        },
        candidate_generator=CandidateGenerator(),
        output_path=tmp_path / "real_error_pairs_validated.csv.gz",
        reports_dir=tmp_path / "reports",
    )

    assert result.rows == []
    assert len(result.holdout_rows) == 1
    metadata = json.loads(result.holdout_rows[0]["metadata"])
    assert metadata["source_split"] == "test"
    assert metadata["raw_split"] == "test"
    assert metadata["holdout_reason"] == "explicit_source_split"
    assert Path(result.holdout_output_path).exists()
    assert pd.read_csv(result.output_path).empty


def test_real_pair_loader_routes_multi_edit_known_rule_to_stress_not_train(tmp_path: Path):
    source_path = tmp_path / "pairs.jsonl"
    source_path.write_text(
        '{"source": "В городе жызнь стала заметно спокойнее и жызнь продолжалась.", "target": "В городе жизнь стала заметно спокойнее и жизнь продолжалась."}\n',
        encoding="utf-8",
    )

    result = load_real_error_pairs(
        {
            "real_sources": {
                "unit_pairs": {
                    "enabled": True,
                    "type": "local_jsonl",
                    "local_path": str(source_path),
                    "max_pairs": 10,
                }
            },
            "validation": {"min_tokens": 5},
        },
        candidate_generator=CandidateGenerator(),
        output_path=tmp_path / "real_error_pairs_validated.csv.gz",
        reports_dir=tmp_path / "reports",
    )

    assert result.rows == []
    assert len(result.stress_rows) == 1
    stress = result.stress_rows[0]
    assert stress["edit_count"] == 2
    assert stress["dataset_layer"] == "stress_multi_error"
    assert stress["is_stress"] is True
    assert stress["count_toward_rule_quota"] is False
    assert stress["loss_weight"] == 0.4
    assert Path(result.stress_output_path).exists()
    assert pd.read_csv(result.output_path).empty


def test_real_pair_loader_routes_unknown_rule_to_mining_not_train(tmp_path: Path):
    source_path = tmp_path / "pairs.jsonl"
    source_path.write_text(
        '{"source": "В городе жызнь стала заметно спокойнее после ошипки.", "target": "В городе жизнь стала заметно спокойнее после ошибки."}\n',
        encoding="utf-8",
    )

    result = load_real_error_pairs(
        {
            "real_sources": {
                "unit_pairs": {
                    "enabled": True,
                    "type": "local_jsonl",
                    "local_path": str(source_path),
                    "max_pairs": 10,
                }
            },
            "validation": {"min_tokens": 5},
        },
        candidate_generator=CandidateGenerator(),
        output_path=tmp_path / "real_error_pairs_validated.csv.gz",
        reports_dir=tmp_path / "reports",
    )

    assert result.rows == []
    assert result.stress_rows == []
    assert len(result.mining_rows) == 1
    assert result.mining_rows[0]["routing_reason"] == "unknown_rule_mining_only"
    assert Path(result.mining_output_path).exists()


def test_candidate_present_requires_all_edits_covered():
    validation = validate_real_error_pair(
        "В городе жызнь стала заметно спокойнее и жызнь продолжалась.",
        "В городе жизнь стала заметно спокойнее и жизнь продолжалась.",
        source_dataset="unit",
        candidate_generator=_FirstKnownEditOnlyCandidateGenerator(),
    )

    assert validation.accepted is False
    assert validation.reason == "candidate_missing"
    assert validation.candidate_present is False


def test_real_pair_loader_writes_split_outputs_and_atomization_report(tmp_path: Path):
    source_path = tmp_path / "pairs.jsonl"
    source_path.write_text(
        '{"source": "В городе жызнь стала заметно спокойнее после проверки.", "target": "В городе жизнь стала заметно спокойнее после проверки."}\n'
        '{"source": "В городе жызнь стала заметно спокойнее и жызнь продолжалась.", "target": "В городе жизнь стала заметно спокойнее и жизнь продолжалась."}\n'
        '{"source": "В городе жызнь стала заметно спокойнее после ошипки.", "target": "В городе жизнь стала заметно спокойнее после ошибки."}\n',
        encoding="utf-8",
    )
    reports_dir = tmp_path / "reports"

    result = load_real_error_pairs(
        {
            "real_sources": {
                "unit_pairs": {
                    "enabled": True,
                    "type": "local_jsonl",
                    "local_path": str(source_path),
                    "max_pairs": 10,
                }
            },
            "validation": {"min_tokens": 5},
        },
        candidate_generator=CandidateGenerator(),
        output_path=tmp_path / "real_error_pairs_validated.csv.gz",
        reports_dir=reports_dir,
    )

    assert Path(result.output_path).exists()
    assert Path(result.atomic_output_path).exists()
    assert Path(result.holdout_output_path).exists()
    assert Path(result.stress_output_path).exists()
    assert Path(result.mining_output_path).exists()
    assert Path(result.rejected_output_path).exists()
    assert (reports_dir / "real_pair_atomization_report.csv").exists()
    filter_report = pd.read_csv(reports_dir / "real_pair_filter_report.csv")
    assert {
        "atomic_train",
        "holdout",
        "stress",
        "mining",
        "rejected",
    } <= set(filter_report.columns)
    assert len(result.rows) == 1
    assert len(result.stress_rows) == 1
    assert len(result.mining_rows) == 1


def test_real_pair_loader_supports_alias_fields_and_writes_canonical_columns(tmp_path: Path):
    source_path = tmp_path / "pairs.jsonl"
    source_path.write_text(
        '{"corrupted": "Жызнь в городе стала заметно спокойнее.", "tgt": "Жизнь в городе стала заметно спокойнее.", "raw_id": "a1"}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "real_error_pairs_validated.csv.gz"
    reports_dir = tmp_path / "reports"

    result = load_real_error_pairs(
        {
            "real_sources": {
                "unit_pairs": {
                    "enabled": True,
                    "type": "local_jsonl",
                    "local_path": str(source_path),
                    "max_pairs": 10,
                }
            },
            "validation": {"min_tokens": 5},
        },
        candidate_generator=CandidateGenerator(),
        output_path=output_path,
        reports_dir=reports_dir,
    )

    frame = pd.read_csv(output_path)
    assert result.accepted_count == 1
    assert {
        "source_subdataset",
        "detected_error_types",
        "candidate_rule_ids",
        "edit_count",
        "char_edit_ratio",
        "token_edit_ratio",
        "metadata",
    } <= set(frame.columns)
    assert frame.loc[0, "source_dataset"] == "unit_pairs"


def test_real_pair_loader_skips_unknown_format_with_reason(tmp_path: Path):
    source_path = tmp_path / "pairs.unknown"
    source_path.write_text("not a supported format", encoding="utf-8")

    result = load_real_error_pairs(
        {
            "real_sources": {
                "unit_unknown": {
                    "enabled": True,
                    "type": "custom_unknown",
                    "local_path": str(source_path),
                }
            }
        },
        candidate_generator=CandidateGenerator(),
        output_path=tmp_path / "real_error_pairs_validated.csv.gz",
        reports_dir=tmp_path / "reports",
    )

    assert result.accepted_count == 0
    assert result.source_reports[0]["status"] == "skipped"
    assert result.source_reports[0]["reason"] == "skipped_format_unknown"


def test_real_pair_loader_reads_m2_local_directory(tmp_path: Path):
    source_dir = tmp_path / "rulec_gec"
    source_dir.mkdir()
    (source_dir / "sample.m2").write_text(
        "S Жызнь в городе стала заметно спокойнее .\n"
        "A 0 1|||SPELL|||Жизнь|||REQUIRED|||-NONE-|||0\n\n",
        encoding="utf-8",
    )

    result = load_real_error_pairs(
        {
            "real_sources": {
                "rulec_gec": {
                    "enabled": True,
                    "type": "m2_local",
                    "local_path": str(source_dir),
                    "max_pairs": 10,
                    "cap_share": 1.0,
                }
            },
            "validation": {"min_tokens": 5},
        },
        candidate_generator=CandidateGenerator(),
        output_path=tmp_path / "real_error_pairs_validated.csv.gz",
        reports_dir=tmp_path / "reports",
    )

    assert result.accepted_count == 1
    assert result.rows[0]["source_dataset"] == "rulec_gec"
    assert result.rows[0]["raw_source_path"].endswith("sample.m2")


def test_sage_materializer_writes_canonical_jsonl_from_local_snapshot(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "data" / "RUSpellRU").mkdir(parents=True)
    (snapshot / "data" / "MedSpellchecker").mkdir(parents=True)
    (snapshot / "data" / "RUSpellRU" / "train.json").write_text(
        '{"source": "Жызнь в городе стала заметно спокойнее.", "correction": "Жизнь в городе стала заметно спокойнее.", "domain": "RUSpellRU"}\n',
        encoding="utf-8",
    )
    (snapshot / "data" / "MedSpellchecker" / "test.json").write_text(
        '{"source": "Пациент отмечает ошипку в короткой записи.", "correction": "Пациент отмечает ошибку в короткой записи.", "domain": "medical"}\n',
        encoding="utf-8",
    )

    result = materialize_sage_jsonl_from_directory(snapshot, tmp_path / "sage")

    ruspell = tmp_path / "sage" / "RUSpellRU.jsonl"
    med = tmp_path / "sage" / "MedSpellChecker.jsonl"
    assert result.written_counts["RUSpellRU"] == 1
    assert result.written_counts["MedSpellChecker"] == 1
    assert ruspell.exists()
    assert med.exists()
    row = json.loads(ruspell.read_text(encoding="utf-8").strip())
    assert row["source"] == "Жызнь в городе стала заметно спокойнее."
    assert row["target"] == "Жизнь в городе стала заметно спокойнее."
    assert row["dataset"] == "RUSpellRU"
    assert row["source_dataset"] == "RUSpellRU"
    assert row["metadata"]["split"] == "train"


def test_sage_materializer_skips_source_equal_target_rows(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "data" / "RUSpellRU").mkdir(parents=True)
    (snapshot / "data" / "RUSpellRU" / "train.json").write_text(
        '{"source": "Жизнь в городе стала заметно спокойнее.", "correction": "Жизнь в городе стала заметно спокойнее.", "domain": "RUSpellRU"}\n'
        '{"source": "Жызнь в городе стала заметно спокойнее.", "correction": "Жизнь в городе стала заметно спокойнее.", "domain": "RUSpellRU"}\n',
        encoding="utf-8",
    )

    result = materialize_sage_jsonl_from_directory(snapshot, tmp_path / "sage")

    rows = [
        json.loads(line)
        for line in (tmp_path / "sage" / "RUSpellRU.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert result.written_counts["RUSpellRU"] == 1
    assert rows[0]["source"] != rows[0]["target"]


def test_punctuation_materializer_writes_combined_local_jsonl(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "data" / "RUSpellRU").mkdir(parents=True)
    (snapshot / "data" / "RUSpellRU" / "test.json").write_text(
        '{"source": "Я думаю что пора идти.", "correction": "Я думаю, что пора идти.", "domain": "RUSpellRU"}\n',
        encoding="utf-8",
    )

    count = materialize_punctuation_jsonl_from_directory(
        snapshot,
        tmp_path / "sage" / "spellcheck_punctuation_benchmark.jsonl",
    )

    row = json.loads((tmp_path / "sage" / "spellcheck_punctuation_benchmark.jsonl").read_text(encoding="utf-8").strip())
    assert count == 1
    assert row["source"] == "Я думаю что пора идти."
    assert row["target"] == "Я думаю, что пора идти."
    assert row["source_dataset"] == "spellcheck_punctuation_benchmark"
    assert row["dataset"] == "RUSpellRU"
    assert row["metadata"]["source_benchmark"] == "spellcheck_punctuation_benchmark"


def test_real_pair_loader_applies_source_caps_and_reports_verdict(tmp_path: Path, monkeypatch):
    major_path = tmp_path / "major.jsonl"
    minor_path = tmp_path / "minor.jsonl"
    major_path.write_text("\n".join(_candidate_backed_rows("major", 8)) + "\n", encoding="utf-8")
    minor_path.write_text("\n".join(_candidate_backed_rows("minor", 2)) + "\n", encoding="utf-8")
    monkeypatch.setenv("RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS", "1")

    result = load_real_error_pairs(
        {
            "sources": {
                "download_policy": {
                    "mode": "local_first_with_controlled_downloads",
                    "allow_downloads_env": "RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS",
                }
            },
            "real_sources": {
                "major": {
                    "enabled": True,
                    "type": "local_jsonl",
                    "local_path": str(major_path),
                    "max_examples": 20,
                    "cap_share": 0.5,
                },
                "minor": {
                    "enabled": True,
                    "type": "local_jsonl",
                    "local_path": str(minor_path),
                    "max_examples": 20,
                    "cap_share": 1.0,
                },
            },
            "validation": {"min_tokens": 5},
        },
        candidate_generator=CandidateGenerator(),
        output_path=tmp_path / "real_error_pairs_validated.csv.gz",
        reports_dir=tmp_path / "reports",
    )

    counts = {report["source_dataset"]: report["accepted"] for report in result.source_reports if report["status"] == "loaded"}
    report = (tmp_path / "reports" / "real_error_source_report.md").read_text(encoding="utf-8")
    assert result.accepted_count == 4
    assert counts == {"major": 2, "minor": 2}
    assert "downloads_enabled: yes" in report
    assert "cap_rejections" in report
    assert "final_verdict: BLOCKED" in report


def test_real_pair_loader_does_not_collapse_when_active_caps_sum_below_one(tmp_path: Path, monkeypatch):
    paths = {}
    for source_name in ("ruspell", "multi", "med", "git"):
        path = tmp_path / f"{source_name}.jsonl"
        path.write_text("\n".join(_candidate_backed_rows(source_name, 10)) + "\n", encoding="utf-8")
        paths[source_name] = path
    monkeypatch.setenv("RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS", "1")

    result = load_real_error_pairs(
        {
            "sources": {
                "download_policy": {
                    "mode": "local_first_with_controlled_downloads",
                    "allow_downloads_env": "RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS",
                }
            },
            "real_sources": {
                "ruspell": {"enabled": True, "type": "local_jsonl", "local_path": str(paths["ruspell"]), "cap_share": 0.30},
                "multi": {"enabled": True, "type": "local_jsonl", "local_path": str(paths["multi"]), "cap_share": 0.50},
                "med": {"enabled": True, "type": "local_jsonl", "local_path": str(paths["med"]), "cap_share": 0.08},
                "git": {"enabled": True, "type": "local_jsonl", "local_path": str(paths["git"]), "cap_share": 0.08},
            },
            "validation": {"min_tokens": 5},
        },
        candidate_generator=CandidateGenerator(),
        output_path=tmp_path / "real_error_pairs_validated.csv.gz",
        reports_dir=tmp_path / "reports",
    )

    counts = {report["source_dataset"]: report["accepted"] for report in result.source_reports if report["status"] == "loaded"}
    assert result.accepted_count >= 20
    assert counts["ruspell"] > 1
    assert counts["multi"] > 1
    assert counts["med"] > 0
    assert counts["git"] > 0


def _candidate_backed_rows(prefix: str, count: int) -> list[str]:
    return [
        json.dumps(
            {
                "source": f"Жызнь в городе стала заметно спокойнее после проверки {prefix}{index}.",
                "target": f"Жизнь в городе стала заметно спокойнее после проверки {prefix}{index}.",
                "domain": "unit",
            },
            ensure_ascii=False,
        )
        for index in range(count)
    ]


class _FirstKnownEditOnlyCandidateGenerator:
    def __init__(self) -> None:
        self._generator = CandidateGenerator()

    def generate(self, source: str):
        first_start = source.find("жызнь")
        return [
            candidate
            for candidate in self._generator.generate(source)
            if candidate.start == first_start
            and candidate.source.lower() == "жызнь"
            and candidate.replacement.lower() == "жизнь"
        ][:1]
