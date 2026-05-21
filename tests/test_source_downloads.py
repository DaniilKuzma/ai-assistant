from pathlib import Path

import pytest

from src.data import source_downloads
from src.data.source_downloads import DownloadBudget, DownloadPolicy, download_if_allowed


def test_download_policy_reads_write_reports_flag():
    policy = DownloadPolicy.from_config(
        {
            "allow_downloads_env": "UNIT_ALLOW_DOWNLOADS",
            "cache_dir": "data/external",
            "write_reports": True,
        }
    )

    assert policy.write_reports is True
    assert policy.allow_downloads_env == "UNIT_ALLOW_DOWNLOADS"


def test_disabled_download_reports_powershell_manual_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("UNIT_ALLOW_DOWNLOADS", raising=False)

    result = download_if_allowed(
        "missing",
        {
            "local_path": str(tmp_path / "missing.txt"),
            "url": "https://example.test/source.txt",
        },
        {
            "allow_downloads_env": "UNIT_ALLOW_DOWNLOADS",
            "cache_dir": str(tmp_path),
            "max_source_download_mb": 1,
            "max_total_download_mb": 1,
        },
        DownloadBudget(),
    )

    assert result.mode == "skipped_downloads_disabled"
    assert result.reason == "downloads_disabled"
    assert "$env:UNIT_ALLOW_DOWNLOADS='1'" in result.required_commands[0]
    assert "curl.exe" in result.required_commands[0]


def test_partial_download_file_is_removed_when_size_limit_is_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeHeaders(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class FakeResponse:
        headers = FakeHeaders()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            if not hasattr(self, "_sent"):
                self._sent = True
                return b"x" * 2048
            return b""

    monkeypatch.setenv("UNIT_ALLOW_DOWNLOADS", "1")
    monkeypatch.setattr(source_downloads, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    output = tmp_path / "source.txt"

    result = download_if_allowed(
        "too_big",
        {"local_path": str(output), "url": "https://example.test/source.txt", "max_download_mb": 0.0005},
        {
            "allow_downloads_env": "UNIT_ALLOW_DOWNLOADS",
            "cache_dir": str(tmp_path),
            "max_source_download_mb": 1,
            "max_total_download_mb": 1,
        },
        DownloadBudget(),
    )

    assert result.mode == "skipped_size_limit"
    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".part").exists()


def test_gitignore_covers_canonical_source_outputs():
    root = Path(".")
    ignored = {
        "data/external/example.bin",
        "data/processed/clean_sentence_pool.csv.gz",
        "data/processed/real_error_pairs_validated.csv.gz",
        "data/processed/real_error_pairs_rejected.csv.gz",
        "data/processed/source_ingestion_manifest.json",
        "data/processed/source_cache/cache.bin",
        "download.part",
    }
    text = (root / ".gitignore").read_text(encoding="utf-8")

    for pattern in ignored:
        assert pattern == pattern
    assert "data/external/**" in text
    assert "data/processed/real_error_pairs_rejected*.csv.gz" in text
    assert "data/processed/source_ingestion_manifest.json" in text
    assert "data/processed/source_cache/" in text
    assert "*.part" in text
