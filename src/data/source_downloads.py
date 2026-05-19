from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_DOWNLOAD_ENV = "RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS"


@dataclass
class DownloadBudget:
    downloaded_bytes: int = 0
    source_results: list["SourceDownloadResult"] = field(default_factory=list)

    def add(self, result: "SourceDownloadResult") -> None:
        self.downloaded_bytes += max(0, int(result.downloaded_size_bytes))
        self.source_results.append(result)


@dataclass(frozen=True)
class DownloadPolicy:
    mode: str = "local_first_with_controlled_downloads"
    allow_downloads_env: str = DEFAULT_DOWNLOAD_ENV
    max_total_download_mb: float = 0.0
    max_source_download_mb: float = 0.0
    cache_dir: str = "data/external"
    fail_if_insufficient_sources: bool = False
    never_commit_downloaded_data: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "DownloadPolicy":
        raw = dict(config or {})
        return cls(
            mode=str(raw.get("mode") or "local_first_with_controlled_downloads"),
            allow_downloads_env=str(raw.get("allow_downloads_env") or DEFAULT_DOWNLOAD_ENV),
            max_total_download_mb=float(raw.get("max_total_download_mb") or 0.0),
            max_source_download_mb=float(raw.get("max_source_download_mb") or 0.0),
            cache_dir=str(raw.get("cache_dir") or "data/external"),
            fail_if_insufficient_sources=bool(raw.get("fail_if_insufficient_sources", False)),
            never_commit_downloaded_data=bool(raw.get("never_commit_downloaded_data", True)),
        )

    @property
    def downloads_allowed(self) -> bool:
        return os.environ.get(self.allow_downloads_env, "").strip().lower() in {"1", "true", "yes", "on"}

    @property
    def max_total_bytes(self) -> int:
        return _mb_to_bytes(self.max_total_download_mb)

    @property
    def max_source_bytes(self) -> int:
        return _mb_to_bytes(self.max_source_download_mb)


@dataclass(frozen=True)
class SourceDownloadResult:
    source_name: str
    mode: str
    path: str = ""
    url: str = ""
    hf_id: str = ""
    downloaded_size_bytes: int = 0
    reason: str = ""
    license_note: str = ""
    used: bool = False
    required_domains: tuple[str, ...] = ()
    required_commands: tuple[str, ...] = ()


def download_if_allowed(
    source_name: str,
    source_config: dict[str, Any],
    policy_config: dict[str, Any] | DownloadPolicy | None,
    budget: DownloadBudget | None = None,
) -> SourceDownloadResult:
    """Resolve a source locally or through an explicitly allowed controlled download."""

    policy = policy_config if isinstance(policy_config, DownloadPolicy) else DownloadPolicy.from_config(policy_config)
    budget = budget or DownloadBudget()
    spec = dict(source_config or {})
    local_path = _path_or_none(spec.get("local_path") or spec.get("path"))
    archive_path = _path_or_none(spec.get("archive_path"))
    url = str(spec.get("url") or "")
    hf_id = str(spec.get("hf_id") or spec.get("repo") or "")
    license_note = str(spec.get("license_note") or spec.get("license_status") or spec.get("license/status") or "")

    for existing_path in (local_path, archive_path):
        if existing_path and existing_path.exists():
            result = SourceDownloadResult(
                source_name=source_name,
                mode="local" if existing_path == local_path else "cached",
                path=str(existing_path),
                url=url,
                hf_id=hf_id,
                downloaded_size_bytes=existing_path.stat().st_size if existing_path.is_file() else _directory_size(existing_path),
                license_note=license_note,
                used=True,
            )
            budget.source_results.append(result)
            return result

    configured_max_bytes = _configured_source_limit_bytes(spec, policy)
    if _exceeds_policy_limit(spec, policy):
        result = _skipped(
            source_name,
            "skipped_size_limit",
            "configured_source_limit_exceeds_policy",
            spec,
            policy,
            path=local_path or archive_path,
        )
        budget.source_results.append(result)
        return result

    if not policy.downloads_allowed:
        reason = "downloads_disabled" if url or hf_id else "missing_local_path"
        mode = "skipped_downloads_disabled" if url or hf_id else "missing_local_path"
        result = _skipped(source_name, mode, reason, spec, policy, path=local_path or archive_path)
        budget.source_results.append(result)
        return result

    total_limit = policy.max_total_bytes
    if total_limit and budget.downloaded_bytes >= total_limit:
        result = _skipped(source_name, "skipped_total_limit", "download_budget_exhausted", spec, policy, path=local_path or archive_path)
        budget.source_results.append(result)
        return result

    if hf_id and not url:
        result = SourceDownloadResult(
            source_name=source_name,
            mode="cached" if _hf_cache_hint_exists(hf_id) else "downloaded",
            path="",
            hf_id=hf_id,
            reason="huggingface_dataset_resolved_by_datasets_cache",
            license_note=license_note,
            used=True,
        )
        budget.source_results.append(result)
        return result

    if not url:
        result = _skipped(source_name, "missing_local_path", "missing_local_path", spec, policy, path=local_path or archive_path)
        budget.source_results.append(result)
        return result

    output_path = archive_path or local_path or Path(policy.cache_dir) / _download_filename(url)
    source_limit = configured_max_bytes or policy.max_source_bytes
    remaining_limit = 0 if not total_limit else max(0, total_limit - budget.downloaded_bytes)
    if remaining_limit and source_limit:
        source_limit = min(source_limit, remaining_limit)
    elif remaining_limit:
        source_limit = remaining_limit
    if total_limit and remaining_limit <= 0:
        result = _skipped(source_name, "skipped_total_limit", "download_budget_exhausted", spec, policy, path=output_path)
        budget.source_results.append(result)
        return result

    result = _download_http(source_name, url, output_path, spec, policy, max_bytes=source_limit)
    if result.mode == "downloaded":
        budget.add(result)
    else:
        budget.source_results.append(result)
    return result


def _download_http(
    source_name: str,
    url: str,
    output_path: Path,
    spec: dict[str, Any],
    policy: DownloadPolicy,
    *,
    max_bytes: int,
) -> SourceDownloadResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        request = Request(url, headers={"User-Agent": "russian-edit-corrector-source-downloader/1.0"})
        with urlopen(request, timeout=45) as response:
            content_length = response.headers.get("Content-Length")
            if max_bytes and content_length and int(content_length) > max_bytes:
                return _skipped(source_name, "skipped_size_limit", "content_length_exceeds_limit", spec, policy, path=output_path)
            downloaded = 0
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if max_bytes and downloaded > max_bytes:
                        handle.close()
                        tmp_path.unlink(missing_ok=True)
                        return _skipped(source_name, "skipped_size_limit", "download_exceeds_limit", spec, policy, path=output_path)
                    handle.write(chunk)
        if spec.get("checksum"):
            expected = str(spec["checksum"])
            actual = _sha256(tmp_path)
            if actual.lower() != expected.lower():
                tmp_path.unlink(missing_ok=True)
                return _skipped(source_name, "skipped_checksum_mismatch", "checksum_mismatch", spec, policy, path=output_path)
        shutil.move(str(tmp_path), str(output_path))
    except (OSError, URLError, TimeoutError, ValueError) as exc:
        tmp_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        result = _skipped(source_name, "skipped_download_failed", f"download_failed:{exc.__class__.__name__}", spec, policy, path=output_path)
        return result

    return SourceDownloadResult(
        source_name=source_name,
        mode="downloaded",
        path=str(output_path),
        url=url,
        downloaded_size_bytes=output_path.stat().st_size,
        license_note=str(spec.get("license_note") or spec.get("license_status") or ""),
        used=True,
    )


def _skipped(
    source_name: str,
    mode: str,
    reason: str,
    spec: dict[str, Any],
    policy: DownloadPolicy,
    *,
    path: Path | None,
) -> SourceDownloadResult:
    url = str(spec.get("url") or "")
    hf_id = str(spec.get("hf_id") or spec.get("repo") or "")
    commands = _required_commands(path, url, hf_id, policy)
    return SourceDownloadResult(
        source_name=source_name,
        mode=mode,
        path=str(path or spec.get("local_path") or spec.get("path") or ""),
        url=url,
        hf_id=hf_id,
        reason=reason,
        license_note=str(spec.get("license_note") or spec.get("license_status") or ""),
        used=False,
        required_domains=_required_domains(url, hf_id),
        required_commands=commands,
    )


def _required_commands(path: Path | None, url: str, hf_id: str, policy: DownloadPolicy) -> tuple[str, ...]:
    prefix = f"export {policy.allow_downloads_env}=1"
    if url:
        output = str(path or Path(policy.cache_dir) / _download_filename(url))
        return (f"{prefix} && mkdir -p {Path(output).parent} && curl -L --fail --retry 4 -o {output} {url}",)
    if hf_id:
        return (f"{prefix} && .venv/bin/python - <<'PY'\nfrom datasets import load_dataset\nload_dataset('{hf_id}')\nPY",)
    return ()


def _required_domains(url: str, hf_id: str) -> tuple[str, ...]:
    domains: list[str] = []
    if "github.com" in url:
        domains.extend(["github.com", "objects.githubusercontent.com"])
    if "storage.yandexcloud.net" in url:
        domains.append("storage.yandexcloud.net")
    if "opencorpora.org" in url:
        domains.append("opencorpora.org")
    if hf_id:
        domains.extend(["huggingface.co", "cdn-lfs.huggingface.co"])
    return tuple(dict.fromkeys(domains))


def _configured_source_limit_bytes(spec: dict[str, Any], policy: DownloadPolicy) -> int:
    source_mb = float(spec.get("max_download_mb") or policy.max_source_download_mb or 0)
    return _mb_to_bytes(source_mb)


def _exceeds_policy_limit(spec: dict[str, Any], policy: DownloadPolicy) -> bool:
    if not policy.max_source_bytes:
        return False
    source_mb = spec.get("max_download_mb")
    return source_mb is not None and _mb_to_bytes(float(source_mb)) > policy.max_source_bytes


def _mb_to_bytes(value: float) -> int:
    return int(value * 1024 * 1024) if value and value > 0 else 0


def _path_or_none(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


def _download_filename(url: str) -> str:
    filename = url.rstrip("/").rsplit("/", 1)[-1]
    return filename or "downloaded_source"


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_cache_hint_exists(hf_id: str) -> bool:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "datasets"
    safe = hf_id.replace("/", "___")
    legacy = hf_id.replace("/", "--")
    return any((cache_root / name).exists() for name in (safe, legacy))
