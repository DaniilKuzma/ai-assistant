from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
import re
import tarfile
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import urlopen
import zipfile
import xml.etree.ElementTree as ElementTree


DEFAULT_CLEAN_CORPUS_SOURCES: list[dict[str, Any]] = [
    {
        "name": "leipzig_news",
        "type": "leipzig",
        "corpus_ids": ["rus_news_2022_1M", "rus_news_2021_1M", "rus_news_2020_1M"],
        "max_sentences": 180_000,
    },
    {
        "name": "leipzig_wikipedia",
        "type": "leipzig",
        "corpus_ids": ["rus_wikipedia_2021_1M", "rus_wikipedia_2016_1M"],
        "max_sentences": 90_000,
    },
    {
        "name": "taiga_rest",
        "type": "hf_dataset",
        "repo": "cointegrated/taiga_stripped_rest",
        "splits": ["Interfax", "Lenta", "NPlus1", "Fontanka", "Arzamas", "KP"],
        "text_fields": ["text"],
        "max_sentences": 90_000,
    },
    {
        "name": "taiga_proza_filtered",
        "type": "hf_dataset",
        "repo": "cointegrated/taiga_stripped_proza",
        "splits": ["train"],
        "text_fields": ["text"],
        "max_sentences": 15_000,
        "strict_formal_filter": True,
    },
    {
        "name": "ud_russian_taiga",
        "type": "ud_conllu",
        "urls": [
            "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/master/ru_taiga-ud-train.conllu",
            "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/master/ru_taiga-ud-dev.conllu",
            "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/master/ru_taiga-ud-test.conllu",
        ],
        "max_sentences": 25_000,
    },
    {
        "name": "opencorpora",
        "type": "opencorpora_xml_zip",
        "url": "https://opencorpora.org/files/export/annot/annot.opcorpora.xml.zip",
        "max_sentences": 40_000,
    },
]


@dataclass(frozen=True)
class CleanCorpusSentence:
    text: str
    source_name: str


@dataclass(frozen=True)
class CleanCorpusLoadResult:
    sentences: list[str]
    source_counts: dict[str, int]
    cache_path: str


def load_clean_corpus_sentences(data_config: dict[str, Any], needed_count: int) -> CleanCorpusLoadResult:
    clean_config = data_config.get("clean_corpus", {})
    if not bool(clean_config.get("enabled", False)) or needed_count <= 0:
        return CleanCorpusLoadResult([], {}, "")

    show_progress = bool(clean_config.get("show_progress", True))
    cache_path = Path(clean_config.get("cache_path", "data/raw/clean_corpus_sentences.txt.gz"))
    rebuild_cache = bool(clean_config.get("rebuild_cache", False))
    if cache_path.exists() and not rebuild_cache:
        if show_progress:
            print(f"Loading clean corpus cache: {cache_path}")
        cached = _read_cache(cache_path, needed_count)
        if len(cached) >= min(needed_count, int(clean_config.get("min_cached_sentences", 1))):
            if show_progress:
                print(f"Loaded {len(cached[:needed_count])} clean sentences from cache")
            return CleanCorpusLoadResult(cached[:needed_count], {"cache": len(cached[:needed_count])}, str(cache_path))

    sources = clean_config.get("sources") or DEFAULT_CLEAN_CORPUS_SOURCES
    source_counts: dict[str, int] = {}
    collected: list[str] = []
    seen: set[str] = set()
    max_scan_sources = int(clean_config.get("max_source_sentences", max(needed_count * 3, needed_count)))

    for source in sources:
        if len(collected) >= needed_count:
            break
        source_name = str(source.get("name") or source.get("type") or "clean_source")
        if show_progress:
            print(f"Collecting clean sentences from {source_name}...")
        source_limit = min(int(source.get("max_sentences", needed_count)), needed_count - len(collected))
        source_added = 0
        for sentence in _iter_source_sentences(source, max_scan_sources=max_scan_sources):
            if source_added >= source_limit or len(collected) >= needed_count:
                break
            if not is_suitable_clean_sentence(sentence, strict=bool(source.get("strict_formal_filter", False))):
                continue
            normalized = normalize_for_dedup(sentence)
            if normalized in seen:
                continue
            seen.add(normalized)
            collected.append(sentence)
            source_added += 1
            if show_progress and source_added % 10_000 == 0:
                print(f"  {source_name}: {source_added} accepted, total {len(collected)}/{needed_count}")
        if source_added:
            source_counts[source_name] = source_added
        if show_progress:
            print(f"Finished {source_name}: accepted {source_added}, total {len(collected)}/{needed_count}")

    if collected:
        if show_progress:
            print(f"Writing clean corpus cache: {cache_path}")
        _write_cache(cache_path, collected)
    return CleanCorpusLoadResult(collected[:needed_count], source_counts, str(cache_path))


def is_suitable_clean_sentence(sentence: str, *, strict: bool = False) -> bool:
    text = normalize_sentence(sentence)
    if len(text) < 60 or len(text) > 260:
        return False
    if not text.endswith((".", "!", "?")):
        return False
    if re.search(r"https?://|www\.|@|<[^>]+>|[{}`]|==|//", text, flags=re.IGNORECASE):
        return False
    if text.startswith(("-", "—", "–", "•", "*", "\"", "«", "»")):
        return False
    if "..." in text or "…" in text:
        return False
    if len(re.findall(r"[\"«»]", text)) > 2:
        return False

    words = re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?", text)
    if len(words) < 7 or len(words) > 36:
        return False

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if not letters:
        return False
    cyrillic = re.findall(r"[А-Яа-яЁё]", text)
    if len(cyrillic) / len(letters) < 0.85:
        return False

    lower = text.lower()
    if any(marker in lower for marker in _REJECT_MARKERS):
        return False
    if strict and any(marker in lower for marker in _STRICT_REJECT_MARKERS):
        return False
    return True


def normalize_sentence(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u2009", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_dedup(text: str) -> str:
    text = normalize_sentence(text).lower()
    text = re.sub(r"\d+", "<NUM>", text)
    return text


def split_text_to_sentences(text: str) -> list[str]:
    text = normalize_sentence(text)
    if not text:
        return []
    try:
        from razdel import sentenize

        return [normalize_sentence(match.text) for match in sentenize(text)]
    except Exception:
        return [normalize_sentence(part) for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def parse_leipzig_sentences(content: str) -> list[str]:
    sentences: list[str] = []
    for line in content.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        sentence = parts[1] if len(parts) == 2 and parts[0].isdigit() else line
        sentences.append(normalize_sentence(sentence))
    return sentences


def parse_ud_conllu_texts(content: str) -> list[str]:
    result: list[str] = []
    for line in content.splitlines():
        if line.startswith("# text = "):
            result.append(normalize_sentence(line.removeprefix("# text = ")))
    return result


def _iter_source_sentences(source: dict[str, Any], *, max_scan_sources: int) -> Iterable[str]:
    source_type = source.get("type")
    try:
        if source_type == "leipzig":
            yield from _iter_leipzig_source(source, max_scan_sources=max_scan_sources)
        elif source_type == "hf_dataset":
            yield from _iter_hf_dataset_source(source, max_scan_sources=max_scan_sources)
        elif source_type == "ud_conllu":
            yield from _iter_ud_source(source, max_scan_sources=max_scan_sources)
        elif source_type == "opencorpora_xml_zip":
            yield from _iter_opencorpora_source(source, max_scan_sources=max_scan_sources)
    except Exception:
        return


def _iter_leipzig_source(source: dict[str, Any], *, max_scan_sources: int) -> Iterable[str]:
    scanned = 0
    for corpus_id in source.get("corpus_ids", []):
        url = f"https://downloads.wortschatz-leipzig.de/corpora/{corpus_id}.tar.gz"
        print(f"  downloading Leipzig corpus: {corpus_id}")
        try:
            with urlopen(url, timeout=30) as response:
                archive_bytes = response.read()
        except (OSError, URLError):
            continue
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.name.endswith("-sentences.txt"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                content = extracted.read().decode("utf-8", errors="ignore")
                for sentence in parse_leipzig_sentences(content):
                    yield sentence
                    scanned += 1
                    if scanned >= max_scan_sources:
                        return


def _iter_hf_dataset_source(source: dict[str, Any], *, max_scan_sources: int) -> Iterable[str]:
    from datasets import load_dataset

    scanned = 0
    fields = source.get("text_fields", ["text"])
    for split in source.get("splits", ["train"]):
        print(f"  streaming HF dataset: {source['repo']} split={split}")
        try:
            dataset = load_dataset(source["repo"], split=split, streaming=True)
        except Exception:
            continue
        for row in dataset:
            for field in fields:
                value = row.get(field)
                if not value:
                    continue
                for sentence in split_text_to_sentences(str(value)):
                    yield sentence
                    scanned += 1
                    if scanned >= max_scan_sources:
                        return


def _iter_ud_source(source: dict[str, Any], *, max_scan_sources: int) -> Iterable[str]:
    scanned = 0
    for url in source.get("urls", []):
        print(f"  downloading UD file: {url}")
        try:
            with urlopen(url, timeout=30) as response:
                content = response.read().decode("utf-8", errors="ignore")
        except (OSError, URLError):
            continue
        for sentence in parse_ud_conllu_texts(content):
            yield sentence
            scanned += 1
            if scanned >= max_scan_sources:
                return


def _iter_opencorpora_source(source: dict[str, Any], *, max_scan_sources: int) -> Iterable[str]:
    url = source.get("url")
    if not url:
        return
    print(f"  downloading OpenCorpora archive: {url}")
    try:
        with urlopen(url, timeout=45) as response:
            archive_bytes = response.read()
    except (OSError, URLError):
        return
    scanned = 0
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        xml_names = [name for name in archive.namelist() if name.endswith(".xml")]
        for xml_name in xml_names:
            with archive.open(xml_name) as xml_file:
                for sentence in _iter_opencorpora_xml_sentences(xml_file):
                    yield sentence
                    scanned += 1
                    if scanned >= max_scan_sources:
                        return


def _iter_opencorpora_xml_sentences(xml_file: Any) -> Iterable[str]:
    for _event, element in ElementTree.iterparse(xml_file, events=("end",)):
        if not element.tag.endswith("sentence"):
            continue
        tokens = [token.attrib.get("text", "") for token in element.iter() if token.tag.endswith("token")]
        sentence = _join_tokens(tokens)
        if sentence:
            yield sentence
        element.clear()


def _join_tokens(tokens: list[str]) -> str:
    text = ""
    no_space_before = set(",.!?:;%)]}»")
    no_space_after = set("([«")
    for token in tokens:
        if not token:
            continue
        if not text or token in no_space_before or text[-1] in no_space_after:
            text += token
        else:
            text += " " + token
    return normalize_sentence(text)


def _read_cache(path: Path, limit: int) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        return [line.strip() for _, line in zip(range(limit), handle) if line.strip()]


def _write_cache(path: Path, sentences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for sentence in sentences:
            handle.write(sentence + "\n")


_REJECT_MARKERS = (
    " блин",
    " хрен",
    " фиг",
    " чувак",
    " чувиха",
    " ёп",
    " епт",
    " нах",
    " хуй",
    " пизд",
    " бляд",
    " сука",
    " говн",
    " vk.com",
    " instagram",
    " telegram",
)

_STRICT_REJECT_MARKERS = (
    " сказал",
    " сказала",
    " спросил",
    " спросила",
    " ответил",
    " ответила",
    " прошептал",
    " крикнул",
    " улыбнулся",
    " подумал",
    " подумала",
)
