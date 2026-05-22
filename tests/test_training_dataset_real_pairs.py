from pathlib import Path

from src.data._training_dataset_builder import should_refresh_real_pair_cache


def test_real_pair_cache_refreshes_when_cached_count_is_below_minimum(tmp_path: Path):
    cache_path = tmp_path / "real_error_pairs_validated.csv.gz"
    cache_path.write_text("source,target\n", encoding="utf-8")

    assert should_refresh_real_pair_cache(cache_path, cached_count=1236, minimum=5000, preferred=30000) is True
    assert should_refresh_real_pair_cache(cache_path, cached_count=6000, minimum=5000, preferred=30000) is False
    assert should_refresh_real_pair_cache(cache_path.with_name("missing.csv.gz"), cached_count=0, minimum=5000, preferred=30000) is True
