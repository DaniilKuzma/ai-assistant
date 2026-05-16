from __future__ import annotations

from typing import Any


def load_hf_dataset(name: str, **kwargs: Any) -> Any:
    """Load an open dataset lazily so tests do not require datasets/HF access."""

    from datasets import load_dataset

    return load_dataset(name, **kwargs)
