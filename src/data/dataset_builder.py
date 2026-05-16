from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict

import pandas as pd

from src.data.synthetic_generator import SyntheticExample, SyntheticGenerator


def build_synthetic_dataset(clean_texts: Iterable[str], seed: int = 13) -> pd.DataFrame:
    generator = SyntheticGenerator(seed=seed)
    examples: list[SyntheticExample] = [generator.generate_from_clean(text) for text in clean_texts]
    examples.extend(generator.add_identity_examples(list(clean_texts)))
    return pd.DataFrame([asdict(example) for example in examples])
