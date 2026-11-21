import numpy as np
import pandas as pd
import pytest


def make_split(split, n, n_posts, seed, sizes=None):
    """Synthetic split. If `sizes` given, it fixes the exact sentiment x sarcasm cell counts."""
    rng = np.random.default_rng(seed)
    if sizes:
        cells = [(sent, sar) for (sent, sar), k in sizes.items() for _ in range(k)]
        rng.shuffle(cells)
        sent, sar = zip(*cells)
    else:
        sent = rng.choice(["negative", "neutral", "positive"], n, p=[.55, .28, .17])
        sar = (rng.random(n) < 0.33).astype(int)
    n = len(sent)
    post = np.r_[np.arange(n_posts), rng.integers(0, n_posts, n - n_posts)]
    return pd.DataFrame({
        "id": [f"{split}_{i}" for i in range(n)],
        "post_id": [f"{split}_p{p}" for p in post],
        "text": ["x"] * n, "context_text": ["c"] * n,
        "script_type": rng.choice(["sinhala", "latin", "mixed"], n, p=[.5, .35, .15]),
        "sentiment": list(sent), "is_sarcastic": list(map(int, sar)), "split": split,
    })


# exact §5.1 test cells: sarcastic 1815/344/118; non-sarcastic = totals minus those
TEST_CELLS = {("negative", 1): 1815, ("neutral", 1): 344, ("positive", 1): 118,
              ("negative", 0): 3763 - 1815, ("neutral", 0): 1908 - 344, ("positive", 0): 1204 - 118}


@pytest.fixture(scope="session")
def synthetic_splits():
    return {
        "train": make_split("train", 32_582, 400, 1),
        "val": make_split("val", 6_938, 100, 2),
        "test": make_split("test", None, 99, 3, sizes=TEST_CELLS),
    }
