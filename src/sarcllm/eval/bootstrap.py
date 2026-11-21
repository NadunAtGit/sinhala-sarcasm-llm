"""Post-level (cluster) paired bootstrap — Dev Plan v2 §6, §11.

Resampling unit = parent post (99 test posts). Each resample draws 99 posts with
replacement; every comment of a drawn post enters with weight = times drawn.
"Paired" = the same resample is applied to every system being compared.

Two-sided bootstrap p-values are also returned (p = 2·min(P(d≤0), P(d≥0)), floored at
1/(B+1)) so that the Holm correction across H1–H5 can be applied to the CI-based
hypotheses as well as the McNemar ones.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .metrics import Arrays, compute

StatFn = Callable[[np.ndarray], dict[str, float]]


def post_weights(post_ids: np.ndarray, n_boot: int, seed: int = 42):
    """Yield row-weight vectors, one per resample. Deterministic given seed."""
    uniq, inv = np.unique(post_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    P = len(uniq)
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, P, P), minlength=P)
        yield counts[inv].astype(float)


def bootstrap(post_ids: np.ndarray, fn: StatFn, n_boot: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generic: fn(weights) -> {stat: value}. Returns one row per resample."""
    return pd.DataFrame([fn(w) for w in post_weights(post_ids, n_boot, seed)])


def summarise(point: dict[str, float], draws: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    B = len(draws)
    rows = []
    for k in draws.columns:
        d = draws[k].dropna().to_numpy()
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2]) if len(d) else (np.nan, np.nan)
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan
        rows.append({"stat": k, "estimate": point[k], "ci_lo": lo, "ci_hi": hi,
                     "p_boot": max(p, 1 / (B + 1)), "n_valid": len(d)})
    return pd.DataFrame(rows).set_index("stat")


# --- the two cases used everywhere -------------------------------------------
def ci_single(a: Arrays, post_ids: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> pd.DataFrame:
    point = compute(a)
    return summarise(point, bootstrap(post_ids, lambda w: compute(a, w), n_boot, seed))


def ci_paired(a: Arrays, b: Arrays, post_ids: np.ndarray, n_boot: int = 10_000,
              seed: int = 42) -> pd.DataFrame:
    """CI of (a − b) for every metric both systems have. Rows of a, b, post_ids must align."""
    _check_aligned(a, b, post_ids)

    def diff(w):
        ma, mb = compute(a, w), compute(b, w)
        return {k: ma[k] - mb[k] for k in ma.keys() & mb.keys()}

    return summarise(diff(np.ones(len(post_ids))), bootstrap(post_ids, diff, n_boot, seed)).sort_index()


def ci_diff_in_gains(oracle_a: Arrays, base_a: Arrays, oracle_b: Arrays, base_b: Arrays,
                     post_ids: np.ndarray, metric: str = "sent_f1_sarcastic",
                     n_boot: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """H4: (oracle_a − base_a) − (oracle_b − base_b), e.g. (L6 − L4) − (E2 − E1).

    All four systems must be aligned on the same rows (restrict to gold-sarcastic rows
    so the eval-set choice cannot confound H4).
    """
    for x in (base_a, oracle_b, base_b):
        _check_aligned(oracle_a, x, post_ids)

    def fn(w):
        g = lambda o, b: compute(o, w)[metric] - compute(b, w)[metric]  # noqa: E731
        ga, gb = g(oracle_a, base_a), g(oracle_b, base_b)
        return {"gain_llm": ga, "gain_encoder": gb, "diff_in_gains": ga - gb}

    return summarise(fn(np.ones(len(post_ids))), bootstrap(post_ids, fn, n_boot, seed))


def _check_aligned(a: Arrays, b: Arrays, post_ids: np.ndarray) -> None:
    if not (len(a.y_sent) == len(b.y_sent) == len(post_ids)):
        raise ValueError("systems are not row-aligned")
    if not (np.array_equal(a.y_sent, b.y_sent) and np.array_equal(a.y_sarc, b.y_sarc)):
        raise ValueError("gold labels differ: systems were aligned on different IDs")