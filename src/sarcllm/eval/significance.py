"""McNemar (continuity-corrected) and Holm — Dev Plan v2 §6, §11.1.

McNemar tests row-level correctness (accuracy-type agreement), so it complements,
not replaces, the Macro-F1 bootstrap CIs. Holm is applied across H1–H5 within each
detector seed, as in ICARC-MS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
from statsmodels.stats.multitest import multipletests


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int((a & ~b).sum())   # a right, b wrong
    n10 = int((~a & b).sum())   # a wrong, b right
    tbl = [[int((a & b).sum()), n01], [n10, int((~a & ~b).sum())]]
    if n01 + n10 == 0:
        return {"a_only": 0, "b_only": 0, "chi2": 0.0, "p": 1.0}
    r = _mcnemar(tbl, exact=False, correction=True)
    return {"a_only": n01, "b_only": n10, "chi2": float(r.statistic), "p": float(r.pvalue)}


def sentiment_correct(arr, mask: np.ndarray | None = None) -> np.ndarray:
    c = arr.p_sent == arr.y_sent
    return c if mask is None else c[mask]


def holm(pvals: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    keys = list(pvals)
    reject, p_adj, _, _ = multipletests([pvals[k] for k in keys], alpha=alpha, method="holm")
    return pd.DataFrame({"p": [pvals[k] for k in keys], "p_holm": p_adj, "reject": reject}, index=keys)