"""All metrics in Dev Plan v2 §6 (eval/metrics.py row).

Every metric is computed from a *weighted* confusion matrix, so the same code serves
the point estimate (weights = 1) and the cluster bootstrap (weights = how many times
each row's post was drawn). This makes 10,000 resamples take seconds, not minutes.

Conventions
-----------
- Sentiment ids: negative=0, neutral=1, positive=2.  INVALID = -1 (parse failure).
  An invalid prediction counts as wrong (a false negative for the gold class) but is
  not a class of its own, so it never creates a spurious 4th F1 term.
- Macro-F1 averages over labels present in gold OR predictions, with zero_division=0,
  which is what sklearn.f1_score(average="macro") does with no `labels` argument.
  `test_reproduce_baselines` checks this matches the ICARC-MS numbers.
- Penalty = non-sarcastic Macro-F1 − sarcastic Macro-F1.
  [CHECK] confirm this is the ICARC-MS definition before Gate G1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INVALID = -1
N_SENT = 3


# --- core ---------------------------------------------------------------------
def confusion(y: np.ndarray, p: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """k x (k+1) weighted confusion; last column holds invalid predictions."""
    p = np.where(p < 0, k, p)
    return np.bincount(y * (k + 1) + p, weights=w, minlength=k * (k + 1)).reshape(k, k + 1)


def f1_per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class F1 and a mask of classes present (in gold or in valid predictions)."""
    k = cm.shape[0]
    tp = np.diag(cm[:, :k])
    support = cm.sum(1)
    pred = cm[:, :k].sum(0)
    denom = support + pred
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return f1, denom > 0


def macro_f1(cm: np.ndarray) -> float:
    f1, present = f1_per_class(cm)
    return float(f1[present].mean()) if present.any() else 0.0


# --- the full metric set ------------------------------------------------------
@dataclass
class Arrays:
    """Row-aligned numpy arrays for one system on one set of test IDs."""
    y_sent: np.ndarray        # gold sentiment 0/1/2
    p_sent: np.ndarray        # predicted sentiment 0/1/2 or -1
    y_sarc: np.ndarray        # gold sarcasm 0/1
    p_sarc: np.ndarray | None  # predicted sarcasm 0/1 or -1; None for sentiment-only systems
    script: np.ndarray        # "sinhala" | "latin" | "mixed"


def compute(a: Arrays, w: np.ndarray | None = None, by_script: bool = False) -> dict[str, float]:
    n = len(a.y_sent)
    w = np.ones(n) if w is None else w
    sar, non = a.y_sarc == 1, a.y_sarc == 0
    out: dict[str, float] = {}

    cm_all = confusion(a.y_sent, a.p_sent, w, N_SENT)
    cm_sar = confusion(a.y_sent[sar], a.p_sent[sar], w[sar], N_SENT)
    cm_non = confusion(a.y_sent[non], a.p_sent[non], w[non], N_SENT)
    out["sent_f1_overall"] = macro_f1(cm_all)
    out["sent_f1_sarcastic"] = macro_f1(cm_sar)
    out["sent_f1_nonsarcastic"] = macro_f1(cm_non)
    out["penalty"] = out["sent_f1_nonsarcastic"] - out["sent_f1_sarcastic"]
    for name, cm in (("overall", cm_all), ("sarcastic", cm_sar)):
        f1, _ = f1_per_class(cm)
        for i, lab in enumerate(("neg", "neu", "pos")):
            out[f"f1_{lab}_{name}"] = float(f1[i])
    out["sent_acc"] = float(np.trace(cm_all[:, :N_SENT]) / cm_all.sum())
    out["parse_fail_rate"] = float(cm_all[:, N_SENT].sum() / cm_all.sum())

    # Negative over-prediction: gold-sarcastic Neutral / Positive rows predicted Negative
    for lab, gid in (("neu", 1), ("pos", 2), ("neupos", None)):
        m = sar & ((a.y_sent == gid) if gid is not None else (a.y_sent > 0))
        tot = w[m].sum()
        out[f"neg_overpred_{lab}"] = float(w[m & (a.p_sent == 0)].sum() / tot) if tot else np.nan

    # Detection
    if a.p_sarc is not None:
        cm_det = confusion(a.y_sarc, a.p_sarc, w, 2)
        f1d, _ = f1_per_class(cm_det)
        out["det_f1_macro"] = macro_f1(cm_det)
        out["det_f1_sarcastic"] = float(f1d[1])
        out["routing_recall"] = float(cm_det[1, 1] / cm_det[1].sum())  # sarcastic-class recall
        out["routed_share"] = float(cm_det[:, 1].sum() / cm_det.sum())

    if by_script:
        for s in np.unique(a.script):
            m = a.script == s
            sub = Arrays(a.y_sent[m], a.p_sent[m], a.y_sarc[m],
                         None if a.p_sarc is None else a.p_sarc[m], a.script[m])
            for k, v in compute(sub, w[m]).items():
                out[f"{s}/{k}"] = v
    return out


# --- convenience: from DataFrames ---------------------------------------------
SENT2ID = {"negative": 0, "neutral": 1, "positive": 2}


def _to_sent_id(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(INVALID).astype(int).to_numpy()
    return s.map(SENT2ID).fillna(INVALID).astype(int).to_numpy()


def align(gold: pd.DataFrame, pred: pd.DataFrame, ids: list[str] | None = None,
          sent_col: str = "sentiment_pred", sarc_col: str | None = "is_sarcastic_pred") -> Arrays:
    """Join predictions onto gold by id. Missing predictions count as INVALID (wrong).

    `gold` needs: id, post_id, sentiment, is_sarcastic, script_type.
    """
    g = gold if ids is None else gold.set_index("id").loc[ids].reset_index()
    if pred["id"].duplicated().any():
        raise ValueError("duplicate ids in predictions")
    m = g.merge(pred, on="id", how="left", validate="one_to_one")
    p_sarc = None
    if sarc_col and sarc_col in m and m[sarc_col].notna().any():
        p_sarc = m[sarc_col].fillna(INVALID).astype(int).to_numpy()
    return Arrays(
        y_sent=_to_sent_id(m["sentiment"]),
        p_sent=_to_sent_id(m[sent_col]),
        y_sarc=m["is_sarcastic"].astype(int).to_numpy(),
        p_sarc=p_sarc,
        script=m["script_type"].to_numpy(),
    )


def score(gold: pd.DataFrame, pred: pd.DataFrame, ids: list[str] | None = None,
          by_script: bool = True, **kw) -> dict[str, float]:
    return compute(align(gold, pred, ids, **kw), by_script=by_script)
