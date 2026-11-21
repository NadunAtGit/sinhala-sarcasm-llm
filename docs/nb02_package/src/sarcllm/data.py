"""Data loading, validation and eval-set construction (Dev Plan v2, §5.1, §6, §9).

Labels are stored as strings in the .jsonl files and mapped to integers here:
    negative=0, neutral=1, positive=2
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

SENTIMENTS = ("negative", "neutral", "positive")
SENT2ID = {s: i for i, s in enumerate(SENTIMENTS)}
SCRIPTS = ("sinhala", "latin", "mixed")

# --- §5.1 required counts -------------------------------------------------
EXPECTED = {
    "split_sizes": {"train": 32_582, "val": 6_938, "test": 6_875},
    "test_sentiment": {"negative": 3_763, "neutral": 1_908, "positive": 1_204},
    "test_sarcastic": 2_277,
    "test_sarcastic_sentiment": {"negative": 1_815, "neutral": 344, "positive": 118},
    "test_posts": 99,
}


class Record(BaseModel):
    id: str
    post_id: str
    text: str
    context_text: str
    script_type: Literal["sinhala", "latin", "mixed"]
    sentiment: Literal["negative", "neutral", "positive"]
    is_sarcastic: int
    split: Literal["train", "val", "test"]

    @field_validator("is_sarcastic")
    @classmethod
    def _binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("is_sarcastic must be 0 or 1")
        return v

    @field_validator("text", "context_text")
    @classmethod
    def _nfc(cls, v: str) -> str:
        return unicodedata.normalize("NFC", v or "")


# --- script detection (fallback only if the thesis data has no script field) --
def detect_script(text: str, hi: float = 0.8, lo: float = 0.2) -> str:
    """Share of Sinhala letters (U+0D80–U+0DFF) among all letters.

    >= hi -> sinhala, <= lo -> latin (Singlish), otherwise mixed.
    [DECISION] Thresholds must match the definition used in ICARC-MS if one exists;
    record whichever is used in RESEARCH_LOG.md.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "latin"
    share = sum("඀" <= c <= "෿" for c in letters) / len(letters)
    if share >= hi:
        return "sinhala"
    if share <= lo:
        return "latin"
    return "mixed"


# --- I/O --------------------------------------------------------------------
def load_split(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(Record(**json.loads(line)).model_dump())
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{path}:{i}: {e}") from e
    df = pd.DataFrame(rows)
    if df["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate ids")
    df["y_sent"] = df["sentiment"].map(SENT2ID).astype(int)
    return df


def write_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    cols = list(Record.model_fields)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in df[cols].to_dict("records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_all(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(processed_dir)
    return {s: load_split(d / f"{s}.jsonl") for s in ("train", "val", "test")}


# --- §5.1 checks ------------------------------------------------------------
def check_counts(splits: dict[str, pd.DataFrame]) -> list[str]:
    """Return a list of mismatches (empty list = all checks pass)."""
    errs: list[str] = []

    def chk(name, got, want):
        if got != want:
            errs.append(f"{name}: got {got}, expected {want}")

    for s, n in EXPECTED["split_sizes"].items():
        chk(f"{s} size", len(splits[s]), n)
    te = splits["test"]
    for lab, n in EXPECTED["test_sentiment"].items():
        chk(f"test {lab}", int((te.sentiment == lab).sum()), n)
    sar = te[te.is_sarcastic == 1]
    chk("test sarcastic", len(sar), EXPECTED["test_sarcastic"])
    for lab, n in EXPECTED["test_sarcastic_sentiment"].items():
        chk(f"test sarcastic {lab}", int((sar.sentiment == lab).sum()), n)
    chk("test posts", te.post_id.nunique(), EXPECTED["test_posts"])

    # leakage-relevant structure: no id or post_id shared across splits
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ids = set(splits[a].id) & set(splits[b].id)
        if ids:
            errs.append(f"{len(ids)} ids shared between {a} and {b}")
        posts = set(splits[a].post_id) & set(splits[b].post_id)
        if posts:
            # Not necessarily an error (depends on how the thesis split was made),
            # but retrieval must then also exclude these post_ids. Flag it.
            errs.append(f"WARNING {len(posts)} post_ids shared between {a} and {b}")
    return errs


# --- eval sets (§3 option B, §9 pilot) ---------------------------------------
def _stratified_take(df: pd.DataFrame, n: int, by: str, rng: np.random.Generator) -> pd.DataFrame:
    """Take n rows, allocating proportionally across `by` (largest remainder)."""
    if n >= len(df):
        return df
    shares = df[by].value_counts(normalize=True).sort_index()
    raw = shares * n
    alloc = np.floor(raw).astype(int)
    for k in (raw - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[k] += 1
    parts = [
        df[df[by] == k].sample(n=int(m), random_state=int(rng.integers(1 << 31)))
        for k, m in alloc.items() if m > 0
    ]
    return pd.concat(parts)


def build_option_b(test: pd.DataFrame, seed: int = 42) -> list[str]:
    """All sarcastic test rows + an equal number of non-sarcastic rows, stratified by script."""
    rng = np.random.default_rng(seed)
    sar = test[test.is_sarcastic == 1]
    non = _stratified_take(test[test.is_sarcastic == 0], len(sar), "script_type", rng)
    return sorted(pd.concat([sar, non]).id.tolist())


# [DECISION] pilot allocation: sarcastic Neutral/Positive over-represented for H5.
PILOT_ALLOC = {
    (1, "negative"): 80, (1, "neutral"): 60, (1, "positive"): 40,
    (0, "negative"): 40, (0, "neutral"): 40, (0, "positive"): 40,
}


def build_pilot(val: pd.DataFrame, alloc: dict = PILOT_ALLOC, seed: int = 42) -> list[str]:
    """300 validation rows, stratified by sarcasm x sentiment, script proportional within cell."""
    rng = np.random.default_rng(seed)
    parts = []
    for (sar, sent), n in alloc.items():
        cell = val[(val.is_sarcastic == sar) & (val.sentiment == sent)]
        if len(cell) < n:
            raise ValueError(f"pilot cell {(sar, sent)} has {len(cell)} rows < {n}")
        parts.append(_stratified_take(cell, n, "script_type", rng))
    return sorted(pd.concat(parts).id.tolist())


def write_ids(ids: list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(ids) + "\n", encoding="utf-8")


def read_ids(path: str | Path) -> list[str]:
    return [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
