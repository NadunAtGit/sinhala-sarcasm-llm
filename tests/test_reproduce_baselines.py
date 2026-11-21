"""§13: the scorer must reproduce the ICARC-MS three-seed means from the exported E1 files.

Skipped until data/processed/test.jsonl and data/baselines/*.csv exist.
Expected file names: data/baselines/<model>_seed<s>_test.csv, model in {xlmr_large_sent, xlmr_large_mt}.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sarcllm.data import load_split
from sarcllm.eval.metrics import score

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data/processed/test.jsonl"
BASE = ROOT / "data/baselines"
SEEDS = (42, 43, 44)
TOL = 5e-4

EXPECTED = {
    ("xlmr_large_sent", "sent_f1_overall"): 0.6461,
    ("xlmr_large_mt", "sent_f1_overall"): 0.6501,
    ("xlmr_large_sent", "sent_f1_sarcastic"): 0.4544,
    ("xlmr_large_mt", "sent_f1_sarcastic"): 0.4502,
    ("xlmr_large_mt", "det_f1_macro"): 0.7547,
}


def _files(model):
    return [BASE / f"{model}_seed{s}_test.csv" for s in SEEDS]


@pytest.mark.parametrize("model,metric", list(EXPECTED))
def test_reproduce(model, metric):
    files = _files(model)
    if not TEST.exists() or not all(f.exists() for f in files):
        pytest.skip("E1 prediction files not exported yet")
    gold = load_split(TEST)
    vals = [score(gold, pd.read_csv(f, dtype={"id": str}), by_script=False)[metric] for f in files]
    assert np.mean(vals) == pytest.approx(EXPECTED[(model, metric)], abs=TOL), vals
