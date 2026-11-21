import numpy as np
import pytest
from sklearn.metrics import f1_score

from sarcllm.eval.bootstrap import ci_diff_in_gains, ci_paired, post_weights
from sarcllm.eval.metrics import INVALID, Arrays, compute
from sarcllm.eval.significance import holm, mcnemar


def _system(test, acc, seed, sarc_acc=0.8):
    rng = np.random.default_rng(seed)
    y = test.sentiment.map({"negative": 0, "neutral": 1, "positive": 2}).to_numpy()
    p = np.where(rng.random(len(y)) < acc, y, rng.integers(0, 3, len(y)))
    ys = test.is_sarcastic.to_numpy()
    ps = np.where(rng.random(len(y)) < sarc_acc, ys, 1 - ys)
    return Arrays(y, p, ys, ps, test.script_type.to_numpy())


def test_macro_f1_matches_sklearn(synthetic_splits):
    te = synthetic_splits["test"]
    a = _system(te, 0.6, 0)
    m = compute(a)
    sar = a.y_sarc == 1
    assert m["sent_f1_overall"] == pytest.approx(f1_score(a.y_sent, a.p_sent, average="macro"))
    assert m["sent_f1_sarcastic"] == pytest.approx(f1_score(a.y_sent[sar], a.p_sent[sar], average="macro"))
    assert m["det_f1_macro"] == pytest.approx(f1_score(a.y_sarc, a.p_sarc, average="macro"))
    assert m["penalty"] == pytest.approx(m["sent_f1_nonsarcastic"] - m["sent_f1_sarcastic"])


def test_invalid_counts_as_wrong_but_not_a_class():
    y = np.array([0, 1, 2, 0]); p = np.array([0, 1, 2, INVALID])
    a = Arrays(y, p, np.zeros(4, int), None, np.array(["latin"] * 4))
    m = compute(a)
    # class 0: tp=1, support=2, pred=1 -> F1 = 2/3 ; classes 1,2 perfect
    assert m["sent_f1_overall"] == pytest.approx((2 / 3 + 1 + 1) / 3)
    assert m["parse_fail_rate"] == 0.25


def test_neg_overprediction():
    y = np.array([1, 1, 2, 2, 0]); p = np.array([0, 1, 0, 0, 0])
    a = Arrays(y, p, np.ones(5, int), None, np.array(["latin"] * 5))
    m = compute(a)
    assert m["neg_overpred_neu"] == 0.5 and m["neg_overpred_pos"] == 1.0
    assert m["neg_overpred_neupos"] == 0.75


def test_weights_are_by_post(synthetic_splits):
    te = synthetic_splits["test"]
    w = next(post_weights(te.post_id.to_numpy(), 1, seed=0))
    # every comment of the same post gets the same weight; total posts drawn = 99
    assert te.assign(w=w).groupby("post_id").w.nunique().max() == 1
    assert te.assign(w=w).groupby("post_id").w.first().sum() == 99


def test_identical_systems_give_zero_ci(synthetic_splits):
    te = synthetic_splits["test"]
    a = _system(te, 0.6, 0)
    r = ci_paired(a, a, te.post_id.to_numpy(), n_boot=200)
    assert (r[["estimate", "ci_lo", "ci_hi"]].abs().max() == 0).all()


def test_better_system_is_detected(synthetic_splits):
    te = synthetic_splits["test"]
    good, bad = _system(te, 0.8, 1), _system(te, 0.5, 2)
    r = ci_paired(good, bad, te.post_id.to_numpy(), n_boot=500)
    assert r.loc["sent_f1_overall", "ci_lo"] > 0


def test_diff_in_gains_runs(synthetic_splits):
    te = synthetic_splits["test"]
    s = (te.is_sarcastic == 1).to_numpy()
    sub = lambda x: Arrays(x.y_sent[s], x.p_sent[s], x.y_sarc[s], x.p_sarc[s], x.script[s])  # noqa: E731
    sysx = [sub(_system(te, acc, i)) for i, acc in enumerate([0.9, 0.5, 0.6, 0.5])]
    r = ci_diff_in_gains(*sysx, post_ids=te.post_id.to_numpy()[s], n_boot=300)
    assert r.loc["diff_in_gains", "ci_lo"] > 0


def test_mcnemar_and_holm():
    a = np.array([1] * 60 + [0] * 40, bool); b = np.array([1] * 40 + [0] * 60, bool)
    r = mcnemar(a, b)
    assert r["a_only"] == 20 and r["b_only"] == 0 and r["p"] < 0.001
    h = holm({"H1": 0.01, "H2": 0.04, "H3": 0.03})
    assert h.loc["H1", "reject"] and not h.loc["H2", "reject"]
