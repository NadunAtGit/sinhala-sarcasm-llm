from sarcllm.data import build_option_b, build_pilot, check_counts, detect_script


def test_counts_pass_on_exact_synthetic(synthetic_splits):
    assert check_counts(synthetic_splits) == []


def test_counts_catch_a_missing_row(synthetic_splits):
    s = dict(synthetic_splits, test=synthetic_splits["test"].iloc[1:])
    assert any("test size" in e for e in check_counts(s))


def test_option_b(synthetic_splits):
    te = synthetic_splits["test"]
    ids = build_option_b(te, seed=42)
    sub = te[te.id.isin(ids)]
    assert (sub.is_sarcastic == 1).sum() == 2277 and (sub.is_sarcastic == 0).sum() == 2277
    assert ids == build_option_b(te, seed=42)  # deterministic
    # script shares of sampled non-sarcastic rows track the population (±1 row per stratum)
    pop = te[te.is_sarcastic == 0].script_type.value_counts(normalize=True) * 2277
    got = sub[sub.is_sarcastic == 0].script_type.value_counts()
    assert ((got - pop).abs() <= 1).all()


def test_pilot(synthetic_splits):
    ids = build_pilot(synthetic_splits["val"])
    assert len(ids) == 300 and len(set(ids)) == 300


def test_detect_script():
    assert detect_script("මට හරිම සතුටුයි") == "sinhala"
    assert detect_script("supiri wade machan") == "latin"
    assert detect_script("supiri machan හරිම සතුටුයි") == "mixed"
