import math
from csd.features import extract_features, FEATURE_ORDER, feature_vector


def test_feature_order_matches_keys():
    feats = extract_features([1.0, 2.0, 3.0, 4.0], ranks=[0, 1, 0, 2])
    assert set(feats.keys()) == set(FEATURE_ORDER)


def test_constant_signal_has_zero_variance_and_acf():
    feats = extract_features([2.0, 2.0, 2.0, 2.0], ranks=[0, 0, 0, 0])
    assert feats["var"] == 0.0
    assert feats["acf_1"] == 0.0           # zero-variance -> defined as 0
    assert feats["frac_argmax"] == 1.0     # all rank 0
    assert feats["spectral_energy"] == 0.0  # mean-removed constant -> no energy


def test_alternating_signal_has_negative_acf1():
    # 1,3,1,3 mean-removed = -1,+1,-1,+1; biased lag-1 autocorr = -3/4 at n=4
    # (the biased estimator normalizes by total energy, so finite n never reaches -1)
    feats = extract_features([1.0, 3.0, 1.0, 3.0], ranks=None)
    assert math.isclose(feats["acf_1"], -0.75, abs_tol=1e-9)


def test_rank_features_default_zero_when_ranks_none():
    feats = extract_features([1.0, 2.0, 3.0], ranks=None)
    assert feats["frac_argmax"] == 0.0 and feats["mean_rank"] == 0.0


def test_feature_vector_is_fixed_length_in_order():
    v = feature_vector(extract_features([1.0, 2.0, 3.0, 4.0], ranks=[0, 1, 2, 3]))
    assert len(v) == len(FEATURE_ORDER)
    assert v[FEATURE_ORDER.index("mean")] == 2.5
