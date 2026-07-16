import math
from csd.stats import (youden_j, clopper_pearson, youden_j_ci, mcnemar, bonferroni_threshold,
                       auc, paired_bootstrap_delta, bootstrap_delta_pvalue, holm_bonferroni)


def test_youden_j_perfect():
    # 15 TP, 0 FN, 0 FP, 15 TN -> TPR 1.0 - FPR 0.0 = 1.0
    assert youden_j(15, 0, 0, 15) == 1.0


def test_youden_j_chance():
    assert youden_j(0, 15, 0, 15) == 0.0


def test_clopper_pearson_full_n15():
    lo, hi = clopper_pearson(15, 15)
    assert math.isclose(lo, 0.7820, abs_tol=1e-3)
    assert hi == 1.0


def test_clopper_pearson_zero_n15():
    lo, hi = clopper_pearson(0, 15)
    assert lo == 0.0
    assert math.isclose(hi, 0.2180, abs_tol=1e-3)


def test_youden_j_ci_perfect_cell():
    # TPR 15/15, FPR 0/15 -> J=1.0, CI [TPR_lo - FPR_hi, TPR_hi - FPR_lo] = [0.564, 1.0]
    j, lo, hi = youden_j_ci(tp=15, fn=0, fp=0, tn=15)
    assert j == 1.0
    assert math.isclose(lo, 0.564, abs_tol=1e-3)
    assert hi == 1.0


def test_mcnemar_significant():
    # 28 discordant one way, 0 the other -> highly significant (paper tag-char cell)
    chi2, p = mcnemar(b=28, c=0)
    assert p < 1e-4


def test_bonferroni_threshold():
    assert math.isclose(bonferroni_threshold(0.05, 64), 0.05 / 64)


def test_auc_perfect_and_chance():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert math.isclose(auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]), 0.5, abs_tol=1e-9)


def test_auc_handles_ties_and_empty_class():
    # one class only -> undefined, return 0.5
    assert auc([0.1, 0.2, 0.3], [0, 0, 0]) == 0.5
    # perfectly anti-separated -> 0.0
    assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_paired_bootstrap_delta_sign_and_ci():
    # a separates perfectly; b barely separates -> delta(AUC) > 0, point within its BCa CI
    a = [0.1, 0.15, 0.2, 0.25, 0.7, 0.8, 0.85, 0.9]
    b = [0.40, 0.45, 0.50, 0.55, 0.42, 0.48, 0.52, 0.58]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    d, lo, hi = paired_bootstrap_delta(a, b, y, metric="auc", n_boot=800, seed=1)
    assert d > 0
    assert lo <= d <= hi


def test_paired_bootstrap_delta_metric_j():
    a = [0.1, 0.15, 0.2, 0.25, 0.7, 0.8, 0.85, 0.9]
    b = [0.40, 0.45, 0.50, 0.55, 0.42, 0.48, 0.52, 0.58]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    d, lo, hi = paired_bootstrap_delta(a, b, y, metric="j", n_boot=400, seed=2)
    assert d > 0 and lo <= d <= hi


def test_bootstrap_delta_pvalue_directional():
    # a separates perfectly (auc=1.0); b is uninformative (auc=0.5)
    a = [0.1 + 0.01 * i for i in range(16)] + [0.7 + 0.01 * i for i in range(16)]
    b = [0.5] * 32
    y = [0] * 16 + [1] * 16
    assert bootstrap_delta_pvalue(a, b, y, metric="auc", n_boot=800, seed=1) < 0.05
    # reversed: a is NOT better than b -> large p (not significant in our direction)
    assert bootstrap_delta_pvalue(b, a, y, metric="auc", n_boot=800, seed=1) > 0.5


def test_holm_bonferroni_orders_and_controls():
    # smallest p rejected first; clearly-null p not rejected; step-down stops at first failure
    assert holm_bonferroni([0.001, 0.02, 0.9], alpha=0.05) == [True, True, False]
    # original-order preserved when inputs are out of order
    assert holm_bonferroni([0.9, 0.001, 0.02], alpha=0.05) == [False, True, True]
    # nothing significant
    assert holm_bonferroni([0.4, 0.6, 0.8], alpha=0.05) == [False, False, False]
