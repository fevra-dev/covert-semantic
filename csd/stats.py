"""Deterministic statistics for detection metrics (matches GLOSSOPETRAE's reporting)."""
from __future__ import annotations
import numpy as np
from scipy.stats import beta, binom, norm, rankdata


def _tpr_fpr(tp: int, fn: int, fp: int, tn: int) -> tuple[float, float]:
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr, fpr


def youden_j(tp: int, fn: int, fp: int, tn: int) -> float:
    """J = TPR - FPR. 0 = chance, 1 = perfect."""
    tpr, fpr = _tpr_fpr(tp, fn, fp, tn)
    return tpr - fpr


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def youden_j_ci(tp: int, fn: int, fp: int, tn: int, alpha: float = 0.05):
    """J with CI propagated as [TPR_lo - FPR_hi, TPR_hi - FPR_lo] (GLOSSOPETRAE §4.2.1)."""
    tpr, fpr = _tpr_fpr(tp, fn, fp, tn)
    tpr_lo, tpr_hi = clopper_pearson(tp, tp + fn, alpha)
    fpr_lo, fpr_hi = clopper_pearson(fp, fp + tn, alpha)
    return (tpr - fpr, tpr_lo - fpr_hi, tpr_hi - fpr_lo)


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Paired-binary test on discordant pairs (b, c). Exact binomial for small n,
    continuity-corrected chi-square otherwise. Returns (statistic, p_value)."""
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    if n < 25:
        k = min(b, c)
        p = min(1.0, 2.0 * binom.cdf(k, n, 0.5))
        return (float(min(b, c)), float(p))
    chi2 = (abs(b - c) - 1) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    p = float(chi2_dist.sf(chi2, df=1))
    return (float(chi2), p)


def bonferroni_threshold(alpha: float, m: int) -> float:
    """Family-wise corrected per-test threshold."""
    return alpha / m


# --- Phase-2 rigor additions: AUC, paired BCa bootstrap, Holm-Bonferroni -------------------

def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based ROC AUC (Mann-Whitney U / (n_pos*n_neg)), tie-safe via average ranks.
    Returns 0.5 when either class is empty (undefined)."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(s)
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _max_j(scores: list[float], labels: list[int]) -> float:
    """Threshold-free Youden J summary: max over candidate thresholds of (TPR - FPR)."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    best = 0.0
    for t in np.unique(s):
        pred = s >= t
        tpr = float((pred & (y == 1)).sum()) / n_pos
        fpr = float((pred & (y == 0)).sum()) / n_neg
        best = max(best, tpr - fpr)
    return best


_METRICS = {"auc": auc, "j": _max_j}


def _delta_stat(scores_a, scores_b, labels, metric):
    """Build the paired delta statistic stat(idx) = metric(a[idx]) - metric(b[idx])."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    y = np.asarray(labels)
    f = _METRICS[metric] if isinstance(metric, str) else metric

    def stat(idx):
        return f(a[idx], y[idx]) - f(b[idx], y[idx])

    return stat, y, len(y)


def _bootstrap(stat, y, n, n_boot, seed):
    """Resample item indices with replacement; skip resamples that collapse to one class."""
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) >= 2:
            boots.append(stat(idx))
    return np.asarray(boots, dtype=float)


def paired_bootstrap_delta(scores_a, scores_b, labels, metric="auc",
                           n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap of the metric difference (a - b) over the SAME resampled items,
    with a BCa confidence interval. `metric` is "auc", "j", or a callable(scores, labels).
    Returns (delta_point, ci_lo, ci_hi). Resamples that collapse to one class are skipped."""
    stat, y, n = _delta_stat(scores_a, scores_b, labels, metric)
    full = stat(np.arange(n))
    boots = _bootstrap(stat, y, n, n_boot, seed)
    if boots.size < 2:
        return float(full), float(full), float(full)

    # bias-correction z0
    prop = float(np.mean(boots < full))
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = norm.ppf(prop)

    # acceleration via jackknife over items
    jack = []
    base = np.arange(n)
    for i in range(n):
        ji = np.delete(base, i)
        jack.append(stat(ji) if len(np.unique(y[ji])) >= 2 else np.nan)
    jack = np.asarray(jack, dtype=float)
    jbar = np.nanmean(jack)
    diff = jbar - jack
    denom = 6.0 * (np.nansum(diff ** 2) ** 1.5)
    accel = float(np.nansum(diff ** 3) / denom) if denom != 0 else 0.0

    def _adj(z):
        num = z0 + z
        return float(norm.cdf(z0 + num / (1 - accel * num)))

    lo_q = _adj(norm.ppf(alpha / 2))
    hi_q = _adj(norm.ppf(1 - alpha / 2))
    if not (np.isfinite(lo_q) and np.isfinite(hi_q)):  # BCa degenerate -> percentile fallback
        lo_q, hi_q = alpha / 2, 1 - alpha / 2
    lo = float(np.quantile(boots, min(lo_q, hi_q)))
    hi = float(np.quantile(boots, max(lo_q, hi_q)))
    return float(full), lo, hi


def bootstrap_delta_pvalue(scores_a, scores_b, labels, metric="auc",
                           n_boot: int = 2000, seed: int = 0) -> float:
    """One-sided bootstrap p-value for H1: metric(a) > metric(b) (directional).
    p = fraction of bootstrap deltas <= 0 (large when a is not better than b)."""
    stat, y, n = _delta_stat(scores_a, scores_b, labels, metric)
    boots = _bootstrap(stat, y, n, n_boot, seed)
    if boots.size < 1:
        return 1.0
    return float((1 + np.sum(boots <= 0)) / (1 + boots.size))


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down FWER control. Returns per-hypothesis reject decisions in input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    for k, idx in enumerate(order):
        if pvalues[idx] <= alpha / (m - k):
            reject[idx] = True
        else:
            break  # step-down: once one fails, all larger p-values also fail
    return reject
