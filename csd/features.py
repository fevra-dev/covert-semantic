"""Hand-engineered distributional features over a text's surprisal + rank streams.

Pure (no model). Feature families: distribution shape (DivEye lineage), spectral
energy (SpecDetect lineage), surprisal autocorrelation (Chromatic doc), rank
statistics (watermark-without-key signal). FEATURE_ORDER fixes the vector layout."""
from __future__ import annotations
import numpy as np
from scipy import stats as _sps

_MAX_LAG = 8

FEATURE_ORDER = (
    ["mean", "var", "skew", "kurtosis", "spectral_energy", "spectral_peak_ratio"]
    + [f"acf_{lag}" for lag in range(1, _MAX_LAG + 1)]
    + ["frac_argmax", "mean_rank", "frac_rank_le2"]
)


def _autocorr(s: np.ndarray, lag: int) -> float:
    n = len(s)
    if n <= lag:
        return 0.0
    s0 = s - s.mean()
    denom = float((s0 * s0).sum())
    if denom == 0.0:
        return 0.0
    return float((s0[:-lag] * s0[lag:]).sum()) / denom


def extract_features(surprisals, ranks=None, max_lag: int = _MAX_LAG) -> dict[str, float]:
    s = np.asarray(surprisals, dtype=np.float64)
    n = len(s)
    feats: dict[str, float] = {}
    feats["mean"] = float(s.mean()) if n else 0.0
    feats["var"] = float(s.var()) if n else 0.0
    feats["skew"] = float(_sps.skew(s)) if n > 2 and s.var() > 0 else 0.0
    feats["kurtosis"] = float(_sps.kurtosis(s)) if n > 3 and s.var() > 0 else 0.0  # excess
    if n:
        d = s - s.mean()
        spec = np.abs(np.fft.rfft(d)) ** 2
        feats["spectral_energy"] = float(spec.sum())
        feats["spectral_peak_ratio"] = (
            float(spec.max() / (spec.mean() + 1e-12)) if spec.size and spec.sum() > 0 else 0.0
        )
    else:
        feats["spectral_energy"] = 0.0
        feats["spectral_peak_ratio"] = 0.0
    for lag in range(1, max_lag + 1):
        feats[f"acf_{lag}"] = _autocorr(s, lag)
    if ranks is not None and len(ranks):
        r = np.asarray(ranks, dtype=np.float64)
        feats["frac_argmax"] = float((r == 0).mean())
        feats["mean_rank"] = float(r.mean())
        feats["frac_rank_le2"] = float((r <= 2).mean())
    else:
        feats["frac_argmax"] = 0.0
        feats["mean_rank"] = 0.0
        feats["frac_rank_le2"] = 0.0
    return feats


def feature_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float64)
