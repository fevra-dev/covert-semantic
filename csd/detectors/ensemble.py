"""Fusion ensembles over the base detectors (design §5).

MaxFusionEnsemble (primary): max of calibrated-normalized perplexity & distributional
scores — a single fused ROC, no extra trainable parameters; the honest "union of
complementary detectors" claim. StackedEnsemble (secondary): a logistic meta-classifier
over [ppl_z, xppl_z, dist_p], trained on cross-fitted out-of-fold base scores (the caller
supplies the rows) so it never trains on its own test items."""
from __future__ import annotations
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from csd.detectors.perplexity_anomaly import Verdict


def _phi(z: float) -> float:
    """Standard-normal CDF: maps an already-cover-standardized z-score into [0, 1]."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class MaxFusionEnsemble:
    """fused = max(Phi(perplexity_z), distributional_p); hidden = fused >= threshold.
    Both base detectors must already be calibrated/fitted (the evaluation harness does
    this on the train partition; the threshold is chosen out-of-sample)."""

    def __init__(self, perplexity, distributional, threshold: float = 0.5):
        self._ppl = perplexity
        self._dist = distributional
        self.threshold = threshold

    def calibrate(self, cover_texts: list[str], stego_texts: list[str]) -> None:
        """Calibrate the perplexity baseline (covers) and fit the distributional base."""
        self._ppl.calibrate(cover_texts)
        self._dist.fit(cover_texts=cover_texts, stego_texts=stego_texts)

    def score(self, text: str) -> Verdict:
        if not text.strip():
            return Verdict(hidden=False, z=0.0, reason="empty input", status="empty")
        pv = self._ppl.score(text)
        dv = self._dist.score(text)
        for v in (pv, dv):
            if v.status != "parsed":
                return Verdict(hidden=False, z=0.0, reason=f"base {v.status}", status=v.status)
        fused = max(_phi(pv.z), dv.z)
        return Verdict(hidden=fused >= self.threshold, z=fused,
                       reason=f"max-fusion={fused:.2f} (Phi(ppl)={_phi(pv.z):.2f}, dist={dv.z:.2f})")


class StackedEnsemble:
    """Logistic meta-classifier over [perplexity_z, cross_perplexity_z, distributional_p].
    `fit` consumes precomputed (cross-fitted) rows + labels; `score` computes the row live
    from the three fitted base detectors."""

    def __init__(self, perplexity, cross, distributional, threshold: float = 0.5,
                 random_state: int = 0):
        self._ppl = perplexity
        self._cross = cross
        self._dist = distributional
        self.threshold = threshold
        self._clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=random_state, max_iter=2000),
        )
        self._fitted = False

    def fit(self, rows: list[list[float]], labels: list[int]) -> None:
        self._clf.fit(np.asarray(rows, dtype=float), np.asarray(labels))
        self._fitted = True

    def calibrate(self, cover_texts: list[str], stego_texts: list[str]) -> None:
        """Calibrate the three base detectors, then fit the meta-classifier on in-train rows.
        (The leakage-free cross-fitted variant lives in evaluation.crossfit_base_scores; the
        outer CV in the evaluation harness already keeps test items out of training.)"""
        self._ppl.calibrate(cover_texts)
        self._cross.calibrate(cover_texts)
        self._dist.fit(cover_texts=cover_texts, stego_texts=stego_texts)
        texts = list(cover_texts) + list(stego_texts)
        labels = [0] * len(cover_texts) + [1] * len(stego_texts)
        self.fit([self._row_from_text(t) for t in texts], labels)

    def _row_from_text(self, text: str) -> list[float]:
        return [self._ppl.score(text).z, self._cross.score(text).z, self._dist.score(text).z]

    def score_from_row(self, row: list[float]) -> Verdict:
        if not self._fitted:
            raise RuntimeError("detector not fitted; call fit() first")
        p = float(self._clf.predict_proba(np.asarray(row, dtype=float).reshape(1, -1))[0, 1])
        return Verdict(hidden=p >= self.threshold, z=p, reason=f"stacked p(stego)={p:.2f}")

    def score(self, text: str) -> Verdict:
        if not text.strip():
            return Verdict(hidden=False, z=0.0, reason="empty input", status="empty")
        for v in (self._ppl.score(text), self._dist.score(text)):
            if v.status != "parsed":
                return Verdict(hidden=False, z=0.0, reason=f"base {v.status}", status=v.status)
        return self.score_from_row(self._row_from_text(text))
