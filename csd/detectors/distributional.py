"""Model-conditional distributional detector (design §4 Phase 2). Fits a logistic-
regression discriminator on hand-engineered features (csd.features) over (cover, stego)
pairs; score() returns the harness Verdict with p(stego) in the `z` slot."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from csd.features import extract_features, feature_vector
from csd.surprisal import ReferenceModel
from csd.detectors.perplexity_anomaly import Verdict


class DistributionalDetector:
    def __init__(self, model: ReferenceModel, C: float = 1.0, random_state: int = 0):
        self._model = model
        self._clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, random_state=random_state, max_iter=2000),
        )
        self._fitted = False

    def _vec(self, text: str) -> np.ndarray:
        surprisals, ranks = self._model.token_stats(text)
        return feature_vector(extract_features(surprisals, ranks))

    def fit(self, cover_texts: list[str], stego_texts: list[str]) -> None:
        x = [self._vec(t) for t in cover_texts] + [self._vec(t) for t in stego_texts]
        y = [0] * len(cover_texts) + [1] * len(stego_texts)
        self._clf.fit(np.array(x), np.array(y))
        self._fitted = True

    def score(self, text: str) -> Verdict:
        if not self._fitted:
            raise RuntimeError("detector not fitted; call fit() first")
        if not text.strip():
            return Verdict(hidden=False, z=0.0, reason="empty input", status="empty")
        p = float(self._clf.predict_proba(self._vec(text).reshape(1, -1))[0, 1])
        return Verdict(hidden=p >= 0.5, z=p,
                       reason=f"distributional p(stego)={p:.2f}")
