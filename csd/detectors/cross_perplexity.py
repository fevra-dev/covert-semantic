"""Binoculars/2603.21567-lineage comparison baseline: ratio of mean surprisal under two
reference models, z-scored against a cover baseline (RQ2 names this as the proxy to beat)."""
from __future__ import annotations
import statistics
from csd.surprisal import ReferenceModel
from csd.detectors.perplexity_anomaly import Verdict


class CrossPerplexityDetector:
    def __init__(self, model_a: ReferenceModel, model_b: ReferenceModel, z_threshold: float = 1.5):
        self._a = model_a
        self._b = model_b
        self._z = z_threshold
        self._mean: float | None = None
        self._std: float | None = None

    def _ratio(self, text: str) -> float:
        mb = self._b.mean_surprisal(text)
        return self._a.mean_surprisal(text) / mb if mb else 0.0

    def calibrate(self, cover_texts: list[str]) -> None:
        ratios = [self._ratio(t) for t in cover_texts]
        self._mean = statistics.fmean(ratios)
        self._std = statistics.pstdev(ratios) or 1e-9

    def score(self, text: str) -> Verdict:
        if self._mean is None or self._std is None:
            raise RuntimeError("detector not calibrated; call calibrate() first")
        if not text.strip():
            return Verdict(hidden=False, z=0.0, reason="empty input", status="empty")
        z = (self._ratio(text) - self._mean) / self._std
        return Verdict(hidden=abs(z) > self._z, z=z,
                       reason=f"cross-perplexity ratio z={z:.2f}")
