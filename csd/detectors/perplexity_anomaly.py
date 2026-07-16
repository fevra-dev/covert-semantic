"""Layer-2 baseline: flag text whose mean surprisal is anomalously high vs a cover
baseline ('anomalously stilted', the only direction GLOSSOPETRAE saw fire)."""
from __future__ import annotations
import statistics
from dataclasses import dataclass
from csd.surprisal import ReferenceModel


@dataclass(frozen=True)
class Verdict:
    hidden: bool
    z: float
    reason: str
    status: str = "parsed"   # parsed | empty | unparsed  (trichotomy, design §5)


class PerplexityAnomalyDetector:
    def __init__(self, model: ReferenceModel, z_threshold: float = 1.5):
        self._model = model
        self._z = z_threshold
        self._mean: float | None = None
        self._std: float | None = None

    def calibrate(self, cover_texts: list[str]) -> None:
        means = [self._model.mean_surprisal(t) for t in cover_texts]
        self._mean = statistics.fmean(means)
        self._std = statistics.pstdev(means) or 1e-9

    def score(self, text: str) -> Verdict:
        if self._mean is None or self._std is None:
            raise RuntimeError("detector not calibrated; call calibrate() first")
        if not text.strip():
            return Verdict(hidden=False, z=0.0, reason="empty input", status="empty")
        z = (self._model.mean_surprisal(text) - self._mean) / self._std
        hidden = z > self._z
        return Verdict(hidden=hidden, z=z,
                       reason=f"mean-surprisal z={z:.2f} vs cover baseline "
                              f"({'stilted' if hidden else 'within-baseline'})")
