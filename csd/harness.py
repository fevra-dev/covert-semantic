"""Run a detector over labelled positives/negatives; report J + CI + the
silent-failure trichotomy (parsed / empty / unparsed) per design §5."""
from __future__ import annotations
from dataclasses import dataclass
from csd.stats import youden_j_ci
from csd.detectors.perplexity_anomaly import Verdict


@dataclass
class DetectionResult:
    tp: int
    fn: int
    fp: int
    tn: int
    empty: int
    unparsed: int
    youden_j: float
    j_lo: float
    j_hi: float


def run_detection(detector, positives: list[str], negatives: list[str]) -> DetectionResult:
    tp = fn = fp = tn = empty = unparsed = 0
    for text in positives:
        v: Verdict = detector.score(text)
        if v.status == "empty":
            empty += 1
            continue
        if v.status == "unparsed":
            unparsed += 1
            continue
        if v.hidden:
            tp += 1
        else:
            fn += 1
    for text in negatives:
        v = detector.score(text)
        if v.status == "empty":
            empty += 1
            continue
        if v.status == "unparsed":
            unparsed += 1
            continue
        if v.hidden:
            fp += 1
        else:
            tn += 1
    j, lo, hi = youden_j_ci(tp, fn, fp, tn)
    return DetectionResult(tp, fn, fp, tn, empty, unparsed, j, lo, hi)
