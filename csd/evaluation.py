"""Out-of-sample evaluation harness (design §6) — the n=20 pilot's #1 honesty fix.

Every number here is produced by a detector that never saw the item it scores:
- in-distribution: stratified k-fold CV, threshold chosen on train folds, scored on the
  held-out fold, predictions pooled out-of-fold (OOF);
- generalization: leave-one-construction-out, calibrated on the training constructions;
- stacker training rows: cross-fitted OOF base scores (no leakage into the meta-classifier).

Detectors are driven through two caller-supplied callables so the harness stays agnostic to
how each detector trains:
  make_detector() -> fresh detector exposing .score(text) -> Verdict
  calibrate(detector, train_texts, train_labels) -> None   (calibrate/fit appropriately)
The harness operates on the continuous Verdict.z and applies its own OOS-chosen threshold,
bypassing each detector's internal fixed threshold."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from csd.stats import auc as _auc


@dataclass
class ScoredItem:
    score: float
    label: int
    status: str  # parsed | empty | unparsed


def collect_scores(detector, texts: list[str], labels: list[int]) -> list[ScoredItem]:
    items = []
    for text, label in zip(texts, labels):
        v = detector.score(text)
        items.append(ScoredItem(score=float(v.z), label=int(label), status=v.status))
    return items


def _parsed(items: list[ScoredItem]) -> list[ScoredItem]:
    return [i for i in items if i.status == "parsed"]


def best_j_threshold(items: list[ScoredItem]) -> tuple[float, float]:
    """Threshold (predict score >= t) maximizing Youden's J. Candidate thresholds are the
    *midpoints* between adjacent unique scores (i.e. placed in the margin) so the chosen
    operating point generalizes to unseen items rather than sitting on a training value."""
    p = _parsed(items)
    pos = [i.score for i in p if i.label == 1]
    neg = [i.score for i in p if i.label == 0]
    if not pos or not neg:
        return (float("inf"), 0.0)
    scores = np.array([i.score for i in p])
    labels = np.array([i.label for i in p])
    n_pos, n_neg = len(pos), len(neg)
    uniq = np.unique(scores)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = [uniq[-1] + 1.0, *mids.tolist()]  # +1 => predict none (J=0 floor)
    best_t, best_j = float(uniq[-1] + 1.0), 0.0
    for t in candidates:
        pred = scores >= t
        tpr = float(np.sum(pred & (labels == 1))) / n_pos
        fpr = float(np.sum(pred & (labels == 0))) / n_neg
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return (best_t, best_j)


def metrics_at(items: list[ScoredItem], threshold: float) -> dict:
    p = _parsed(items)
    tp = fn = fp = tn = 0
    for i in p:
        pred = i.score >= threshold
        if i.label == 1:
            tp += pred
            fn += not pred
        else:
            fp += pred
            tn += not pred
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    empty = sum(i.status == "empty" for i in items)
    unparsed = sum(i.status == "unparsed" for i in items)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "tpr": tpr, "fpr": fpr,
            "j": tpr - fpr, "empty": empty, "unparsed": unparsed, "n": len(items)}


def auc_of(items: list[ScoredItem]) -> float:
    p = _parsed(items)
    return _auc([i.score for i in p], [i.label for i in p])


def kfold_indices(n: int, k: int = 5, seed: int = 0,
                  labels: list[int] | None = None) -> list[tuple[list[int], list[int]]]:
    """Stratified k-fold (round-robin per class) when labels given, else plain shuffled."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if labels is None:
        rng.shuffle(idx)
        fold_members = [list(f) for f in np.array_split(idx, k)]
    else:
        y = np.asarray(labels)
        fold_members = [[] for _ in range(k)]
        for cls in np.unique(y):
            cls_idx = idx[y == cls].copy()
            rng.shuffle(cls_idx)
            for j, ix in enumerate(cls_idx):
                fold_members[j % k].append(int(ix))
    out = []
    for i in range(k):
        test = sorted(fold_members[i])
        test_set = set(test)
        train = [j for j in range(n) if j not in test_set]
        out.append((train, test))
    return out


def _subset(texts, labels, indices):
    return [texts[i] for i in indices], [labels[i] for i in indices]


def eval_in_distribution(make_detector, calibrate, texts: list[str], labels: list[int],
                         k: int = 5) -> dict:
    """Stratified k-fold CV with OOS threshold selection; predictions pooled out-of-fold."""
    n = len(texts)
    oof_scores: list[float | None] = [None] * n
    oof_status: list[str] = ["parsed"] * n
    oof_pred: list[int | None] = [None] * n
    for train_idx, test_idx in kfold_indices(n, k=k, labels=labels):
        train_texts, train_labels = _subset(texts, labels, train_idx)
        det = make_detector()
        calibrate(det, train_texts, train_labels)
        thr, _ = best_j_threshold(collect_scores(det, train_texts, train_labels))
        test_texts, test_labels = _subset(texts, labels, test_idx)
        for local, gi in enumerate(test_idx):
            it = collect_scores(det, [test_texts[local]], [test_labels[local]])[0]
            oof_scores[gi] = it.score
            oof_status[gi] = it.status
            oof_pred[gi] = int(it.score >= thr) if it.status == "parsed" else None
    pooled = [ScoredItem(s if s is not None else 0.0, lab, st)
              for s, lab, st in zip(oof_scores, labels, oof_status)]
    # pooled J at per-fold thresholds (operating-point), pooled AUC (threshold-free)
    tp = fn = fp = tn = 0
    for pred, lab, st in zip(oof_pred, labels, oof_status):
        if st != "parsed" or pred is None:
            continue
        if lab == 1:
            tp += pred
            fn += not pred
        else:
            fp += pred
            tn += not pred
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"auc": auc_of(pooled), "j": tpr - fpr, "fpr": fpr, "tpr": tpr,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "empty": oof_status.count("empty"), "unparsed": oof_status.count("unparsed"),
            "n": n, "oof_scores": [float(s) if s is not None else 0.0 for s in oof_scores],
            "labels": list(labels)}


def eval_leave_one_construction_out(make_detector, calibrate,
                                    by_constr: dict[str, tuple[list[str], list[int]]]) -> dict:
    """Train on all-but-one construction, evaluate on the held-out construction."""
    constructions = list(by_constr)
    out: dict[str, dict] = {}
    for held in constructions:
        train_texts: list[str] = []
        train_labels: list[int] = []
        for c in constructions:
            if c == held:
                continue
            tx, ly = by_constr[c]
            train_texts += tx
            train_labels += ly
        det = make_detector()
        calibrate(det, train_texts, train_labels)
        thr, _ = best_j_threshold(collect_scores(det, train_texts, train_labels))
        test_texts, test_labels = by_constr[held]
        items = collect_scores(det, test_texts, test_labels)
        cell = metrics_at(items, thr)
        cell["auc"] = auc_of(items)
        cell["oof_scores"] = [float(i.score) for i in items]
        cell["labels"] = list(test_labels)
        out[held] = cell
    return out


def crossfit_base_scores(make_bases, by_constr: dict[str, tuple[list[str], list[int]]],
                         k: int = 5) -> tuple[list[list[float]], list[int]]:
    """Pool all constructions, k-fold CV, fit bases on train, emit OOF rows
    [ppl_z, xppl_z, dist_p] for every item — leakage-free training data for the stacker."""
    texts: list[str] = []
    labels: list[int] = []
    for tx, ly in by_constr.values():
        texts += tx
        labels += ly
    n = len(texts)
    rows: list[list[float] | None] = [None] * n
    for train_idx, test_idx in kfold_indices(n, k=k, labels=labels):
        train_texts, train_labels = _subset(texts, labels, train_idx)
        ppl, xpl, dist = make_bases(train_texts, train_labels)
        for gi in test_idx:
            t = texts[gi]
            rows[gi] = [float(ppl.score(t).z), float(xpl.score(t).z), float(dist.score(t).z)]
    return [r for r in rows if r is not None], labels
