from csd.detectors.perplexity_anomaly import Verdict
from csd.evaluation import (ScoredItem, collect_scores, best_j_threshold, metrics_at,
                            auc_of, kfold_indices, eval_in_distribution,
                            eval_leave_one_construction_out, crossfit_base_scores)


class _Stub:
    """Detector whose continuous score is encoded in the text (for harness-logic tests)."""
    def __init__(self, offset=0.0):
        self.threshold = 0.5
        self._offset = offset

    def score(self, text):
        z = float(text) + self._offset
        return Verdict(hidden=z >= self.threshold, z=z, reason="stub")


def _sep(n_each=10):
    """Linearly separable: negatives near 0.1, positives near 0.9."""
    texts = [f"{0.1 + 0.01*i:.4f}" for i in range(n_each)] + \
            [f"{0.9 - 0.01*i:.4f}" for i in range(n_each)]
    labels = [0]*n_each + [1]*n_each
    return texts, labels


def test_collect_scores_records_score_label_status():
    items = collect_scores(_Stub(), ["0.1", "0.9"], [0, 1])
    assert [round(i.score, 1) for i in items] == [0.1, 0.9]
    assert [i.label for i in items] == [0, 1]
    assert all(i.status == "parsed" for i in items)


def test_best_j_threshold_separable():
    items = [ScoredItem(0.1, 0, "parsed"), ScoredItem(0.2, 0, "parsed"),
             ScoredItem(0.8, 1, "parsed"), ScoredItem(0.9, 1, "parsed")]
    thr, j = best_j_threshold(items)
    assert j == 1.0
    assert 0.2 < thr <= 0.8


def test_metrics_at_confusion():
    items = [ScoredItem(0.1, 0, "parsed"), ScoredItem(0.4, 0, "parsed"),
             ScoredItem(0.6, 1, "parsed"), ScoredItem(0.9, 1, "parsed")]
    m = metrics_at(items, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 0, 0, 2)
    assert m["j"] == 1.0 and m["fpr"] == 0.0


def test_auc_of_matches_separable():
    items = [ScoredItem(0.1, 0, "parsed"), ScoredItem(0.2, 0, "parsed"),
             ScoredItem(0.8, 1, "parsed"), ScoredItem(0.9, 1, "parsed")]
    assert auc_of(items) == 1.0


def test_kfold_indices_stratified_partition():
    labels = [0]*5 + [1]*5
    folds = kfold_indices(10, k=5, seed=0, labels=labels)
    assert len(folds) == 5
    all_test = []
    for train, test in folds:
        assert sorted(train + test) == list(range(10))     # partition
        assert set(train).isdisjoint(test)                  # disjoint
        assert {labels[i] for i in test} == {0, 1}          # stratified: both classes present
        all_test += test
    assert sorted(all_test) == list(range(10))              # every item tested once


def test_eval_in_distribution_separable_is_near_perfect():
    texts, labels = _sep(10)
    res = eval_in_distribution(lambda: _Stub(), lambda d, tx, ly: None, texts, labels, k=5)
    assert res["auc"] == 1.0
    assert res["j"] == 1.0
    assert len(res["oof_scores"]) == len(texts)            # one OOF score per item (for bootstrap)
    assert res["labels"] == labels


def test_eval_leave_one_construction_out_one_cell_each():
    a, b, c = _sep(6), _sep(6), _sep(6)
    by_constr = {"a": a, "b": b, "c": c}
    res = eval_leave_one_construction_out(lambda: _Stub(), lambda d, tx, ly: None, by_constr)
    assert set(res) == {"a", "b", "c"}
    for cell in res.values():
        assert {"j", "auc", "fpr", "oof_scores", "labels"} <= set(cell)
        assert cell["auc"] == 1.0


def test_crossfit_base_scores_shape_and_no_leakage():
    a, b = _sep(6), _sep(6)
    by_constr = {"a": a, "b": b}

    def make_bases(train_texts, train_labels):
        return (_Stub(0.0), _Stub(0.1), _Stub(-0.1))  # three "base" detectors

    rows, labels = crossfit_base_scores(make_bases, by_constr, k=4)
    assert len(rows) == 24 and len(labels) == 24
    assert all(len(r) == 3 for r in rows)                 # [ppl_z, xppl_z, dist_p]
