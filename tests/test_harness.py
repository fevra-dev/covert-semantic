from csd.surprisal import MockReferenceModel
from csd.detectors.perplexity_anomaly import PerplexityAnomalyDetector
from csd.harness import run_detection


def test_run_detection_reports_j_and_trichotomy():
    det = PerplexityAnomalyDetector(MockReferenceModel(), z_threshold=1.0)
    det.calibrate(["the cat sat on the warm mat", "a dog ran in the green park",
                   "the sun was warm", "he sat and was glad"])
    positives = ["the chromatic feline reposed upon the thermal textile",
                 "the obfuscatory lexicon bewildered subsequent interlocutors"]
    negatives = ["the cat sat on the mat", "a dog ran in the park"]
    res = run_detection(det, positives, negatives)
    assert res.tp + res.fn == len(positives)
    assert res.fp + res.tn == len(negatives)
    assert -1.0 <= res.youden_j <= 1.0
    # empties tracked separately, never folded into "clean"
    assert res.empty == 0 and res.unparsed == 0


def test_empty_replies_counted_separately():
    det = PerplexityAnomalyDetector(MockReferenceModel(), z_threshold=1.0)
    det.calibrate(["the cat sat on the warm mat", "he sat and was glad"])
    res = run_detection(det, positives=["", "the verbose periphrasis"], negatives=["the cat sat"])
    assert res.empty == 1  # the "" positive is an empty, not a miss
