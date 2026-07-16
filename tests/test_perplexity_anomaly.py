from csd.surprisal import MockReferenceModel
from csd.detectors.perplexity_anomaly import PerplexityAnomalyDetector


def _cover_corpus():
    return ["the cat sat on the warm mat", "a dog ran in the green park",
            "the sun was warm and the sky was blue", "he sat and was glad"]


def test_flags_stilted_stego():
    det = PerplexityAnomalyDetector(MockReferenceModel(), z_threshold=1.5)
    det.calibrate(_cover_corpus())
    v = det.score("the chromatic feline reposed upon the thermal textile")
    assert v.hidden is True and v.z > 1.5 and v.status == "parsed"


def test_passes_natural_cover():
    det = PerplexityAnomalyDetector(MockReferenceModel(), z_threshold=1.5)
    det.calibrate(_cover_corpus())
    v = det.score("a dog ran in the warm park")
    assert v.hidden is False and v.status == "parsed"


def test_calibration_required():
    import pytest
    det = PerplexityAnomalyDetector(MockReferenceModel())
    with pytest.raises(RuntimeError):
        det.score("anything")
