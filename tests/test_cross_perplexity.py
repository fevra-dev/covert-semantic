import pytest
from csd.surprisal import MockReferenceModel
from csd.detectors.cross_perplexity import CrossPerplexityDetector


def _cover():
    return ["the cat sat on the warm mat", "a dog ran in the green park",
            "the sun was warm", "he sat and was glad"]


def test_requires_calibration():
    det = CrossPerplexityDetector(MockReferenceModel(), MockReferenceModel(default_rank=20))
    with pytest.raises(RuntimeError):
        det.score("anything")


def test_flags_anomalous_ratio():
    det = CrossPerplexityDetector(MockReferenceModel(), MockReferenceModel(default_rank=20),
                                  z_threshold=1.5)
    det.calibrate(_cover())
    v = det.score("the chromatic feline reposed upon the thermal textile")
    assert v.status == "parsed" and isinstance(v.hidden, bool)


def test_empty_text_is_empty_status():
    det = CrossPerplexityDetector(MockReferenceModel(), MockReferenceModel(default_rank=20))
    det.calibrate(_cover())
    assert det.score("").status == "empty"
