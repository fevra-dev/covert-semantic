import pytest
from csd.surprisal import MockReferenceModel
from csd.detectors.distributional import DistributionalDetector
from csd.harness import run_detection


def _cover():
    return ["the cat sat on the warm mat", "a dog ran in the green park",
            "the sun was warm and the sky was blue", "he sat and was glad today",
            "the boy ate the ripe red plum", "she put the cup on the shelf"]


def _stego():
    # stilted, ornate -> rank/variance/spectral signature differs from cover
    return ["chromatic feline reposed upon thermal textile sublime",
            "obfuscatory lexicon bewildered subsequent interlocutors greatly",
            "luminescent perambulation traversed nocturnal boulevards quietly",
            "the azure report noted a vivid signal and a stark outcome",
            "grandiose elucidation precipitated unforeseen epistemic ramifications",
            "the umber report noted a keen signal and a noble outcome"]


def test_requires_fit_before_score():
    det = DistributionalDetector(MockReferenceModel())
    with pytest.raises(RuntimeError):
        det.score("anything")


def test_separates_cover_from_stego_after_fit():
    det = DistributionalDetector(MockReferenceModel())
    det.fit(cover_texts=_cover(), stego_texts=_stego())
    assert det.score(_stego()[0]).hidden is True
    assert det.score(_cover()[0]).hidden is False


def test_empty_text_is_empty_status():
    det = DistributionalDetector(MockReferenceModel())
    det.fit(cover_texts=_cover(), stego_texts=_stego())
    assert det.score("   ").status == "empty"


def test_harness_compatible_reports_j():
    det = DistributionalDetector(MockReferenceModel())
    det.fit(cover_texts=_cover(), stego_texts=_stego())
    res = run_detection(det, positives=_stego(), negatives=_cover())
    assert -1.0 <= res.youden_j <= 1.0
    assert res.empty == 0 and res.unparsed == 0
