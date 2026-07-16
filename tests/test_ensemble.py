import math

from csd.constructions import Construction, encode_text, make_cover
from csd.payloads import TUPLE3_SCHEMA, decode_payload
from csd.surprisal import MockReferenceModel
from csd.detectors.perplexity_anomaly import PerplexityAnomalyDetector
from csd.detectors.cross_perplexity import CrossPerplexityDetector
from csd.detectors.distributional import DistributionalDetector
from csd.detectors.ensemble import MaxFusionEnsemble, StackedEnsemble


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _fitted_bases():
    ref_a, ref_b = MockReferenceModel(), MockReferenceModel(default_rank=20)
    covers = [make_cover(i) for i in range(8)]
    stegos = [encode_text(decode_payload(i, TUPLE3_SCHEMA), Construction.SYNONYM)
              for i in range(8)]
    ppl = PerplexityAnomalyDetector(ref_a)
    ppl.calibrate(covers)
    xpl = CrossPerplexityDetector(ref_a, ref_b)
    xpl.calibrate(covers)
    dist = DistributionalDetector(ref_a)
    dist.fit(cover_texts=covers, stego_texts=stegos)
    return ppl, xpl, dist, covers, stegos


def test_max_fusion_is_max_of_normalized_components():
    ppl, _xpl, dist, _c, stegos = _fitted_bases()
    ens = MaxFusionEnsemble(ppl, dist)
    t = stegos[0]
    v = ens.score(t)
    assert 0.0 <= v.z <= 1.0
    expected = max(_phi(ppl.score(t).z), dist.score(t).z)
    assert abs(v.z - expected) < 1e-9


def test_max_fusion_threshold_controls_hidden():
    ppl, _xpl, dist, _c, stegos = _fitted_bases()
    ens = MaxFusionEnsemble(ppl, dist)
    t = stegos[0]
    ens.threshold = 0.0
    assert ens.score(t).hidden is True
    ens.threshold = 1.01
    assert ens.score(t).hidden is False


def test_max_fusion_empty_input_propagates():
    ppl, _xpl, dist, _c, _s = _fitted_bases()
    v = MaxFusionEnsemble(ppl, dist).score("   ")
    assert v.status == "empty" and v.hidden is False


def test_stacked_separates_linearly_separable_rows():
    ppl, xpl, dist, _c, _s = _fitted_bases()
    st = StackedEnsemble(ppl, xpl, dist)
    rows = [[3.0, 2.0, 0.92], [2.6, 1.6, 0.88], [2.9, 1.9, 0.95], [2.4, 1.4, 0.90],
            [-1.0, -0.6, 0.08], [-0.8, -0.4, 0.12], [-1.2, -0.7, 0.05], [-0.9, -0.5, 0.10]]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    st.fit(rows, labels)
    pos = st.score_from_row([3.1, 2.1, 0.96])
    neg = st.score_from_row([-1.1, -0.7, 0.04])
    assert pos.hidden is True and pos.z > 0.5
    assert neg.hidden is False and neg.z < 0.5


def test_stacked_score_from_text_returns_unit_interval():
    ppl, xpl, dist, _c, stegos = _fitted_bases()
    st = StackedEnsemble(ppl, xpl, dist)
    rows = [[3.0, 2.0, 0.92], [2.6, 1.6, 0.88], [-1.0, -0.6, 0.08], [-0.8, -0.4, 0.12]]
    st.fit(rows, [1, 1, 0, 0])
    v = st.score(stegos[0])
    assert 0.0 <= v.z <= 1.0 and v.status == "parsed"


def _fresh_bases():
    from csd.surprisal import MockReferenceModel
    ref_a, ref_b = MockReferenceModel(), MockReferenceModel(default_rank=20)
    return (PerplexityAnomalyDetector(ref_a), CrossPerplexityDetector(ref_a, ref_b),
            DistributionalDetector(ref_a))


def test_max_fusion_calibrate_fits_bases_end_to_end():
    ppl, _xpl, dist = _fresh_bases()                 # uncalibrated/unfitted
    covers = [make_cover(i) for i in range(8)]
    stegos = [encode_text(decode_payload(i, TUPLE3_SCHEMA), Construction.SYNONYM)
              for i in range(8)]
    ens = MaxFusionEnsemble(ppl, dist)
    ens.calibrate(covers, stegos)                    # should calibrate ppl + fit dist
    v = ens.score(stegos[0])
    assert 0.0 <= v.z <= 1.0 and v.status == "parsed"


def test_stacked_calibrate_fits_bases_and_meta():
    ppl, xpl, dist = _fresh_bases()
    covers = [make_cover(i) for i in range(8)]
    stegos = [encode_text(decode_payload(i, TUPLE3_SCHEMA), Construction.SYNONYM)
              for i in range(8)]
    st = StackedEnsemble(ppl, xpl, dist)
    st.calibrate(covers, stegos)
    v = st.score(stegos[0])
    assert 0.0 <= v.z <= 1.0 and v.status == "parsed"
