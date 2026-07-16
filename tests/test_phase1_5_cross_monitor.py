from csd.corpus import CorpusRecord, append_record, record_id
from csd.monitors import Monitor
from experiments.phase1_5_cross_monitor import run_phase1_5


class _TextMonitor(Monitor):
    """Label-unaware (sees only text); 'self' is blind, 'good' fires on the marker."""
    def __init__(self, kind):
        self.kind = kind

    def judge(self, text):
        if self.kind == "self":
            return "NO"
        return "YES" if "XPAYLOAD" in text else "NO"


def _tiny_corpus(path, n=5):
    for con in ["acrostic", "synonym"]:
        for i in range(n):
            append_record(path, CorpusRecord(
                id=record_id(con, "stego", i, i), construction=con, label="stego",
                payload_code=i, payload_fields={}, response=f"hidden XPAYLOAD text {i}",
                model="mock", max_tokens=16, temperature=0.0, decoded_ok=True, ts="t"))
            append_record(path, CorpusRecord(
                id=record_id(con, "cover", i, i), construction=con, label="cover",
                payload_code=i, payload_fields={}, response=f"plain ordinary text {i}",
                model="mock", max_tokens=16, temperature=0.0, decoded_ok=None, ts="t"))


def test_phase1_5_e2e_structure_and_values(tmp_path):
    corpus = str(tmp_path / "c.jsonl")
    cache = str(tmp_path / "cache.jsonl")
    _tiny_corpus(corpus, n=5)
    models = ["self", "good"]
    res = run_phase1_5(corpus, cache, models, self_model="self",
                       monitor_factory=lambda m: _TextMonitor(m))
    # structure
    assert set(res["per_model"]) == {"self", "good"}
    assert {"verdict", "holm", "contrasts", "self_model"} <= set(res)
    # self is the null; the marker-aware monitor is perfect
    assert res["per_model"]["self"]["acrostic"]["youden_j"] == 0.0
    assert res["per_model"]["good"]["acrostic"]["youden_j"] == 1.0
    assert res["per_model"]["good"]["pooled"]["youden_j"] == 1.0
    # a Holm-corrected contrast of good-vs-self should reject somewhere
    assert any(c["candidate"] == "good" and c["reject"] for c in res["contrasts"])


def test_phase1_5_judging_is_resumable(tmp_path):
    corpus = str(tmp_path / "c.jsonl")
    cache = str(tmp_path / "cache.jsonl")
    _tiny_corpus(corpus, n=4)
    models = ["self", "good"]
    r1 = run_phase1_5(corpus, cache, models, self_model="self",
                      monitor_factory=lambda m: _TextMonitor(m))
    r2 = run_phase1_5(corpus, cache, models, self_model="self",
                      monitor_factory=lambda m: _TextMonitor(m))
    assert r1["judged"] == 4 * 2 * 2 * 2       # n=4 x 2 constructions x {stego,cover} x 2 models
    assert r2["judged"] == 0                    # all cached -> no re-judging


def test_fully_cached_rerun_builds_no_monitor(tmp_path):
    corpus = str(tmp_path / "c.jsonl")
    cache = str(tmp_path / "cache.jsonl")
    _tiny_corpus(corpus, n=3)
    models = ["self", "good"]
    run_phase1_5(corpus, cache, models, self_model="self",
                 monitor_factory=lambda m: _TextMonitor(m))

    def boom(_m):  # a fully-cached re-analysis must not construct a (live) monitor
        raise AssertionError("monitor_factory called despite a full cache")

    r2 = run_phase1_5(corpus, cache, models, self_model="self", monitor_factory=boom)
    assert r2["judged"] == 0 and "per_model" in r2
