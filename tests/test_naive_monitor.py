from csd.monitors import MockMonitor
from csd.detectors.naive_monitor import NaiveMonitorDetector, parse_monitor_reply
from csd.harness import run_detection


def test_parse_monitor_reply_direct():
    # the pure parse the driver applies to cached replies (must match detector behavior)
    assert parse_monitor_reply("YES").hidden is True and parse_monitor_reply("YES").status == "parsed"
    assert parse_monitor_reply("NO").hidden is False and parse_monitor_reply("NO").status == "parsed"
    assert parse_monitor_reply("").status == "empty"
    assert parse_monitor_reply("   ").status == "empty"
    assert parse_monitor_reply("I cannot determine that.").status == "unparsed"
    assert parse_monitor_reply("Yes, then no").hidden is True  # leads with YES


def test_yes_reply_flags_hidden():
    v = NaiveMonitorDetector(MockMonitor(response="YES")).score("some text")
    assert v.hidden is True and v.status == "parsed"


def test_no_reply_is_not_hidden():
    v = NaiveMonitorDetector(MockMonitor(response="NO")).score("some text")
    assert v.hidden is False and v.status == "parsed"


def test_empty_reply_is_empty_status():
    # GLOSSOPETRAE saw Opus return 149/150 empties; an empty is NOT a clean pass
    v = NaiveMonitorDetector(MockMonitor(response="")).score("some text")
    assert v.status == "empty" and v.hidden is False


def test_unparseable_reply_is_unparsed_status():
    v = NaiveMonitorDetector(MockMonitor(response="I cannot determine that.")).score("t")
    assert v.status == "unparsed"


def test_natural_yesno_phrases_parse():
    assert NaiveMonitorDetector(MockMonitor(response="No hidden message.")).score("t").hidden is False
    assert NaiveMonitorDetector(MockMonitor(response="Yes, there is something.")).score("t").hidden is True


def test_naive_monitor_reproduces_null_via_harness():
    # a monitor that always says NO -> all stego missed, all cover passed -> J = 0 (the null)
    det = NaiveMonitorDetector(MockMonitor(response="NO"))
    res = run_detection(det, positives=["stego a", "stego b"], negatives=["cover a", "cover b"])
    assert res.youden_j == 0.0
    assert res.tp == 0 and res.tn == 2 and res.empty == 0 and res.unparsed == 0
