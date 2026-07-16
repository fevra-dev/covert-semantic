import os
import tempfile

from csd.monitor_cache import (JudgmentRecord, append_judgment, load_judgments,
                               existing_keys, judgment_key)


def _rec(model, rid, reply="NO"):
    return JudgmentRecord(model=model, record_id=rid, raw=reply, ts="2026-06-28T00:00:00Z")


def test_append_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "j.jsonl")
        append_judgment(p, _rec("openai/gpt-5.5", "acrostic:stego:0:0", "YES"))
        append_judgment(p, _rec("anthropic/claude-opus-4.8", "acrostic:cover:0:0", "NO"))
        recs = load_judgments(p)
        assert len(recs) == 2
        assert recs[0].model == "openai/gpt-5.5" and recs[0].raw == "YES"


def test_existing_keys_supports_resume():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "j.jsonl")
        append_judgment(p, _rec("openai/gpt-5.5", "acrostic:stego:0:0", "YES"))
        keys = existing_keys(p)
        assert judgment_key("openai/gpt-5.5", "acrostic:stego:0:0") in keys
        # same record_id, different model -> not yet cached (per-model judgments)
        assert judgment_key("google/gemini-3.1-pro-preview", "acrostic:stego:0:0") not in keys


def test_existing_keys_missing_file_is_empty():
    assert existing_keys("/nonexistent/monitor_cache.jsonl") == set()
