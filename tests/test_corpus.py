import os
import tempfile

from csd.corpus import (CorpusRecord, append_record, load_records, existing_ids,
                        record_id, by_construction)


def _rec(i, construction="acrostic", label="stego"):
    return CorpusRecord(
        id=record_id(construction, label, i, i), construction=construction, label=label,
        payload_code=i, payload_fields={"color": "red", "number": 0, "shape": "square"},
        response="A short sentence.", model="m", max_tokens=256, temperature=0.7,
        decoded_ok=True, ts="2026-06-28T00:00:00Z")


def test_append_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.jsonl")
        append_record(p, _rec(0))
        append_record(p, _rec(1))
        recs = load_records(p)
        assert [r.payload_code for r in recs] == [0, 1]
        assert recs[0].construction == "acrostic"
        assert recs[0].decoded_ok is True
        assert recs[1].payload_fields["shape"] == "square"


def test_existing_ids_supports_resume():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.jsonl")
        append_record(p, _rec(0))
        ids = existing_ids(p)
        assert record_id("acrostic", "stego", 0, 0) in ids
        assert record_id("acrostic", "stego", 9, 9) not in ids


def test_existing_ids_missing_file_is_empty():
    assert existing_ids("/nonexistent/path/corpus.jsonl") == set()


def test_by_construction_groups_and_filters_label():
    recs = [_rec(0, "acrostic", "stego"), _rec(1, "acrostic", "cover"),
            _rec(2, "synonym", "stego")]
    grouped = by_construction(recs)
    assert set(grouped) == {"acrostic", "synonym"}
    assert len(grouped["acrostic"]) == 2
    stego_only = by_construction(recs, label="stego")
    assert len(stego_only["acrostic"]) == 1 and stego_only["acrostic"][0].label == "stego"
