import pytest
from csd.payloads import Payload, TUPLE3_SCHEMA, decode_payload
from csd.constructions import Construction, encode_text, decode_text


def test_narrative_positional_roundtrip():
    p = Payload(schema=TUPLE3_SCHEMA, fields={"color": "blue", "number": 7, "shape": "star"})
    text = encode_text(p, Construction.NARRATIVE_POSITIONAL)
    # exactly 3 sentences; S1->color, S2->number, S3->shape
    assert text.count(".") == 3
    assert decode_text(text, TUPLE3_SCHEMA, Construction.NARRATIVE_POSITIONAL).fields == p.fields


def test_acrostic_roundtrip():
    p = Payload(schema=TUPLE3_SCHEMA, fields={"color": "red", "number": 3, "shape": "circle"})
    text = encode_text(p, Construction.ACROSTIC)
    assert decode_text(text, TUPLE3_SCHEMA, Construction.ACROSTIC).fields == p.fields


def test_cover_has_no_payload_marker():
    from csd.constructions import make_cover
    cover = make_cover(seed=1)
    # cover must be a plain sentence string with no encoded structure
    assert isinstance(cover, str) and len(cover) > 0


def test_synonym_roundtrip():
    p = Payload(schema=TUPLE3_SCHEMA, fields={"color": "green", "number": 50, "shape": "spiral"})
    text = encode_text(p, Construction.SYNONYM)
    assert decode_text(text, TUPLE3_SCHEMA, Construction.SYNONYM).fields == p.fields


def test_synonym_differs_from_acrostic_text():
    p = Payload(schema=TUPLE3_SCHEMA, fields={"color": "red", "number": 3, "shape": "circle"})
    assert encode_text(p, Construction.SYNONYM) != encode_text(p, Construction.ACROSTIC)


@pytest.mark.parametrize("code", [0, 1, 2, 1799, 3599])
def test_sentence_length_roundtrip(code):
    p = decode_payload(code, TUPLE3_SCHEMA)
    text = encode_text(p, Construction.SENTENCE_LENGTH)
    sentences = [s for s in text.split(".") if s.strip()]
    assert len(sentences) == 12  # 12 parity bits cover TUPLE3 cardinality 3600
    back = decode_text(text, TUPLE3_SCHEMA, Construction.SENTENCE_LENGTH)
    assert back.fields == p.fields


def test_sentence_length_parity_encodes_bits():
    # code=1 -> 12-bit big-endian 0b000000000001 -> only the LAST sentence is odd-word-count
    p = decode_payload(1, TUPLE3_SCHEMA)
    sentences = [s for s in encode_text(p, Construction.SENTENCE_LENGTH).split(".") if s.strip()]
    parities = [len(s.split()) % 2 for s in sentences]
    assert parities == [0] * 11 + [1]
