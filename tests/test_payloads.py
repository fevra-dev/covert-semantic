import math
from csd.payloads import (Payload, TUPLE3_SCHEMA, TUPLE5_SCHEMA,
                          encode_payload, decode_payload, payload_bits, payload_chance)


def test_tuple3_bits_and_chance():
    assert math.isclose(payload_bits(TUPLE3_SCHEMA), 11.81, abs_tol=0.01)
    assert math.isclose(payload_chance(TUPLE3_SCHEMA), 1 / 3600, rel_tol=1e-6)


def test_tuple5_bits_and_chance():
    assert math.isclose(payload_bits(TUPLE5_SCHEMA), 17.40, abs_tol=0.01)
    assert math.isclose(payload_chance(TUPLE5_SCHEMA), 1 / 172800, rel_tol=1e-6)


def test_roundtrip_tuple3():
    p = Payload(schema=TUPLE3_SCHEMA, fields={"color": "red", "number": 42, "shape": "square"})
    code = encode_payload(p)
    assert isinstance(code, int) and 0 <= code < 3600
    assert decode_payload(code, TUPLE3_SCHEMA).fields == p.fields


def test_invalid_field_rejected():
    import pytest
    with pytest.raises(ValueError):
        encode_payload(Payload(schema=TUPLE3_SCHEMA,
                               fields={"color": "octarine", "number": 0, "shape": "square"}))
