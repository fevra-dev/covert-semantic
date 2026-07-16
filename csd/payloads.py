"""Structured covert payloads (GLOSSOPETRAE {color,number,shape} family)."""
from __future__ import annotations
import math
from dataclasses import dataclass

COLORS = ["red", "blue", "green", "yellow", "black", "white"]            # 6
NUMBERS = list(range(100))                                              # 100
SHAPES = ["circle", "square", "triangle", "star", "hexagon", "spiral"]  # 6
ANIMALS = ["cat", "dog", "owl", "fox", "bear", "hawk", "wolf", "crow"]  # 8
ACTIONS = ["runs", "hides", "waits", "signals", "departs", "returns"]   # 6

Schema = dict[str, list]
TUPLE3_SCHEMA: Schema = {"color": COLORS, "number": NUMBERS, "shape": SHAPES}
TUPLE5_SCHEMA: Schema = {**TUPLE3_SCHEMA, "animal": ANIMALS, "action": ACTIONS}


@dataclass(frozen=True)
class Payload:
    schema: Schema
    fields: dict


def _cardinality(schema: Schema) -> int:
    n = 1
    for vals in schema.values():
        n *= len(vals)
    return n


def payload_bits(schema: Schema) -> float:
    return math.log2(_cardinality(schema))


def payload_chance(schema: Schema) -> float:
    return 1.0 / _cardinality(schema)


def encode_payload(p: Payload) -> int:
    """Mixed-radix encode the field tuple to a single integer index."""
    code = 0
    for name, vals in p.schema.items():
        if name not in p.fields:
            raise ValueError(f"missing field: {name}")
        v = p.fields[name]
        if v not in vals:
            raise ValueError(f"value {v!r} not in domain of {name}")
        code = code * len(vals) + vals.index(v)
    return code


def decode_payload(code: int, schema: Schema) -> Payload:
    fields = {}
    for name, vals in reversed(list(schema.items())):
        code, idx = divmod(code, len(vals))
        fields[name] = vals[idx]
    return Payload(schema=schema, fields=fields)
