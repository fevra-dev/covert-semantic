"""Versioned, append-only, resumable JSONL corpus store for the Phase-2 rigor pass.

The frontier generations are the only irreversible cost (design §3.1). Each record is
flushed as it is produced; a re-run skips ids already present (`existing_ids`), so a
mid-run crash or rate-limit never loses spend. Stage B (`analyze`) reads the corpus
offline and re-runnably."""
from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CorpusRecord:
    id: str
    construction: str
    label: str                      # "stego" | "cover"
    payload_code: int
    payload_fields: dict
    response: str
    model: str
    max_tokens: int
    temperature: float
    decoded_ok: bool | None         # stego: did decode_text recover the payload? cover: None
    ts: str


def record_id(construction: str, label: str, payload_code: int, idx: int) -> str:
    """Deterministic id for resume/dedup. idx distinguishes repeated payloads."""
    return f"{construction}:{label}:{idx}:{payload_code}"


def append_record(path: str, rec: CorpusRecord) -> None:
    """Append one record as a JSON line, flushed immediately (resumability)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec)) + "\n")
        f.flush()


def load_records(path: str) -> list[CorpusRecord]:
    if not os.path.exists(path):
        return []
    out: list[CorpusRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(CorpusRecord(**json.loads(line)))
    return out


def existing_ids(path: str) -> set[str]:
    """Ids already written — read only the `id` field so resume is cheap on large files."""
    if not os.path.exists(path):
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def by_construction(records: list[CorpusRecord],
                    label: str | None = None) -> dict[str, list[CorpusRecord]]:
    """Group records by construction, optionally filtering to one label."""
    out: dict[str, list[CorpusRecord]] = {}
    for r in records:
        if label is not None and r.label != label:
            continue
        out.setdefault(r.construction, []).append(r)
    return out
