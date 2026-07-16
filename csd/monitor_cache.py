"""Resumable cache of naive-monitor judgments (Phase 1.5).

Monitor calls cost money and a panel × corpus is many; each raw YES/NO reply is cached
by (model, record_id) and flushed immediately, so a crash — or a later panel / informed
expansion — re-spends nothing. Parsing the cached reply is done downstream by
csd.detectors.naive_monitor.parse_monitor_reply (pure)."""
from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class JudgmentRecord:
    model: str
    record_id: str        # the CorpusRecord id the text came from
    raw: str              # the monitor's raw reply (parsed downstream)
    ts: str


def judgment_key(model: str, record_id: str) -> str:
    return f"{model}\t{record_id}"


def append_judgment(path: str, rec: JudgmentRecord) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec)) + "\n")
        f.flush()


def load_judgments(path: str) -> list[JudgmentRecord]:
    if not os.path.exists(path):
        return []
    out: list[JudgmentRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(JudgmentRecord(**json.loads(line)))
    return out


def existing_keys(path: str) -> set[str]:
    """(model, record_id) pairs already judged — cheap resume on large caches."""
    if not os.path.exists(path):
        return set()
    keys: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                keys.add(judgment_key(d["model"], d["record_id"]))
    return keys
