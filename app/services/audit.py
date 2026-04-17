from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "data" / "structured" / "audit_logs.jsonl"


def generate_trace_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    short = uuid4().hex[:8]
    return f"{prefix}_{ts}_{short}"


def append_audit_log(event_type: str, payload: dict, trace_id: str | None = None) -> dict:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trace_id": trace_id or generate_trace_id(event_type),
        "event_type": event_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "payload": payload,
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def list_audit_logs(limit: int = 20) -> list[dict]:
    if not AUDIT_PATH.exists():
        return []

    records = []
    with AUDIT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    records.reverse()
    return records[: max(1, min(limit, 100))]
