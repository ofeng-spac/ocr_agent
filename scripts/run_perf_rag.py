#!/usr/bin/env python3
"""
Run a dedicated performance test for /api/rag/ask.

Usage:
    python3 scripts/run_perf_rag.py
"""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "eval" / "perf_rag"
SUMMARY_PATH = OUTPUT_DIR / "rag_perf_summary.json"
REPORT_PATH = OUTPUT_DIR / "rag_perf_report.md"
HOST = "http://127.0.0.1:8091"


def wait_http(url: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):  # noqa: S310
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"timeout waiting for {url}")


def warmup() -> None:
    payloads = [
        {"canonical_name": "双黄连口服液", "question": "规格是什么"},
        {"canonical_name": "贝伐珠单抗注射液", "question": "这个药主要用于哪些场景"},
    ]
    for payload in payloads:
        req = urllib.request.Request(  # noqa: S310
            f"{HOST}/api/rag/ask",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15):  # noqa: S310
            pass


def parse_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def parse_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def load_stats_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict], name: str) -> dict | None:
    for row in rows:
        if row.get("Name") == name:
            return row
    return None


def summarize_row(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {
        "name": row.get("Name"),
        "request_count": parse_int(row.get("Request Count", "0")),
        "failure_count": parse_int(row.get("Failure Count", "0")),
        "avg_response_ms": parse_float(row.get("Average Response Time", "0")),
        "median_response_ms": parse_float(row.get("Median Response Time", "0")),
        "min_response_ms": parse_float(row.get("Min Response Time", "0")),
        "max_response_ms": parse_float(row.get("Max Response Time", "0")),
        "rps": parse_float(row.get("Requests/s", "0")),
        "p95_ms": parse_float(row.get("95%", "0")),
        "p99_ms": parse_float(row.get("99%", "0")),
    }


def build_report(summary: dict) -> str:
    lines = [
        "# RAG Performance Report",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Target host: {summary['host']}",
        f"- Users: {summary['users']}",
        f"- Spawn rate: {summary['spawn_rate']}",
        f"- Duration: {summary['run_time']}",
        "",
        "## Aggregated",
        "",
    ]

    agg = summary["aggregated"]
    lines.extend(
        [
            f"- Request count: {agg['request_count']}",
            f"- Failure count: {agg['failure_count']}",
            f"- Avg response: {agg['avg_response_ms']} ms",
            f"- Median response: {agg['median_response_ms']} ms",
            f"- P95 response: {agg['p95_ms']} ms",
            f"- P99 response: {agg['p99_ms']} ms",
            f"- Max response: {agg['max_response_ms']} ms",
            f"- Throughput: {agg['rps']} req/s",
        ]
    )

    lines.extend(["", "## Per Route", ""])
    for key in ["field", "semantic"]:
        item = summary["routes"].get(key)
        if not item:
            continue
        lines.extend(
            [
                f"### {item['name']}",
                f"- Request count: {item['request_count']}",
                f"- Failure count: {item['failure_count']}",
                f"- Avg response: {item['avg_response_ms']} ms",
                f"- Median response: {item['median_response_ms']} ms",
                f"- P95 response: {item['p95_ms']} ms",
                f"- P99 response: {item['p99_ms']} ms",
                f"- Max response: {item['max_response_ms']} ms",
                f"- Throughput: {item['rps']} req/s",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    users = 3
    spawn_rate = 1
    run_time = "20s"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_prefix = OUTPUT_DIR / "rag_perf"

    backend_proc = subprocess.Popen(
        ["micromamba", "run", "-n", "ocr", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", "8091"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ},
    )

    try:
        wait_http(f"{HOST}/api/health", timeout=40)
        warmup()

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "locust",
                "-f",
                str(ROOT / "perf" / "locustfile.py"),
                "--host",
                HOST,
                "--headless",
                "--tags",
                "rag",
                "-u",
                str(users),
                "-r",
                str(spawn_rate),
                "--run-time",
                run_time,
                "--csv",
                str(csv_prefix),
                "--only-summary",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stats_rows = load_stats_csv(Path(f"{csv_prefix}_stats.csv"))
        aggregated = summarize_row(find_row(stats_rows, "Aggregated"))
        field_row = summarize_row(find_row(stats_rows, "POST /api/rag/ask [field]"))
        semantic_row = summarize_row(find_row(stats_rows, "POST /api/rag/ask [semantic]"))

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "host": HOST,
            "users": users,
            "spawn_rate": spawn_rate,
            "run_time": run_time,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "aggregated": aggregated,
            "routes": {
                "field": field_row,
                "semantic": semantic_row,
            },
            "csv_prefix": str(csv_prefix),
        }

        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT_PATH.write_text(build_report(summary), encoding="utf-8")

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"summary saved to {SUMMARY_PATH}")
        print(f"report saved to {REPORT_PATH}")

        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
    finally:
        backend_proc.terminate()
        backend_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
