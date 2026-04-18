#!/usr/bin/env python3
"""
Run the project's core regression checks in one command.

Usage:
    python3 scripts/run_regression.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "eval" / "regression_summary.json"


def run_step(name: str, cmd: list[str]) -> dict:
    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "name": name,
        "command": cmd,
        "started_at": started_at,
        "elapsed_ms": elapsed_ms,
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def extract_pytest_summary(stdout: str) -> dict:
    summary = {"passed": None, "failed": None, "raw": ""}
    for line in stdout.splitlines():
        if "passed" in line or "failed" in line or "error" in line:
            summary["raw"] = line.strip()
    m = re.search(r"(?P<passed>\d+)\s+passed", summary["raw"])
    if m:
        summary["passed"] = int(m.group("passed"))
    m = re.search(r"(?P<failed>\d+)\s+failed", summary["raw"])
    if m:
        summary["failed"] = int(m.group("failed"))
    return summary


def extract_verifier_summary(stdout: str) -> dict:
    m = re.search(r"passed\s+(?P<passed>\d+)/(?P<total>\d+)\s+cases", stdout)
    if not m:
        return {"passed": None, "total": None}
    return {
        "passed": int(m.group("passed")),
        "total": int(m.group("total")),
    }


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    python = sys.executable

    steps = [
        run_step("pytest", [python, "-m", "pytest", "-q", "tests"]),
        run_step("verifier_regression", [python, "scripts/check_name_verifier.py"]),
        run_step("build_rag_eval_cases", [python, "scripts/build_rag_eval_cases.py"]),
        run_step("rag_eval", [python, "scripts/eval_rag.py"]),
    ]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": python,
        "all_passed": all(step["passed"] for step in steps),
        "steps": steps,
        "artifacts": {
            "rag_eval_results": load_json_if_exists(ROOT / "data" / "eval" / "rag_eval_results.json"),
            "rag_eval_report_path": str(ROOT / "data" / "eval" / "rag_eval_report.md"),
        },
        "highlights": {
            "pytest": extract_pytest_summary(steps[0]["stdout"]),
            "verifier_regression": extract_verifier_summary(steps[1]["stdout"]),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Regression Summary ===")
    for step in steps:
        mark = "PASS" if step["passed"] else "FAIL"
        print(f"- {step['name']}: {mark} ({step['elapsed_ms']} ms)")

    pytest_info = summary["highlights"]["pytest"]
    verifier_info = summary["highlights"]["verifier_regression"]
    print("\n=== Highlights ===")
    print(f"- pytest: {pytest_info}")
    print(f"- verifier: {verifier_info}")
    rag_results = summary["artifacts"]["rag_eval_results"]
    if rag_results:
        print(f"- rag overall pass rate: {rag_results['summary']['overall_pass_rate']:.2%}")
        print(f"- rag avg latency: {rag_results['summary']['avg_latency_ms']} ms")

    print(f"\nsummary saved to {OUTPUT_PATH}")

    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
