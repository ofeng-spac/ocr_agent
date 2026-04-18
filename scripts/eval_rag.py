#!/usr/bin/env python3
"""
Evaluate the second-layer dual-track RAG system.

Usage:
    python3 scripts/eval_rag.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag import LeafletQAService


CASES_PATH = ROOT / "data" / "eval" / "rag_eval_cases.jsonl"
RESULTS_PATH = ROOT / "data" / "eval" / "rag_eval_results.json"
REPORT_PATH = ROOT / "data" / "eval" / "rag_eval_report.md"


def load_cases() -> list[dict]:
    cases = []
    with CASES_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def check_case(case: dict, result: dict, latency_ms: float) -> dict:
    checks = {}
    checks["route_mode"] = result.get("route_mode") == case.get("expected_route_mode")
    checks["status"] = result.get("status") == case.get("expected_status")

    expected_retrieval_mode = case.get("expected_retrieval_mode")
    if expected_retrieval_mode is not None:
        checks["retrieval_mode"] = result.get("retrieval_mode") == expected_retrieval_mode

    expected_target_field = case.get("expected_target_field")
    if expected_target_field is not None:
        checks["target_field"] = result.get("target_field") == expected_target_field

    expected_substrings = case.get("expected_substrings") or []
    if expected_substrings:
        answer = result.get("answer", "")
        checks["answer_contains"] = all(substr in answer for substr in expected_substrings)

    expected_citation_field = case.get("expected_citation_field")
    if expected_citation_field:
        checks["citation_field"] = any(
            item.get("field_name") == expected_citation_field
            for item in result.get("citations", [])
        )

    expected_citation_section = case.get("expected_citation_section")
    if expected_citation_section:
        checks["citation_section"] = any(
            item.get("section") == expected_citation_section
            for item in result.get("citations", [])
        )

    checks["latency_recorded"] = latency_ms >= 0
    passed = all(checks.values())

    return {
        "case_id": case["case_id"],
        "canonical_name": case["canonical_name"],
        "question": case["question"],
        "latency_ms": round(latency_ms, 2),
        "checks": checks,
        "passed": passed,
        "result": {
            "status": result.get("status"),
            "route_mode": result.get("route_mode"),
            "retrieval_mode": result.get("retrieval_mode"),
            "target_field": result.get("target_field"),
            "answer_preview": (result.get("answer") or "")[:200],
            "citations_count": len(result.get("citations", [])),
        },
    }


def summarize(case_results: list[dict]) -> dict:
    total = len(case_results)
    passed = sum(1 for item in case_results if item["passed"])

    metric_counts = Counter()
    metric_totals = Counter()
    route_groups = defaultdict(list)
    refusal_cases = []

    for item in case_results:
        for metric_name, metric_passed in item["checks"].items():
            metric_totals[metric_name] += 1
            if metric_passed:
                metric_counts[metric_name] += 1

        route_mode = item["result"]["route_mode"] or "unknown"
        route_groups[route_mode].append(item["latency_ms"])

        if item["result"]["status"] in {"doc_unavailable", "field_unavailable", "semantic_unavailable"}:
            refusal_cases.append(item)

    metrics = {
        "total_cases": total,
        "passed_cases": passed,
        "overall_pass_rate": round(passed / total, 4) if total else 0.0,
        "route_accuracy": round(metric_counts["route_mode"] / metric_totals["route_mode"], 4) if metric_totals["route_mode"] else 0.0,
        "status_accuracy": round(metric_counts["status"] / metric_totals["status"], 4) if metric_totals["status"] else 0.0,
        "retrieval_mode_accuracy": round(metric_counts["retrieval_mode"] / metric_totals["retrieval_mode"], 4) if metric_totals["retrieval_mode"] else 0.0,
        "target_field_accuracy": round(metric_counts["target_field"] / metric_totals["target_field"], 4) if metric_totals["target_field"] else 0.0,
        "answer_hit_rate": round(metric_counts["answer_contains"] / metric_totals["answer_contains"], 4) if metric_totals["answer_contains"] else 0.0,
        "citation_field_hit_rate": round(metric_counts["citation_field"] / metric_totals["citation_field"], 4) if metric_totals["citation_field"] else 0.0,
        "citation_section_hit_rate": round(metric_counts["citation_section"] / metric_totals["citation_section"], 4) if metric_totals["citation_section"] else 0.0,
        "refusal_accuracy": round(sum(1 for item in refusal_cases if item["passed"]) / len(refusal_cases), 4) if refusal_cases else 0.0,
        "avg_latency_ms": round(sum(item["latency_ms"] for item in case_results) / total, 2) if total else 0.0,
        "avg_latency_by_route_ms": {
            route: round(sum(values) / len(values), 2)
            for route, values in route_groups.items()
        },
    }
    return metrics


def build_report(summary: dict, case_results: list[dict]) -> str:
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Total cases: {summary['total_cases']}",
        f"- Passed cases: {summary['passed_cases']}",
        f"- Overall pass rate: {summary['overall_pass_rate']:.2%}",
        f"- Route accuracy: {summary['route_accuracy']:.2%}",
        f"- Status accuracy: {summary['status_accuracy']:.2%}",
        f"- Retrieval mode accuracy: {summary['retrieval_mode_accuracy']:.2%}",
        f"- Target field accuracy: {summary['target_field_accuracy']:.2%}",
        f"- Answer hit rate: {summary['answer_hit_rate']:.2%}",
        f"- Citation field hit rate: {summary['citation_field_hit_rate']:.2%}",
        f"- Citation section hit rate: {summary['citation_section_hit_rate']:.2%}",
        f"- Refusal accuracy: {summary['refusal_accuracy']:.2%}",
        f"- Avg latency: {summary['avg_latency_ms']} ms",
        "",
        "## Avg Latency by Route",
        "",
    ]

    for route, latency in sorted(summary["avg_latency_by_route_ms"].items()):
        lines.append(f"- {route}: {latency} ms")

    lines.extend(["", "## Case Results", ""])
    lines.append("| Case | Passed | Route | Status | Retrieval | Latency(ms) |")
    lines.append("|:--|:--:|:--|:--|:--|--:|")
    for item in case_results:
        lines.append(
            f"| {item['case_id']} | {'Y' if item['passed'] else 'N'} | "
            f"{item['result']['route_mode']} | {item['result']['status']} | "
            f"{item['result']['retrieval_mode']} | {item['latency_ms']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    cases = load_cases()
    service = LeafletQAService()
    service.ensure_index()

    case_results = []
    for case in cases:
        t0 = time.perf_counter()
        result = service.ask(case["canonical_name"], case["question"])
        latency_ms = (time.perf_counter() - t0) * 1000
        case_results.append(check_case(case, result, latency_ms))

    summary = summarize(case_results)
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "cases": case_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(build_report(summary, case_results), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results saved to {RESULTS_PATH}")
    print(f"report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
