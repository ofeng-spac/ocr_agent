#!/usr/bin/env python3
"""
Check whether Qdrant rebuild keeps index metadata and retrieval behavior stable.

Usage:
    python3 scripts/check_reindex_consistency.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag import CHUNK_COLLECTION_NAME, FIELD_COLLECTION_NAME, LeafletQAService


REPORT_PATH = ROOT / "data" / "eval" / "reindex_consistency_report.json"
SAMPLE_QUERIES = [
    ("双黄连口服液", "规格是什么"),
    ("贝伐珠单抗注射液", "这个药主要用于哪些场景"),
    ("狂犬病灭活疫苗", "适应症是什么"),
]


def snapshot_collection(client, collection_name: str) -> dict:
    info = client.get_collection(collection_name)
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    sample_payloads = []
    for point in points:
        sample_payloads.append(
            {
                "id": point.id,
                "payload": point.payload,
            }
        )
    return {
        "points_count": info.points_count,
        "vector_size": info.config.params.vectors.size,
        "sample_payloads": sample_payloads,
    }


def run_sample_queries(service: LeafletQAService) -> list[dict]:
    results = []
    for canonical_name, question in SAMPLE_QUERIES:
        result = service.ask(canonical_name, question)
        results.append(
            {
                "canonical_name": canonical_name,
                "question": question,
                "status": result.get("status"),
                "route_mode": result.get("route_mode"),
                "retrieval_mode": result.get("retrieval_mode"),
                "target_field": result.get("target_field"),
                "answer_preview": (result.get("answer") or "")[:120],
                "first_citation": (result.get("citations") or [None])[0],
            }
        )
    return results


def compare_snapshots(before: dict, after: dict) -> list[str]:
    errors = []
    for collection_name in [FIELD_COLLECTION_NAME, CHUNK_COLLECTION_NAME]:
        before_info = before[collection_name]
        after_info = after[collection_name]
        if before_info["points_count"] != after_info["points_count"]:
            errors.append(
                f"{collection_name} 点数变化: before={before_info['points_count']} after={after_info['points_count']}"
            )
        if before_info["vector_size"] != after_info["vector_size"]:
            errors.append(
                f"{collection_name} 向量维度变化: before={before_info['vector_size']} after={after_info['vector_size']}"
            )
        if before_info["sample_payloads"] != after_info["sample_payloads"]:
            errors.append(f"{collection_name} 样本 payload 发生变化")
    return errors


def compare_behaviors(before: list[dict], after: list[dict]) -> list[str]:
    errors = []
    for before_item, after_item in zip(before, after):
        for key in ["status", "route_mode", "retrieval_mode", "target_field", "answer_preview"]:
            if before_item.get(key) != after_item.get(key):
                errors.append(
                    f"{before_item['canonical_name']} / {before_item['question']} 的 {key} 发生变化: "
                    f"before={before_item.get(key)} after={after_item.get(key)}"
                )
        if before_item.get("first_citation") != after_item.get("first_citation"):
            errors.append(
                f"{before_item['canonical_name']} / {before_item['question']} 的首条 citation 发生变化"
            )
    return errors


def main() -> None:
    service = LeafletQAService()
    try:
        service.ensure_index()
        client = service._get_client()
        if client is None:
            raise RuntimeError("Qdrant client unavailable")

        before_snapshot = {
            FIELD_COLLECTION_NAME: snapshot_collection(client, FIELD_COLLECTION_NAME),
            CHUNK_COLLECTION_NAME: snapshot_collection(client, CHUNK_COLLECTION_NAME),
        }
        before_behavior = run_sample_queries(service)

        rebuild_info = service.rebuild_index()

        after_snapshot = {
            FIELD_COLLECTION_NAME: snapshot_collection(client, FIELD_COLLECTION_NAME),
            CHUNK_COLLECTION_NAME: snapshot_collection(client, CHUNK_COLLECTION_NAME),
        }
        after_behavior = run_sample_queries(service)

        errors = compare_snapshots(before_snapshot, after_snapshot)
        errors.extend(compare_behaviors(before_behavior, after_behavior))

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "passed": not errors,
            "rebuild_info": rebuild_info,
            "before_snapshot": before_snapshot,
            "after_snapshot": after_snapshot,
            "before_behavior": before_behavior,
            "after_behavior": after_behavior,
            "errors": errors,
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"report saved to {REPORT_PATH}")

        if errors:
            raise SystemExit(1)
    finally:
        service.close()


if __name__ == "__main__":
    main()
