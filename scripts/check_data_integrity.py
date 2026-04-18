#!/usr/bin/env python3
"""
Check structured data integrity and Qdrant index consistency.

Usage:
    python3 scripts/check_data_integrity.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient

from app.services.rag import CHUNK_COLLECTION_NAME, FIELD_COLLECTION_NAME, QDRANT_PATH
from app.services.vectorizer import get_vector_size


CATALOG_PATH = ROOT / "data" / "structured" / "drug_catalog.json"
FIELDS_PATH = ROOT / "data" / "structured" / "leaflet_fields.jsonl"
CHUNKS_PATH = ROOT / "data" / "structured" / "leaflet_chunks.jsonl"
REPORT_PATH = ROOT / "data" / "eval" / "data_integrity_report.json"

CATALOG_REQUIRED = {
    "drug_id",
    "canonical_name",
    "generic_name",
    "brand_names",
    "aliases",
    "doc_paths",
    "doc_status",
    "doc_quality",
    "domain",
    "category",
    "sample_count",
    "active",
}

FIELD_REQUIRED = {
    "drug_id",
    "canonical_name",
    "field_name",
    "field_value",
    "source_file",
    "source_type",
}

CHUNK_REQUIRED = {
    "chunk_id",
    "drug_id",
    "canonical_name",
    "section",
    "chunk_text",
    "source_file",
    "source_type",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def check_catalog() -> dict:
    errors = []
    warnings = []
    catalog = load_json(CATALOG_PATH)

    seen_ids = set()
    seen_names = set()

    for idx, item in enumerate(catalog, start=1):
        missing = sorted(CATALOG_REQUIRED - set(item))
        if missing:
            errors.append(f"catalog[{idx}] 缺少字段: {missing}")
        if item.get("drug_id") in seen_ids:
            errors.append(f"catalog[{idx}] drug_id 重复: {item.get('drug_id')}")
        if item.get("canonical_name") in seen_names:
            errors.append(f"catalog[{idx}] canonical_name 重复: {item.get('canonical_name')}")
        seen_ids.add(item.get("drug_id"))
        seen_names.add(item.get("canonical_name"))

        for doc_path in item.get("doc_paths", []):
            full_path = ROOT / doc_path
            if not full_path.exists():
                errors.append(f"{item.get('canonical_name')} 引用的文档不存在: {doc_path}")

        if item.get("doc_status") == "available" and not item.get("doc_paths"):
            warnings.append(f"{item.get('canonical_name')} 标记为 available，但 doc_paths 为空")

    return {
        "count": len(catalog),
        "errors": errors,
        "warnings": warnings,
        "catalog": catalog,
    }


def check_fields(valid_drugs: set[str], valid_ids: set[str]) -> dict:
    errors = []
    warnings = []
    records = load_jsonl(FIELDS_PATH)

    for idx, record in enumerate(records, start=1):
        missing = sorted(FIELD_REQUIRED - set(record))
        if missing:
            errors.append(f"field[{idx}] 缺少字段: {missing}")
            continue

        if record["canonical_name"] not in valid_drugs:
            errors.append(f"field[{idx}] canonical_name 不在 catalog 中: {record['canonical_name']}")
        if record["drug_id"] not in valid_ids:
            errors.append(f"field[{idx}] drug_id 不在 catalog 中: {record['drug_id']}")
        if not str(record["field_value"]).strip():
            errors.append(f"field[{idx}] field_value 为空")

        full_path = ROOT / record["source_file"]
        if not full_path.exists():
            errors.append(f"field[{idx}] source_file 不存在: {record['source_file']}")

        if record["source_type"] == "non_standard_reference":
            warnings.append(f"{record['canonical_name']} 使用非标准参考资料字段: {record['field_name']}")

    return {
        "count": len(records),
        "errors": errors,
        "warnings": warnings,
        "records": records,
    }


def check_chunks(valid_drugs: set[str], valid_ids: set[str]) -> dict:
    errors = []
    records = load_jsonl(CHUNKS_PATH)
    seen_chunk_ids = set()

    for idx, record in enumerate(records, start=1):
        missing = sorted(CHUNK_REQUIRED - set(record))
        if missing:
            errors.append(f"chunk[{idx}] 缺少字段: {missing}")
            continue

        if record["canonical_name"] not in valid_drugs:
            errors.append(f"chunk[{idx}] canonical_name 不在 catalog 中: {record['canonical_name']}")
        if record["drug_id"] not in valid_ids:
            errors.append(f"chunk[{idx}] drug_id 不在 catalog 中: {record['drug_id']}")
        if not str(record["chunk_text"]).strip():
            errors.append(f"chunk[{idx}] chunk_text 为空")
        if record["chunk_id"] in seen_chunk_ids:
            errors.append(f"chunk[{idx}] chunk_id 重复: {record['chunk_id']}")
        seen_chunk_ids.add(record["chunk_id"])

        full_path = ROOT / record["source_file"]
        if not full_path.exists():
            errors.append(f"chunk[{idx}] source_file 不存在: {record['source_file']}")

    return {
        "count": len(records),
        "errors": errors,
        "records": records,
    }


def check_qdrant_consistency(field_count: int, chunk_count: int) -> dict:
    errors = []
    summary = {}
    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        collections = {c.name for c in client.get_collections().collections}
        if FIELD_COLLECTION_NAME not in collections:
            errors.append(f"缺少 Qdrant 集合: {FIELD_COLLECTION_NAME}")
        if CHUNK_COLLECTION_NAME not in collections:
            errors.append(f"缺少 Qdrant 集合: {CHUNK_COLLECTION_NAME}")

        if FIELD_COLLECTION_NAME in collections:
            info = client.get_collection(FIELD_COLLECTION_NAME)
            summary[FIELD_COLLECTION_NAME] = {
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
            }
            if info.points_count != field_count:
                errors.append(
                    f"{FIELD_COLLECTION_NAME} 点数不一致: qdrant={info.points_count}, jsonl={field_count}"
                )
            if info.config.params.vectors.size != get_vector_size():
                errors.append(
                    f"{FIELD_COLLECTION_NAME} 向量维度不一致: qdrant={info.config.params.vectors.size}, expected={get_vector_size()}"
                )

        if CHUNK_COLLECTION_NAME in collections:
            info = client.get_collection(CHUNK_COLLECTION_NAME)
            summary[CHUNK_COLLECTION_NAME] = {
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
            }
            if info.points_count != chunk_count:
                errors.append(
                    f"{CHUNK_COLLECTION_NAME} 点数不一致: qdrant={info.points_count}, jsonl={chunk_count}"
                )
            if info.config.params.vectors.size != get_vector_size():
                errors.append(
                    f"{CHUNK_COLLECTION_NAME} 向量维度不一致: qdrant={info.config.params.vectors.size}, expected={get_vector_size()}"
                )
    finally:
        client.close()

    return {
        "summary": summary,
        "errors": errors,
    }


def main() -> None:
    catalog_result = check_catalog()
    valid_drugs = {item["canonical_name"] for item in catalog_result["catalog"]}
    valid_ids = {item["drug_id"] for item in catalog_result["catalog"]}

    field_result = check_fields(valid_drugs, valid_ids)
    chunk_result = check_chunks(valid_drugs, valid_ids)
    qdrant_result = check_qdrant_consistency(field_result["count"], chunk_result["count"])

    errors = (
        catalog_result["errors"]
        + field_result["errors"]
        + chunk_result["errors"]
        + qdrant_result["errors"]
    )
    warnings = catalog_result["warnings"] + field_result["warnings"]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "passed": not errors,
        "catalog_count": catalog_result["count"],
        "field_count": field_result["count"],
        "chunk_count": chunk_result["count"],
        "qdrant": qdrant_result["summary"],
        "errors": errors,
        "warnings": warnings,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report saved to {REPORT_PATH}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
