from __future__ import annotations

import json
from pathlib import Path

from app.services.verifier import normalize_name


ROOT = Path(__file__).resolve().parents[2]
FIELDS_PATH = ROOT / "data" / "structured" / "leaflet_fields.jsonl"

QUESTION_FIELD_RULES = [
    ("specification", ["规格", "多少毫克", "多少mg", "多大规格", "每片", "每支"]),
    ("indications", ["适应症", "主治", "用途", "治什么", "适用于什么"]),
    ("dosage", ["用法", "用量", "怎么吃", "怎么用", "剂量", "服用"]),
    ("precautions", ["注意事项", "注意什么", "慎用", "提醒"]),
    ("contraindications", ["禁忌", "不能用", "禁用"]),
    ("appearance", ["性状", "外观", "长什么样"]),
    ("generic_name", ["药品名称", "通用名", "正式名称", "叫什么"]),
    ("brand_name", ["商品名", "品牌名", "品牌"]),
]


def load_leaflet_fields() -> list[dict]:
    records = []
    if not FIELDS_PATH.exists():
        return records
    with FIELDS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def infer_field_name(question: str) -> str:
    normalized_question = normalize_name(question)
    for field_name, keywords in QUESTION_FIELD_RULES:
        for keyword in keywords:
            if normalize_name(keyword) in normalized_question:
                return field_name
    return "generic_name"


class LeafletQAService:
    def __init__(self, records: list[dict] | None = None):
        self.records = records if records is not None else load_leaflet_fields()

    def ask(self, canonical_name: str, question: str) -> dict:
        canonical_name = canonical_name.strip()
        question = question.strip()

        if not canonical_name:
            return {
                "status": "error",
                "reason": "canonical_name 不能为空。",
                "answer": "",
                "citations": [],
            }

        if not question:
            return {
                "status": "error",
                "reason": "question 不能为空。",
                "answer": "",
                "citations": [],
            }

        target_field = infer_field_name(question)
        candidates = [
            record
            for record in self.records
            if normalize_name(record["canonical_name"]) == normalize_name(canonical_name)
        ]

        if not candidates:
            return {
                "status": "doc_unavailable",
                "reason": f"{canonical_name} 当前没有可用的结构化说明书字段。",
                "answer": "",
                "citations": [],
                "target_field": target_field,
            }

        hits = [record for record in candidates if record["field_name"] == target_field]
        if not hits and target_field == "brand_name":
            hits = [record for record in candidates if record["field_name"] == "generic_name"]

        if not hits:
            return {
                "status": "field_unavailable",
                "reason": f"{canonical_name} 当前没有字段 {target_field} 的结构化结果。",
                "answer": "",
                "citations": [],
                "target_field": target_field,
            }

        answer = "；".join(hit["field_value"] for hit in hits)
        citations = [
            {
                "field_name": hit["field_name"],
                "field_value": hit["field_value"],
                "source_file": hit["source_file"],
                "source_type": hit["source_type"],
            }
            for hit in hits
        ]

        return {
            "status": "ok",
            "reason": f"已基于字段 {target_field} 返回结果。",
            "target_field": target_field,
            "answer": answer,
            "citations": citations,
        }
