from __future__ import annotations

import atexit
import json
from pathlib import Path

from app.services.verifier import normalize_name
from app.services.vectorizer import VECTOR_SIZE, vectorize_text

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    qmodels = None


ROOT = Path(__file__).resolve().parents[2]
FIELDS_PATH = ROOT / "data" / "structured" / "leaflet_fields.jsonl"
QDRANT_PATH = ROOT / "data" / "qdrant"
COLLECTION_NAME = "leaflet_fields_v1"

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
        self._client = None

    def _get_client(self):
        if not QDRANT_AVAILABLE:
            return None
        if self._client is None:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(QDRANT_PATH))
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _payload_for_record(self, record: dict) -> dict:
        return {
            "drug_id": record["drug_id"],
            "canonical_name": record["canonical_name"],
            "field_name": record["field_name"],
            "field_value": record["field_value"],
            "source_file": record["source_file"],
            "source_type": record["source_type"],
        }

    def rebuild_index(self) -> dict:
        client = self._get_client()
        if client is None:
            return {
                "status": "unavailable",
                "reason": "qdrant_client 不可用，无法构建索引。",
                "collection_name": COLLECTION_NAME,
            }

        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
        )

        points = []
        for idx, record in enumerate(self.records):
            content = f"{record['canonical_name']} {record['field_name']} {record['field_value']}"
            points.append(
                qmodels.PointStruct(
                    id=idx,
                    vector=vectorize_text(content),
                    payload=self._payload_for_record(record),
                )
            )

        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)

        return {
            "status": "ok",
            "collection_name": COLLECTION_NAME,
            "points": len(points),
            "vector_size": VECTOR_SIZE,
        }

    def ensure_index(self) -> bool:
        client = self._get_client()
        if client is None:
            return False

        if client.collection_exists(COLLECTION_NAME):
            info = client.get_collection(COLLECTION_NAME)
            if info.points_count == len(self.records):
                return True

        self.rebuild_index()
        return True

    def _search_qdrant(self, canonical_name: str, target_field: str, question: str) -> list[dict]:
        client = self._get_client()
        if client is None:
            return []

        if not self.ensure_index():
            return []

        fields = [target_field]
        if target_field == "brand_name":
            fields.append("generic_name")

        results = []
        seen = set()
        query_vector = vectorize_text(question)

        for field_name in fields:
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="canonical_name",
                            match=qmodels.MatchValue(value=canonical_name),
                        ),
                        qmodels.FieldCondition(
                            key="field_name",
                            match=qmodels.MatchValue(value=field_name),
                        ),
                    ]
                ),
                limit=3,
            )
            for hit in response.points:
                payload = hit.payload or {}
                key = (
                    payload.get("canonical_name"),
                    payload.get("field_name"),
                    payload.get("field_value"),
                    payload.get("source_file"),
                )
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "field_name": payload.get("field_name", ""),
                        "field_value": payload.get("field_value", ""),
                        "source_file": payload.get("source_file", ""),
                        "source_type": payload.get("source_type", ""),
                        "score": round(hit.score or 0.0, 4),
                    }
                )
        return results

    @staticmethod
    def _direct_field_hits(candidates: list[dict], target_field: str) -> list[dict]:
        hits = [record for record in candidates if record["field_name"] == target_field]
        if not hits and target_field == "brand_name":
            hits = [record for record in candidates if record["field_name"] == "generic_name"]
        return hits

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

        qdrant_hits = self._search_qdrant(canonical_name, target_field, question)
        if qdrant_hits:
            answer = "；".join(hit["field_value"] for hit in qdrant_hits)
            return {
                "status": "ok",
                "reason": f"已基于 Qdrant 检索并返回字段 {target_field} 的结果。",
                "target_field": target_field,
                "answer": answer,
                "citations": qdrant_hits,
                "retrieval_mode": "qdrant",
            }

        hits = self._direct_field_hits(candidates, target_field)

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
            "retrieval_mode": "structured_fields",
        }


_QA_SERVICE: LeafletQAService | None = None


def get_leaflet_qa_service() -> LeafletQAService:
    global _QA_SERVICE
    if _QA_SERVICE is None:
        _QA_SERVICE = LeafletQAService()
    return _QA_SERVICE


def _close_global_qa_service() -> None:
    global _QA_SERVICE
    if _QA_SERVICE is not None:
        _QA_SERVICE.close()
        _QA_SERVICE = None


atexit.register(_close_global_qa_service)
