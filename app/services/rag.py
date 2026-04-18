from __future__ import annotations

import atexit
import json
from pathlib import Path

from app.services.embedding import get_embedding_model_name
from app.services.verifier import normalize_name
from app.services.vectorizer import get_vector_size, vectorize_text, vectorize_texts

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
CHUNKS_PATH = ROOT / "data" / "structured" / "leaflet_chunks.jsonl"
QDRANT_PATH = ROOT / "data" / "qdrant"
FIELD_COLLECTION_NAME = "leaflet_fields_v1"
CHUNK_COLLECTION_NAME = "leaflet_chunks_v1"

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

SEMANTIC_SECTION_RULES = [
    (["适应症", "主治", "用途", "用于", "哪些场景", "适用于", "治疗", "疾病", "感染"], ["indications"]),
    (["注意", "风险", "慎用", "不良反应", "副作用"], ["precautions", "contraindications"]),
    (["怎么用", "给药", "输注", "服用", "剂量", "频次"], ["dosage"]),
    (["外观", "长什么样"], ["appearance"]),
]

SEMANTIC_ROUTE_PATTERNS = [
    "主要用于",
    "哪些场景",
    "什么场景",
    "一般用于",
    "有哪些风险",
    "风险和注意事项",
    "使用提醒",
    "慎用提醒",
    "治疗哪些感染",
]


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_leaflet_fields() -> list[dict]:
    return _load_jsonl(FIELDS_PATH)


def load_leaflet_chunks() -> list[dict]:
    return _load_jsonl(CHUNKS_PATH)


def infer_field_name(question: str) -> str | None:
    normalized_question = normalize_name(question)
    for field_name, keywords in QUESTION_FIELD_RULES:
        for keyword in keywords:
            if normalize_name(keyword) in normalized_question:
                return field_name
    return None


def route_question(question: str) -> tuple[str, str | None]:
    normalized_question = normalize_name(question)
    if any(normalize_name(pattern) in normalized_question for pattern in SEMANTIC_ROUTE_PATTERNS):
        return "semantic_query", None
    field_name = infer_field_name(question)
    if field_name:
        return "field_query", field_name
    return "semantic_query", None


def infer_semantic_sections(question: str) -> list[str]:
    normalized_question = normalize_name(question)
    sections = []
    for keywords, section_names in SEMANTIC_SECTION_RULES:
        for keyword in keywords:
            if normalize_name(keyword) in normalized_question:
                sections.extend(section_names)
                break
    deduped = []
    seen = set()
    for section in sections:
        if section not in seen:
            deduped.append(section)
            seen.add(section)
    return deduped


class LeafletQAService:
    def __init__(self, records: list[dict] | None = None, chunks: list[dict] | None = None):
        self.records = records if records is not None else load_leaflet_fields()
        self.chunks = chunks if chunks is not None else load_leaflet_chunks()
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
                "field_collection": FIELD_COLLECTION_NAME,
                "chunk_collection": CHUNK_COLLECTION_NAME,
            }

        for collection_name in [FIELD_COLLECTION_NAME, CHUNK_COLLECTION_NAME]:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)

        for collection_name in [FIELD_COLLECTION_NAME, CHUNK_COLLECTION_NAME]:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=get_vector_size(), distance=qmodels.Distance.COSINE),
            )

        field_points = []
        field_texts = [
            f"{record['canonical_name']} {record['field_name']} {record['field_value']}"
            for record in self.records
        ]
        field_vectors = vectorize_texts(field_texts) if field_texts else []
        for idx, record in enumerate(self.records):
            field_points.append(
                qmodels.PointStruct(
                    id=idx,
                    vector=field_vectors[idx],
                    payload=self._payload_for_record(record),
                )
            )

        chunk_points = []
        chunk_texts = [
            f"{chunk['canonical_name']} {chunk['section']} {chunk['chunk_text']}"
            for chunk in self.chunks
        ]
        chunk_vectors = vectorize_texts(chunk_texts) if chunk_texts else []
        for idx, chunk in enumerate(self.chunks):
            chunk_points.append(
                qmodels.PointStruct(
                    id=idx,
                    vector=chunk_vectors[idx],
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "drug_id": chunk["drug_id"],
                        "canonical_name": chunk["canonical_name"],
                        "section": chunk["section"],
                        "chunk_text": chunk["chunk_text"],
                        "source_file": chunk["source_file"],
                        "source_type": chunk["source_type"],
                    },
                )
            )

        if field_points:
            client.upsert(collection_name=FIELD_COLLECTION_NAME, points=field_points)
        if chunk_points:
            client.upsert(collection_name=CHUNK_COLLECTION_NAME, points=chunk_points)

        return {
            "status": "ok",
            "field_collection": FIELD_COLLECTION_NAME,
            "chunk_collection": CHUNK_COLLECTION_NAME,
            "field_points": len(field_points),
            "chunk_points": len(chunk_points),
            "vector_size": get_vector_size(),
            "embedding_model": get_embedding_model_name(),
        }

    def ensure_index(self) -> bool:
        client = self._get_client()
        if client is None:
            return False

        field_ok = False
        chunk_ok = False
        if client.collection_exists(FIELD_COLLECTION_NAME):
            info = client.get_collection(FIELD_COLLECTION_NAME)
            field_ok = info.points_count == len(self.records)
        if client.collection_exists(CHUNK_COLLECTION_NAME):
            info = client.get_collection(CHUNK_COLLECTION_NAME)
            chunk_ok = info.points_count == len(self.chunks)
        if field_ok and chunk_ok:
            return True

        self.rebuild_index()
        return True

    def _search_qdrant_fields(self, canonical_name: str, target_field: str, question: str) -> list[dict]:
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
                collection_name=FIELD_COLLECTION_NAME,
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

    def _search_qdrant_chunks(self, canonical_name: str, question: str, sections: list[str] | None = None) -> list[dict]:
        client = self._get_client()
        if client is None:
            return []

        if not self.ensure_index():
            return []

        results = []
        seen = set()
        query_vector = vectorize_text(question)

        query_sections = sections or [None]
        for section_name in query_sections:
            must_conditions = [
                qmodels.FieldCondition(
                    key="canonical_name",
                    match=qmodels.MatchValue(value=canonical_name),
                )
            ]
            if section_name:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="section",
                        match=qmodels.MatchValue(value=section_name),
                    )
                )

            response = client.query_points(
                collection_name=CHUNK_COLLECTION_NAME,
                query=query_vector,
                query_filter=qmodels.Filter(must=must_conditions),
                limit=3,
            )

            for hit in response.points:
                payload = hit.payload or {}
                key = (payload.get("chunk_id"), payload.get("chunk_text"))
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "chunk_id": payload.get("chunk_id", ""),
                        "section": payload.get("section", ""),
                        "chunk_text": payload.get("chunk_text", ""),
                        "source_file": payload.get("source_file", ""),
                        "source_type": payload.get("source_type", ""),
                        "score": round(hit.score or 0.0, 4),
                    }
                )
        results.sort(key=lambda item: item["score"], reverse=True)
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

        route_mode, target_field = route_question(question)
        candidates = [
            record
            for record in self.records
            if normalize_name(record["canonical_name"]) == normalize_name(canonical_name)
        ]

        chunk_candidates = [
            chunk
            for chunk in self.chunks
            if normalize_name(chunk["canonical_name"]) == normalize_name(canonical_name)
        ]

        if not candidates and not chunk_candidates:
            return {
                "status": "doc_unavailable",
                "reason": f"{canonical_name} 当前没有可用的结构化说明书字段。",
                "answer": "",
                "citations": [],
                "target_field": target_field,
                "route_mode": route_mode,
            }

        if route_mode == "field_query":
            qdrant_hits = self._search_qdrant_fields(canonical_name, target_field, question)
            if qdrant_hits:
                answer = "；".join(hit["field_value"] for hit in qdrant_hits)
                return {
                    "status": "ok",
                    "reason": f"已基于 Qdrant 检索并返回字段 {target_field} 的结果。",
                    "target_field": target_field,
                    "answer": answer,
                    "citations": qdrant_hits,
                    "retrieval_mode": "qdrant_fields",
                    "route_mode": route_mode,
                }

            hits = self._direct_field_hits(candidates, target_field)

            if not hits:
                return {
                    "status": "field_unavailable",
                    "reason": f"{canonical_name} 当前没有字段 {target_field} 的结构化结果。",
                    "answer": "",
                    "citations": [],
                    "target_field": target_field,
                    "route_mode": route_mode,
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
                "route_mode": route_mode,
            }

        semantic_sections = infer_semantic_sections(question)
        chunk_hits = self._search_qdrant_chunks(canonical_name, question, sections=semantic_sections)
        if chunk_hits:
            answer = "\n\n".join(hit["chunk_text"] for hit in chunk_hits[:2])
            citations = [
                {
                    "section": hit["section"],
                    "chunk_id": hit["chunk_id"],
                    "chunk_text": hit["chunk_text"],
                    "source_file": hit["source_file"],
                    "source_type": hit["source_type"],
                    "score": hit["score"],
                }
                for hit in chunk_hits
            ]
            return {
                "status": "ok",
                "reason": "已基于 Qdrant chunk 检索返回语义问答结果。",
                "target_field": target_field,
                "answer": answer,
                "citations": citations,
                "retrieval_mode": "qdrant_chunks",
                "route_mode": route_mode,
            }

        return {
            "status": "semantic_unavailable",
            "reason": f"{canonical_name} 当前没有可用的 chunk 级语义检索结果。",
            "target_field": target_field,
            "answer": "",
            "citations": [],
            "retrieval_mode": "qdrant_chunks",
            "route_mode": route_mode,
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
