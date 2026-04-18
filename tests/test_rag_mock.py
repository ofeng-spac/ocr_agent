from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.rag import LeafletQAService


def build_service():
    records = [
        {
            "drug_id": "drug_001",
            "canonical_name": "双黄连口服液",
            "field_name": "specification",
            "field_value": "每支装 10 毫升。",
            "source_file": "data/source_docs/text/双黄连口服液.txt",
            "source_type": "standard_leaflet",
        },
        {
            "drug_id": "drug_001",
            "canonical_name": "双黄连口服液",
            "field_name": "generic_name",
            "field_value": "双黄连口服液",
            "source_file": "data/source_docs/text/双黄连口服液.txt",
            "source_type": "standard_leaflet",
        },
    ]
    chunks = [
        {
            "chunk_id": "drug_001_chunk_001",
            "drug_id": "drug_001",
            "canonical_name": "双黄连口服液",
            "section": "indications",
            "chunk_text": "疏风解表，清热解毒。用于外感风热所致的感冒。",
            "source_file": "data/source_docs/text/双黄连口服液.txt",
            "source_type": "standard_leaflet",
        }
    ]
    return LeafletQAService(records=records, chunks=chunks)


def test_rag_returns_doc_unavailable_when_no_records_and_no_chunks():
    service = LeafletQAService(records=[], chunks=[])

    result = service.ask("狂犬病灭活疫苗", "适应症是什么")

    assert result["status"] == "doc_unavailable"
    assert result["route_mode"] == "field_query"


@patch.object(LeafletQAService, "_search_qdrant_fields")
def test_rag_field_query_prefers_qdrant_field_hits(mock_search_fields):
    service = build_service()
    mock_search_fields.return_value = [
        {
            "field_name": "specification",
            "field_value": "每支装 10 毫升。",
            "source_file": "data/source_docs/text/双黄连口服液.txt",
            "source_type": "standard_leaflet",
            "score": 0.98,
        }
    ]

    result = service.ask("双黄连口服液", "规格是什么")

    assert result["status"] == "ok"
    assert result["route_mode"] == "field_query"
    assert result["retrieval_mode"] == "qdrant_fields"
    assert "每支装 10 毫升" in result["answer"]
    mock_search_fields.assert_called_once()


@patch.object(LeafletQAService, "_search_qdrant_fields")
def test_rag_field_query_falls_back_to_structured_fields_when_qdrant_returns_empty(mock_search_fields):
    service = build_service()
    mock_search_fields.return_value = []

    result = service.ask("双黄连口服液", "规格是什么")

    assert result["status"] == "ok"
    assert result["route_mode"] == "field_query"
    assert result["retrieval_mode"] == "structured_fields"
    assert result["target_field"] == "specification"
    assert result["citations"][0]["field_name"] == "specification"


@patch.object(LeafletQAService, "_search_qdrant_chunks")
def test_rag_semantic_query_uses_chunk_retrieval(mock_search_chunks):
    service = build_service()
    mock_search_chunks.return_value = [
        {
            "chunk_id": "drug_001_chunk_001",
            "section": "indications",
            "chunk_text": "疏风解表，清热解毒。用于外感风热所致的感冒。",
            "source_file": "data/source_docs/text/双黄连口服液.txt",
            "source_type": "standard_leaflet",
            "score": 0.91,
        }
    ]

    result = service.ask("双黄连口服液", "这个药主要用于哪些场景")

    assert result["status"] == "ok"
    assert result["route_mode"] == "semantic_query"
    assert result["retrieval_mode"] == "qdrant_chunks"
    assert result["citations"][0]["section"] == "indications"
    mock_search_chunks.assert_called_once()


@patch.object(LeafletQAService, "_search_qdrant_chunks")
def test_rag_semantic_query_returns_semantic_unavailable_when_no_chunk_hits(mock_search_chunks):
    service = build_service()
    mock_search_chunks.return_value = []

    result = service.ask("双黄连口服液", "这个药主要用于哪些场景")

    assert result["status"] == "semantic_unavailable"
    assert result["route_mode"] == "semantic_query"
    assert result["retrieval_mode"] == "qdrant_chunks"


def test_ensure_index_rebuilds_when_collection_counts_do_not_match():
    service = build_service()
    fake_client = MagicMock()
    fake_client.collection_exists.side_effect = [True, True]
    fake_client.get_collection.side_effect = [
        MagicMock(points_count=1),
        MagicMock(points_count=0),
    ]
    service._client = fake_client
    service.rebuild_index = MagicMock(return_value={"status": "ok"})

    ok = service.ensure_index()

    assert ok is True
    service.rebuild_index.assert_called_once()
