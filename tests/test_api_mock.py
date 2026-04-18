from __future__ import annotations

from unittest.mock import patch


def test_health_endpoint_returns_ok(client):
    resp = client.get("/api/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["service"] == "drug-recognition-agent"


@patch("app.api.main.list_videos")
def test_videos_endpoint_returns_video_list(mock_list_videos, client):
    mock_list_videos.return_value = ["video001.mp4", "video002.mp4"]

    resp = client.get("/api/videos")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["videos"][0]["name"] == "video001.mp4"
    assert data["videos"][1]["url"] == "/videos/video002.mp4"


@patch("app.api.main.invoke_recognition_workflow")
def test_recognize_endpoint_returns_mocked_workflow_result(mock_workflow, client):
    mock_workflow.return_value = {
        "video_name": "video001.mp4",
        "model": "mock-model",
        "elapsed": 1.23,
        "result": "RAW_RESULT",
        "raw_name": "注射用头孢噻呋钠",
        "evidence_text": "兽用;批号:20240723",
        "uncertainty_text": "低",
        "uncertainty_level": "low",
        "canonical_name": "注射用头孢噻呋钠",
        "verify_status": "verified_exact",
        "verify_match_type": "canonical",
        "verify_reason": "已匹配到标准药名 注射用头孢噻呋钠。",
        "trace_id": "trace-001",
        "workflow_trace": [{"node": "recognize_node", "status": "ok", "summary": "done"}],
    }

    resp = client.post(
        "/api/recognize",
        json={"video_name": "video001.mp4", "model": "qwen3-vl-8b-instruct-awq-4bit"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["video_name"] == "video001.mp4"
    assert data["raw_name"] == "注射用头孢噻呋钠"
    assert data["canonical_name"] == "注射用头孢噻呋钠"
    assert data["verify_status"] == "verified_exact"
    assert data["trace_id"] == "trace-001"
    assert data["workflow_trace"][0]["node"] == "recognize_node"

    mock_workflow.assert_called_once_with(
        request_type="recognize",
        video_path="/home/vision/users/flb/Kestrel-Recognition/video/video001.mp4",
        model="qwen3-vl-8b-instruct-awq-4bit",
        knowledge=True,
        guide=True,
        expected_drug_name=None,
    )


def test_recognize_endpoint_rejects_path_like_video_name(client):
    resp = client.post("/api/recognize", json={"video_name": "../video001.mp4"})

    assert resp.status_code == 400
    assert "file name" in resp.json()["detail"]


def test_recognize_endpoint_returns_404_for_missing_video(client):
    resp = client.post("/api/recognize", json={"video_name": "missing.mp4"})

    assert resp.status_code == 404
    assert "Video not found" in resp.json()["detail"]


@patch("app.api.main.invoke_recognition_workflow", side_effect=KeyError("bad-model"))
def test_recognize_endpoint_maps_unknown_model_to_http_400(mock_workflow, client):
    resp = client.post(
        "/api/recognize",
        json={"video_name": "video001.mp4", "model": "bad-model"},
    )

    assert resp.status_code == 400
    assert "Unknown model preset" in resp.json()["detail"]
    mock_workflow.assert_called_once()


@patch("app.api.main.invoke_recognition_workflow")
def test_verify_endpoint_returns_structured_response(mock_workflow, client):
    mock_workflow.return_value = {
        "video_name": "video001.mp4",
        "model": "mock-model",
        "elapsed": 1.11,
        "result": "RAW_RESULT",
        "raw_name": "注射用头孢噻呋钠",
        "canonical_name": "注射用头孢噻呋钠",
        "verify_status": "verified_exact",
        "verify_match_type": "canonical",
        "verify_reason": "已匹配到标准药名 注射用头孢噻呋钠。",
        "expected_check": {
            "expected_drug_name": "注射用头孢噻呋钠",
            "status": "match",
            "reason": "识别后的标准药名与期望药名一致。",
        },
        "trace_id": "trace-verify-001",
        "workflow_trace": [{"node": "verify_node", "status": "ok", "summary": "verified"}],
    }

    resp = client.post(
        "/api/verify",
        json={
            "video_name": "video001.mp4",
            "expected_drug_name": "注射用头孢噻呋钠",
            "model": "qwen3-vl-8b-instruct-awq-4bit",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["canonical_name"] == "注射用头孢噻呋钠"
    assert data["verify_status"] == "verified_exact"
    assert data["expected_check"]["status"] == "match"
    assert data["trace_id"] == "trace-verify-001"

    mock_workflow.assert_called_once_with(
        request_type="verify",
        video_path="/home/vision/users/flb/Kestrel-Recognition/video/video001.mp4",
        model="qwen3-vl-8b-instruct-awq-4bit",
        knowledge=True,
        guide=True,
        expected_drug_name="注射用头孢噻呋钠",
    )


def test_verify_endpoint_rejects_empty_expected_name(client):
    resp = client.post(
        "/api/verify",
        json={"video_name": "video001.mp4", "expected_drug_name": "   "},
    )

    assert resp.status_code == 400
    assert "must not be empty" in resp.json()["detail"]


def test_verify_endpoint_returns_404_for_missing_video(client):
    resp = client.post(
        "/api/verify",
        json={"video_name": "missing.mp4", "expected_drug_name": "注射用头孢噻呋钠"},
    )

    assert resp.status_code == 404
    assert "Video not found" in resp.json()["detail"]


@patch("app.api.main.invoke_recognition_workflow", side_effect=KeyError("bad-model"))
def test_verify_endpoint_maps_unknown_model_to_http_400(mock_workflow, client):
    resp = client.post(
        "/api/verify",
        json={
            "video_name": "video001.mp4",
            "expected_drug_name": "注射用头孢噻呋钠",
            "model": "bad-model",
        },
    )

    assert resp.status_code == 400
    assert "Unknown model preset" in resp.json()["detail"]
    mock_workflow.assert_called_once()


@patch("app.api.main.invoke_rag_workflow")
def test_rag_ask_endpoint_returns_mocked_result(mock_rag_workflow, client):
    mock_rag_workflow.return_value = {
        "status": "ok",
        "reason": "已基于 Qdrant chunk 检索返回语义问答结果。",
        "target_field": None,
        "answer": "用于外感风热所致的感冒。",
        "citations": [{"section": "indications"}],
        "retrieval_mode": "qdrant_chunks",
        "route_mode": "semantic_query",
        "trace_id": "rag-trace-001",
        "workflow_trace": [{"node": "rag_node", "status": "ok", "summary": "done"}],
    }

    resp = client.post(
        "/api/rag/ask",
        json={"canonical_name": "双黄连口服液", "question": "这个药主要用于哪些场景"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["retrieval_mode"] == "qdrant_chunks"
    assert data["trace_id"] == "rag-trace-001"
    assert data["workflow_trace"][0]["node"] == "rag_node"

    mock_rag_workflow.assert_called_once_with(
        canonical_name="双黄连口服液",
        question="这个药主要用于哪些场景",
    )


@patch("app.api.main.invoke_rag_workflow")
def test_rag_ask_endpoint_maps_error_to_http_400(mock_rag_workflow, client):
    mock_rag_workflow.return_value = {
        "status": "error",
        "reason": "canonical_name 不能为空。",
        "answer": "",
        "citations": [],
    }

    resp = client.post(
        "/api/rag/ask",
        json={"canonical_name": "", "question": "规格是什么"},
    )

    assert resp.status_code == 400
    assert "不能为空" in resp.json()["detail"]


@patch("app.api.main.list_audit_logs")
def test_audit_logs_endpoint_returns_logs(mock_list_audit_logs, client):
    mock_list_audit_logs.return_value = [
        {"trace_id": "trace-001", "event_type": "recognize", "payload": {"raw_name": "注射用头孢噻呋钠"}}
    ]

    resp = client.get("/api/audit_logs?limit=5")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["logs"][0]["trace_id"] == "trace-001"
    mock_list_audit_logs.assert_called_once_with(limit=5)


@patch("app.api.main.load_evaluation_summary")
def test_eval_summary_endpoint_returns_summary(mock_load_summary, client):
    mock_load_summary.return_value = {
        "available": True,
        "recommended_model": "Qwen3-VL-8B-Instruct-AWQ-4bit",
        "recommended_config": {"config": "kb+guide+cot"},
    }

    resp = client.get("/api/eval/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["recommended_model"] == "Qwen3-VL-8B-Instruct-AWQ-4bit"
    assert data["recommended_config"]["config"] == "kb+guide+cot"
    mock_load_summary.assert_called_once()
