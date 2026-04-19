from __future__ import annotations

import time
from datetime import datetime

from langgraph.graph import END, START, StateGraph

from app.graph.state import RagWorkflowState, RecognitionWorkflowState
from app.services.audit import append_audit_log, generate_trace_id
from app.services.rag import get_leaflet_qa_service
from app.services.recognizer import apply_verification, run_recognition


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _append_trace(
    state: dict,
    node: str,
    status: str,
    summary: str,
    *,
    started_at: str | None = None,
    finished_at: str | None = None,
    duration_ms: float | None = None,
) -> list[dict]:
    trace = list(state.get("workflow_trace", []))
    item = {"node": node, "status": status, "summary": summary}
    if started_at is not None:
        item["started_at"] = started_at
    if finished_at is not None:
        item["finished_at"] = finished_at
    if duration_ms is not None:
        item["duration_ms"] = round(duration_ms, 2)
    trace.append(item)
    return trace


def recognize_node(state: RecognitionWorkflowState) -> RecognitionWorkflowState:
    started_at = _now_iso()
    t0 = time.perf_counter()
    recognition_result = run_recognition(
        video_path=state["video_path"],
        model=state["model"],
        knowledge=state["knowledge"],
        guide=state["guide"],
    )
    finished_at = _now_iso()
    trace = _append_trace(
        state,
        "recognize_node",
        "ok",
        f"识别完成，raw_name={recognition_result.get('raw_name') or '未提取'}",
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )
    return {
        "recognition_result": recognition_result,
        "workflow_trace": trace,
    }


def verify_node(state: RecognitionWorkflowState) -> RecognitionWorkflowState:
    started_at = _now_iso()
    t0 = time.perf_counter()
    response = apply_verification(state["recognition_result"], state.get("expected_drug_name"))
    response["trace_id"] = state["trace_id"]

    summary = (
        f"标准名校验完成，status={response.get('verify_status')}，"
        f"canonical_name={response.get('canonical_name') or '未确认'}"
    )
    finished_at = _now_iso()
    trace = _append_trace(
        state,
        "verify_node",
        "ok",
        summary,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )
    response["workflow_trace"] = trace
    return {
        "response": response,
        "workflow_trace": trace,
    }


def audit_node(state: RecognitionWorkflowState) -> RecognitionWorkflowState:
    started_at = _now_iso()
    t0 = time.perf_counter()
    event_type = f"{state['request_type']}_workflow"
    append_audit_log(event_type, state["response"], trace_id=state["trace_id"])
    finished_at = _now_iso()
    trace = _append_trace(
        state,
        "audit_node",
        "ok",
        "审计日志已写入。",
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )
    response = dict(state["response"])
    response["workflow_trace"] = trace
    return {
        "response": response,
        "workflow_trace": trace,
    }


def rag_node(state: RagWorkflowState) -> RagWorkflowState:
    started_at = _now_iso()
    t0 = time.perf_counter()
    qa_service = get_leaflet_qa_service()
    result = qa_service.ask(state["canonical_name"], state["question"])
    result["trace_id"] = state["trace_id"]
    result["canonical_name"] = state["canonical_name"]
    result["question"] = state["question"]

    summary = (
        f"说明书问答完成，status={result.get('status')}，"
        f"target_field={result.get('target_field') or 'unknown'}"
    )
    finished_at = _now_iso()
    trace = _append_trace(
        state,
        "rag_node",
        "ok",
        summary,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )
    result["workflow_trace"] = trace
    return {
        "rag_result": result,
        "response": result,
        "workflow_trace": trace,
    }


def build_recognition_graph():
    graph = StateGraph(RecognitionWorkflowState)
    graph.add_node("recognize_node", recognize_node)
    graph.add_node("verify_node", verify_node)
    graph.add_node("audit_node", audit_node)
    graph.add_edge(START, "recognize_node")
    graph.add_edge("recognize_node", "verify_node")
    graph.add_edge("verify_node", "audit_node")
    graph.add_edge("audit_node", END)
    return graph.compile()


def build_rag_graph():
    graph = StateGraph(RagWorkflowState)
    graph.add_node("rag_node", rag_node)
    graph.add_node("audit_node", audit_node)
    graph.add_edge(START, "rag_node")
    graph.add_edge("rag_node", "audit_node")
    graph.add_edge("audit_node", END)
    return graph.compile()


_RECOGNITION_GRAPH = None
_RAG_GRAPH = None


def get_recognition_graph():
    global _RECOGNITION_GRAPH
    if _RECOGNITION_GRAPH is None:
        _RECOGNITION_GRAPH = build_recognition_graph()
    return _RECOGNITION_GRAPH


def get_rag_graph():
    global _RAG_GRAPH
    if _RAG_GRAPH is None:
        _RAG_GRAPH = build_rag_graph()
    return _RAG_GRAPH


def invoke_recognition_workflow(
    *,
    request_type: str,
    video_path: str,
    model: str,
    knowledge: bool,
    guide: bool,
    expected_drug_name: str | None = None,
) -> dict:
    trace_id = generate_trace_id(request_type)
    state: RecognitionWorkflowState = {
        "request_type": request_type,
        "video_path": video_path,
        "video_name": video_path.split("/")[-1],
        "model": model,
        "knowledge": knowledge,
        "guide": guide,
        "expected_drug_name": expected_drug_name,
        "trace_id": trace_id,
        "workflow_trace": [],
    }
    final_state = get_recognition_graph().invoke(state)
    return final_state["response"]


def invoke_rag_workflow(*, canonical_name: str, question: str) -> dict:
    trace_id = generate_trace_id("rag")
    state: RagWorkflowState = {
        "request_type": "rag",
        "canonical_name": canonical_name,
        "question": question,
        "trace_id": trace_id,
        "workflow_trace": [],
    }
    final_state = get_rag_graph().invoke(state)
    return final_state["response"]
