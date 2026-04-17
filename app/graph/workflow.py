from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.state import RecognitionWorkflowState
from app.services.audit import append_audit_log, generate_trace_id
from app.services.recognizer import apply_verification, run_recognition


def _append_trace(state: RecognitionWorkflowState, node: str, status: str, summary: str) -> list[dict]:
    trace = list(state.get("workflow_trace", []))
    trace.append({"node": node, "status": status, "summary": summary})
    return trace


def recognize_node(state: RecognitionWorkflowState) -> RecognitionWorkflowState:
    recognition_result = run_recognition(
        video_path=state["video_path"],
        model=state["model"],
        knowledge=state["knowledge"],
        guide=state["guide"],
    )
    trace = _append_trace(
        state,
        "recognize_node",
        "ok",
        f"识别完成，raw_name={recognition_result.get('raw_name') or '未提取'}",
    )
    return {
        "recognition_result": recognition_result,
        "workflow_trace": trace,
    }


def verify_node(state: RecognitionWorkflowState) -> RecognitionWorkflowState:
    response = apply_verification(state["recognition_result"], state.get("expected_drug_name"))
    response["trace_id"] = state["trace_id"]

    summary = (
        f"标准名校验完成，status={response.get('verify_status')}，"
        f"canonical_name={response.get('canonical_name') or '未确认'}"
    )
    trace = _append_trace(state, "verify_node", "ok", summary)
    response["workflow_trace"] = trace
    return {
        "response": response,
        "workflow_trace": trace,
    }


def audit_node(state: RecognitionWorkflowState) -> RecognitionWorkflowState:
    event_type = f"{state['request_type']}_workflow"
    append_audit_log(event_type, state["response"], trace_id=state["trace_id"])
    trace = _append_trace(state, "audit_node", "ok", "审计日志已写入。")
    response = dict(state["response"])
    response["workflow_trace"] = trace
    return {
        "response": response,
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


_GRAPH = None


def get_recognition_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_recognition_graph()
    return _GRAPH


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
