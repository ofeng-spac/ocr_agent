from __future__ import annotations

from typing import Literal, TypedDict


class RecognitionWorkflowState(TypedDict, total=False):
    request_type: Literal["recognize", "verify"]
    video_path: str
    video_name: str
    model: str
    knowledge: bool
    guide: bool
    expected_drug_name: str | None
    trace_id: str
    workflow_trace: list[dict]
    recognition_result: dict
    response: dict


class RagWorkflowState(TypedDict, total=False):
    request_type: Literal["rag"]
    canonical_name: str
    question: str
    trace_id: str
    workflow_trace: list[dict]
    rag_result: dict
    response: dict
