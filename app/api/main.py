from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models.schemas import AskRequest, RecognizeRequest, VerifyRequest
from app.services.audit import append_audit_log, generate_trace_id, list_audit_logs
from app.services.evaluation import load_evaluation_summary
from app.services.rag import LeafletQAService
from app.services.recognizer import VIDEO_DIR, list_videos, recognize_video


app = FastAPI(title="Drug Recognition Agent API", version="0.1.0")
qa_service = LeafletQAService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")


@app.get("/api/health")
def health():
    return {"ok": True, "service": "drug-recognition-agent"}


@app.get("/api/videos")
def get_videos():
    videos = list_videos()
    return {
        "count": len(videos),
        "videos": [{"name": name, "url": f"/videos/{name}"} for name in videos],
    }


@app.post("/api/recognize")
async def recognize(req: RecognizeRequest):
    video_name = req.video_name.strip()
    if "/" in video_name or "\\" in video_name:
        raise HTTPException(status_code=400, detail="video_name must be a file name, not a path")

    video_path = VIDEO_DIR / video_name
    if not video_path.exists() or video_path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail=f"Video not found: {video_name}")

    try:
        result = await asyncio.to_thread(
            recognize_video,
            str(video_path),
            req.model,
            req.knowledge,
            req.guide,
            req.expected_drug_name,
        )
        trace_id = generate_trace_id("recognize")
        result["trace_id"] = trace_id
        append_audit_log("recognize", result, trace_id=trace_id)
        return result
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown model preset: {req.model}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/verify")
async def verify(req: VerifyRequest):
    video_name = req.video_name.strip()
    if "/" in video_name or "\\" in video_name:
        raise HTTPException(status_code=400, detail="video_name must be a file name, not a path")

    expected_drug_name = req.expected_drug_name.strip()
    if not expected_drug_name:
        raise HTTPException(status_code=400, detail="expected_drug_name must not be empty")

    video_path = VIDEO_DIR / video_name
    if not video_path.exists() or video_path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail=f"Video not found: {video_name}")

    try:
        result = await asyncio.to_thread(
            recognize_video,
            str(video_path),
            req.model,
            req.knowledge,
            req.guide,
            expected_drug_name,
        )
        response = {
            "video_name": result["video_name"],
            "model": result["model"],
            "elapsed": result["elapsed"],
            "raw_name": result["raw_name"],
            "canonical_name": result["canonical_name"],
            "verify_status": result["verify_status"],
            "verify_match_type": result["verify_match_type"],
            "verify_reason": result["verify_reason"],
            "expected_check": result.get("expected_check"),
            "result": result["result"],
        }
        trace_id = generate_trace_id("verify")
        response["trace_id"] = trace_id
        append_audit_log("verify", response, trace_id=trace_id)
        return response
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown model preset: {req.model}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/rag/ask")
def ask_leaflet(req: AskRequest):
    result = qa_service.ask(req.canonical_name, req.question)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["reason"])
    trace_id = generate_trace_id("rag")
    result["trace_id"] = trace_id
    append_audit_log(
        "rag_ask",
        {
            "canonical_name": req.canonical_name,
            "question": req.question,
            **result,
        },
        trace_id=trace_id,
    )
    return result


@app.get("/api/audit_logs")
def get_audit_logs(limit: int = 20):
    logs = list_audit_logs(limit=limit)
    return {
        "count": len(logs),
        "logs": logs,
    }


@app.get("/api/eval/summary")
def get_eval_summary():
    return load_evaluation_summary()
