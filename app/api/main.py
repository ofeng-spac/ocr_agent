from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models.schemas import RecognizeRequest
from app.services.recognizer import VIDEO_DIR, list_videos, recognize_video


app = FastAPI(title="Drug Recognition Agent API", version="0.1.0")

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
        return await asyncio.to_thread(
            recognize_video,
            str(video_path),
            req.model,
            req.knowledge,
            req.guide,
            req.expected_drug_name,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown model preset: {req.model}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
