from __future__ import annotations

import asyncio
import base64
import re
import time
import tomllib
from pathlib import Path

import cv2
import numpy as np
import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parent
VIDEO_DIR = ROOT_DIR / "video"
CONFIG_PATH = BACKEND_DIR / "config.toml"
PROMPT_PATH = BACKEND_DIR / "prompt.md"
DEMO_MODEL = "qwen3-vl-8b-instruct-awq-4bit"


class RecognizeRequest(BaseModel):
    video_name: str


def sample_frames(video_path: str, fps: int = 5, max_frames: int = 14):
    """Sample frames from video at target fps, return list of RGB frames."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    interval = max(1, round(src_fps / fps))
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, bgr = cap.read()
        if not ret or bgr is None:
            break
        if idx % interval == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def crop_background(img, border: int = 10, k: float = 3.2, area_ratio: float = 0.02, pad_ratio: float = 0.06):
    """Crop solid-color background using border pixel statistics, keep foreground."""
    img = np.ascontiguousarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    b = max(1, min(border, h // 4, w // 4))
    bg = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med))) + 1e-6
    thr = med + k * 1.4826 * mad

    mask = (cv2.GaussianBlur(gray, (5, 5), 0) >= thr).astype(np.uint8)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < h * w * area_ratio:
        return img
    x, y, ww, hh = cv2.boundingRect(c)
    pad = round(pad_ratio * max(ww, hh))
    return img[max(0, y - pad):min(h, y + hh + pad), max(0, x - pad):min(w, x + ww + pad)]


def encode_frame(frame, max_side: int = 1280, quality: int = 85):
    """Encode RGB frame to JPEG, return base64 data URL."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("failed to encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def load_prompt(prompt_path: Path, guide: bool = True, kb: bool = True, cot: bool = True):
    """Load prompt, split by ## headings, selectively join based on flags."""
    text = prompt_path.read_text(encoding="utf-8").strip()
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    skip = set()
    if not guide:
        skip.update(["识别规则", "安全准则"])
    if not kb:
        skip.add("知识库")

    parts = []
    for section in sections:
        title = section.split("\n", 1)[0].strip().lstrip("#").strip()
        title_clean = re.sub(r"[（(].+?[）)]", "", title).strip()
        if title_clean in skip:
            continue
        if not cot and title_clean == "输出格式":
            lines = section.splitlines()
            section = "\n".join(l for l in lines if not re.match(r"^(关键信息摘录|不确定性)[：:]", l))
        parts.append(section.strip())

    return "\n\n".join(parts)


def call_vlm(urls, prompt: str, api_key: str, base_url: str, model: str, max_tokens: int = 8192):
    """Send images + prompt to VLM API, return raw response text."""
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    content = [{"type": "text", "text": prompt}]
    for url in urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def _load_config():
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _list_videos():
    return sorted(p.name for p in VIDEO_DIR.glob("*.mp4"))


def _extract_drug_name(result: str) -> str:
    matches = re.findall(r"药品名称[：:]\s*(.+)", result or "")
    return matches[-1].strip() if matches else ""


def _recognize_video(video_name: str):
    if "/" in video_name or "\\" in video_name:
        raise HTTPException(status_code=400, detail="video_name must be a file name, not a path")

    video_path = VIDEO_DIR / video_name
    if not video_path.exists() or video_path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail=f"Video not found: {video_name}")

    cfg = _load_config()
    api_cfg = cfg.get("api", {}).get(DEMO_MODEL)
    if not api_cfg:
        raise HTTPException(status_code=500, detail=f"Missing API preset: {DEMO_MODEL}")

    frames = sample_frames(str(video_path), cfg["video"]["fps"], cfg["video"]["max_frames"])
    if not frames:
        raise HTTPException(status_code=400, detail=f"Cannot sample frames from: {video_name}")

    crop = cfg.get("crop")
    if crop:
        frames = [crop_background(frame, **crop) for frame in frames]

    urls = [encode_frame(frame, **cfg.get("image", {})) for frame in frames]
    prompt = load_prompt(PROMPT_PATH, guide=True, kb=True, cot=True)

    t0 = time.perf_counter()
    result = call_vlm(urls, prompt, **api_cfg)
    elapsed = round(time.perf_counter() - t0, 3)

    return {
        "video_name": video_name,
        "model": DEMO_MODEL,
        "elapsed": elapsed,
        "drug_name": _extract_drug_name(result),
        "raw_result": result,
    }


app = FastAPI(title="Kestrel Demo API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")


@app.get("/api/health")
def health():
    return {"ok": True, "model": DEMO_MODEL}


@app.get("/api/videos")
def list_videos():
    videos = _list_videos()
    return {
        "count": len(videos),
        "videos": [{"name": name, "url": f"/videos/{name}"} for name in videos],
    }


@app.post("/api/recognize")
async def recognize_video(req: RecognizeRequest):
    return await asyncio.to_thread(_recognize_video, req.video_name)

