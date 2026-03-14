import re
import base64
import cv2
import numpy as np
import openai
from pathlib import Path


def sample_frames(video_path, fps=5, max_frames=14):
    """Sample frames from video at target fps, return list of RGB frames."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    interval = max(1, round(src_fps / fps))
    frames, idx = [], 0
    while len(frames) < max_frames:
        ret, bgr = cap.read()
        if not ret or bgr is None:
            break
        if idx % interval == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def crop_background(img, border=10, k=3.2, area_ratio=0.02, pad_ratio=0.06):
    """Crop solid-color background using border pixel statistics, keep foreground."""
    img = np.ascontiguousarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    b = max(1, min(border, h // 4, w // 4))
    bg = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(),
                         gray[:, :b].ravel(), gray[:, -b:].ravel()])
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


def encode_frame(frame, max_side=1280, quality=85):
    """Encode RGB frame to JPEG, return base64 data URL."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def load_prompt(prompt_path, guide=True, kb=True, cot=True):
    """Load prompt, split by ## headings, selectively join based on flags."""
    text = Path(prompt_path).read_text(encoding="utf-8").strip()
    sections = re.split(r'(?=^## )', text, flags=re.MULTILINE)
    skip = set()
    if not guide:
        skip.update(["识别规则", "安全准则"])
    if not kb:
        skip.add("知识库")
    parts = []
    for s in sections:
        title = s.split('\n', 1)[0].strip().lstrip('#').strip()
        title_clean = re.sub(r'[（(].+?[）)]', '', title).strip()
        if title_clean not in skip:
            if not cot and title_clean == "输出格式":
                # strip CoT lines, keep only 药品名称 line
                lines = s.splitlines()
                s = "\n".join(
                    l for l in lines
                    if not re.match(r'^(关键信息摘录|不确定性)[：:]', l)
                )
            parts.append(s.strip())
    return "\n\n".join(parts)


def call_vlm(urls, prompt, api_key, base_url, model, max_tokens=8192):
    """Send images + prompt to VLM API, return raw response text."""
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    content = [{"type": "text", "text": prompt}]
    for u in urls:
        content.append({"type": "image_url", "image_url": {"url": u}})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()
