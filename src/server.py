import re
import base64
import cv2
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
