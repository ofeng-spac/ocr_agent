import io
from pathlib import Path
import cv2
import base64
import numpy as np
from PIL import Image, ImageOps
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

def video_downsample(video_path: str, target_fps: int) -> List[np.ndarray]:
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    if not Path(video_path).expanduser().exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    interval = max(1, int(round(fps / float(target_fps))) if fps > 0 else 1)

    frames: List[np.ndarray] = []
    frame_idx = 0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            frame_idx += 1
    finally:
        cap.release()
    return frames

def crop_background(img: np.ndarray, border: int, k: float, area_ratio: float, pad_ratio: float) -> np.ndarray:
    if img is None or img.size == 0:
        return img

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    b = max(1, min(border, h // 4, w // 4))
    top = gray[:b, :]
    bottom = gray[(h - b):, :]
    left = gray[:, :b]
    right = gray[:, (w - b):]
    bg = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
    
    bg_med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - bg_med))) + 1e-6
    bg_sigma = 1.4826 * mad
    thr = bg_med + k * bg_sigma

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = (blur >= thr).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float32)
    max_idx = int(np.argmax(areas))
    max_area = float(areas[max_idx])

    if max_area < h * w * area_ratio:
        return img

    x, y, ww, hh = cv2.boundingRect(contours[max_idx])
    pad = int(round(pad_ratio * max(ww, hh)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + ww + pad)
    y1 = min(h, y + hh + pad)

    return img[y0:y1, x0:x1]

ImgLike = Union[Image.Image, np.ndarray]


def to_urls(images: List[ImgLike], max_side: int, jpeg_quality: int) -> List[str]:
    def _encode_ndarray_to_data_url(arr: np.ndarray) -> str:
        if arr.ndim == 2:
            img = arr
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported ndarray shape: {arr.shape!r}")

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
        img = np.ascontiguousarray(img)

        h, w = img.shape[:2]
        if max_side and max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = round(w * scale), round(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        q = int(jpeg_quality)
        q = 1 if q < 1 else 100 if q > 100 else q
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise ValueError("Failed to encode jpeg")

        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _encode_pil_to_data_url(pil_img: Image.Image) -> str:
        pil = ImageOps.exif_transpose(pil_img)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")

        w, h = pil.size
        if max_side and max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil = pil.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)

        with io.BytesIO() as out:
            pil.save(out, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
            buf = out.getvalue()

        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _one(img: ImgLike) -> str:
        if isinstance(img, np.ndarray):
            return _encode_ndarray_to_data_url(img)
        return _encode_pil_to_data_url(img)

    if len(images) <= 1:
        return [_one(images[0])] if images else []

    max_workers = min(len(images), 8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, images))
