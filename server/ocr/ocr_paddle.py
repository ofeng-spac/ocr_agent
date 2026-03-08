import numpy as np
from PIL import Image, ImageOps
from paddleocr import PaddleOCR  # type: ignore
from typing import Sequence

from server.ingestion import ImgLike

_OCR_CACHE: dict[str, PaddleOCR] = {}

def _to_bgr_uint8(img: ImgLike) -> np.ndarray:
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unsupported ndarray shape: {arr.shape!r}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        rgb = np.ascontiguousarray(arr)
    else:
        pil = ImageOps.exif_transpose(img)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))

    return np.ascontiguousarray(rgb[..., ::-1])

def paddle_ocr(images: Sequence[ImgLike], *, lang: str = "ch") -> str:
    ocr = _OCR_CACHE.get(lang)
    if ocr is None:
        ocr = PaddleOCR(lang=lang, use_angle_cls=True)
        _OCR_CACHE[lang] = ocr
    blocks: list[str] = []

    for i, im in enumerate(images):
        bgr = _to_bgr_uint8(im)
        raw = ocr.ocr(bgr, cls=True)

        if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
            raw = raw[0]

        lines: list[str] = []
        if raw:
            for item in raw:
                if not item or len(item) < 2:
                    continue
                rec = item[1]
                if rec and len(rec) >= 1:
                    lines.append(str(rec[0]))

        blocks.append(f"[{i}] " + (" | ".join(lines) if lines else "(empty)"))

    return "\n".join(blocks)
