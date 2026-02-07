import pytesseract
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import shutil
from typing import Sequence

from server.ingestion import ImgLike

_CMD_CACHE: dict[str, str] = {}

def tesseract_ocr(images: Sequence[ImgLike], *, lang: str = "chi_sim+eng", psm=None, oem=None, tesseract_cmd=None, extra_config=None) -> str:
    cache_key = str(tesseract_cmd) if tesseract_cmd else ""
    cmd = _CMD_CACHE.get(cache_key)
    if cmd is None:
        if tesseract_cmd:
            tcmd = str(tesseract_cmd)
            if Path(tcmd).exists():
                cmd = tcmd
            else:
                cmd = shutil.which(tcmd) or shutil.which("tesseract")
                if not cmd:
                    raise FileNotFoundError(f"tesseract_cmd not found: {tesseract_cmd!r}")
        else:
            cmd = shutil.which("tesseract")
            if not cmd:
                raise FileNotFoundError('tesseract not found. Set Config.tesseract_cmd to full path (e.g. "...\\\\scoop\\\\shims\\\\tesseract.exe") or add it to PATH.')
        _CMD_CACHE[cache_key] = cmd

    pytesseract.pytesseract.tesseract_cmd = cmd

    config_parts: list[str] = []
    if oem is not None:
        config_parts.extend(["--oem", str(int(oem))])
    if psm is not None:
        config_parts.extend(["--psm", str(int(psm))])
    if extra_config:
        config_parts.append(str(extra_config).strip())
    config_str = " ".join(config_parts)

    blocks: list[str] = []
    for i, im in enumerate(images):
        if isinstance(im, np.ndarray):
            arr = im
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
            if arr.ndim == 3 and arr.shape[2] == 3:
                pil = Image.fromarray(np.ascontiguousarray(arr), mode="RGB")
            elif arr.ndim == 2:
                pil = Image.fromarray(np.ascontiguousarray(arr), mode="L")
            else:
                raise ValueError(f"Unsupported ndarray shape: {arr.shape!r}")
        else:
            pil = ImageOps.exif_transpose(im)

        if pil.mode != "L":
            pil = pil.convert("L")

        text = pytesseract.image_to_string(pil, lang=lang, config=config_str)
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        blocks.append(f"[{i}] " + (" | ".join(lines) if lines else "(empty)"))

    return "\n".join(blocks)
