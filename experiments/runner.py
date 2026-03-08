"""
Shared experiment runner: video preprocessing and OCR inference.
Centralizes logic shared between exp_comparison.py and server/main.py.
"""
import re
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.config import Config
from server.ingestion import ImgLike, crop_background, video_downsample


def strip_kb(prompt: str) -> str:
    """
    Remove the knowledge-base constraint block from a prompt string.
    Used for the ablation experiment (open vs. constrained condition).

    Removes everything from the output-rules section up to (not including)
    the output-format section, then relaxes the drug-name field description.
    """
    # Remove the KB constraint block (including trailing blank lines)
    prompt = re.sub(
        r"药品名称输出规则（强制执行）：.*?(?=输出格式)",
        "",
        prompt,
        flags=re.DOTALL,
    )
    # Relax the output-format field description
    prompt = prompt.replace(
        "药品名称：<从合法列表中选择的完整条目>",
        "药品名称：<从图片中识别出的完整药品名称>",
    )
    return prompt.strip()


def preprocess(video_path: str, cfg: Config) -> List[np.ndarray]:
    """Extract frames from a video and apply background cropping."""
    frames = video_downsample(video_path, cfg.target_fps)
    frames = frames[: cfg.max_images]

    if cfg.enable_crop_background and frames:
        def _one(arr: np.ndarray) -> np.ndarray:
            return crop_background(
                arr,
                border=cfg.crop_border,
                k=cfg.crop_k,
                area_ratio=cfg.crop_area_ratio,
                pad_ratio=cfg.crop_pad_ratio,
            )

        if len(frames) == 1:
            frames = [_one(frames[0])]
        else:
            max_workers = min(len(frames), 8, os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                frames = list(ex.map(_one, frames))

    return frames


def run_one(video_path: str, mode: str, cfg: Config) -> str:
    """Preprocess a video and run OCR in the specified mode. Returns the result string."""
    frames = preprocess(video_path, cfg)

    if mode == "vlm":
        from server.ocr.ocr_vlm import vlm_ocr
        return vlm_ocr(frames, cfg)
    elif mode == "paddle":
        from server.ocr.ocr_paddle import paddle_ocr
        return paddle_ocr(frames)
    elif mode == "tesseract":
        from server.ocr.ocr_tesseract import tesseract_ocr
        return tesseract_ocr(
            frames,
            lang=cfg.tesseract_lang,
            psm=cfg.tesseract_psm,
            oem=cfg.tesseract_oem,
            tesseract_cmd=cfg.tesseract_cmd,
            extra_config=cfg.tesseract_extra_config,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
