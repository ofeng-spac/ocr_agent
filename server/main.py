import sys
import time
import argparse
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.config import Config
from server.ingestion import crop_background, video_downsample

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="视频路径")
    parser.add_argument("--mode", choices=["vlm", "paddle", "tesseract"], default="vlm")
    args = parser.parse_args()

    cfg = Config()

    t0 = time.perf_counter()
    frames = video_downsample(args.video, cfg.target_fps)
    frames = frames[: cfg.max_images]

    if cfg.enable_crop_background:
        def _one(arr: np.ndarray) -> np.ndarray:
            return crop_background(arr, border=cfg.crop_border, k=cfg.crop_k, area_ratio=cfg.crop_area_ratio, pad_ratio=cfg.crop_pad_ratio)

        if len(frames) <= 1:
            frames = [_one(frames[0])] if frames else []
        else:
            max_workers = min(len(frames), 8, (os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                frames = list(ex.map(_one, frames))

    def _run_vlm() -> str:
        from server.ocr.ocr_vlm import vlm_ocr
        return vlm_ocr(frames, cfg)

    def _run_paddle() -> str:
        from server.ocr.ocr_paddle import paddle_ocr
        return paddle_ocr(frames)

    def _run_tesseract() -> str:
        from server.ocr.ocr_tesseract import tesseract_ocr
        return tesseract_ocr(frames, lang=getattr(cfg, "tesseract_lang", "chi_sim+eng"), psm=getattr(cfg, "tesseract_psm", None), oem=getattr(cfg, "tesseract_oem", None), tesseract_cmd=getattr(cfg, "tesseract_cmd", None), extra_config=getattr(cfg, "tesseract_extra_config", None))

    runners = {
        "vlm": _run_vlm,
        "paddle": _run_paddle,
        "tesseract": _run_tesseract,
    }
    result = runners[args.mode]()
    print(result)
    t1 = time.perf_counter()
    print(f"用时: {t1 - t0:.3f}s", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
