import sys
import time
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

from config import Config
from server.ingestion import crop_background, video_downsample
from ocr.vlm_ocr import vlm_ocr

def main() -> int:
    cfg = Config()
    t0 = time.perf_counter()
    frames = video_downsample(cfg.video_path, cfg.target_fps)
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

    result = vlm_ocr(frames, cfg)
    print(result)
    t1 = time.perf_counter()
    print(f"用时: {t1 - t0:.3f}s", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
