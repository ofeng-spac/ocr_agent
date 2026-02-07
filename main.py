import sys
import time
import numpy as np
from PIL import Image

from config import Config
from server.ingestion import crop_background, video_downsample
from ocr.vlm_ocr import vlm_ocr

def main() -> int:
    cfg = Config()
    t0 = time.perf_counter()
    frames = video_downsample(cfg.video_path, cfg.target_fps)
    frames = frames[: cfg.max_images]

    if cfg.enable_crop_background:
        processed: list[Image.Image] = []
        for im in frames:
            arr = np.asarray(im)
            cropped = crop_background(
                arr,
                border=cfg.crop_border,
                k=cfg.crop_k,
                area_ratio=cfg.crop_area_ratio,
                pad_ratio=cfg.crop_pad_ratio,
            )
            processed.append(Image.fromarray(cropped))
        frames = processed

    result = vlm_ocr(frames, cfg)
    print(result)
    t1 = time.perf_counter()
    print(f"用时: {t1 - t0:.3f}s", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
