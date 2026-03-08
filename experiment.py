import sys
import time
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.config import Config
from server.ingestion import crop_background, video_downsample

import os
from concurrent.futures import ThreadPoolExecutor

VIDEO_DIR = _ROOT / "video"

def run_one(video_path: Path, mode: str, cfg: Config) -> str:
    frames = video_downsample(str(video_path), cfg.target_fps)
    frames = frames[: cfg.max_images]

    if cfg.enable_crop_background:
        def _one(arr):
            return crop_background(arr, border=cfg.crop_border, k=cfg.crop_k,
                                   area_ratio=cfg.crop_area_ratio, pad_ratio=cfg.crop_pad_ratio)
        if len(frames) <= 1:
            frames = [_one(frames[0])] if frames else []
        else:
            max_workers = min(len(frames), 8, (os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                frames = list(ex.map(_one, frames))

    if mode == "vlm":
        from server.ocr.ocr_vlm import vlm_ocr
        return vlm_ocr(frames, cfg)
    elif mode == "paddle":
        from server.ocr.ocr_paddle import paddle_ocr
        return paddle_ocr(frames)
    else:
        from server.ocr.ocr_tesseract import tesseract_ocr
        return tesseract_ocr(frames, lang=cfg.tesseract_lang, psm=cfg.tesseract_psm,
                             oem=cfg.tesseract_oem, tesseract_cmd=cfg.tesseract_cmd,
                             extra_config=cfg.tesseract_extra_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vlm", "paddle", "tesseract"], default="vlm")
    args = parser.parse_args()

    cfg = Config()
    log_path = _ROOT / f"experiment_log_{args.mode}.txt"
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    total = len(videos)
    print(f"共 {total} 个视频，模式: {args.mode}，日志写入 {log_path}")

    with log_path.open("w", encoding="utf-8") as log:
        for i, video_path in enumerate(videos, 1):
            header = f"[{i:03d}/{total}] {video_path.name}"
            print(header)
            log.write(header + "\n")

            t0 = time.perf_counter()
            result = run_one(video_path, args.mode, cfg)
            elapsed = time.perf_counter() - t0

            body = f"{result}\n用时: {elapsed:.3f}s\n"
            print(body)
            log.write(body + "\n")

    print(f"完成，日志已保存至 {log_path}")


if __name__ == "__main__":
    main()
