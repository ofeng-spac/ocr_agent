"""
Main comparison experiment: VLM vs PaddleOCR vs Tesseract

Usage:
    python experiments/exp_comparison.py --mode vlm
    python experiments/exp_comparison.py --mode paddle
    python experiments/exp_comparison.py --mode tesseract

Logs are written to logs/comparison_<mode>.txt
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.config import Config
from experiments.runner import run_one

VIDEO_DIR = _ROOT / "video"
LOG_DIR = _ROOT / "logs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Main comparison: VLM vs PaddleOCR vs Tesseract")
    parser.add_argument("--mode", choices=["vlm", "paddle", "tesseract"], default="vlm")
    args = parser.parse_args()

    cfg = Config()
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"comparison_{args.mode}.txt"

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    total = len(videos)
    print(f"{total} videos  mode: {args.mode}")

    with log_path.open("w", encoding="utf-8") as log:
        for i, video_path in enumerate(videos, 1):
            header = f"[{i:03d}/{total}] {video_path.name}"
            print(header)
            log.write(header + "\n")

            t0 = time.perf_counter()
            result = run_one(str(video_path), args.mode, cfg)
            elapsed = time.perf_counter() - t0

            body = f"{result}\n用时: {elapsed:.3f}s\n"
            print(body)
            log.write(body + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
