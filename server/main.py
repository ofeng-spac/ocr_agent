"""
Single-video CLI tool.

Usage:
    python server/main.py <video_path> [--mode vlm|paddle|tesseract]
"""

import sys
import time
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.config import Config
from experiments.runner import run_one


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-video drug label recognition")
    parser.add_argument("video", help="path to video file")
    parser.add_argument("--mode", choices=["vlm", "paddle", "tesseract"], default="vlm")
    args = parser.parse_args()

    cfg = Config()

    t0 = time.perf_counter()
    result = run_one(args.video, args.mode, cfg)
    elapsed = time.perf_counter() - t0

    print(result)
    print(f"elapsed: {elapsed:.3f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
