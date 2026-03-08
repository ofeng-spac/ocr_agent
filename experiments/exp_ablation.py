"""
Ablation experiment: effect of knowledge-base (KB) constraints

Compares two conditions:
  constrained  — standard prompt (with KB list and alias mappings)
  open         — KB constraint block dynamically stripped; everything else identical

Both conditions use the same VLM (Qwen3-VL) and the same video set.
The ERR gap between conditions quantifies the contribution of KB constraints.

Usage:
    python experiments/exp_ablation.py

Logs are written to:
    logs/ablation_open.txt
"""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.config import Config
from server.ocr.ocr_vlm import vlm_ocr
from experiments.runner import preprocess, strip_kb

VIDEO_DIR = _ROOT / "video"
LOG_DIR = _ROOT / "logs"
PROMPT_PATH = _ROOT / "server" / "prompt.md"


def run_group(label: str, use_kb: bool) -> None:
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not use_kb:
        prompt_text = strip_kb(prompt_text)

    cfg = Config()
    cfg.prompt_md_path = ""  # inject prompt directly; do not read from disk

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"ablation_{label}.txt"

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    total = len(videos)
    print(f"group: {label}  kb: {'yes' if use_kb else 'no'}  {total} videos")

    with log_path.open("w", encoding="utf-8") as log:
        for i, video_path in enumerate(videos, 1):
            header = f"[{i:03d}/{total}] {video_path.name}"
            print(header)
            log.write(header + "\n")

            t0 = time.perf_counter()
            frames = preprocess(str(video_path), cfg)
            result = vlm_ocr(frames, cfg, prompt_override=prompt_text)
            elapsed = time.perf_counter() - t0

            body = f"{result}\n用时: {elapsed:.3f}s\n"
            print(body)
            log.write(body + "\n")

    print(f"group [{label}] done.")


def main() -> None:
    run_group("open", use_kb=False)
    print("\nAblation experiment complete.")


if __name__ == "__main__":
    main()
