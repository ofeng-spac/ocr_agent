"""
Usage:
    python main.py -v <video_path> -a <api_preset> [-k] [-g]

Examples:
    python main.py -v ../video/v1.mp4 -a qwen3-vl-flash
    python main.py -v ../video/v1.mp4 -a qwen3-vl-plus -k -g
"""
import argparse
import json
import time
import tomllib
from pathlib import Path
from server import *


def main():
    p = argparse.ArgumentParser(description="Drug label recognition via VLM")
    p.add_argument("-v", required=True, help="video path")
    p.add_argument("-a", required=True, help="API preset name (matches [api.xxx] in config.toml)")
    p.add_argument("-k", action="store_true", help="disable knowledge base in prompt")
    p.add_argument("-g", action="store_true", help="disable recognition guide and safety rules in prompt")
    args = p.parse_args()

    base = Path(__file__).parent
    cfg = tomllib.loads((base / "config.toml").read_text("utf-8"))
    api_cfg = cfg["api"][args.a]

    # 1. sample frames
    frames = sample_frames(args.v, cfg["video"]["fps"], cfg["video"]["max_frames"])
    # 2. crop background
    crop = cfg.get("crop")
    if crop:
        frames = [crop_background(f, **crop) for f in frames]
    # 3. encode to base64
    urls = [encode_frame(f, **cfg.get("image", {})) for f in frames]
    # 4. load prompt
    prompt = load_prompt(base / "prompt.md", guide=not args.g, kb=not args.k)
    # 5. call VLM
    t0 = time.perf_counter()
    result = call_vlm(urls, prompt, **api_cfg)
    elapsed = round(time.perf_counter() - t0, 3)

    print(json.dumps({"result": result, "elapsed": elapsed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
