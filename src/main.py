"""
Usage:
    python main.py -video <path> -model <api_preset> [-knowledge on|off] [-guide on|off]

Examples:
    python main.py -video ../video/v1.mp4 -model qwen3-vl-flash
    python main.py -video ../video/v1.mp4 -model qwen3-vl-plus -knowledge off -guide off
"""
import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.recognition_parser import parse_recognition_result
from app.services.verifier import DrugCatalogVerifier
from server import *


def main():
    p = argparse.ArgumentParser(description="Drug label recognition via VLM")
    p.add_argument("-video", required=True, help="video path")
    p.add_argument("-model", required=True, help="API preset name (matches [api.xxx] in config.toml)")
    p.add_argument("-knowledge", default="on", choices=["on", "off"], help="knowledge base: on (default) or off")
    p.add_argument("-guide", default="on", choices=["on", "off"], help="recognition guide: on (default) or off")
    args = p.parse_args()

    base = Path(__file__).parent
    cfg = tomllib.loads((base / "config.toml").read_text("utf-8"))
    api_cfg = cfg["api"][args.model]

    # 1. sample frames
    frames = sample_frames(args.video, cfg["video"]["fps"], cfg["video"]["max_frames"])
    # 3. encode to base64
    urls = [encode_frame(f, **cfg.get("image", {})) for f in frames]
    # 4. load prompt
    prompt = load_prompt(base / "prompt.md", guide=(args.guide == "on"), kb=(args.knowledge == "on"))
    # 5. call VLM
    t0 = time.perf_counter()
    result = call_vlm(urls, prompt, **api_cfg)
    elapsed = round(time.perf_counter() - t0, 3)

    parsed = parse_recognition_result(result)
    verifier = DrugCatalogVerifier()
    verification = verifier.verify(parsed["raw_name"], parsed["evidence_text"])

    output = {
        "result": result,
        "elapsed": elapsed,
        "raw_name": parsed["raw_name"],
        "evidence_text": parsed["evidence_text"],
        "uncertainty_text": parsed["uncertainty_text"],
        "uncertainty_level": parsed["uncertainty_level"],
        "canonical_name": verification["canonical_name"],
        "verify_status": verification["status"],
        "verify_match_type": verification["match_type"],
        "verify_reason": verification["reason"],
    }

    if verification.get("candidate_name"):
        output["candidate_name"] = verification["candidate_name"]

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
