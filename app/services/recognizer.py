from __future__ import annotations

import sys
import time
import tomllib
from pathlib import Path

from app.services.recognition_parser import parse_recognition_result
from app.services.verifier import DrugCatalogVerifier, normalize_name


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
CONFIG_PATH = SRC_DIR / "config.toml"
PROMPT_PATH = SRC_DIR / "prompt.md"
VIDEO_DIR = ROOT / "video"


def load_config() -> dict:
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_server_functions():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from server import call_vlm, encode_frame, load_prompt, sample_frames  # noqa: E402

    return sample_frames, encode_frame, load_prompt, call_vlm


def compare_expected(expected_drug_name: str | None, canonical_name: str, verify_status: str) -> dict | None:
    if not expected_drug_name:
        return None

    expected = expected_drug_name.strip()
    if not expected:
        return None

    if not canonical_name or not verify_status.startswith("verified"):
        return {
            "expected_drug_name": expected,
            "status": "review_required",
            "reason": "当前识别结果未通过标准名强校验，不能直接做一致性确认。",
        }

    if normalize_name(expected) == normalize_name(canonical_name):
        return {
            "expected_drug_name": expected,
            "status": "match",
            "reason": "识别后的标准药名与期望药名一致。",
        }

    return {
        "expected_drug_name": expected,
        "status": "mismatch",
        "reason": f"识别后的标准药名为 {canonical_name}，与期望药名不一致。",
    }


def list_videos() -> list[str]:
    return sorted(p.name for p in VIDEO_DIR.glob("*.mp4"))


def recognize_video(
    video_path: str,
    model: str = "qwen3-vl-8b-instruct-awq-4bit",
    knowledge: bool = True,
    guide: bool = True,
    expected_drug_name: str | None = None,
) -> dict:
    cfg = load_config()
    api_cfg = cfg["api"][model]
    sample_frames, encode_frame, load_prompt, call_vlm = load_server_functions()

    frames = sample_frames(video_path, cfg["video"]["fps"], cfg["video"]["max_frames"])
    urls = [encode_frame(frame, **cfg.get("image", {})) for frame in frames]
    prompt = load_prompt(PROMPT_PATH, guide=guide, kb=knowledge)

    t0 = time.perf_counter()
    raw_result = call_vlm(urls, prompt, **api_cfg)
    elapsed = round(time.perf_counter() - t0, 3)

    parsed = parse_recognition_result(raw_result)
    verification = DrugCatalogVerifier().verify(parsed["raw_name"], parsed["evidence_text"])
    expected_check = compare_expected(expected_drug_name, verification["canonical_name"], verification["status"])

    output = {
        "video_name": Path(video_path).name,
        "model": model,
        "elapsed": elapsed,
        "result": raw_result,
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

    if expected_check:
        output["expected_check"] = expected_check

    return output
