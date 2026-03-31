"""
Simple grouped experiment runner.

Usage:
    python run.py <model_idx_hex>   # run 8 text configs for one model, write one log
    python run.py --analyze          # analyze logs in logs/ and write logs/analysis.md

Examples:
    python run.py 0
    python run.py 1
    python run.py a
"""

import csv
import re
import sys
import time
import tomllib
from pathlib import Path

from server import call_vlm, crop_background, encode_frame, load_prompt, sample_frames


CONFIGS = [
    {"name": "baseline", "kb": False, "guide": False, "cot": False},
    {"name": "kb", "kb": True, "guide": False, "cot": False},
    {"name": "guide", "kb": False, "guide": True, "cot": False},
    {"name": "cot", "kb": False, "guide": False, "cot": True},
    {"name": "kb+guide", "kb": True, "guide": True, "cot": False},
    {"name": "kb+cot", "kb": True, "guide": False, "cot": True},
    {"name": "guide+cot", "kb": False, "guide": True, "cot": True},
    {"name": "kb+guide+cot", "kb": True, "guide": True, "cot": True},
]

CONFIG_ORDER = [c["name"] for c in CONFIGS]

UNKNOWN_VALUES = {
    "",
    "未知",
    "无法识别",
    "不确定",
    "unknown",
    "Unknown",
    "UNKNOWN",
    "n/a",
    "N/A",
}

RE_META = re.compile(r"^(config|config_label|code|model|knowledge|guide|cot|total):\s*(.+)$")
RE_HEADER = re.compile(r"^\[(\d+)/(\d+)\]\s+(\S+)\s+\|\s+(.+)$")
RE_TIME = re.compile(r"^time:\s*([\d.]+)s$")
RE_RESULT_EN = re.compile(r"^result:\s*(correct|wrong)$", re.IGNORECASE)
RE_RESULT_ZH = re.compile(r"^结果[：:]\s*(正确|错误)$")


def _output(text: str, log_file: Path) -> None:
    print(text)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _extract_drug_name(text: str) -> str:
    patterns = [
        r"药品名称[：:]\s*(.+)",
        r"Drug\s*Name[：:]\s*(.+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            name = matches[-1].strip()
            if name:
                return name
    return ""


def _entry_category(is_correct: bool, predicted: str) -> str:
    if is_correct:
        return "correct"
    if predicted in UNKNOWN_VALUES:
        return "unknown"
    return "misid"


def _group_stats(entries):
    n = len(entries)
    correct = unknown = misid = 0
    for is_correct, predicted, *_ in entries:
        cat = _entry_category(is_correct, predicted)
        if cat == "correct":
            correct += 1
        elif cat == "unknown":
            unknown += 1
        else:
            misid += 1
    return n, correct, unknown, misid


def _pretty_model(name: str) -> str:
    parts = name.split("-")
    out = []
    for p in parts:
        low = p.lower()
        if low.startswith("qwen"):
            out.append("Qwen" + p[4:])
        elif low == "vl":
            out.append("VL")
        elif low == "awq":
            out.append("AWQ")
        elif low == "instruct":
            out.append("Instruct")
        elif re.fullmatch(r"\d+b", low):
            out.append(p[:-1] + "B")
        else:
            out.append(p)
    return "-".join(out)


def _cfg_label_from_code(code: str) -> str:
    bits = code[1:] if len(code) == 4 else code
    mapping = {
        "000": "baseline",
        "100": "kb",
        "010": "guide",
        "001": "cot",
        "110": "kb+guide",
        "101": "kb+cot",
        "011": "guide+cot",
        "111": "kb+guide+cot",
    }
    return mapping.get(bits, bits)


def _parse_log_experiments(log_path: Path):
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    experiments = []
    current_meta = None
    current_entries = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("config:") or line.startswith("code:"):
            if current_meta and current_entries:
                experiments.append((current_meta, current_entries))
            current_meta = {}
            current_entries = []
            if line.startswith("config:"):
                current_meta["config"] = line.split(":", 1)[1].strip()
            else:
                code = line.split(":", 1)[1].strip()
                current_meta["code"] = code
                current_meta["config"] = _cfg_label_from_code(code)
            i += 1
            continue

        if current_meta is None:
            i += 1
            continue

        mm = RE_META.match(line)
        if mm:
            k = mm.group(1)
            v = mm.group(2).strip()
            if k == "config_label" and "config" not in current_meta:
                current_meta["config"] = v
            else:
                current_meta[k] = v
            if k == "code" and "config" not in current_meta:
                current_meta["config"] = _cfg_label_from_code(v)
            i += 1
            continue

        mh = RE_HEADER.match(line)
        if mh:
            ground_truth = mh.group(4).strip()
            predicted = ""
            is_correct = None
            elapsed = None

            j = i + 1
            chunk = []
            while j < len(lines):
                nxt = lines[j].strip()
                if RE_HEADER.match(nxt) or nxt.startswith("config:") or nxt.startswith("code:"):
                    break
                chunk.append(nxt)
                j += 1

            for c in chunk:
                if c:
                    ext = _extract_drug_name(c)
                    if ext:
                        predicted = ext

                mt = RE_TIME.match(c)
                if mt:
                    elapsed = float(mt.group(1))

                mr_en = RE_RESULT_EN.match(c)
                if mr_en:
                    is_correct = mr_en.group(1).lower() == "correct"

                mr_zh = RE_RESULT_ZH.match(c)
                if mr_zh:
                    is_correct = mr_zh.group(1) == "正确"

            if is_correct is None and predicted:
                is_correct = predicted == ground_truth

            if is_correct is not None:
                current_entries.append((is_correct, predicted, elapsed, ground_truth))

            i = j
            continue

        i += 1

    if current_meta and current_entries:
        experiments.append((current_meta, current_entries))

    return experiments


def analyze_all() -> None:
    base = Path(__file__).parent
    log_dir = base.parent / "logs"
    log_files = sorted(log_dir.glob("*.txt"))

    if not log_files:
        print("No log files found in logs/")
        return

    rows = []
    for lf in log_files:
        for meta, entries in _parse_log_experiments(lf):
            times = [e[2] for e in entries if e[2] is not None]
            config = meta.get("config")
            if not config and meta.get("code"):
                config = _cfg_label_from_code(meta["code"])

            rows.append(
                {
                    "model": meta.get("model", "?"),
                    "config": config or "?",
                    "stats": _group_stats(entries),
                    "times": times,
                }
            )

    if not rows:
        print("No parseable entries found in logs/")
        return

    lookup = {}
    for r in rows:
        lookup[(r["model"], r["config"])] = r

    all_models = list(dict.fromkeys(r["model"] for r in rows))
    model_labels = [_pretty_model(m) for m in all_models]
    n_total = next((r["stats"][0] for r in rows if r["stats"][0] > 0), 0)

    out_md = log_dir / "analysis.md"
    lines = []
    lines.append("# Ablation Results")
    lines.append("")
    if n_total:
        lines.append(f"n={n_total}, metrics = correct% / unknown% / misid%")
    else:
        lines.append("metrics = correct% / unknown% / misid%")
    lines.append("")

    header = "| Config | " + " | ".join(model_labels) + " |"
    sep = "|:---:|" + ":---:|" * len(model_labels)
    lines.append(header)
    lines.append(sep)

    for cfg in CONFIG_ORDER:
        cells = []
        for model in all_models:
            r = lookup.get((model, cfg))
            if r is None:
                cells.append("-")
            else:
                n, c, u, m = r["stats"]
                if n == 0:
                    cells.append("-")
                else:
                    cells.append(f"{c/n*100:.0f} / {u/n*100:.0f} / {m/n*100:.0f}")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Counts (correct / unknown / misid)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for cfg in CONFIG_ORDER:
        cells = []
        for model in all_models:
            r = lookup.get((model, cfg))
            if r is None:
                cells.append("-")
            else:
                _, c, u, m = r["stats"]
                cells.append(f"{c} / {u} / {m}")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Avg Response Time (s)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for cfg in CONFIG_ORDER:
        cells = []
        for model in all_models:
            r = lookup.get((model, cfg))
            if r is None or not r["times"]:
                cells.append("-")
            else:
                cells.append(f"{sum(r['times']) / len(r['times']):.2f}")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analysis saved to {out_md}")


def run_model_group(model_idx: int) -> None:
    base = Path(__file__).parent
    cfg = tomllib.loads((base / "config.toml").read_text(encoding="utf-8"))

    models = cfg["models"]
    if model_idx >= len(models):
        print(f"Error: model index {model_idx} out of range (have {len(models)} models)")
        sys.exit(1)

    model_name = models[model_idx]
    pretty_model_name = _pretty_model(model_name)
    api_cfg = cfg["api"][model_name]

    video_dir = base.parent / "video"
    csv_path = video_dir / "video.csv"
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    log_dir = base.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{pretty_model_name}.txt"
    log_path.write_text("", encoding="utf-8")

    total = len(rows)

    _output(f"model: {model_name}", log_path)
    _output(f"model_pretty: {pretty_model_name}", log_path)
    _output(f"total: {total}", log_path)
    _output("", log_path)

    for conf in CONFIGS:
        config_name = conf["name"]
        use_kb = conf["kb"]
        use_guide = conf["guide"]
        use_cot = conf["cot"]

        prompt = load_prompt(base / "prompt.md", guide=use_guide, kb=use_kb, cot=use_cot)

        _output("=" * 48, log_path)
        _output(f"config: {config_name}", log_path)
        _output(f"model: {model_name}", log_path)
        _output(f"knowledge: {'on' if use_kb else 'off'}", log_path)
        _output(f"guide: {'on' if use_guide else 'off'}", log_path)
        _output(f"cot: {'on' if use_cot else 'off'}", log_path)
        _output(f"total: {total}", log_path)
        _output("", log_path)

        times = []
        correct = 0

        for i, row in enumerate(rows, 1):
            video_name, ground_truth = row[0], row[1]
            video_path = str(video_dir / video_name)

            frames = sample_frames(video_path, cfg["video"]["fps"], cfg["video"]["max_frames"])
            crop = cfg.get("crop")
            if crop:
                frames = [crop_background(f, **crop) for f in frames]
            urls = [encode_frame(f, **cfg.get("image", {})) for f in frames]

            t0 = time.perf_counter()
            result = call_vlm(urls, prompt, **api_cfg)
            elapsed = round(time.perf_counter() - t0, 3)
            times.append(elapsed)

            predicted = _extract_drug_name(result)
            is_correct = predicted == ground_truth
            if is_correct:
                correct += 1

            _output(f"[{i:03d}/{total}] {video_name} | {ground_truth}", log_path)
            _output(result, log_path)
            _output(f"time: {elapsed}s", log_path)
            _output(f"result: {'correct' if is_correct else 'wrong'}", log_path)
            _output("", log_path)

        avg_time = round(sum(times) / len(times), 3) if times else 0
        accuracy = round(correct / total * 100, 1) if total else 0
        _output(f"correct: {correct}/{total} ({accuracy}%)", log_path)
        _output(f"avg_time: {avg_time}s", log_path)
        _output(f"min_time: {min(times):.3f}s", log_path)
        _output(f"max_time: {max(times):.3f}s", log_path)
        _output("", log_path)

    _output(f"Done. Group log saved to {log_path}", log_path)


def _parse_model_index(arg: str) -> int:
    if len(arg) != 1 or arg.lower() not in "0123456789abcdef":
        raise ValueError("model_idx must be one hex char, e.g. 0, 1, a")
    return int(arg, 16)


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--analyze":
        analyze_all()
        return

    if len(sys.argv) == 2:
        try:
            model_idx = _parse_model_index(sys.argv[1])
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        run_model_group(model_idx)
        return

    print("Usage:")
    print("  python run.py <model_idx_hex>   # run 8 text configs for one model, e.g. 0")
    print("  python run.py --analyze         # analyze logs and compare")
    sys.exit(1)


if __name__ == "__main__":
    main()
