"""
Batch experiment runner with binary-style encoding.

Code format: <model><knowledge><guide><cot>
  - model:     0-9/a-f (index into config.toml models array)
  - knowledge: 0=off, 1=on
  - guide:     0=off, 1=on
  - cot:       0=off, 1=on  (关键信息摘录 + 不确定性 lines)

Usage:
    python run.py <code>          # run experiment
    python run.py --analyze       # analyze all logs in logs/ and compare

Examples:
    python run.py 0111   → model 0, kb=on,  guide=on,  cot=on  → logs/log_0111.txt
    python run.py 0110   → model 0, kb=on,  guide=on,  cot=off → logs/log_0110.txt
    python run.py 0100   → model 0, kb=on,  guide=off, cot=off → logs/log_0100.txt
    python run.py 1111   → model 1, kb=on,  guide=on,  cot=on  → logs/log_1111.txt
--analyze
Result categories ():
    正确: predicted == ground_truth
    未知: predicted is "未知" / empty  →  safe, triggers human review
    误识: predicted is a specific wrong drug name  →  DANGEROUS
"""
import csv
import re as _re
import sys
import time
import tomllib
from pathlib import Path
from server import *

_UNKNOWN_VALUES = {"未知", "无法识别", "无", ""}

_RE_HEADER = _re.compile(r'^\[(\d+)/(\d+)\] (\S+) \| (.+)$')
_RE_DRUG   = _re.compile(r'^药品名称[：:]\s*(.*)$')
_RE_RESULT = _re.compile(r'^结果：(正确|错误)$')
_RE_META   = _re.compile(r'^(code|model|knowledge|guide|cot):\s*(.+)$')
_RE_TIME   = _re.compile(r'^time: ([\d.]+)s$')


def _extract_drug_name(result):
    """Extract drug name from VLM output by matching '药品名称：xxx' pattern."""
    matches = _re.findall(r'药品名称[：:]\s*(.+)', result)
    return matches[-1].strip() if matches else ""


def _output(text, log_file):
    """Print to console and append to log file simultaneously."""
    print(text)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _parse_log(log_path):
    """Parse log file. Returns (meta, entries).
    meta: dict with code/model/knowledge/guide
    entries: list of (is_correct, predicted, elapsed, ground_truth)
    """
    lines = log_path.read_text(encoding="utf-8").splitlines()
    meta = {}
    entries = []
    i = 0
    while i < len(lines):
        # collect header metadata from first few lines
        mm = _RE_META.match(lines[i])
        if mm:
            meta[mm.group(1)] = mm.group(2).strip()
            i += 1
            continue
        # entry header
        mh = _RE_HEADER.match(lines[i])
        if mh:
            ground_truth = mh.group(4).strip()
            predicted = None
            is_correct = None
            j = i + 1
            while j < len(lines) and not _RE_HEADER.match(lines[j]):
                md = _RE_DRUG.match(lines[j])
                if md:
                    predicted = md.group(1).strip()
                mr = _RE_RESULT.match(lines[j])
                if mr:
                    is_correct = mr.group(1) == "正确"
                j += 1
            elapsed = None
            for line in lines[i+1:j]:
                mt = _RE_TIME.match(line)
                if mt:
                    elapsed = float(mt.group(1))
            if is_correct is not None:
                entries.append((is_correct, predicted or "", elapsed, ground_truth))
            i = j
        else:
            i += 1
    return meta, entries


def _entry_category(is_correct, predicted):
    """Classify one result: 正确 / 未知 / 误识."""
    if is_correct:
        return "正确"
    if predicted in _UNKNOWN_VALUES:
        return "未知"
    return "误识"


def _group_stats(entries):
    """Return (n, correct, unknown, misid) for a list of entries."""
    n = len(entries)
    correct = unknown = misid = 0
    for ok, pred, *_ in entries:
        cat = _entry_category(ok, pred)
        if cat == "正确":
            correct += 1
        elif cat == "未知":
            unknown += 1
        else:
            misid += 1
    return n, correct, unknown, misid


def _pretty_model(name):
    return (name.replace("qwen3", "Qwen3").replace("qwen2.5", "Qwen2.5")
               .replace("-vl-", "-VL-").replace("instruct", "Instruct")
               .replace("awq", "AWQ").replace("4bit", "4bit"))


def _cfg_label(code):
    cfg = code[1:]
    parts = []
    if cfg[0] == "1": parts.append("kb")
    if cfg[1] == "1": parts.append("guide")
    if cfg[2] == "1": parts.append("cot")
    return "+".join(parts) if parts else "baseline"


def analyze_all():

    base = Path(__file__).parent
    log_dir = base.parent / "logs"
    log_files = sorted(log_dir.glob("log????.txt"))

    if not log_files:
        print("No log files found in logs/")
        return

    rows = []
    for lf in log_files:
        meta, entries = _parse_log(lf)
        if not entries:
            continue
        times = [e[2] for e in entries if e[2] is not None]
        model = meta.get("model", "?")
        rows.append({
            "code":  meta.get("code", lf.stem[3:]),
            "model": model,
            "kb":    meta.get("knowledge", "?"),
            "guide": meta.get("guide", "?"),
            "cot":   meta.get("cot", "?"),
            "stats": _group_stats(entries),
            "times": times,
        })

    if not rows:
        print("No parseable entries found.")
        return

    n_total = rows[0]["stats"][0]

    # ── ablation table (消融表) ───────────────────────────────────────────────
    # cfg order: canonical 2³ order (kb/guide/cot bits)
    CFG_ORDER = [
        "baseline",
        "kb", "guide", "cot",
        "kb+guide", "kb+cot", "guide+cot",
        "kb+guide+cot",
    ]

    # build lookup: (model, cfg_label) -> row data
    lookup = {}
    for r in rows:
        lookup[(r["model"], _cfg_label(r["code"]))] = r

    all_models = list(dict.fromkeys(r["model"] for r in rows))
    model_labels = [_pretty_model(m) for m in all_models]

    out_md = base.parent / "logs" / "analysis.md"
    lines = []

    # ── Table 1: 百分比 ────────────────────────────────────────────────────────
    lines.append("# 消融实验结果\n")
    lines.append(f"n={n_total}　**正确% / 拒识% / 误识%**\n")

    header = "| 配置 | " + " | ".join(model_labels) + " |"
    sep    = "|:---:|" + ":---:|" * len(all_models)
    lines.append(header)
    lines.append(sep)

    for cfg in CFG_ORDER:
        cells = []
        for model in all_models:
            r = lookup.get((model, cfg))
            if r is None:
                cells.append("-")
            else:
                n, c, u, m = r["stats"]
                cells.append(f"{c/n*100:.0f} / {u/n*100:.0f} / {m/n*100:.0f}")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")

    # ── Table 2: 绝对数值 ───────────────────────────────────────────────────────
    lines.append("\n## 绝对数值 (正确 / 拒识 / 误识)\n")

    header = "| 配置 | " + " | ".join(model_labels) + " |"
    sep    = "|:---:|" + ":---:|" * len(all_models)
    lines.append(header)
    lines.append(sep)

    for cfg in CFG_ORDER:
        cells = []
        for model in all_models:
            r = lookup.get((model, cfg))
            if r is None:
                cells.append("-")
            else:
                n, c, u, m = r["stats"]
                cells.append(f"{c} / {u} / {m}")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")

    # ── Table 3: 响应时间 ───────────────────────────────────────────────────────
    lines.append("\n## 平均响应时间 (秒)\n")

    header = "| 配置 | " + " | ".join(model_labels) + " |"
    sep    = "|:---:|" + ":---:|" * len(all_models)
    lines.append(header)
    lines.append(sep)

    for cfg in CFG_ORDER:
        cells = []
        for model in all_models:
            r = lookup.get((model, cfg))
            if r is None or not r["times"]:
                cells.append("-")
            else:
                avg_t = sum(r["times"]) / len(r["times"])
                cells.append(f"{avg_t:.2f}")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analysis saved to {out_md}")


def run_experiment(code):
    if len(code) != 4:
        print("Error: code must be 4 characters, e.g. 0111")
        sys.exit(1)

    model_idx = int(code[0], 16)  # hex, supports 0-f
    use_kb = code[1] == "1"
    use_guide = code[2] == "1"
    use_cot = code[3] == "1"

    base = Path(__file__).parent
    cfg = tomllib.loads((base / "config.toml").read_text("utf-8"))

    models = cfg["models"]
    if model_idx >= len(models):
        print(f"Error: model index {model_idx} out of range (have {len(models)} models)")
        sys.exit(1)

    model_name = models[model_idx]
    api_cfg = cfg["api"][model_name]

    # load prompt once
    prompt = load_prompt(base / "prompt.md", guide=use_guide, kb=use_kb, cot=use_cot)

    # read video list
    video_dir = base.parent / "video"
    csv_path = video_dir / "video.csv"
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    # prepare log
    log_dir = base.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"log{code}.txt"

    total = len(rows)
    kb_str = "on" if use_kb else "off"
    guide_str = "on" if use_guide else "off"
    cot_str = "on" if use_cot else "off"

    # write log header
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")
    _output(f"code: {code}", log_path)
    _output(f"model: {model_name}", log_path)
    _output(f"knowledge: {kb_str}", log_path)
    _output(f"guide: {guide_str}", log_path)
    _output(f"cot: {cot_str}", log_path)
    _output(f"total: {total}", log_path)
    _output("", log_path)

    times = []
    correct = 0
    for i, row in enumerate(rows, 1):
        video_name, ground_truth = row[0], row[1]
        video_path = str(video_dir / video_name)

        # process
        frames = sample_frames(video_path, cfg["video"]["fps"], cfg["video"]["max_frames"])
        crop = cfg.get("crop")
        if crop:
            frames = [crop_background(f, **crop) for f in frames]
        urls = [encode_frame(f, **cfg.get("image", {})) for f in frames]

        t0 = time.perf_counter()
        result = call_vlm(urls, prompt, **api_cfg)
        elapsed = round(time.perf_counter() - t0, 3)
        times.append(elapsed)

        # check correctness
        predicted = _extract_drug_name(result)
        is_correct = predicted == ground_truth
        if is_correct:
            correct += 1
        mark = "正确" if is_correct else "错误"

        # output to both console and log
        _output(f"[{i:03d}/{total}] {video_name} | {ground_truth}", log_path)
        _output(result, log_path)
        _output(f"time: {elapsed}s", log_path)
        _output(f"结果：{mark}", log_path)
        _output("", log_path)

    # summary
    avg_time = round(sum(times) / len(times), 3) if times else 0
    accuracy = round(correct / total * 100, 1) if total else 0
    _output(f"correct: {correct}/{total} ({accuracy}%)", log_path)
    _output(f"avg_time: {avg_time}s", log_path)
    _output(f"min_time: {min(times):.3f}s", log_path)
    _output(f"max_time: {max(times):.3f}s", log_path)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--analyze":
        analyze_all()
    elif len(sys.argv) == 2:
        run_experiment(sys.argv[1])
    else:
        print("Usage:")
        print("  python run.py <code>      # run experiment, e.g. 0111")
        print("  python run.py --analyze   # analyze all logs and compare")
        sys.exit(1)
