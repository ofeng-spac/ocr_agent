"""
Batch experiment runner with binary-style encoding.

Code format: <model><knowledge><guide>
  - model:     0-9/a-f (index into config.toml models array)
  - knowledge: 0=off, 1=on
  - guide:     0=off, 1=on

Usage:
    python run.py <code>          # run experiment
    python run.py --analyze       # analyze all logs in logs/ and compare
    python run.py --time          # violin plot of response times

Examples:
    python run.py 011    → model 0 (flash), kb=on,  guide=on  → logs/log_011.txt
    python run.py 010    → model 0 (flash), kb=on,  guide=off → logs/log_010.txt
    python run.py 100    → model 1 (plus),  kb=off, guide=off → logs/log_100.txt
    python run.py 111    → model 1 (plus),  kb=on,  guide=on  → logs/log_111.txt
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
_RE_META   = _re.compile(r'^(code|model|knowledge|guide):\s*(.+)$')
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
    entries: list of (is_correct, predicted, elapsed)
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
                entries.append((is_correct, predicted or "", elapsed))
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


def analyze_all():
    base = Path(__file__).parent
    log_dir = base.parent / "logs"
    log_files = sorted(log_dir.glob("log???.txt"))

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
        model_short = model.split("-")[-1] if "-" in model else model
        rows.append({
            "code":  meta.get("code", lf.stem[3:]),
            "model": model_short,
            "kb":    meta.get("knowledge", "?"),
            "guide": meta.get("guide", "?"),
            "stats": _group_stats(entries),
            "times": times,
        })

    if not rows:
        print("No parseable entries found.")
        return

    def pct(a, b):
        return f"{a/b*100:.1f}%" if b else "-"

    n_total = rows[0]["stats"][0]

    base = Path(__file__).parent
    out = base.parent / "logs" / "analysis.md"
    lines = []
    lines.append("# 实验结果对比\n")
    lines.append(f"- n={n_total}")
    lines.append("- **正确** = 识别正确　**未知** = 拒识/高不确定性（安全，触发人工复核）　**误识** = 给出错误药名（**危险**）\n")

    lines.append("| code | 模型 | 知识库 | 引导 | 正确 | 未知 | 误识 | 均时 |")
    lines.append("|------|------|--------|------|-----:|-----:|-----:|-----:|")
    for r in rows:
        n, c, u, m = r["stats"]
        avg = f"{sum(r['times'])/len(r['times']):.3f}s" if r["times"] else "-"
        lines.append(
            f"| {r['code']} | {r['model']} | {r['kb']} | {r['guide']} "
            f"| {pct(c,n)} | {pct(u,n)} | {pct(m,n)} | {avg} |"
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_times():
    import matplotlib.pyplot as plt

    base = Path(__file__).parent
    log_dir = base.parent / "logs"
    log_files = sorted(log_dir.glob("log???.txt"))

    if not log_files:
        print("No log files found in logs/")
        return

    labels, data = [], []
    for lf in log_files:
        meta, entries = _parse_log(lf)
        times = [e[2] for e in entries if e[2] is not None]
        if not times:
            continue
        code = meta.get("code", lf.stem[3:])
        model = meta.get("model", "?").split("-")[-1]
        kb = "kb" if meta.get("knowledge") == "on" else ""
        guide = "g" if meta.get("guide") == "on" else ""
        label = f"{code}\n{model}"
        labels.append(label)
        data.append(times)

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, positions=range(1, len(data)+1),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#d0e8ff")
        pc.set_alpha(0.8)
    parts["cmedians"].set_color("#c0392b")
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Response time (s)")

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out = base.parent / "logs" / "time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def run_experiment(code):
    if len(code) != 3:
        print("Error: code must be 3 characters, e.g. 011")
        sys.exit(1)

    model_idx = int(code[0], 16)  # hex, supports 0-f
    use_kb = code[1] == "1"
    use_guide = code[2] == "1"

    base = Path(__file__).parent
    cfg = tomllib.loads((base / "config.toml").read_text("utf-8"))

    models = cfg["models"]
    if model_idx >= len(models):
        print(f"Error: model index {model_idx} out of range (have {len(models)} models)")
        sys.exit(1)

    model_name = models[model_idx]
    api_cfg = cfg["api"][model_name]

    # load prompt once
    prompt = load_prompt(base / "prompt.md", guide=use_guide, kb=use_kb)

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

    # write log header
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")
    _output(f"code: {code}", log_path)
    _output(f"model: {model_name}", log_path)
    _output(f"knowledge: {kb_str}", log_path)
    _output(f"guide: {guide_str}", log_path)
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
    elif len(sys.argv) == 2 and sys.argv[1] == "--time":
        plot_times()
    elif len(sys.argv) == 2:
        run_experiment(sys.argv[1])
    else:
        print("Usage:")
        print("  python run.py <code>      # run experiment, e.g. 011")
        print("  python run.py --analyze   # analyze all logs and compare")
        print("  python run.py --time      # violin plot of response times")
        sys.exit(1)
