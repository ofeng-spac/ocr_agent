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
    python run.py --time          # violin plot of response times

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


def _ned(a, b):
    """Normalized edit distance between two strings (0=identical, 1=totally different)."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    # standard Levenshtein via DP
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (0 if ca == cb else 1))
        prev = curr
    return prev[lb] / max(la, lb)


def _misid_avg_ned(entries):
    """Mean NED of misidentified entries (excludes 正确 and 未知)."""
    neds = [
        _ned(pred, gt)
        for ok, pred, _, gt in entries
        if _entry_category(ok, pred) == "误识"
    ]
    return sum(neds) / len(neds) if neds else None


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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

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
            "ned": _misid_avg_ned(entries),
        })

    if not rows:
        print("No parseable entries found.")
        return

    def cell(n, c, u, m):
        if n == 0:
            return "-"
        return f"{c/n*100:.0f} / {u/n*100:.0f} / {m/n*100:.0f}"

    n_total = rows[0]["stats"][0]

    # ── ablation table (消融表) ───────────────────────────────────────────────
    # cfg order: canonical 2³ order (kb/guide/cot bits)
    CFG_ORDER = [
        "baseline",
        "kb", "guide", "cot",
        "kb+guide", "kb+cot", "guide+cot",
        "kb+guide+cot",
    ]

    # build lookup: (model, cfg_label) -> (n, correct, unknown, misid)
    lookup = {}
    for r in rows:
        lookup[(r["model"], _cfg_label(r["code"]))] = r["stats"]

    all_models = list(dict.fromkeys(r["model"] for r in rows))
    model_labels = [_pretty_model(m) for m in all_models]

    out_md = base.parent / "logs" / "analysis.md"
    lines = []
    lines.append("# 消融实验结果\n")
    lines.append(f"n={n_total}　**正确% / 拒识% / 误识%**\n")

    # header
    header = "| 配置 | " + " | ".join(model_labels) + " |"
    sep    = "|:---:|" + ":---:|" * len(all_models)
    lines.append(header)
    lines.append(sep)

    for cfg in CFG_ORDER:
        cells = []
        for model in all_models:
            stats = lookup.get((model, cfg))
            if stats is None:
                cells.append("-")
            else:
                n, c, u, m = stats
                cells.append(cell(n, c, u, m))
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── shared data prep ──────────────────────────────────────────────────────
    all_models = list(dict.fromkeys(r["model"] for r in rows))
    all_cfgs   = list(dict.fromkeys(_cfg_label(r["code"]) for r in rows))

    # accuracy matrix  [model × cfg]
    acc_matrix  = np.full((len(all_models), len(all_cfgs)), np.nan)
    misid_matrix = np.full((len(all_models), len(all_cfgs)), np.nan)
    for r in rows:
        mi = all_models.index(r["model"])
        ci = all_cfgs.index(_cfg_label(r["code"]))
        n, c, u, m = r["stats"]
        acc_matrix[mi, ci]   = c / n * 100
        misid_matrix[mi, ci] = m / n * 100

    model_labels = [_pretty_model(m) for m in all_models]

    # ── heatmap ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.5, len(all_models) * 0.9 + 1.5)))

    for ax, matrix, title, cmap, fmt in [
        (axes[0], acc_matrix,   "Accuracy (%)",        "YlGn",  ".1f"),
        (axes[1], misid_matrix, "Misidentification (%)", "YlOrRd", ".1f"),
    ]:
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(all_cfgs)))
        ax.set_xticklabels(all_cfgs, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(all_models)))
        ax.set_yticklabels(model_labels, fontsize=8)
        ax.set_title(title, fontsize=10, pad=8)
        for i in range(len(all_models)):
            for j in range(len(all_cfgs)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "black" if val < 60 else ("white" if val > 80 else "black")
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=8, color=text_color, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_hm = base.parent / "logs" / "heatmap.png"
    fig.savefig(out_hm, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── scatter: avg_time vs accuracy ─────────────────────────────────────────
    # model: color (muted academic palette), cfg: marker shape
    # Nature-style palette (npg: Nature Publishing Group)
    color_map  = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]
    marker_map = {"baseline": "o", "kb": "s", "guide": "^", "cot": "D",
                  "kb+guide": "p", "kb+cot": "h", "guide+cot": "X", "kb+guide+cot": "P"}

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")
    ax2.set_xlabel("Average response time (s)", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.tick_params(labelsize=9)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#aaaaaa")

    rng = np.random.default_rng(0)
    plotted = {}  # track (model, cfg) for legend dedup
    for r in rows:
        if not r["times"]:
            continue
        avg_t = sum(r["times"]) / len(r["times"])
        n, c, u, m = r["stats"]
        acc   = c / n * 100
        mi    = all_models.index(r["model"])
        cfg   = _cfg_label(r["code"])
        color  = color_map[mi % len(color_map)]
        marker = marker_map.get(cfg, "o")
        jx = rng.uniform(-0.15, 0.15)
        jy = rng.uniform(-0.4, 0.4)
        ax2.scatter(avg_t + jx, acc + jy,
                    color=color, marker=marker,
                    s=50, edgecolors="black", linewidths=0.6,
                    zorder=3, alpha=0.75)
        plotted.setdefault("model", set()).add(mi)
        plotted.setdefault("cfg",   set()).add(cfg)

    # y-axis: focus on meaningful range
    all_acc = [r["stats"][1] / r["stats"][0] * 100 for r in rows]
    ax2.set_xlim(left=0)
    ax2.set_ylim(max(0, min(all_acc) - 8), 103)

    # legend: models (color) + configs (shape), merged into one box
    from matplotlib.lines import Line2D
    legend_items = []
    for i, m in enumerate(all_models):
        legend_items.append(Line2D([0], [0], marker="o", linestyle="none",
                                   color=color_map[i % len(color_map)],
                                   markersize=6, label=_pretty_model(m)))
    legend_items.append(Line2D([0], [0], linestyle="none", label=""))  # spacer
    for cfg, mk in marker_map.items():
        if cfg in plotted.get("cfg", set()):
            legend_items.append(Line2D([0], [0], marker=mk, linestyle="none",
                                       color="#666666", markersize=6, label=cfg))

    ax2.legend(handles=legend_items, fontsize=7.5, framealpha=1.0,
               edgecolor="#bbbbbb", loc="upper left",
               bbox_to_anchor=(1.02, 1), borderaxespad=0,
               handletextpad=0.5, borderpad=0.8)

    fig2.tight_layout(pad=1.2)
    fig2.subplots_adjust(right=0.72)
    out_sc = base.parent / "logs" / "scatter.png"
    fig2.savefig(out_sc, dpi=150, bbox_inches="tight")
    plt.close(fig2)


def plot_times():
    import matplotlib.pyplot as plt

    base = Path(__file__).parent
    log_dir = base.parent / "logs"
    log_files = sorted(log_dir.glob("log????.txt"))

    if not log_files:
        print("No log files found in logs/")
        return

    from collections import OrderedDict
    # model_name -> [(config_label, times, accuracy_pct)]
    groups = OrderedDict()
    for lf in log_files:
        meta, entries = _parse_log(lf)
        times = [e[2] for e in entries if e[2] is not None]
        if not times:
            continue
        code  = meta.get("code", lf.stem[3:])
        model = meta.get("model", "?")
        cfg_label = _cfg_label(code)
        n, c, u, m = _group_stats(entries)
        acc = c / n * 100 if n else 0.0
        groups.setdefault(model, []).append((cfg_label, times, acc))

    models = list(groups.keys())
    if not models:
        print("No parseable entries found.")
        return

    colors     = ["#d0e8ff", "#ffe0d0", "#d0ffd8", "#f0d0ff"]
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#aaaaaa")

    all_cfgs = []
    for model in models:
        for cfg_label, _, _ in groups[model]:
            if cfg_label not in all_cfgs:
                all_cfgs.append(cfg_label)

    positions  = []
    tick_pos   = []
    tick_labels = []
    bar_colors = []
    all_data   = []
    acc_labels = []   # (x_pos, acc_pct) for annotation
    x = 1
    for cfg in all_cfgs:
        group_start = x
        for idx, model in enumerate(models):
            for cfg_label, times, acc in groups[model]:
                if cfg_label == cfg:
                    positions.append(x)
                    all_data.append(times)
                    bar_colors.append(colors[idx % len(colors)])
                    acc_labels.append((x, acc))
                    x += 1
                    break
        tick_pos.append((group_start + x - 1) / 2)
        tick_labels.append(cfg)
        x += 1  # gap between groups

    if all_data:
        parts = ax.violinplot(all_data, positions=positions,
                              showmedians=True, showextrema=True, widths=0.8)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(bar_colors[i])
            pc.set_alpha(0.8)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)
        for key in ("cmins", "cmaxes", "cbars"):
            parts[key].set_linewidth(0.8)
            parts[key].set_color("#888888")

        # annotate accuracy above each violin
        y_top = ax.get_ylim()[1]
        for xpos, acc in acc_labels:
            max_t = max(all_data[positions.index(xpos)])
            label_y = max_t + (y_top - max_t) * 0.05 + 0.3
            color = "#1a7a1a" if acc >= 95 else ("#d4700a" if acc >= 80 else "#c0392b")
            ax.text(xpos, label_y, f"{acc:.0f}%",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=color)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("Response time (s)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(False)

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=colors[i], label=_pretty_model(m))
                    for i, m in enumerate(models)]
    ax.legend(handles=legend_items, fontsize=8, framealpha=1.0,
              edgecolor="#bbbbbb", loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              borderpad=0.8, handletextpad=0.5)

    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(right=0.78)
    out = base.parent / "logs" / "time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    elif len(sys.argv) == 2 and sys.argv[1] == "--time":
        plot_times()
    elif len(sys.argv) == 2:
        run_experiment(sys.argv[1])
    else:
        print("Usage:")
        print("  python run.py <code>      # run experiment, e.g. 0111")
        print("  python run.py --analyze   # analyze all logs and compare")
        print("  python run.py --time      # violin plot of response times")
        sys.exit(1)
