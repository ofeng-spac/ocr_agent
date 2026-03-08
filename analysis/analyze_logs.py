"""
Log analysis script: compute timing statistics and ERR, produce figures.

Input logs:
  logs/comparison_vlm.txt       -- main experiment VLM (also serves as ablation constrained group)
  logs/comparison_paddle.txt    -- main experiment PaddleOCR
  logs/comparison_tesseract.txt -- main experiment Tesseract
  logs/ablation_open.txt        -- ablation open group (no KB constraint)

Output:
  analysis/timing.png           -- timing comparison across three methods (mean +/- std)

Usage:
    python analysis/analyze_logs.py
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LOG_DIR = _ROOT / "logs"
FIG_DIR = Path(__file__).resolve().parent

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

# Known drug names -- must match the whitelist in server/prompt.md
KNOWN_NAMES = [
    "注射用头孢曲松钠",
    "贝伐珠单抗注射液",
    "注射用头孢噻呋钠",
    "瑞比克狂犬病灭活疫苗（HCP-SAD株）",
    "卫佳捌犬瘟热、腺病毒2型、副流感、细小病毒病四联活疫苗",
    "猫鼻气管炎、杯状病毒病、泛白细胞减少症三联灭活疫苗",
    "贝倍旺狂犬病灭活疫苗（PV/BHK-21株）",
    "犬瘟热、犬副流感、犬腺病毒与犬细小病毒病四联活疫苗",
    "金喵乐猫泛白细胞减少症、鼻气管炎、杯状病毒病三联灭活疫苗（FP/15株+FH/AS株+FC/HF株）",
    "妙三多Fel-0-VaxPCT 猫鼻气管炎、嵌杯病毒病、泛白细胞减少症三联灭活疫苗",
]


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_log(log_path: Path) -> list[tuple[str, list[str], float]]:
    """
    Parse a log file. Returns a list of (video_name, result_lines, elapsed_seconds).
    """
    entries = []
    header = None
    lines = []
    elapsed = None

    with log_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = re.match(r"\[\d+/\d+\] (.+)", line)
            if m:
                if header is not None:
                    entries.append((header, lines, elapsed))
                header, lines, elapsed = m.group(1), [], None
            elif re.match(r"用时: [\d.]+s", line):
                elapsed = float(re.search(r"[\d.]+", line).group())
            elif header is not None:
                lines.append(line)

    if header is not None:
        entries.append((header, lines, elapsed))
    return entries


def extract_drug_name(result_lines: list[str]) -> str | None:
    """Extract the drug name field from VLM output lines."""
    for line in result_lines:
        m = re.match(r"药品名称[：:](.+)", line.strip())
        if m:
            return m.group(1).strip()
    return None


def best_similarity(name: str) -> float:
    """Return the highest string similarity between name and any known drug name."""
    if not name or name == "未知":
        return 0.0
    return max(SequenceMatcher(None, name, k).ratio() for k in KNOWN_NAMES)


def is_exact(name: str) -> bool:
    return name in KNOWN_NAMES


def is_complete(name: str, threshold: float = 0.85) -> bool:
    """Return True if name is sufficiently similar to a known drug name (open group judge)."""
    return best_similarity(name) >= threshold


# ── Main comparison analysis ──────────────────────────────────────────────────

def analyze_comparison() -> None:
    configs = [
        ("vlm",       "VLM"),
        ("paddle",    "PaddleOCR"),
        ("tesseract", "Tesseract"),
    ]

    avgs, stds, labels = [], [], []

    for mode, label in configs:
        path = LOG_DIR / f"comparison_{mode}.txt"
        if not path.exists():
            print(f"[skip] comparison_{mode}.txt not found")
            continue

        entries = parse_log(path)
        # Exclude first sample (cold-start warm-up) -- consistent with paper
        times = [e[2] for e in entries[1:] if e[2] is not None]

        avg, std = np.mean(times), np.std(times)
        avgs.append(avg)
        stds.append(std)
        labels.append(label)

        extra = ""
        if mode == "vlm":
            names = [extract_drug_name(e[1]) for e in entries]
            err = sum(1 for n in names if n and is_exact(n)) / len(entries)
            extra = f"  ERR={err:.1%}"
        print(f"{mode}: n={len(times)}  avg={avg:.3f}s  std={std:.3f}s{extra}")

    if not avgs:
        return

    colors = ["steelblue", "coral", "gray"][:len(labels)]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    ax.bar(x, avgs, yerr=stds, capsize=6, width=0.5,
           color=colors,
           error_kw={"elinewidth": 1.2, "ecolor": "#555555"})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Average processing time (s)", fontsize=11)
    ax.set_ylim(0, max(avgs) * 1.35)
    for i, (a, s) in enumerate(zip(avgs, stds)):
        ax.text(i, a + s + 0.15, f"{a:.2f}s", ha="center", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "timing.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"-> timing.png")
    plt.close(fig)


# ── Ablation analysis ─────────────────────────────────────────────────────────

def analyze_ablation() -> None:
    groups = [
        ("constrained", LOG_DIR / "comparison_vlm.txt", is_exact),
        ("open",        LOG_DIR / "ablation_open.txt",  is_complete),
    ]

    for label, path, judge in groups:
        if not path.exists():
            print(f"[skip] {path.name} not found")
            continue
        entries = parse_log(path)
        names = [extract_drug_name(e[1]) for e in entries]
        complete = sum(1 for n in names if n and judge(n))
        total = len(entries)
        print(f"{label}: {complete}/{total} complete formatted output ({complete/total:.1%})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Main comparison:")
    analyze_comparison()
    print("\nAblation:")
    analyze_ablation()


if __name__ == "__main__":
    main()
