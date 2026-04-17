from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_PATH = ROOT / "logs" / "analysis.md"


def _split_row(line: str) -> list[str]:
    parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return parts


def _parse_metric_triplet(value: str) -> dict:
    parts = [part.strip() for part in value.split("/")]
    if len(parts) != 3:
        return {"correct": None, "unknown": None, "misid": None}
    return {
        "correct": float(parts[0]) if "." in parts[0] else int(parts[0]),
        "unknown": float(parts[1]) if "." in parts[1] else int(parts[1]),
        "misid": float(parts[2]) if "." in parts[2] else int(parts[2]),
    }


def _parse_markdown_table(lines: list[str], start_idx: int) -> tuple[list[str], list[dict], int]:
    headers = _split_row(lines[start_idx])
    rows: list[dict] = []
    idx = start_idx + 2
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("|"):
            break
        values = _split_row(line)
        row = {headers[i]: values[i] for i in range(min(len(headers), len(values)))}
        rows.append(row)
        idx += 1
    return headers, rows, idx


def _score_entry(entry: dict) -> tuple[float, float]:
    metrics = entry["metrics_pct"]
    return (metrics["correct"], -entry["avg_time_sec"])


def load_evaluation_summary() -> dict:
    if not ANALYSIS_PATH.exists():
        return {
            "available": False,
            "reason": "analysis.md 不存在。",
            "models": {},
            "recommended_model": "",
            "recommended_config": None,
        }

    lines = ANALYSIS_PATH.read_text(encoding="utf-8").splitlines()
    tables = {}
    i = 0
    current_section = None
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("## Counts"):
            current_section = "counts"
        elif line.startswith("## Avg Response Time"):
            current_section = "avg_time_sec"
        elif line.startswith("| Config |"):
            section_key = current_section or "metrics_pct"
            headers, rows, i = _parse_markdown_table(lines, i)
            tables[section_key] = {"headers": headers, "rows": rows}
            current_section = current_section or "metrics_pct"
            continue
        i += 1

    metrics_rows = tables.get("metrics_pct", {}).get("rows", [])
    counts_rows = tables.get("counts", {}).get("rows", [])
    times_rows = tables.get("avg_time_sec", {}).get("rows", [])

    if not metrics_rows:
        return {
            "available": False,
            "reason": "analysis.md 中没有可解析的指标表。",
            "models": {},
            "recommended_model": "",
            "recommended_config": None,
        }

    models = [key for key in metrics_rows[0].keys() if key != "Config"]
    counts_map = {row["Config"]: row for row in counts_rows}
    times_map = {row["Config"]: row for row in times_rows}

    model_summaries: dict[str, list[dict]] = {}
    best_by_model: dict[str, dict] = {}

    for model in models:
        entries = []
        for metric_row in metrics_rows:
            config_name = metric_row["Config"]
            counts_row = counts_map.get(config_name, {})
            times_row = times_map.get(config_name, {})
            entry = {
                "config": config_name,
                "metrics_pct": _parse_metric_triplet(metric_row.get(model, "")),
                "counts": _parse_metric_triplet(counts_row.get(model, "")),
                "avg_time_sec": float(times_row.get(model, "0") or 0),
            }
            entries.append(entry)

        ranked = sorted(
            entries,
            key=lambda item: (-item["metrics_pct"]["correct"], item["avg_time_sec"], item["metrics_pct"]["misid"]),
        )
        model_summaries[model] = ranked
        best_by_model[model] = ranked[0]

    recommended_model = min(
        best_by_model,
        key=lambda model: (-best_by_model[model]["metrics_pct"]["correct"], best_by_model[model]["avg_time_sec"]),
    )

    return {
        "available": True,
        "source_file": str(ANALYSIS_PATH.relative_to(ROOT)),
        "models": model_summaries,
        "best_by_model": best_by_model,
        "recommended_model": recommended_model,
        "recommended_config": best_by_model[recommended_model],
    }
