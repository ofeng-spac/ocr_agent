#!/usr/bin/env python3
"""
Build chunk-level leaflet data for semantic retrieval.

Usage:
    python3 scripts/build_leaflet_chunks.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "structured" / "drug_catalog.json"
OUTPUT_PATH = ROOT / "data" / "structured" / "leaflet_chunks.jsonl"

SECTION_ALIASES = {
    "drug_name": ["药品名称", "通用名称", "通用名"],
    "ingredients": ["成份", "成分"],
    "appearance": ["性状"],
    "indications": ["适应症", "功能主治", "功能与主治"],
    "specification": ["规格"],
    "dosage": ["用法用量", "用法与用量"],
    "contraindications": ["禁忌"],
    "precautions": ["注意事项"],
    "interactions": ["药物相互作用"],
    "pharmacology": ["药理毒理"],
    "pharmacokinetics": ["药代动力学"],
    "storage": ["贮藏", "贮藏条件"],
    "packaging": ["包装"],
    "manufacturer": ["生产企业"],
    "reference_text": ["适应症", "用法用量", "产品特性", "作用机制", "药代动力学", "不良反应及毒性"],
}

CHUNK_DOC_QUALITY = {"standard_leaflet", "reference_text", "pharmacopoeia_text"}
MAX_CHARS = 220
OVERLAP = 40
STOP_ONLY_HEADERS = [
    "不良反应",
    "药物相互作用",
    "贮藏",
    "包装",
    "有效期",
    "批准文号",
    "说明书修订日期",
    "生产企业",
    "药理毒理",
    "药代动力学",
    "作用类别",
]


def read_catalog() -> list[dict]:
    with CATALOG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\x0c", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _merge_broken_headings(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _merge_broken_headings(text: str) -> str:
    aliases = [alias for group in SECTION_ALIASES.values() for alias in group] + STOP_ONLY_HEADERS
    for alias in aliases:
        chars = r"\s*".join(re.escape(ch) for ch in alias)
        text = re.sub(rf"[\[【]\s*{chars}\s*[\]】]", f"[{alias}]", text, flags=re.MULTILINE)
        text = re.sub(rf"^\s*{chars}\s*$", alias, text, flags=re.MULTILINE)
    return text


def normalize_heading(line: str) -> str:
    line = line.replace("【", "").replace("】", "").replace("[", "").replace("]", "")
    line = re.sub(r"\s+", "", line)
    return line.strip("：:")


def detect_section(line: str) -> tuple[str, str] | None:
    normalized = normalize_heading(line)
    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            alias_norm = re.sub(r"\s+", "", alias)
            if normalized.startswith(alias_norm):
                tail = normalized[len(alias_norm) :].lstrip("：:")
                return section, tail
    return None


def clean_section_content(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^[\]\[【】]+", "", value).strip()
    for marker in STOP_ONLY_HEADERS:
        idx = value.find(marker)
        if idx > 0:
            value = value[:idx].strip(" [【】]")
            break
    value = re.sub(r"\s+", " ", value).strip()
    return value


def split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_section = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"]", "[", "】", "【"}:
            continue

        detected = detect_section(line)
        if detected:
            if current_section and current_lines:
                sections.append((current_section, current_lines))
            current_section, inline_tail = detected
            current_lines = [inline_tail] if inline_tail else []
            continue

        if current_section is None:
            continue
        current_lines.append(line)

    if current_section and current_lines:
        sections.append((current_section, current_lines))

    normalized_sections = []
    for section, lines in sections:
        value = " ".join(line for line in lines if line).strip()
        value = clean_section_content(value)
        if value:
            normalized_sections.append((section, value))
    return normalized_sections


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            pieces.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return pieces


def build_chunks_for_item(item: dict) -> list[dict]:
    if item["doc_status"] != "available" or not item["doc_paths"]:
        return []
    if item["doc_quality"] not in CHUNK_DOC_QUALITY:
        return []

    doc_path = ROOT / item["doc_paths"][0]
    text = clean_text(doc_path.read_text(encoding="utf-8", errors="ignore"))
    source_file = str(doc_path.relative_to(ROOT))
    sections = split_sections(text)

    records = []
    idx = 0
    for section_name, section_text in sections:
        for chunk_text_value in chunk_text(section_text):
            idx += 1
            records.append(
                {
                    "chunk_id": f"{item['drug_id']}_chunk_{idx:03d}",
                    "drug_id": item["drug_id"],
                    "canonical_name": item["canonical_name"],
                    "section": section_name,
                    "chunk_text": chunk_text_value,
                    "source_file": source_file,
                    "source_type": item["doc_quality"],
                }
            )
    return records


def main() -> None:
    catalog = read_catalog()
    records = []
    for item in catalog:
        records.extend(build_chunks_for_item(item))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} chunk records to {OUTPUT_PATH}")
    by_drug = {}
    for record in records:
        by_drug.setdefault(record["canonical_name"], 0)
        by_drug[record["canonical_name"]] += 1
    for item in catalog:
        print(f"{item['canonical_name']}: {by_drug.get(item['canonical_name'], 0)} chunks")


if __name__ == "__main__":
    main()
