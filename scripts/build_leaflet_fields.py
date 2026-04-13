#!/usr/bin/env python3
"""
Build structured leaflet fields from the current text corpus.

Usage:
    python3 scripts/build_leaflet_fields.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "structured" / "drug_catalog.json"
OUTPUT_PATH = ROOT / "data" / "structured" / "leaflet_fields.jsonl"

HEADER_ALIASES = {
    "appearance": ["性状"],
    "specification": ["规格"],
    "indications": ["适应症", "功能主治", "功能与主治"],
    "dosage": ["用法用量", "用法与用量"],
    "precautions": ["注意事项"],
    "contraindications": ["禁忌"],
}

STOP_ONLY_HEADERS = [
    "成份",
    "不良反应",
    "药物相互作用",
    "贮藏",
    "包装",
    "有效期",
    "批准文号",
    "生产企业",
    "药理毒理",
    "药代动力学",
    "产品特性",
    "作用机制",
    "检查",
    "含量测定",
    "鉴别",
    "处方",
    "制法",
]


def read_catalog() -> list[dict]:
    with CATALOG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\x0c", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ：:\n\t")
    return value.strip()


def normalize_label_chars(text: str) -> str:
    return re.sub(r"\s+", "", text)


def extract_line_value(text: str, labels: list[str]) -> str:
    lines = text.splitlines()
    normalized_labels = [normalize_label_chars(label) for label in labels]
    for line in lines:
        compact_line = normalize_label_chars(line)
        for label in normalized_labels:
            if compact_line.startswith(label + "：") or compact_line.startswith(label + ":"):
                raw = re.split(r"[：:]", line, maxsplit=1)
                if len(raw) == 2:
                    return compact(raw[1])
    return ""


def header_pattern(alias: str) -> str:
    chars = [re.escape(ch) + r"\s*" for ch in alias]
    return "".join(chars).rstrip(r"\s*")


def build_all_section_pattern() -> str:
    aliases = [alias for alias_group in HEADER_ALIASES.values() for alias in alias_group]
    aliases.extend(STOP_ONLY_HEADERS)
    return "|".join(header_pattern(alias) for alias in aliases)


ALL_SECTION_PATTERN = build_all_section_pattern()


def extract_section(text: str, aliases: list[str]) -> str:
    starts = []
    for alias in aliases:
        pat = re.compile(
            rf"(?m)^(?P<header>[【\[]?\s*{header_pattern(alias)}\s*[】\]]?)\s*(?P<tail>[^\n]*)"
        )
        for match in pat.finditer(text):
            starts.append((match.start(), match.end(), match.group("tail")))

    if not starts:
        return ""

    starts.sort(key=lambda item: item[0])
    start_pos, end_pos, tail = starts[0]
    next_match = re.search(
        rf"(?m)^(?:[【\[]?\s*(?:{ALL_SECTION_PATTERN})\s*[】\]]?)\s*[^\n]*",
        text[end_pos:],
    )
    stop_pos = end_pos + next_match.start() if next_match else len(text)

    body = tail.strip()
    remainder = text[end_pos:stop_pos].strip()
    if body and remainder:
        merged = body + "\n" + remainder
    else:
        merged = body or remainder

    return compact(merged)


def split_brand_values(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    parts = re.split(r"[,，/]+", raw_value)
    out = []
    for part in parts:
        normalized = compact(part.replace("®", ""))
        if normalized:
            out.append(normalized)
    deduped = []
    seen = set()
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def extract_fields_for_item(item: dict) -> list[dict]:
    if item["doc_status"] != "available" or not item["doc_paths"]:
        return []

    doc_path = ROOT / item["doc_paths"][0]
    text = clean_text(doc_path.read_text(encoding="utf-8", errors="ignore"))
    source_file = str(doc_path.relative_to(ROOT))

    records: list[dict] = []

    generic_name = extract_line_value(text, ["通用名称", "通用名"])
    if not generic_name:
        generic_name = item["generic_name"]

    brand_raw = extract_line_value(text, ["商品名", "商品名称"])
    brand_names = split_brand_values(brand_raw)

    section_values = {
        "specification": extract_section(text, HEADER_ALIASES["specification"]),
        "appearance": extract_section(text, HEADER_ALIASES["appearance"]),
        "indications": extract_section(text, HEADER_ALIASES["indications"]),
        "dosage": extract_section(text, HEADER_ALIASES["dosage"]),
        "precautions": extract_section(text, HEADER_ALIASES["precautions"]),
        "contraindications": extract_section(text, HEADER_ALIASES["contraindications"]),
    }

    def add_record(field_name: str, field_value: str) -> None:
        field_value = compact(field_value)
        if not field_value:
            return
        records.append(
            {
                "drug_id": item["drug_id"],
                "canonical_name": item["canonical_name"],
                "field_name": field_name,
                "field_value": field_value,
                "source_file": source_file,
                "source_type": item["doc_quality"],
            }
        )

    add_record("generic_name", generic_name)
    for brand_name in brand_names:
        add_record("brand_name", brand_name)
    for field_name, field_value in section_values.items():
        add_record(field_name, field_value)

    return records


def main() -> None:
    catalog = read_catalog()
    output_records = []
    for item in catalog:
        output_records.extend(extract_fields_for_item(item))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_drug = {}
    for record in output_records:
        by_drug.setdefault(record["canonical_name"], set()).add(record["field_name"])

    print(f"wrote {len(output_records)} records to {OUTPUT_PATH}")
    for item in catalog:
        fields = sorted(by_drug.get(item["canonical_name"], set()))
        print(f"{item['canonical_name']}: {', '.join(fields) if fields else 'no_fields'}")


if __name__ == "__main__":
    main()
