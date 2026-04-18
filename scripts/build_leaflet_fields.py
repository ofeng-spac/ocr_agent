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
    "作用类别",
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

QUALITY_FIELD_POLICY = {
    "standard_leaflet": {"specification", "appearance", "indications", "dosage", "precautions", "contraindications"},
    "reference_text": {"indications", "dosage"},
    "pharmacopoeia_text": {"specification", "appearance", "indications", "dosage"},
    "non_standard_reference": set(),
}

FIELD_OVERRIDES = {
    "双黄连口服液": {
        "dosage": "口服。一次 20 毫升（2 支），一日 3 次；小儿酌减或遵医嘱。",
        "precautions": (
            "忌烟、酒及辛辣、生冷、油腻食物。"
            "不宜在服药期间同时服用滋补性中药。"
            "风寒感冒者不适用。"
            "糖尿病患者及有高血压、心脏病、肝病、肾病等慢性病严重者应在医师指导下服用。"
            "儿童、孕妇、哺乳期妇女、年老体弱及脾虚便溏者应在医师指导下服用。"
            "发热体温超过 38.5℃ 的患者，应去医院就诊。"
            "服药 3 天症状无缓解，应去医院就诊。"
            "对本品过敏者禁用，过敏体质者慎用。"
            "本品性状发生改变时禁止使用。"
            "儿童必须在成人监护下使用。"
            "如正在使用其他药品，使用本品前请咨询医师或药师。"
        ),
    },
    "盐酸小檗碱片": {
        "appearance": "本品为黄色片、糖衣片或薄膜衣片，除去包衣后显黄色。",
        "dosage": (
            "口服，成人：一次 0.1 克～0.3 克（1 片～3 片），一日 3 次；"
            "儿童用量见说明书分龄分体重表。"
        ),
        "contraindications": "溶血性贫血患者及葡萄糖-6-磷酸脱氢酶缺乏患者禁用。",
    },
}


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


def clean_extracted_section(value: str, aliases: list[str]) -> str:
    value = value.replace("【", "[").replace("】", "]")
    value = compact(value)

    for alias in aliases:
        for variant in [
            alias,
            f"[{alias}]",
            f"{alias}]",
            f"{alias}】",
            f"[{alias}",
            f"【{alias}】",
        ]:
            if value.startswith(variant):
                value = compact(value[len(variant):])

    value = re.sub(r"^(?:[\]\[【】]|注意事项\])+", "", value).strip()

    inline_stops = [
        "不良反应",
        "药物相互作用",
        "贮 藏",
        "贮藏",
        "包 装",
        "包装",
        "有 效 期",
        "有效期",
        "批准文号",
        "说明书修订日期",
        "生产企业",
        "作用类别",
    ]
    for marker in inline_stops:
        idx = value.find(marker)
        if idx > 0:
            value = value[:idx].strip(" [【】]")
            break

    value = re.sub(r"(用法用量|注意事项|禁忌)\s*[】\]]\s*$", "", value)
    value = compact(value)
    return value


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

    return clean_extracted_section(merged, aliases)


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

    allowed_fields = QUALITY_FIELD_POLICY.get(item["doc_quality"], set())
    if allowed_fields:
        section_values = {
            field_name: value
            for field_name, value in section_values.items()
            if field_name in allowed_fields
        }
    else:
        section_values = {}

    overrides = FIELD_OVERRIDES.get(item["canonical_name"], {})
    for field_name, field_value in overrides.items():
        section_values[field_name] = field_value

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
