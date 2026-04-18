#!/usr/bin/env python3
"""
Build a larger mixed RAG evaluation set.

Usage:
    python3 scripts/build_rag_eval_cases.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS_PATH = ROOT / "data" / "structured" / "leaflet_fields.jsonl"
OUTPUT_PATH = ROOT / "data" / "eval" / "rag_eval_cases.jsonl"

FIELD_TEMPLATES = {
    "generic_name": ["药品名称是什么", "正式名称叫什么"],
    "specification": ["规格是什么", "这个药规格多大"],
    "appearance": ["外观是什么样", "性状是什么"],
    "indications": ["适应症是什么", "这个药适用于什么"],
    "dosage": ["用法用量是什么", "这个药怎么用"],
    "precautions": ["注意事项是什么", "使用时需要注意什么"],
    "contraindications": ["禁忌是什么", "哪些人不能用"],
    "brand_name": ["商品名是什么", "品牌名是什么"],
}

CURATED_SEMANTIC_CASES = [
    ("双黄连口服液", "这个药主要用于哪些场景", ["外感风热所致的感冒"], "indications"),
    ("双黄连口服液", "这个药有哪些使用提醒", ["风寒感冒者不适用"], "precautions"),
    ("奥美拉唑肠溶胶囊", "这个药一般治疗哪些问题", ["胃溃疡"], "indications"),
    ("奥美拉唑肠溶胶囊", "这个药有什么慎用提醒", ["肾功能不全"], "precautions"),
    ("注射用头孢噻呋钠", "这个药一般治疗哪些感染", ["呼吸道"], "indications"),
    ("注射用头孢噻呋钠", "这个药通常怎么给药", ["皮下注射"], "dosage"),
    ("注射用头孢曲松钠", "这个药一般治疗哪些感染", ["下呼吸道感染"], "indications"),
    ("注射用头孢曲松钠", "这个药通常怎么给药", ["肌内注射"], "dosage"),
    ("盐酸小檗碱片", "这个药一般用于什么情况", ["肠道感染"], "indications"),
    ("蒲地蓝消炎口服液", "这个药主要用于哪些场景", ["疖肿"], "indications"),
    ("贝伐珠单抗注射液", "这个药主要用于哪些场景", ["转移性结直肠癌"], "indications"),
    ("贝伐珠单抗注射液", "这个药有哪些风险和注意事项", ["胃肠道穿孔"], "precautions"),
]

REFUSAL_CASES = [
    ("狂犬病灭活疫苗", "适应症是什么", "field_query", "doc_unavailable", "indications"),
    ("狂犬病灭活疫苗", "这个药主要用于什么", "semantic_query", "doc_unavailable", None),
    ("犬瘟热、腺病毒2型、副流感、细小病毒病四联活疫苗", "规格是什么", "field_query", "doc_unavailable", "specification"),
    ("犬瘟热、腺病毒2型、副流感、细小病毒病四联活疫苗", "这个药主要用于什么", "semantic_query", "doc_unavailable", None),
    ("猫鼻气管炎、杯状病毒病、泛白细胞减少症三联灭活疫苗", "适应症是什么", "field_query", "doc_unavailable", "indications"),
    ("猫鼻气管炎、杯状病毒病、泛白细胞减少症三联灭活疫苗", "这个药主要用于什么", "semantic_query", "doc_unavailable", None),
    ("孟鲁司特钠片", "适应症是什么", "field_query", "field_unavailable", "indications"),
    ("孟鲁司特钠片", "这个药主要用于什么场景", "semantic_query", "semantic_unavailable", None),
]


def load_fields() -> list[dict]:
    records = []
    with FIELDS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def first_signal(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    parts = re.split(r"[。；;]", text)
    for part in parts:
        part = part.strip()
        if len(part) >= 4:
            return part[:40]
    return text[:40]


def build_field_cases(records: list[dict]) -> list[dict]:
    cases = []
    idx = 1
    for record in records:
        templates = FIELD_TEMPLATES.get(record["field_name"], [])
        for template in templates:
            case = {
                "case_id": f"field_{idx:03d}",
                "case_type": "field",
                "canonical_name": record["canonical_name"],
                "question": template,
                "expected_route_mode": "field_query",
                "expected_status": "ok",
                "expected_retrieval_mode": "qdrant_fields",
                "expected_target_field": record["field_name"],
                "expected_substrings": [first_signal(record["field_value"])],
                "expected_citation_field": record["field_name"],
            }
            cases.append(case)
            idx += 1
    return cases


def build_semantic_cases(start_idx: int) -> list[dict]:
    cases = []
    idx = start_idx
    for canonical_name, question, substrings, section in CURATED_SEMANTIC_CASES:
        cases.append(
            {
                "case_id": f"semantic_{idx:03d}",
                "case_type": "semantic",
                "canonical_name": canonical_name,
                "question": question,
                "expected_route_mode": "semantic_query",
                "expected_status": "ok",
                "expected_retrieval_mode": "qdrant_chunks",
                "expected_target_field": None,
                "expected_substrings": substrings,
                "expected_citation_section": section,
            }
        )
        idx += 1
    return cases


def build_refusal_cases(start_idx: int) -> list[dict]:
    cases = []
    idx = start_idx
    for canonical_name, question, route_mode, status, target_field in REFUSAL_CASES:
        cases.append(
            {
                "case_id": f"refusal_{idx:03d}",
                "case_type": "refusal",
                "canonical_name": canonical_name,
                "question": question,
                "expected_route_mode": route_mode,
                "expected_status": status,
                "expected_retrieval_mode": None,
                "expected_target_field": target_field,
            }
        )
        idx += 1
    return cases


def main() -> None:
    records = load_fields()
    field_cases = build_field_cases(records)
    semantic_cases = build_semantic_cases(start_idx=1)
    refusal_cases = build_refusal_cases(start_idx=1)

    cases = field_cases + semantic_cases + refusal_cases
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"wrote {len(cases)} cases to {OUTPUT_PATH}")
    print(f"field cases: {len(field_cases)}")
    print(f"semantic cases: {len(semantic_cases)}")
    print(f"refusal cases: {len(refusal_cases)}")


if __name__ == "__main__":
    main()
