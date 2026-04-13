from __future__ import annotations

import re


def _extract_last_value(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        matches = re.findall(pattern, text or "", flags=re.IGNORECASE)
        if matches:
            value = matches[-1].strip()
            if value:
                return value
    return ""


def normalize_uncertainty(value: str) -> str:
    value = (value or "").strip().lower()
    mapping = {
        "低": "low",
        "高": "high",
        "unknown": "unknown",
        "low": "low",
        "high": "high",
    }
    return mapping.get(value, "unknown")


def parse_recognition_result(result_text: str) -> dict:
    result_text = (result_text or "").strip()

    evidence_text = _extract_last_value(
        [
            r"关键信息摘录[：:]\s*(.+)",
            r"Key\s*Evidence[：:]\s*(.+)",
        ],
        result_text,
    )
    uncertainty_raw = _extract_last_value(
        [
            r"不确定性[：:]\s*(.+)",
            r"Uncertainty[：:]\s*(.+)",
        ],
        result_text,
    )
    raw_name = _extract_last_value(
        [
            r"药品名称[：:]\s*(.+)",
            r"Drug\s*Name[：:]\s*(.+)",
        ],
        result_text,
    )

    return {
        "raw_name": raw_name,
        "evidence_text": evidence_text,
        "uncertainty_text": uncertainty_raw,
        "uncertainty_level": normalize_uncertainty(uncertainty_raw),
        "raw_result": result_text,
    }
