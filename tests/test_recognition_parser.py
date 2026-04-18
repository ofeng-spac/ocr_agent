from __future__ import annotations

import pytest

from app.services.recognition_parser import normalize_uncertainty, parse_recognition_result


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("低", "low"),
        ("高", "high"),
        ("low", "low"),
        ("HIGH", "high"),
        ("中", "unknown"),
        ("", "unknown"),
    ],
)
def test_normalize_uncertainty_values(raw_value, expected):
    assert normalize_uncertainty(raw_value) == expected


def test_parse_recognition_result_extracts_all_core_fields(zh_recognition_text):
    result = parse_recognition_result(zh_recognition_text)

    assert result["raw_name"] == "注射用头孢噻呋钠"
    assert result["evidence_text"] == "注射用头孢噻呋钠;兽用;生产批号:20240723"
    assert result["uncertainty_text"] == "低"
    assert result["uncertainty_level"] == "low"
    assert result["raw_result"] == zh_recognition_text


def test_parse_recognition_result_supports_english_labels(en_recognition_text):
    result = parse_recognition_result(en_recognition_text)

    assert result["raw_name"] == "奥美拉唑肠溶胶囊"
    assert result["evidence_text"] == "Omeprazole Enteric-coated Capsules"
    assert result["uncertainty_text"] == "high"
    assert result["uncertainty_level"] == "high"


def test_parse_recognition_result_handles_missing_fields():
    result = parse_recognition_result("没有结构化标签的普通文本")

    assert result["raw_name"] == ""
    assert result["evidence_text"] == ""
    assert result["uncertainty_text"] == ""
    assert result["uncertainty_level"] == "unknown"
