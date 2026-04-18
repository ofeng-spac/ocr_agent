from __future__ import annotations

from app.services.recognition_parser import normalize_uncertainty, parse_recognition_result


def test_normalize_uncertainty_supports_chinese_values():
    assert normalize_uncertainty("低") == "low"
    assert normalize_uncertainty("高") == "high"


def test_normalize_uncertainty_supports_english_values():
    assert normalize_uncertainty("low") == "low"
    assert normalize_uncertainty("HIGH") == "high"


def test_normalize_uncertainty_unknown_value_falls_back_to_unknown():
    assert normalize_uncertainty("中") == "unknown"
    assert normalize_uncertainty("") == "unknown"


def test_parse_recognition_result_extracts_all_core_fields():
    text = (
        "关键信息摘录：注射用头孢噻呋钠;兽用;生产批号:20240723\n"
        "不确定性：低\n"
        "药品名称：注射用头孢噻呋钠"
    )
    result = parse_recognition_result(text)

    assert result["raw_name"] == "注射用头孢噻呋钠"
    assert result["evidence_text"] == "注射用头孢噻呋钠;兽用;生产批号:20240723"
    assert result["uncertainty_text"] == "低"
    assert result["uncertainty_level"] == "low"
    assert result["raw_result"] == text


def test_parse_recognition_result_supports_english_labels():
    text = (
        "Key Evidence: Omeprazole Enteric-coated Capsules\n"
        "Uncertainty: high\n"
        "Drug Name: 奥美拉唑肠溶胶囊"
    )
    result = parse_recognition_result(text)

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
