from __future__ import annotations

import pytest

from app.services.verifier import normalize_name


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        (" 安维汀 ® ", "安维汀"),
        (" 狂犬病灭活疫苗 （PV/BHK-21株） ", "狂犬病灭活疫苗(PV/BHK-21株)"),
    ],
)
def test_normalize_name_values(raw_name, expected):
    assert normalize_name(raw_name) == expected


@pytest.mark.parametrize(
    ("raw_name", "expected_status", "expected_canonical_name", "expected_match_type"),
    [
        ("注射用头孢噻呋钠", "verified_exact", "注射用头孢噻呋钠", "canonical"),
        ("头孢曲松钠", "verified_alias", "注射用头孢曲松钠", "alias"),
        ("安维汀", "verified_brand", "贝伐珠单抗注射液", "brand"),
        ("完全不存在的药名", "unknown", "", "unmatched"),
        ("", "unknown", "", "empty"),
    ],
)
def test_verifier_common_paths(verifier, raw_name, expected_status, expected_canonical_name, expected_match_type):
    result = verifier.verify(raw_name)

    assert result["status"] == expected_status
    assert result["canonical_name"] == expected_canonical_name
    assert result["match_type"] == expected_match_type


def test_verifier_known_confusion_returns_review_required(verifier):
    result = verifier.verify("注射用奥美拉唑钠")

    assert result["status"] == "review_required"
    assert result["match_type"] == "known_confusion"
    assert result["candidate_name"] == "奥美拉唑肠溶胶囊"
    assert "不能直接归一" in result["reason"]
