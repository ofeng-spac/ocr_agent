from __future__ import annotations

from app.services.verifier import DrugCatalogVerifier, normalize_name


def test_normalize_name_removes_spaces_and_trademarks():
    assert normalize_name(" 安维汀 ® ") == "安维汀"
    assert normalize_name(" 狂犬病灭活疫苗 （PV/BHK-21株） ") == "狂犬病灭活疫苗(PV/BHK-21株)"


def test_verifier_exact_match_returns_verified_exact():
    verifier = DrugCatalogVerifier()
    result = verifier.verify("注射用头孢噻呋钠")

    assert result["status"] == "verified_exact"
    assert result["canonical_name"] == "注射用头孢噻呋钠"
    assert result["match_type"] == "canonical"


def test_verifier_alias_match_returns_verified_alias():
    verifier = DrugCatalogVerifier()
    result = verifier.verify("头孢曲松钠")

    assert result["status"] == "verified_alias"
    assert result["canonical_name"] == "注射用头孢曲松钠"
    assert result["match_type"] == "alias"


def test_verifier_brand_match_returns_verified_brand():
    verifier = DrugCatalogVerifier()
    result = verifier.verify("安维汀")

    assert result["status"] == "verified_brand"
    assert result["canonical_name"] == "贝伐珠单抗注射液"
    assert result["match_type"] == "brand"


def test_verifier_known_confusion_returns_review_required():
    verifier = DrugCatalogVerifier()
    result = verifier.verify("注射用奥美拉唑钠")

    assert result["status"] == "review_required"
    assert result["match_type"] == "known_confusion"
    assert result["candidate_name"] == "奥美拉唑肠溶胶囊"


def test_verifier_unknown_name_returns_unknown():
    verifier = DrugCatalogVerifier()
    result = verifier.verify("完全不存在的药名")

    assert result["status"] == "unknown"
    assert result["match_type"] == "unmatched"
    assert result["canonical_name"] == ""


def test_verifier_empty_name_returns_unknown_empty():
    verifier = DrugCatalogVerifier()
    result = verifier.verify("")

    assert result["status"] == "unknown"
    assert result["match_type"] == "empty"
    assert "无法校验" in result["reason"]
