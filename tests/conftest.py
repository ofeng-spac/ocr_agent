from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.verifier import DrugCatalogVerifier


@pytest.fixture(scope="module")
def verifier():
    return DrugCatalogVerifier()


@pytest.fixture
def zh_recognition_text():
    return (
        "关键信息摘录：注射用头孢噻呋钠;兽用;生产批号:20240723\n"
        "不确定性：低\n"
        "药品名称：注射用头孢噻呋钠"
    )


@pytest.fixture
def en_recognition_text():
    return (
        "Key Evidence: Omeprazole Enteric-coated Capsules\n"
        "Uncertainty: high\n"
        "Drug Name: 奥美拉唑肠溶胶囊"
    )
