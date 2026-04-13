#!/usr/bin/env python3
"""
Run simple regression checks for the catalog verifier.

Usage:
    python3 scripts/check_name_verifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "structured" / "verifier_cases.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.verifier import DrugCatalogVerifier


def main() -> None:
    verifier = DrugCatalogVerifier()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    passed = 0
    for idx, case in enumerate(cases, start=1):
        result = verifier.verify(case["raw_name"])
        ok = (
            result["status"] == case["expected_status"]
            and result["canonical_name"] == case["expected_canonical_name"]
        )
        mark = "PASS" if ok else "FAIL"
        print(
            f"[{idx:02d}] {mark} | raw_name={case['raw_name']} | "
            f"status={result['status']} | canonical={result['canonical_name']}"
        )
        if not ok:
            print(f"      expected_status={case['expected_status']} expected_canonical={case['expected_canonical_name']}")
            print(f"      reason={result['reason']}")
        else:
            passed += 1

    print(f"\npassed {passed}/{len(cases)} cases")
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
