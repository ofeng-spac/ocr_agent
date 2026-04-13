from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "data" / "structured" / "drug_catalog.json"


def normalize_name(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("，", ",").replace("、", "、")
    text = text.replace("®", "").replace("™", "")
    text = re.sub(r"\s+", "", text)
    return text


@dataclass
class DrugEntry:
    drug_id: str
    canonical_name: str
    generic_name: str
    brand_names: list[str]
    aliases: list[str]
    known_confusions: list[str]
    active: bool


class DrugCatalogVerifier:
    def __init__(self, catalog_path: Path = CATALOG_PATH):
        with catalog_path.open(encoding="utf-8") as f:
            raw_items = json.load(f)

        self.entries = [
            DrugEntry(
                drug_id=item["drug_id"],
                canonical_name=item["canonical_name"],
                generic_name=item["generic_name"],
                brand_names=item.get("brand_names", []),
                aliases=item.get("aliases", []),
                known_confusions=item.get("known_confusions", []),
                active=item.get("active", True),
            )
            for item in raw_items
            if item.get("active", True)
        ]

        self.canonical_index: dict[str, DrugEntry] = {}
        self.generic_index: dict[str, DrugEntry] = {}
        self.brand_index: dict[str, DrugEntry] = {}
        self.alias_index: dict[str, DrugEntry] = {}
        self.confusion_index: dict[str, DrugEntry] = {}

        for entry in self.entries:
            self.canonical_index[normalize_name(entry.canonical_name)] = entry
            self.generic_index[normalize_name(entry.generic_name)] = entry
            for brand in entry.brand_names:
                self.brand_index[normalize_name(brand)] = entry
            for alias in entry.aliases:
                self.alias_index[normalize_name(alias)] = entry
            for confusion in entry.known_confusions:
                self.confusion_index[normalize_name(confusion)] = entry

    def verify(self, raw_name: str, evidence_text: str = "") -> dict:
        raw_name = (raw_name or "").strip()
        evidence_text = (evidence_text or "").strip()

        if not raw_name:
            return {
                "status": "unknown",
                "match_type": "empty",
                "canonical_name": "",
                "reason": "raw_name 为空，无法校验。",
                "raw_name": raw_name,
                "evidence_text": evidence_text,
            }

        normalized = normalize_name(raw_name)

        if normalized in self.canonical_index:
            entry = self.canonical_index[normalized]
            return self._result("verified_exact", "canonical", entry, raw_name, evidence_text)

        if normalized in self.generic_index:
            entry = self.generic_index[normalized]
            return self._result("verified_exact", "generic", entry, raw_name, evidence_text)

        if normalized in self.brand_index:
            entry = self.brand_index[normalized]
            return self._result("verified_brand", "brand", entry, raw_name, evidence_text)

        if normalized in self.alias_index:
            entry = self.alias_index[normalized]
            return self._result("verified_alias", "alias", entry, raw_name, evidence_text)

        if normalized in self.confusion_index:
            entry = self.confusion_index[normalized]
            return {
                "status": "review_required",
                "match_type": "known_confusion",
                "canonical_name": "",
                "candidate_name": entry.canonical_name,
                "reason": f"命中了已知混淆项，不能直接归一到 {entry.canonical_name}。",
                "raw_name": raw_name,
                "evidence_text": evidence_text,
            }

        best = self._best_fuzzy_match(normalized)
        if best:
            score, entry = best
            return {
                "status": "review_required",
                "match_type": "fuzzy_candidate",
                "canonical_name": "",
                "candidate_name": entry.canonical_name,
                "reason": f"存在相似候选项 {entry.canonical_name}，相似度 {score:.3f}，需要人工复核。",
                "raw_name": raw_name,
                "evidence_text": evidence_text,
            }

        return {
            "status": "unknown",
            "match_type": "unmatched",
            "canonical_name": "",
            "reason": "未在当前标准目录中找到可靠匹配项。",
            "raw_name": raw_name,
            "evidence_text": evidence_text,
        }

    def _best_fuzzy_match(self, normalized_name: str) -> tuple[float, DrugEntry] | None:
        candidates: list[tuple[float, DrugEntry]] = []

        for entry in self.entries:
            for candidate in [entry.canonical_name, entry.generic_name, *entry.aliases]:
                score = SequenceMatcher(None, normalized_name, normalize_name(candidate)).ratio()
                if score >= 0.86:
                    candidates.append((score, entry))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_entry = candidates[0]
        if len(candidates) > 1 and abs(best_score - candidates[1][0]) < 0.03:
            return None
        return best_score, best_entry

    @staticmethod
    def _result(status: str, match_type: str, entry: DrugEntry, raw_name: str, evidence_text: str) -> dict:
        return {
            "status": status,
            "match_type": match_type,
            "canonical_name": entry.canonical_name,
            "candidate_name": entry.canonical_name,
            "reason": f"已匹配到标准药名 {entry.canonical_name}。",
            "raw_name": raw_name,
            "evidence_text": evidence_text,
        }

