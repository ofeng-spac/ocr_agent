#!/usr/bin/env python3
"""
Build local Qdrant index from structured leaflet fields.

Usage:
    python3 scripts/build_qdrant_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag import LeafletQAService


def main() -> None:
    service = LeafletQAService()
    info = service.rebuild_index()
    print(info)


if __name__ == "__main__":
    main()
