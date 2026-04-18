#!/usr/bin/env python3
"""
Inspect the local Qdrant collection contents.

Examples:
    python3 scripts/inspect_qdrant.py --summary
    python3 scripts/inspect_qdrant.py --limit 5
    python3 scripts/inspect_qdrant.py --drug "盐酸小檗碱片"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


QDRANT_PATH = ROOT / "data" / "qdrant"
COLLECTION_NAME = "leaflet_fields_v1"


def get_client() -> QdrantClient:
    return QdrantClient(path=str(QDRANT_PATH))


def show_summary(client: QdrantClient) -> None:
    cols = client.get_collections().collections
    print("collections:", [c.name for c in cols])
    if not any(c.name == COLLECTION_NAME for c in cols):
        print(f"collection {COLLECTION_NAME!r} not found")
        return

    info = client.get_collection(COLLECTION_NAME)
    print("collection_name:", COLLECTION_NAME)
    print("points_count:", info.points_count)
    print("vector_size:", info.config.params.vectors.size)
    print("distance:", info.config.params.vectors.distance)


def load_points(client: QdrantClient, limit: int, drug: str | None = None):
    kwargs = {
        "collection_name": COLLECTION_NAME,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    }
    if drug:
        kwargs["scroll_filter"] = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="canonical_name",
                    match=qmodels.MatchValue(value=drug),
                )
            ]
        )

    resp = client.scroll(**kwargs)
    points = resp[0] if isinstance(resp, tuple) else resp.points
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local Qdrant collection payloads.")
    parser.add_argument("--summary", action="store_true", help="show collection summary only")
    parser.add_argument("--limit", type=int, default=10, help="number of points to print")
    parser.add_argument("--drug", type=str, default="", help="filter by canonical drug name")
    args = parser.parse_args()

    client = get_client()
    try:
        show_summary(client)
        if args.summary:
            return

        drug = args.drug.strip() or None
        points = load_points(client, limit=max(1, min(args.limit, 100)), drug=drug)
        for idx, point in enumerate(points, start=1):
            print(f"--- point {idx}")
            print("id:", point.id)
            print(json.dumps(point.payload, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
