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
DEFAULT_COLLECTION = "leaflet_fields_v1"


def get_client() -> QdrantClient:
    return QdrantClient(path=str(QDRANT_PATH))


def show_summary(client: QdrantClient, collection_name: str) -> None:
    cols = client.get_collections().collections
    print("collections:", [c.name for c in cols])
    if not any(c.name == collection_name for c in cols):
        print(f"collection {collection_name!r} not found")
        return

    info = client.get_collection(collection_name)
    print("collection_name:", collection_name)
    print("points_count:", info.points_count)
    print("vector_size:", info.config.params.vectors.size)
    print("distance:", info.config.params.vectors.distance)


def load_points(
    client: QdrantClient,
    collection_name: str,
    limit: int,
    drug: str | None = None,
    with_vectors: bool = False,
):
    kwargs = {
        "collection_name": collection_name,
        "limit": limit,
        "with_payload": True,
        "with_vectors": with_vectors,
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
    parser.add_argument("--with-vectors", action="store_true", help="also print stored vectors")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="collection name")
    args = parser.parse_args()

    client = get_client()
    try:
        show_summary(client, args.collection)
        if args.summary:
            return

        drug = args.drug.strip() or None
        points = load_points(
            client,
            collection_name=args.collection,
            limit=max(1, min(args.limit, 100)),
            drug=drug,
            with_vectors=args.with_vectors,
        )
        for idx, point in enumerate(points, start=1):
            print(f"--- point {idx}")
            print("id:", point.id)
            print(json.dumps(point.payload, ensure_ascii=False, indent=2))
            if args.with_vectors:
                vector = getattr(point, "vector", None)
                if vector is None:
                    print("vector: <not loaded>")
                else:
                    print("vector_dim:", len(vector))
                    print("vector:", json.dumps(vector, ensure_ascii=False))
    finally:
        client.close()


if __name__ == "__main__":
    main()
