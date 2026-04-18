# RAG Evaluation Report

- Generated at: 2026-04-18T13:48:57
- Total cases: 12
- Passed cases: 12
- Overall pass rate: 100.00%
- Route accuracy: 100.00%
- Status accuracy: 100.00%
- Retrieval mode accuracy: 100.00%
- Target field accuracy: 100.00%
- Answer hit rate: 100.00%
- Citation field hit rate: 100.00%
- Citation section hit rate: 100.00%
- Refusal accuracy: 100.00%
- Avg latency: 0.99 ms

## Avg Latency by Route

- field_query: 0.77 ms
- semantic_query: 1.43 ms

## Case Results

| Case | Passed | Route | Status | Retrieval | Latency(ms) |
|:--|:--:|:--|:--|:--|--:|
| field_001 | Y | field_query | ok | qdrant_fields | 1.36 |
| field_002 | Y | field_query | ok | qdrant_fields | 0.83 |
| field_003 | Y | field_query | ok | qdrant_fields | 0.81 |
| field_004 | Y | field_query | ok | qdrant_fields | 1.22 |
| field_005 | Y | field_query | ok | qdrant_fields | 0.71 |
| semantic_001 | Y | semantic_query | ok | qdrant_chunks | 1.37 |
| semantic_002 | Y | semantic_query | ok | qdrant_chunks | 1.74 |
| semantic_003 | Y | semantic_query | ok | qdrant_chunks | 1.35 |
| semantic_004 | Y | semantic_query | ok | qdrant_chunks | 1.25 |
| refusal_001 | Y | field_query | doc_unavailable | None | 0.24 |
| refusal_002 | Y | field_query | doc_unavailable | None | 0.27 |
| refusal_003 | Y | field_query | field_unavailable | None | 0.69 |
