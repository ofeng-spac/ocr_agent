from __future__ import annotations

from app.services.embedding import get_embedding_dimension, get_embedding_service


def vectorize_text(text: str) -> list[float]:
    return get_embedding_service().embed_text(text)


def vectorize_texts(texts: list[str]) -> list[list[float]]:
    return get_embedding_service().embed_texts(texts)


def get_vector_size() -> int:
    return get_embedding_dimension()
