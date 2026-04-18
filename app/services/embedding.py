from __future__ import annotations

import atexit
import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
DEFAULT_EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = DEFAULT_EMBEDDING_DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def embed_text(self, text: str) -> list[float]:
        if not (text or "").strip():
            return [0.0] * self.dimension
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text or "" for text in texts]
        vectors = self.model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )
        return [vector.tolist() for vector in vectors]

    @property
    def dimension(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None)
        if getter is None:
            getter = self.model.get_sentence_embedding_dimension
        dim = getter()
        if dim is None:
            dim = len(self.embed_text("测试"))
        return int(dim)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_embedding_dimension() -> int:
    return get_embedding_service().dimension


def get_embedding_model_name() -> str:
    return get_embedding_service().model_name


def _shutdown_embedding_service() -> None:
    try:
        get_embedding_service.cache_clear()
    except Exception:
        pass


atexit.register(_shutdown_embedding_service)
