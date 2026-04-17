from __future__ import annotations

import hashlib
import math
import re

VECTOR_SIZE = 256


def _stable_bucket(token: str, size: int = VECTOR_SIZE) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % size


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"\s+", "", (text or "").strip().lower())
    if not text:
        return []

    tokens = []
    chars = list(text)
    tokens.extend(chars)

    for n in (2, 3):
        for i in range(len(chars) - n + 1):
            tokens.append("".join(chars[i : i + n]))

    word_tokens = re.findall(r"[a-z0-9]+", text)
    tokens.extend(word_tokens)
    return tokens


def vectorize_text(text: str, size: int = VECTOR_SIZE) -> list[float]:
    vector = [0.0] * size
    tokens = _tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        idx = _stable_bucket(token, size=size)
        vector[idx] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector
