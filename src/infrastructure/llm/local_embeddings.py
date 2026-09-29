from __future__ import annotations

import hashlib
import math
import re

from src.rag.embeddings import EmbeddingProvider


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding fallback for local development.

    It is not a replacement for a production semantic model, but keeps ingestion,
    authorization, hybrid retrieval and evaluation executable without credentials.
    """

    def __init__(self, *, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = re.findall(r"[\w.-]+", text.casefold())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self._dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
