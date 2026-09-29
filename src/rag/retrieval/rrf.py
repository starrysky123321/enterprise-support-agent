from __future__ import annotations

from dataclasses import replace

from src.rag.models import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievedChunk]], *, rrf_k: int = 60,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    scores: dict[tuple[str, str], float] = {}
    chunks: dict[tuple[str, str], RetrievedChunk] = {}
    sources: dict[tuple[str, str], set[str]] = {}
    raw_scores: dict[tuple[str, str], dict[str, float]] = {}
    for route, ranked in ranked_lists.items():
        seen: set[tuple[str, str]] = set()
        for rank, chunk in enumerate(ranked, start=1):
            key = (chunk.doc_id, chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(key, chunk)
            sources.setdefault(key, set()).add(route)
            raw_scores.setdefault(key, {})[route] = chunk.score
    ordered = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))
    if limit is not None:
        ordered = ordered[:limit]
    return [
        replace(
            chunks[key], score=scores[key],
            retrieval_sources=tuple(sorted(sources[key])), raw_scores=raw_scores[key],
        )
        for key in ordered
    ]
