from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

import chromadb

from src.rag.models import RAGChunk, RetrievedChunk, RetrievalFilter
from src.rag.vectorstore.interface import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, *, persist_dir: str, collection_name: str) -> None:
        path = Path(persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    async def upsert_chunks(
        self,
        *,
        chunks: list[RAGChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ids = [f"{chunk.doc_id}:{chunk.chunk_id}" for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            self._build_metadata(chunk)
            for chunk in chunks
        ]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=cast(Any, metadatas),
            embeddings=cast(Any, embeddings),
        )

    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        doc_id: str | None = None,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        conditions: list[dict] = []
        if doc_id:
            conditions.append({"doc_id": doc_id})
        if filters:
            if filters.allowed_doc_ids:
                conditions.append({"doc_id": {"$in": list(filters.allowed_doc_ids)}})
            if filters.product_name:
                conditions.append({"product_name": filters.product_name})
            if filters.product_version:
                conditions.append({"product_version": filters.product_version})
            if filters.document_types:
                conditions.append({"document_type": {"$in": list(filters.document_types)}})
            if filters.knowledge_space_id:
                conditions.append({"knowledge_space_id": filters.knowledge_space_id})
            if filters.department:
                conditions.append({"department": filters.department})
        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif conditions:
            where = {"$and": conditions}
        response: Any = self._collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            score = 1.0 / (1.0 + float(distance))
            results.append(
                RetrievedChunk(
                    doc_id=str((metadata or {}).get("doc_id", "")),
                    chunk_id=str((metadata or {}).get("chunk_id", "")),
                    source=str((metadata or {}).get("source", "unknown")),
                    text=str(document),
                    score=score,
                    page_number=self._parse_page_number(metadata or {}),
                    document_type=str((metadata or {}).get("document_type", "manual")),
                    product_name=self._parse_optional_string(metadata or {}, "product_name"),
                    product_version=self._parse_optional_string(metadata or {}, "product_version"),
                    department=self._parse_optional_string(metadata or {}, "department"),
                    knowledge_space_id=self._parse_optional_string(metadata or {}, "knowledge_space_id"),
                    visibility=str((metadata or {}).get("visibility", "private")),
                    retrieval_sources=("dense",),
                    raw_scores={"dense": score},
                )
            )
        return results

    async def list_chunks(self, *, doc_id: str) -> list[RAGChunk]:
        if not doc_id:
            return []

        response: Any = self._collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        ids = response.get("ids", []) or []
        documents = response.get("documents", []) or []
        metadatas = response.get("metadatas", []) or []

        chunks: list[RAGChunk] = []
        for raw_id, document, metadata in zip(ids, documents, metadatas):
            chunk_id = str((metadata or {}).get("chunk_id", "")) or str(raw_id).split(":", 1)[-1]
            chunks.append(
                RAGChunk(
                    doc_id=str((metadata or {}).get("doc_id", doc_id)),
                    chunk_id=chunk_id,
                    source=str((metadata or {}).get("source", "unknown")),
                    text=str(document),
                    page_number=self._parse_page_number(metadata or {}),
                    chunking_strategy=self._parse_optional_string(metadata or {}, "chunking_strategy"),
                    chunk_size=self._parse_optional_int((metadata or {}).get("chunk_size")),
                    chunk_overlap=self._parse_optional_int((metadata or {}).get("chunk_overlap")),
                    document_type=str((metadata or {}).get("document_type", "manual")),
                    product_name=self._parse_optional_string(metadata or {}, "product_name"),
                    product_version=self._parse_optional_string(metadata or {}, "product_version"),
                    department=self._parse_optional_string(metadata or {}, "department"),
                    knowledge_space_id=self._parse_optional_string(metadata or {}, "knowledge_space_id"),
                    visibility=str((metadata or {}).get("visibility", "private")),
                )
            )

        chunks.sort(key=self._chunk_sort_key)
        return chunks

    async def delete_by_doc_id(self, *, doc_id: str) -> None:
        if not doc_id:
            return
        self._collection.delete(where={"doc_id": doc_id})

    @staticmethod
    def _build_metadata(chunk: RAGChunk) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
        }
        if chunk.page_number is not None:
            metadata["page_number"] = int(chunk.page_number)
        if chunk.chunking_strategy is not None:
            metadata["chunking_strategy"] = chunk.chunking_strategy
        if chunk.chunk_size is not None:
            metadata["chunk_size"] = int(chunk.chunk_size)
        if chunk.chunk_overlap is not None:
            metadata["chunk_overlap"] = int(chunk.chunk_overlap)
        metadata["document_type"] = chunk.document_type
        metadata["visibility"] = chunk.visibility
        for key in ("product_name", "product_version", "department", "knowledge_space_id"):
            value = getattr(chunk, key)
            if value is not None:
                metadata[key] = value
        return metadata

    @staticmethod
    def _parse_page_number(metadata: Mapping[str, Any]) -> int | None:
        raw = metadata.get("page_number")
        return ChromaVectorStore._parse_optional_int(raw)

    @staticmethod
    def _parse_optional_int(raw: Any) -> int | None:
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_optional_string(metadata: Mapping[str, Any], key: str) -> str | None:
        raw = metadata.get(key)
        if raw is None:
            return None
        value = str(raw).strip()
        return value or None

    @staticmethod
    def _chunk_sort_key(chunk: RAGChunk) -> tuple[int, str]:
        parts = chunk.chunk_id.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (int(parts[1]), chunk.chunk_id)
        return (2**31 - 1, chunk.chunk_id)
