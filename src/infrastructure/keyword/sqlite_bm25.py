from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from src.rag.keyword import KeywordIndex
from src.rag.models import RAGChunk, RetrievedChunk, RetrievalFilter


class SQLiteBM25Index(KeywordIndex):
    """Local persistent BM25 backend using SQLite FTS5.

    The adapter is intentionally behind ``KeywordIndex`` so OpenSearch can replace it
    without changing fusion or permission code.
    """

    def __init__(self, *, database_path: str) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_key UNINDEXED, doc_id UNINDEXED, chunk_id UNINDEXED,
                source UNINDEXED, text, page_number UNINDEXED,
                document_type UNINDEXED, product_name UNINDEXED,
                product_version UNINDEXED, department UNINDEXED,
                knowledge_space_id UNINDEXED, visibility UNINDEXED,
                tokenize='unicode61'
                )"""
            )

    async def upsert_chunks(self, *, chunks: list[RAGChunk]) -> None:
        await asyncio.to_thread(self._upsert_sync, chunks)

    def _upsert_sync(self, chunks: list[RAGChunk]) -> None:
        if not chunks:
            return
        with self._connect() as conn:
            for chunk in chunks:
                key = f"{chunk.doc_id}:{chunk.chunk_id}"
                conn.execute("DELETE FROM chunks_fts WHERE chunk_key = ?", (key,))
                conn.execute(
                    """INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        key, chunk.doc_id, chunk.chunk_id, chunk.source, chunk.text,
                        chunk.page_number, chunk.document_type, chunk.product_name,
                        chunk.product_version, chunk.department,
                        chunk.knowledge_space_id, chunk.visibility,
                    ),
                )

    async def search(
        self, *, query: str, top_k: int, filters: RetrievalFilter | None = None
    ) -> list[RetrievedChunk]:
        if top_k < 1 or not query.strip():
            return []
        return await asyncio.to_thread(self._search_sync, query, top_k, filters)

    def _search_sync(
        self, query: str, top_k: int, filters: RetrievalFilter | None
    ) -> list[RetrievedChunk]:
        # Quoted tokens avoid exposing FTS operators and work for error codes such as E_CONN_42.
        tokens = [token.replace('"', '""') for token in query.split() if token.strip()]
        if not tokens:
            return []
        match_query = " OR ".join(f'"{token}"' for token in tokens)
        clauses = ["chunks_fts MATCH ?"]
        params: list[object] = [match_query]
        if filters:
            mappings = (
                ("product_name", filters.product_name),
                ("product_version", filters.product_version),
                ("knowledge_space_id", filters.knowledge_space_id),
                ("department", filters.department),
            )
            for column, value in mappings:
                if value:
                    clauses.append(f"{column} = ?")
                    params.append(value)
            if filters.document_types:
                marks = ",".join("?" for _ in filters.document_types)
                clauses.append(f"document_type IN ({marks})")
                params.extend(filters.document_types)
            if filters.allowed_doc_ids:
                marks = ",".join("?" for _ in filters.allowed_doc_ids)
                clauses.append(f"doc_id IN ({marks})")
                params.extend(filters.allowed_doc_ids)
        params.append(top_k)
        sql = f"""SELECT *, bm25(chunks_fts) AS rank FROM chunks_fts
                  WHERE {' AND '.join(clauses)} ORDER BY rank LIMIT ?"""
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        results: list[RetrievedChunk] = []
        for row in rows:
            # SQLite BM25 is lower-is-better and commonly negative; expose a monotonic score.
            raw = float(row["rank"])
            score = 1.0 / (1.0 + abs(raw))
            results.append(
                RetrievedChunk(
                    doc_id=row["doc_id"], chunk_id=row["chunk_id"], source=row["source"],
                    text=row["text"], score=score, page_number=row["page_number"],
                    document_type=row["document_type"] or "manual",
                    product_name=row["product_name"], product_version=row["product_version"],
                    department=row["department"], knowledge_space_id=row["knowledge_space_id"],
                    visibility=row["visibility"] or "private", retrieval_sources=("bm25",),
                    raw_scores={"bm25": raw},
                )
            )
        return results

    async def delete_by_doc_id(self, *, doc_id: str) -> None:
        await asyncio.to_thread(self._delete_sync, doc_id)

    def _delete_sync(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
