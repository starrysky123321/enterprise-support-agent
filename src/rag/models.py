from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DocumentType = Literal[
    "manual", "api_doc", "sop", "ticket", "release_note", "faq", "config_doc"
]
DocumentVisibility = Literal["private", "department", "workspace"]


@dataclass(slots=True, frozen=True)
class RetrievalFilter:
    """Filters applied inside every retrieval backend before candidates are returned."""

    allowed_doc_ids: tuple[str, ...] = ()
    product_name: str | None = None
    product_version: str | None = None
    document_types: tuple[str, ...] = ()
    knowledge_space_id: str | None = None
    department: str | None = None
    deny_all: bool = False

    def matches(self, chunk: "RAGChunk | RetrievedChunk") -> bool:
        if self.deny_all:
            return False
        return not any(
            (
                self.allowed_doc_ids and chunk.doc_id not in self.allowed_doc_ids,
                self.product_name and chunk.product_name != self.product_name,
                self.product_version and chunk.product_version != self.product_version,
                self.document_types and chunk.document_type not in self.document_types,
                self.knowledge_space_id
                and chunk.knowledge_space_id != self.knowledge_space_id,
                self.department and chunk.department != self.department,
            )
        )


@dataclass(slots=True)
class RAGChunk:
    doc_id: str
    chunk_id: str
    source: str
    text: str
    page_number: int | None = None
    chunking_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    document_type: str = "manual"
    product_name: str | None = None
    product_version: str | None = None
    department: str | None = None
    knowledge_space_id: str | None = None
    visibility: str = "private"


@dataclass(slots=True)
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    source: str
    text: str
    score: float
    page_number: int | None = None
    document_type: str = "manual"
    product_name: str | None = None
    product_version: str | None = None
    department: str | None = None
    knowledge_space_id: str | None = None
    visibility: str = "private"
    retrieval_sources: tuple[str, ...] = ()
    raw_scores: dict[str, float] | None = None
